"""
Flow matching equivalents of Agent and ResolveDualConflict from agent_planners.py.
Drop-in replacements: same public interface, Euler ODE inference instead of DDPM.

Key differences from diffusion planners:
  - Loads "velocity_network" key from checkpoint (not "noise_predictor_network")
  - Inference: n_steps Euler ODE steps (default 10) instead of 100 DDPM steps
  - No DDPMScheduler dependency
"""

import os
import torch
import numpy as np
from itertools import islice
from collections import deque

from core.dataset import normalize_data, unnormalize_data
from core.models.diffusionNet import ConditionalUnet1D
from application.ur5_robotiq_controller import UR5RobotiqPybulletController


class FlowAgent:
    """
    Single-arm flow matching planner.
    Drop-in replacement for Agent in agent_planners.py.
    """

    def __init__(self, id, arm, parameters):
        self.id = id
        self.pybullet_id = (
            arm.id if isinstance(arm, UR5RobotiqPybulletController) else arm.body_id
        )
        self.arm = arm
        self.current_task = None
        self.parameters = parameters
        self.observation_deque = deque(maxlen=self.parameters["observation_horizon"])
        _dev = parameters.get("device", None)
        self.device = torch.device(_dev) if _dev else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_steps = self.parameters.get("n_steps", 10)

        # Reuse the same ConditionalUnet1D backbone as the diffusion planner
        self.model = ConditionalUnet1D(
            input_dim=self.parameters["action_dim"],
            global_cond_dim=self.parameters["observation_dim"]
            * self.parameters["observation_horizon"],
        )
        self.model.to(self.device)
        self.model.eval()

        ckpt = torch.load(
            self.parameters["single_agent_model"], map_location=self.device
        )
        networks = ckpt["networks"]
        self.model.load_state_dict(networks["velocity_network"])

        # Load normalization stats (.npz saved alongside checkpoint).
        # Use splitext so this works whether or not the path ends in ".pth".
        stats_path = os.path.splitext(self.parameters["single_agent_model"])[0] + ".npz"
        self.stats = np.load(stats_path, allow_pickle=True)

        # Precompute ODE timestep tensors once — reused every predict_plan call
        num_samples = self.parameters["num_samples"]
        self._dt = 1.0 / self.n_steps
        self._t_tensors = [
            torch.full(
                (num_samples,),
                i / self.n_steps * 1000.0,
                device=self.device,
                dtype=torch.float32,
            )
            for i in range(self.n_steps)
        ]
        action_stats = dict(self.stats["actions"].flatten()[0])
        observation_stats = dict(self.stats["obs"].flatten()[0])
        self.stats = dict(obs=observation_stats, action=action_stats)

    def update_deque(self, observation):
        if not self.observation_deque:
            self.observation_deque.extend(
                [observation] * self.parameters["observation_horizon"]
            )
        else:
            self.observation_deque.append(observation)

    def set_task(self, task):
        self.current_task = task

    def predict_plan(self):
        """
        Generate num_samples action trajectory candidates via Euler ODE.
        Returns: (num_samples, prediction_horizon, action_dim)
        """
        observation = np.stack(self.observation_deque)
        observation = normalize_data(observation, stats=self.stats["obs"])
        observation = torch.from_numpy(observation).to(self.device, dtype=torch.float32)

        num_samples = self.parameters["num_samples"]
        pred_horizon = self.parameters["prediction_horizon"]
        action_dim = self.parameters["action_dim"]

        with torch.no_grad():
            observation = observation.unsqueeze(0).flatten(start_dim=1)
            observation = observation.repeat(num_samples, 1)

            # Start from Gaussian noise x_0
            x = torch.randn(
                (num_samples, pred_horizon, action_dim),
                device=self.device,
            )

            # Euler ODE: integrate v_theta from t=0 to t=1
            # t_tensors and dt are precomputed at __init__ time
            for t_tensor in self._t_tensors:
                velocity = self.model(
                    sample=x, timestep=t_tensor, global_cond=observation
                )
                x = x + self._dt * velocity

        naction = x.detach().cpu().numpy()
        naction = unnormalize_data(naction, stats=self.stats["action"])
        return naction


class FlowResolveDualConflict:
    """
    Dual-arm flow matching conflict resolver.
    Drop-in replacement for ResolveDualConflict in agent_planners.py.
    """

    def __init__(self, arms, get_observation_fn, preprocess_observation_fn, parameters):
        self.arms = arms
        self.parameters = parameters
        self.get_observation_fn = get_observation_fn
        self.preprocess_observation_fn = preprocess_observation_fn
        _dev = parameters.get("device", None)
        self.device = torch.device(_dev) if _dev else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_steps = self.parameters.get("n_steps", 10)

        # Dual-arm model: observation_dim * observation_horizon * 2 (two agents)
        self.model = ConditionalUnet1D(
            input_dim=self.parameters["action_dim"],
            global_cond_dim=self.parameters["observation_dim"]
            * self.parameters["observation_horizon"]
            * 2,
        )
        self.model.to(self.device)
        self.model.eval()

        ckpt = torch.load(
            self.parameters["dual_agent_model"], map_location=self.device
        )
        networks = ckpt["networks"]
        self.model.load_state_dict(networks["velocity_network"])

        # Load normalization stats (.npz saved alongside checkpoint).
        # Use splitext so this works whether or not the path ends in ".pth".
        stats_path = os.path.splitext(self.parameters["dual_agent_model"])[0] + ".npz"
        self.stats = np.load(stats_path, allow_pickle=True)
        action_stats = dict(self.stats["actions"].flatten()[0])
        observation_stats = dict(self.stats["obs"].flatten()[0])
        self.stats = dict(obs=observation_stats, action=action_stats)
        self.conflict_cache = set()

        # Precompute ODE timestep tensors once — reused every predict_plan call
        num_samples = self.parameters["num_samples"]
        self._dt = 1.0 / self.n_steps
        self._t_tensors = [
            torch.full(
                (num_samples,),
                i / self.n_steps * 1000.0,
                device=self.device,
                dtype=torch.float32,
            )
            for i in range(self.n_steps)
        ]

    def predict_plan(self, conflict, agents_deque):
        """
        Generate num_samples coordinated action trajectories for the ego arm
        given a detected conflict with another arm.

        conflict:     (ego_agent_id, other_agent_id)
        agents_deque: deque of past observation states
        Returns:      (num_samples, prediction_horizon, action_dim)
        """
        self.conflict_cache.add(conflict)
        ego_agent, other_agent = conflict
        ego_arm = self.arms[ego_agent]
        other_arm = self.arms[other_agent]

        latest_observations = islice(
            agents_deque,
            max(0, len(agents_deque) - self.parameters["observation_horizon"]),
            len(agents_deque),
        )
        ego_observation_list = []
        for obs_state in latest_observations:
            obs = self.get_observation_fn(ego_arm, obs_state, [other_arm, ego_arm])
            obs = self.preprocess_observation_fn(obs)
            ego_observation_list.append(obs)
        ego_observation = np.concatenate(ego_observation_list, axis=-1)
        ego_observation = normalize_data(ego_observation, stats=self.stats["obs"])
        ego_observation = torch.from_numpy(ego_observation).to(
            self.device, dtype=torch.float32
        )

        num_samples = self.parameters["num_samples"]
        pred_horizon = self.parameters["prediction_horizon"]
        action_dim = self.parameters["action_dim"]

        with torch.no_grad():
            ego_observation = ego_observation.unsqueeze(0).flatten(start_dim=1)
            ego_observation = ego_observation.repeat(num_samples, 1)

            # Start from Gaussian noise x_0
            x = torch.randn(
                (num_samples, pred_horizon, action_dim),
                device=self.device,
            )

            # Euler ODE: integrate v_theta from t=0 to t=1
            # t_tensors and dt are precomputed at __init__ time
            for t_tensor in self._t_tensors:
                velocity = self.model(
                    sample=x, timestep=t_tensor, global_cond=ego_observation
                )
                x = x + self._dt * velocity

        naction = x.detach().cpu().numpy()
        naction = unnormalize_data(naction, stats=self.stats["action"])
        return naction
