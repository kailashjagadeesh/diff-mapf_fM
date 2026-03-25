import torch
import torch.nn as nn
from diffusers.training_utils import EMAModel

from core.models.diffusionNet import ConditionalUnet1D


class FlowActor(object):
    """
    Flow matching actor that learns a velocity field v_theta(x_t, t, cond)
    using the Rectified Flow / Linear Conditional Flow Matching objective.

    Training:
        x_0 ~ N(0, I)  (noise)
        x_1 = action   (data)
        x_t = (1 - t) * x_0 + t * x_1,  t ~ U[0, 1]
        loss = ||v_theta(x_t, t, cond) - (x_1 - x_0)||^2

    Inference (Euler ODE, n_steps steps):
        x <- x_0 ~ N(0, I)
        for i in range(n_steps):
            t = i / n_steps
            x <- x + (1 / n_steps) * v_theta(x, t, cond)
        return x
    """

    def __init__(self, observation_dim, action_dim, network_config):
        self.action_dim = action_dim
        self.network_config = network_config
        self.observation_dim = observation_dim
        self.n_steps = network_config.get("n_steps", 10)

        # Reuse the same ConditionalUnet1D backbone as DiffusionActor.
        # The network predicts the velocity field instead of noise.
        self.velocity_network = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=observation_dim,
            down_dims=network_config["unet_layers"],
            diffusion_step_embed_dim=network_config["time_dim"],
        )

        self.ema_model = EMAModel(
            parameters=self.velocity_network.parameters(), power=0.75
        )

    def loss(self, observations, actions):
        """
        Compute rectified flow loss.

        observations: (B, observation_dim)
        actions:      (B, prediction_horizon, action_dim)
        """
        device = actions.device
        B = actions.shape[0]

        # Sample source distribution x_0 ~ N(0, I)
        x_0 = torch.randn_like(actions, device=device, dtype=actions.dtype)

        # Sample time t ~ U[0, 1]
        t = torch.rand(B, device=device, dtype=actions.dtype)
        t_broadcast = t.view(B, 1, 1)

        # Linearly interpolate: x_t = (1-t)*x_0 + t*x_1
        x_t = (1.0 - t_broadcast) * x_0 + t_broadcast * actions

        # Target velocity is constant along the path: u = x_1 - x_0
        target_velocity = actions - x_0

        # Scale t to [0, 1000] to match the sinusoidal embedding's expected range
        # (ConditionalUnet1D was designed for integer timesteps 0..num_train_timesteps)
        t_scaled = t * 1000.0

        velocity_pred = self.velocity_network(
            x_t, t_scaled, global_cond=observations
        )

        return nn.functional.mse_loss(velocity_pred, target_velocity)

    @torch.no_grad()
    def sample_action(self, noise, observations):
        """
        Generate actions via Euler ODE integration.

        noise:        (B, prediction_horizon, action_dim) — starting sample x_0
        observations: (B, observation_dim)
        returns:      (B, prediction_horizon, action_dim)
        """
        x = noise
        dt = 1.0 / self.n_steps
        device = noise.device
        B = noise.shape[0]

        for i in range(self.n_steps):
            t_val = i / self.n_steps
            # Scale to [0, 1000] to match sinusoidal embedding range
            t_scaled = torch.full(
                (B,), t_val * 1000.0, device=device, dtype=noise.dtype
            )
            velocity = self.velocity_network(x, t_scaled, global_cond=observations)
            x = x + dt * velocity  # Euler step

        return x
