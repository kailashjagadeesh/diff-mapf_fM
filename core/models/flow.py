import os
import numpy as np
import torch
from tqdm import tqdm
from time import time
from torch.optim import AdamW
from diffusers.optimization import get_scheduler

from core.models.utils import EarlyStopping
from core.models.basePolicyAlgo import BasePolicyAlgo

DEBUG_LOG = False


class FlowLearner(BasePolicyAlgo):
    """
    Behavioral cloning trainer for the flow matching policy.
    Mirrors DiffusionLearner, replacing the DDPM-based actor with FlowActor.
    """

    def __init__(
        self,
        policy_key,
        network_fns,
        algo_config,
        writer,
        device="cuda",
        grad_norm=0.0,
        load_path=None,
        save_interval=1,
        early_stop=False,
        len_dataloader=None,
        dataset_stats=None,
    ):
        super().__init__(
            policy_key=policy_key,
            writer=writer,
            load_path=load_path,
            save_interval=save_interval,
            device=device,
        )

        self.step = 0
        self.grad_norm = grad_norm
        self.early_stop = early_stop
        self.network_fns = network_fns
        self.dataset_stats = dataset_stats  # saved alongside checkpoint for planner use

        self.actor_lr = algo_config["pi_lr"]
        self.lr_decay = algo_config["lr_decay"]

        self.save_interval = save_interval
        self.batch_size = algo_config["batch_size"]
        self.num_epochs = algo_config["num_epochs"]

        self.prediction_horizon = algo_config["prediction_horizon"]
        self.observation_horizon = algo_config["observation_horizon"]

        assert len_dataloader is not None
        self.len_dataloader = len_dataloader

        self.setup(load_path)

    def setup(self, load_path):
        self.stats = {"update_steps": 0}

        self.actor = self.network_fns["actor"]()
        self.actor.ema_model.to(self.device)
        self.actor.velocity_network.to(self.device)

        self.optimizer = AdamW(
            self.actor.velocity_network.parameters(),
            lr=self.actor_lr,
            weight_decay=1e-6,
        )

        if self.lr_decay:
            self.lr_scheduler = get_scheduler(
                name="cosine",
                optimizer=self.optimizer,
                num_warmup_steps=1000,
                num_training_steps=self.len_dataloader * self.num_epochs,
            )

        if load_path is not None:
            print("[FlowLearner] Loading model from {}".format(load_path))
            checkpoint = torch.load(load_path, map_location=self.device)
            networks = checkpoint["networks"]

            self.actor.velocity_network.load_state_dict(networks["velocity_network"])
            self.actor.ema_model.load_state_dict(networks["ema_model"])
            self.optimizer.load_state_dict(networks["optimizer"])
            self.lr_scheduler.load_state_dict(networks["lr_scheduler"])

            self.stats["update_steps"] = checkpoint["stats"]["update_steps"]

            if "success_rate" in checkpoint["stats"]:
                self.stats["success_rate"] = checkpoint["stats"]["success_rate"]

            print(
                "[FlowLearner] Continue training from update steps: {}".format(
                    self.stats["update_steps"]
                )
            )

        self.update_time = time()

    def log_scalars(self, scalars, timestamp):
        self.writer.add_scalars(scalars, timestamp)

    def flatten_obs(self, obs, horizon):
        return obs[:, :horizon, :].flatten(start_dim=1)

    def train_batch(self, batch):
        observations = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)

        observations_cond = self.flatten_obs(observations, self.observation_horizon)

        fm_loss = self.actor.loss(observations_cond, actions)
        self.optimizer.zero_grad()
        fm_loss.backward()
        if self.grad_norm > 0.0:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor.velocity_network.parameters(), max_norm=1.0
            )
        self.optimizer.step()

        self.lr_scheduler.step()
        self.actor.ema_model.step(self.actor.velocity_network.parameters())

        result = {
            "Training/FM_Loss": fm_loss.item(),
        }
        if self.grad_norm > 0.0:
            result["Training/Actor_Grad_Norm"] = actor_grad_norm.item()

        return result

    def train(self, dataloader, return_stats=False):
        scalar_summaries = {
            "Training/FM_Loss": 0.0,
        }
        if self.grad_norm > 0.0:
            scalar_summaries["Training/Actor_Grad_Norm"] = 0.0

        fm_loss = np.inf
        stop_check = EarlyStopping(tolerance=1.0, min_delta=0.0)

        with tqdm(range(self.num_epochs), desc="Epoch") as tglobal:
            for _ in tglobal:
                flow_loss_list = list()
                with tqdm(dataloader, desc="Batch", leave=False) as tepoch:
                    for nbatch in tepoch:
                        scalars = self.train_batch(nbatch)
                        for key, value in scalars.items():
                            scalar_summaries[key] += value
                        flow_loss_list.append(scalars["Training/FM_Loss"])
                        tepoch.set_postfix(
                            {
                                "FM Loss": scalars["Training/FM_Loss"],
                            }
                        )

                    self.step += 1
                    self.stats["update_steps"] += 1
                    for key in scalar_summaries:
                        scalar_summaries[key] /= self.len_dataloader
                    self.log_scalars(
                        scalar_summaries, timestamp=self.stats["update_steps"]
                    )

                    if self.stats["update_steps"] % self.save_interval == 0:
                        self.save()

                    if DEBUG_LOG:
                        output = "\r[FlowLearner] Update Steps: {}".format(
                            self.stats["update_steps"]
                        )
                        output += " | FM: {:.4f} | Time: {:2f}".format(
                            scalar_summaries["Training/FM_Loss"],
                            float(time() - self.update_time),
                        )
                        self.update_time = time()
                        print(output)

                    if self.early_stop:
                        if stop_check(fm_loss, np.mean(flow_loss_list)):
                            print(
                                "[FlowLearner] Early stopping at update steps: {}".format(
                                    self.stats["update_steps"]
                                )
                            )
                            break
                    fm_loss = np.mean(flow_loss_list)

        if return_stats:
            return scalar_summaries

    def get_state_dicts_to_save(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "ema_model": self.actor.ema_model.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "velocity_network": self.actor.velocity_network.state_dict(),
        }

    def save(self, eval=None, best_model_params=None):
        """
        Override BasePolicyAlgo.save() to also persist dataset normalization
        stats as a .npz file alongside the checkpoint. This is required by the
        FlowAgent / FlowResolveDualConflict planners at inference time.
        """
        super().save(eval=eval, best_model_params=best_model_params)

        # Also save normalization stats so the planner can load them
        if self.dataset_stats is not None and self.logdir is not None:
            if eval is not None:
                stats_path = "{}/best_ckpt_{}.npz".format(
                    self.logdir, self.policy_key
                )
            else:
                stats_path = "{}/ckpt_{}_{:05d}.npz".format(
                    self.logdir,
                    self.policy_key,
                    int(self.stats["update_steps"] / self.save_interval),
                )
            np.savez(
                stats_path,
                obs=self.dataset_stats["obs"],
                actions=self.dataset_stats["actions"],
            )
