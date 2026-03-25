"""
Training entry point for the flow matching pipeline.
Mirrors core/agent_manager.py — use this script to train FlowActor models.

Usage (single-arm):
    python core/flow_agent_manager.py \\
        --config configs/flow.json \\
        --offline_dataset datasets/single_agent \\
        --num_agents 1 \\
        --num_epochs 100 \\
        --name flow_single_agent

Usage (dual-arm):
    python core/flow_agent_manager.py \\
        --config configs/flow.json \\
        --offline_dataset datasets/dual_agent \\
        --num_agents 2 \\
        --num_epochs 100 \\
        --name flow_dual_agent
"""

import torch
from torch.utils.data import DataLoader

from core.dataset import MultiArmDataset
from core.flow_utils import (
    create_flow_agent,
    setup_problem,
    prepare_logger,
    train_agent,
)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    device, args, config = setup_problem()
    logger = prepare_logger(args, config)

    assert (
        args.offline_dataset is not None
    ), "[FlowAgentManager] Error: --offline_dataset is required"

    env_config = config["environment"]
    training_config = config["training"]
    hyperparameters = training_config["hyperparameters"]
    hyperparameters["num_epochs"] = args.num_epochs

    dataset = MultiArmDataset(
        dataset_path=args.offline_dataset,
        action_horizon=hyperparameters["action_horizon"],
        pred_horizon=hyperparameters["prediction_horizon"],
        obs_horizon=hyperparameters["observation_horizon"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=hyperparameters["batch_size"],
        num_workers=0,
        shuffle=False,
        pin_memory=True,
        persistent_workers=False,
    )

    motion_planning_learner = create_flow_agent(
        training_config,
        device,
        logger,
        args,
        len(dataloader),
        dataset_stats=dataset.stats,  # pass stats for .npz saving alongside checkpoints
    )
    train_agent(motion_planning_learner, dataloader)
