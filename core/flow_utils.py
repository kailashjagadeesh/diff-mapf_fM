"""
Flow matching utilities — mirrors core/utils.py for the flow matching pipeline.
Shared helpers (CLI parsing, logging, etc.) are re-exported from core.utils so
the two pipelines stay in sync without duplicating code.
"""

from core.utils import (
    parse_args,
    load_config,
    setup_problem,
    prepare_logger,
    get_device,
    train_agent,
    get_observation_dimensions,
)
from core.models.flowNet import FlowActor
from core.models.flow import FlowLearner


def create_flow_agent(training_config, device, logger, args, len_dataloader, dataset_stats=None):
    """
    Factory function for FlowLearner — analogous to create_agent() in core/utils.py.

    training_config:  dict loaded from configs/flow.json (training section)
    device:           torch device string or None (auto-detected)
    logger:           Logger instance
    args:             parsed CLI args (from parse_args())
    len_dataloader:   number of batches per epoch
    dataset_stats:    normalization stats from MultiArmDataset.stats (optional but
                      recommended — saved alongside checkpoints for planner use)
    """
    action_dim = 6
    hyperparameters = training_config["hyperparameters"]
    observation_horizon = hyperparameters["observation_horizon"]
    observation_dim = get_observation_dimensions(training_config["observations"])

    if device is None:
        device = get_device()

    def create_actor():
        actor_observation_dim = observation_dim * observation_horizon * args.num_agents
        return FlowActor(
            observation_dim=actor_observation_dim,
            action_dim=action_dim,
            network_config=training_config["network"]["actor"],
        )

    algo = training_config.get("algo", "flowNet")
    if algo != "flowNet":
        raise ValueError(
            f"create_flow_agent expects algo='flowNet', got '{algo}'. "
            "Use create_agent() from core.utils for diffusion variants."
        )

    policy_key = "single_agent_model" if args.num_agents == 1 else "dual_agent_model"
    network_fns = {"actor": create_actor}

    motion_planning_learner = FlowLearner(
        policy_key=policy_key,
        network_fns=network_fns,
        algo_config=hyperparameters,
        writer=logger,
        device=device,
        load_path=args.load,
        save_interval=args.save_interval or training_config.get("save_interval", 10),
        grad_norm=args.grad_norm,
        early_stop=args.early_stop,
        len_dataloader=len_dataloader,
        dataset_stats=dataset_stats,
    )

    return motion_planning_learner
