import os
import concurrent.futures
from pathlib import Path
import time
import pickle
import argparse
import numpy as np
import pybullet as p
from tqdm import tqdm
from math import degrees
from collections import namedtuple
from itertools import combinations
from dataclasses import dataclass, asdict
from core.recorder import PybulletRecorder
from random import seed as random_seed, uniform, choice

import application.pybullet_utils as pu
from application.executer import Executer
from application.ur5_robotiq_controller import UR5RobotiqPybulletController
from application.misc_utils import (
    OtherObject,
    TargetObject,
    write_csv_line,
    suppress_stdout,
    configure_pybullet,
)
from application.task import (
    PolicyTask,
    SetTargetTask,
    ControlArmTask,
    OpenGripperTask,
    CloseGripperTask,
    AttachToGripperTask,
    DetachToGripperTask,
    CartesianControlTask,
    UR5AsyncTaskRunner as TaskRunner,
)

ConfigInfo = namedtuple(
    "ConfigInfo", ["joint_config", "ee_pose", "pos_distance", "quat_distance"]
)

d_robot = 0.6
d_target = 1.0
robot_base_poses = [
    [[d_robot, d_robot, 0.01], [0, 0, 0, 1]],
    [[d_robot, -d_robot, 0.01], [0, 0, 0, 1]],
    [[-d_robot, d_robot, 0.01], pu.quaternion_from_euler([0, 0, np.pi])],
    [[-d_robot, -d_robot, 0.01], pu.quaternion_from_euler([0, 0, np.pi])],
]
dump_positions = [
    [0.2, 0.2, 0.5],
    [0.2, -0.2, 0.5],
    [-0.2, 0.2, 0.5],
    [-0.2, -0.2, 0.5],
]
target_xys = [[d_target, 0], [0, -d_target], [0, d_target], [-d_target, 0]]


@dataclass
class Parameters:
    search: str = "cbs"
    timeout: int = 600
    action_dim: int = 6
    num_samples: int = 10
    n_timesteps: int = 100   # used by diffusion backbone
    n_steps: int = 10        # used by flow matching backbone
    backbone: str = "diffusion"  # "diffusion" | "flow"
    cbs_strategy: str = "standard"  # "standard" | "cardinal"
    action_horizon: int = 1
    observation_dim: int = 57
    prediction_horizon: int = 16
    observation_horizon: int = 2
    dual_agent_model: str = "runs/plain_diffusion/mini_custom_diffusion_2.pth"
    single_agent_model: str = "runs/plain_diffusion/mini_custom_diffusion_1.pth"
    device: str = "cpu"


def set_initial_random_configs(ur5s, randomization_magnitude=0.4):
    above_threshold = 0.2
    while True:
        for ur5 in ur5s:
            ur5.reset()
            curr = np.array(ur5.get_arm_joint_values())
            ur5.set_arm_joints(
                curr
                + np.array(
                    [
                        uniform(-randomization_magnitude, randomization_magnitude)
                        for _ in range(6)
                    ]
                )
            )

        if not any([ur5.check_collision_with_info()[0] for ur5 in ur5s]) and all(
            [ur5.get_eef_pose()[0][2] > above_threshold for ur5 in ur5s]
        ):
            break


def prepare_task_runners(ur5s, targets):
    task_runners = []
    for i, (ur5, robot_targets) in enumerate(zip(ur5s, targets)):
        folder_path = os.path.join("application/tasks", "robot" + str(i))
        grasps = [
            pickle.load(open(os.path.join(folder_path, t.object_name, "grasp.p"), "rb"))
            for t in robot_targets
        ]
        grasp_configs = [
            pickle.load(
                open(os.path.join(folder_path, t.object_name, "grasp_config.p"), "rb")
            )
            for t in robot_targets
        ]
        dump_configs = [
            pickle.load(
                open(os.path.join(folder_path, t.object_name, "dump_config.p"), "rb")
            )
            for t in robot_targets
        ]
        dump_jvs = [
            pickle.load(
                open(os.path.join(folder_path, t.object_name, "dump_jv.p"), "rb")
            )
            for t in robot_targets
        ]
        initial_poses = [
            pickle.load(
                open(os.path.join(folder_path, t.object_name, "initial_pose.p"), "rb")
            )
            for t in robot_targets
        ]

        ur5_tasks = []
        for target, grasp, grasp_config, dump_config, dump_jv, initial_pose in zip(
            robot_targets, grasps, grasp_configs, dump_configs, dump_jvs, initial_poses
        ):
            gcee_pose = grasp_config.ee_pose
            if gcee_pose[0][2] < 0.2:
                gcee_pose[0][2] += 0.2
            dcee_pose = dump_config.ee_pose
            if dcee_pose[0][2] < 0.2:
                dcee_pose[0][2] += 0.2
            ur5_single_target_task = [
                SetTargetTask(ur5=ur5, target_id=target.id, initial_pose=initial_pose),
                PolicyTask(
                    ur5=ur5,
                    target_pose=gcee_pose,
                    position_tolerance=0.2,
                    orientation_tolerance=0.5,
                    # visualize=True,
                ),
                # ControlArmTask(ur5=ur5, target_config=grasp_config.joint_config),
                ControlArmTask(ur5=ur5, target_config=grasp.pre_grasp_jv),
                ControlArmTask(ur5=ur5, target_config=grasp.grasp_jv),
                CloseGripperTask(ur5=ur5),
                AttachToGripperTask(ur5=ur5, target_id=target.id),
                CartesianControlTask(ur5=ur5, axis="z", value=0.3),
                PolicyTask(
                    ur5=ur5,
                    target_pose=dcee_pose,
                    position_tolerance=0.2,
                    orientation_tolerance=0.5,
                    # visualize=True,
                ),
                ControlArmTask(ur5=ur5, target_config=dump_jv),
                CartesianControlTask(ur5=ur5, axis="z", value=-0.05),
                DetachToGripperTask(ur5=ur5),
                OpenGripperTask(ur5=ur5),
                CartesianControlTask(ur5=ur5, axis="z", value=0.05),
            ]
            ur5_tasks += ur5_single_target_task
        task_runners.append(
            TaskRunner(
                ur5=ur5,
                tasks=ur5_tasks + [ControlArmTask(ur5=ur5, target_config=ur5.RESET)],
            )
        )
    return task_runners


def create_target_xyss(d_target, delta, num_objects):
    result = []
    result.append([[d_target + delta * i, 0] for i in range(num_objects)])
    result.append([[0, -d_target - delta * i] for i in range(num_objects)])
    result.append([[0, d_target + delta * i] for i in range(num_objects)])
    result.append([[-d_target - delta * i, 0] for i in range(num_objects)])
    return result


def create_scene(
    random=True, target_object_names=None, initial_configs=None, num_targets_per_arm=1
):
    with suppress_stdout():
        _ = p.loadURDF("plane.urdf")
    plastic_bin = OtherObject(
        "application/assets/tote/tote.urdf", initial_pose=[[0, 0, 0], [0, 0, 0, 1]]
    )

    if target_object_names is None:
        pickable_objects = [
            os.listdir(os.path.join("application/tasks", robot))
            for robot in ["robot0", "robot1", "robot2", "robot3"]
        ]
        target_object_names = [
            choice(list((combinations(po, num_targets_per_arm))))
            for po in pickable_objects
        ]

    target_xyss = create_target_xyss(
        d_target, delta=0.2, num_objects=num_targets_per_arm
    )

    ur5s = [UR5RobotiqPybulletController(base_pose=rb) for rb in robot_base_poses]
    [
        p.addUserDebugText(str(i), [pose[0][0], pose[0][1], 0.6], (1, 0, 0), textSize=2)
        for i, pose in enumerate(robot_base_poses)
    ]
    targets = []
    for i, (names, target_xys) in enumerate(zip(target_object_names, target_xyss)):
        initial_poses = [
            pickle.load(
                open(os.path.join("application/tasks", "robot" + str(i), n, "initial_pose.p"), "rb")
            )
            for n in names
        ]
        rpys = [
            [degrees(e) for e in pu.euler_from_quaternion(pose[1])]
            for pose in initial_poses
        ]
        ur5_targets = [
            TargetObject(n, target_xy, 0, rpy)
            for n, target_xy, rpy in zip(names, target_xys, rpys)
        ]
        targets.append(ur5_targets)

    if random and initial_configs is None:
        set_initial_random_configs(ur5s)
    else:
        assert initial_configs is not None
        for ur5, c in zip(ur5s, initial_configs):
            ur5.set_arm_joints(c)

    pu.step(3)
    return ur5s, targets, plastic_bin, dump_positions


def check_success(targets, bin):
    bbox = p.getAABB(bin.id, -1)
    target_in_bins = []
    for tt in targets:
        for t in tt:
            target_pose = t.get_pose()
            target_in_bin = (
                bbox[0][0] < target_pose[0][0] < bbox[1][0]
                and bbox[0][1] < target_pose[0][1] < bbox[1][1]
            )
            target_in_bins.append(target_in_bin)
    return all(target_in_bins)


def demo_with_seed(seed, result_dir, recorder_dir, parameters, rendering=False):
    configure_pybullet(rendering=rendering, debug=False)
    random_seed(seed)

    benchmark_dir = os.path.join("application/tasks", "benchmark", str(seed))
    if not os.path.exists(benchmark_dir):
        raise ValueError("Benchmark dataset does not exist!")

    target_object_names = pickle.load(
        open(os.path.join(benchmark_dir, "target_object_names.p"), "rb")
    )
    intitial_configs = pickle.load(
        open(os.path.join(benchmark_dir, "intitial_configs.p"), "rb")
    )

    ur5s, targets, plastic_bin, _ = create_scene(
        target_object_names=target_object_names, initial_configs=intitial_configs
    )

    recorder = PybulletRecorder()
    for ur5 in ur5s:
        recorder.register_object(
            body_id=ur5.id, urdf_path="application/assets/ur5/ur5_robotiq.urdf", color=ur5.color
        )
    for ur5_targets in targets:
        for target in ur5_targets:
            recorder.register_object(body_id=target.id, urdf_path=target.urdf_path)
    recorder.register_object(body_id=plastic_bin.id, urdf_path=plastic_bin.urdf)

    task_runners = prepare_task_runners(ur5s=ur5s, targets=targets)
    executer = Executer(
        task_runners=task_runners,
        recorder=recorder,
        recorder_dir=recorder_dir,
        parameters=asdict(parameters),
    )
    executer_success, step_count, info = executer.run()
    success = False if not executer_success else check_success(targets, plastic_bin)
    p.disconnect()

    pm = executer.planning_metrics
    result = {
        "info": info,
        "experiment": seed,
        "success": success,
        "limit": executer.limit,
        "step_count": step_count,
        "simulation_output_path": executer.simulation_output_path,
        "total_planning_time": pm["total_planning_time"],
        "num_planning_calls": pm["num_planning_calls"],
        "avg_planning_time": (
            pm["total_planning_time"] / pm["num_planning_calls"]
            if pm["num_planning_calls"] > 0 else 0.0
        ),
        "total_cbs_expanded": pm["total_cbs_expanded"],
        "total_cbs_rebranch": pm["total_cbs_rebranch"],
        "total_cbs_repair": pm["total_cbs_repair"],
    }
    # Skip writing the CSV file here to prevent race conditions during multiprocessing
    # write_csv_line(result_filepath, result)
    return result, executer.simulation_output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action_dim", type=int, default=6, help="Action dimension")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples")
    parser.add_argument(
        "--n_timesteps", type=int, default=100, help="Number of diffusion denoising steps (diffusion backbone)"
    )
    parser.add_argument(
        "--n_steps", type=int, default=10, help="Number of ODE integration steps (flow matching backbone)"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="diffusion",
        choices=["diffusion", "flow"],
        help="Policy backbone: 'diffusion' (DDPM) or 'flow' (flow matching)",
    )
    parser.add_argument(
        "--cbs_strategy",
        type=str,
        default="standard",
        choices=["standard", "cardinal"],
        help="CBS conflict selection strategy: 'standard' (first-found) or 'cardinal' (cardinality-aware)",
    )
    parser.add_argument("--action_horizon", type=int, default=1, help="Action horizon")
    parser.add_argument(
        "--observation_dim", type=int, default=57, help="Observation dimension"
    )
    parser.add_argument(
        "--prediction_horizon", type=int, default=16, help="Prediction horizon"
    )
    parser.add_argument(
        "--observation_horizon", type=int, default=2, help="Observation horizon"
    )
    parser.add_argument(
        "--dual_agent_model",
        type=str,
        default="runs/plain_diffusion/mini_custom_diffusion_2.pth",
        help="Dual agent model path",
    )
    parser.add_argument(
        "--single_agent_model",
        type=str,
        default="runs/plain_diffusion/mini_custom_diffusion_1.pth",
        help="Single agent model path",
    )
    parser.add_argument(
        "--search",
        type=str,
        default="cbs",
        choices=["cbs"],
        help="Search algorithm to use",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout for the application (in seconds)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Enable PyBullet GUI rendering",
    )
    parser.add_argument(
        "--num_experiments",
        type=int,
        default=100,
        help="Number of experiments to run",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers for experiments",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device for model inference in workers (default: cpu to avoid GPU OOM)",
    )
    args = parser.parse_args()

    parameters = Parameters(
        search=args.search,
        timeout=args.timeout,
        action_dim=args.action_dim,
        num_samples=args.num_samples,
        n_timesteps=args.n_timesteps,
        n_steps=args.n_steps,
        backbone=args.backbone,
        cbs_strategy=args.cbs_strategy,
        action_horizon=args.action_horizon,
        observation_dim=args.observation_dim,
        dual_agent_model=args.dual_agent_model,
        single_agent_model=args.single_agent_model,
        prediction_horizon=args.prediction_horizon,
        observation_horizon=args.observation_horizon,
        device=args.device,
    )

    parent_dir = Path(parameters.dual_agent_model).parent
    timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    result_dir = parent_dir.as_posix() + "/results" + "_" + timestr
    recorder_dir = parent_dir.as_posix() + "/simulation" + "_" + timestr
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(recorder_dir):
        os.makedirs(recorder_dir)

    num_valids = 0
    num_successes = 0
    num_experiments = args.num_experiments
    max_workers = args.num_workers

    # Note: We must restrict concurrent execution from trying to use PyBullet GUI
    # simultaneously if rendering is enabled. ProcessPoolExecutor can crash easily
    # with multiple GUI instances.
    if args.render and max_workers > 1:
        print("Warning: Setting max_workers=1 because --render is active.")
        max_workers = 1

    failed_seeds = set()

    def _record_result(result, simulation_pkl_path):
        write_csv_line(os.path.join(result_dir, "results.csv"), result)
        info = result["info"]
        if info is not None:
            robot_collided = info[1] == "ur5_robotiq" and info[2] == "ur5_robotiq"
            robot_collided = robot_collided or info[2] == "plane"
            is_valid = robot_collided or result["success"]
        else:
            is_valid = True
        if is_valid:
            print(f'\nExperiment {result["experiment"]}:')
            print(f'\tSuccess: {result["success"]}')
            print(f"\tPath: {simulation_pkl_path}")
        return int(is_valid), int(result["success"])

    with tqdm(
        total=num_experiments, dynamic_ncols=True, desc="Running Application"
    ) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    demo_with_seed,
                    seed=experiment_id,
                    result_dir=result_dir,
                    recorder_dir=recorder_dir,
                    parameters=parameters,
                    rendering=args.render
                ): experiment_id
                for experiment_id in range(num_experiments)
            }

            for future in concurrent.futures.as_completed(futures):
                experiment_id = futures[future]
                try:
                    result, simulation_pkl_path = future.result()
                    v, s = _record_result(result, simulation_pkl_path)
                    num_valids += v
                    num_successes += s
                except Exception as exc:
                    print(f'\nExperiment {experiment_id} failed: {exc} — will retry')
                    failed_seeds.add(experiment_id)

                pbar.update(1)
                if num_valids > 0:
                    pbar.set_description(f"Success Rate: {num_successes/num_valids:.04f}")

    if failed_seeds:
        print(f"\nRetrying {len(failed_seeds)} failed experiments sequentially...")
        for seed in sorted(failed_seeds):
            try:
                result, simulation_pkl_path = demo_with_seed(
                    seed=seed,
                    result_dir=result_dir,
                    recorder_dir=recorder_dir,
                    parameters=parameters,
                    rendering=args.render,
                )
                v, s = _record_result(result, simulation_pkl_path)
                num_valids += v
                num_successes += s
                print(f"Retry experiment {seed}: ok")
            except Exception as exc:
                print(f"Retry experiment {seed} failed again (skipping): {exc}")

    if num_valids > 0:
        print(f"Success Rate: {num_successes/num_valids:.04f}")
    else:
        print("Success Rate: N/A (0 valid experiments)")
    print(f"Total Valid: {num_valids}")
    print(f"Total Success: {num_successes}")
    print(f"Total Experiments: {num_experiments}")
