"""RoboDojo stack-blocks scenes, served together to the online learner.

Run this inside the simulator image, after the Isaac app has started. The
learner connects over TCP from the host conda environment. ``--num-envs``
scenes share one physics step.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Serve stack-blocks envs to the online learner.")
parser.add_argument("--task", required=True)
parser.add_argument("--env-cfg", required=True)
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--once", action="store_true")
parser.add_argument("--layout-source", choices=("random", "eval"), default="random")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--domain-randomization", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

from env.camera_manager.capture.render_sync import add_zero_delay_kit_args

add_zero_delay_kit_args(args)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from omegaconf import OmegaConf

from env.global_configs import BENCHMARK, ENV_CONFIG_PATH, ROOT_DIR
from utils.load_file import load_yaml
from utils.pipeline_utils import process_config, process_randomization, resolve_random_task_num_envs


def main() -> None:
    repo = os.environ.get("STACK_BLOCKS_ROOT", "/stack-blocks")
    src = os.path.join(repo, "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from starter.envs.isaac_batch import bind_eval_layouts, install_parallel_control
    from starter.envs.remote import serve
    from starter.envs.stack_blocks import ParallelAdapter

    task_registry = importlib.import_module(f"task.{BENCHMARK}.task_registry")
    task_name, task_class = task_registry.load_task_class(args.task)
    eval_cfg = load_yaml(os.path.join(ENV_CONFIG_PATH, args.env_cfg + ".yml"))
    env_cfg = OmegaConf.create(
        {
            "sim": load_yaml(os.path.join(ENV_CONFIG_PATH, "sim", eval_cfg["config"]["sim"] + ".yml")),
            "scene": load_yaml(os.path.join(ENV_CONFIG_PATH, "scene", eval_cfg["config"]["scene"] + ".yml")),
            "camera": load_yaml(os.path.join(ENV_CONFIG_PATH, "camera", eval_cfg["config"]["camera"] + ".yml")),
            "robot": load_yaml(os.path.join(ENV_CONFIG_PATH, "robot", eval_cfg["config"]["robot"] + ".yml")),
            "task_env": load_yaml(task_registry.task_config_path(os.path.join(ROOT_DIR, "task", BENCHMARK, "config"), task_name)),
        }
    )
    requested = int(args.num_envs)
    if requested < 1:
        raise SystemExit("num_envs must be positive")
    width = resolve_random_task_num_envs(task_name, requested, env_cfg.sim)
    if width != requested:
        raise SystemExit(
            f"num_envs {requested} is above the limit {width} for {task_name}. "
            "Lower num_envs in the training config."
        )
    OmegaConf.update(env_cfg, "sim.scene.num_envs", width, force_add=True)
    # process_randomization reads eval_cfg. The online server is not an eval client,
    # so domain randomization stays off unless that block is filled in later.
    randomization = {key: bool(args.domain_randomization) for key in
                     ("random_table", "random_ground", "random_background")}
    OmegaConf.update(env_cfg, "eval_cfg", {"domain_randomization": randomization}, force_add=True)
    env_cfg = process_randomization(env_cfg)
    env_cfg, _eval_num = process_config(env_cfg, task_name=task_name)
    # Keep only the permitted head-camera capture for online learning.
    for camera in ("cam_left_wrist", "cam_right_wrist"):
        OmegaConf.update(env_cfg, f"camera.annotator.{camera}.enabled", False, force_add=True)
    OmegaConf.update(env_cfg, "sim.scene.num_envs", width, force_add=True)
    # Official eval sets this from observation.collect_freq. Without it,
    # CameraManager treats default_frequency as a local and crashes while
    # creating the wrist and head cameras.
    observation = eval_cfg.get("observation") or {}
    OmegaConf.update(
        env_cfg,
        "camera.default_frequency",
        observation.get("collect_freq", 0),
        force_add=True,
    )
    env_cfg.sim.seed = [args.seed + index for index in range(width)]
    OmegaConf.update(env_cfg, "sim.device", args.device, force_add=True)
    task = task_class(env_cfg, simulation_app)
    install_parallel_control(task, eval_cfg.get("observation", {}), task_name)
    if args.layout_source == "eval":
        bind_eval_layouts(task, task_name, args.env_cfg, seed=args.seed)
    print(f"[env] task {task_name} ready num_envs {width} layouts={args.layout_source}", flush=True)
    try:
        serve(ParallelAdapter(task), args.host, args.port, once=args.once)
    finally:
        task.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
