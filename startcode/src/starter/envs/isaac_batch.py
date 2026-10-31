"""Step several RoboDojo scenes in one Isaac process.

Eval already does this in ``take_action_batch``. The online server cannot use
``EvalEnv`` itself: that class opens a policy websocket. This attaches the same
end-effector control and camera read to a plain task object.
"""

from __future__ import annotations

from copy import deepcopy


def install_parallel_control(task, observation_cfg: dict, task_name: str) -> None:
    """Add ``get_obs_batch`` and ``take_action_batch`` to ``task``."""

    import numpy as np

    from env.observation_manager.obs_manager import ObsManager

    num_envs = int(task.num_envs)
    seeds = list(getattr(task, "env_seed_list", None) or [0] * num_envs)
    obs = ObsManager(
        obs_config=dict(observation_cfg),
        num_envs=num_envs,
        dt=float(task.dt),
        task_name=task_name,
        description_cfg={},
        seeds_per_env=seeds,
    )
    obs.initialize(task)
    task.online_obs = obs

    def get_obs_batch(env_idx_list=None):
        if env_idx_list is None:
            env_idx_list = list(range(num_envs))
        task.online_obs.render_for_capture()
        data = task.online_obs.get_obs(env_idx_list=list(env_idx_list))
        return [data[index] for index in env_idx_list]

    def take_action_batch(actions_list, env_idx_list=None):
        if env_idx_list is None:
            env_idx_list = list(range(num_envs))
        env_idx_list = list(env_idx_list)
        sequences = []
        for action, env_idx in zip(actions_list, env_idx_list):
            sequences.append(_control_sequence(task, action, int(env_idx), np))
        if not sequences:
            return
        task.robot_manager.control_manager.push(env_idx_list, sequences)
        while _controls_remain(task, env_idx_list):
            meta = task.robot_manager.control_manager.pop(env_idx_list)
            task.robot_manager.control_robot(meta_control_list=meta)
            task.sim_step(render=False)
        if hasattr(task.reward_manager, "step"):
            task.reward_manager.step(env_idx_list=env_idx_list)

    task.get_obs_batch = get_obs_batch
    task.take_action_batch = take_action_batch


def bind_eval_layouts(task, task_name: str, config_name: str, seed: int = 0):
    """Point each scene at an official eval layout before ``reset``.

    Training seeds are layout indices modulo the number of saved layouts.
    """

    from env.seed_manager.seed_manager import SeedManager

    manager = SeedManager(
        {
            "num_envs": int(task.num_envs),
            "task_name": task_name,
            "config_name": config_name,
            "seed": int(seed),
        }
    )
    manager.init_eval()
    layouts = list(manager.seed_list)
    if not layouts:
        raise RuntimeError(f"No eval layouts for {task_name} / {config_name}.")
    original = task.reset

    def reset(seed=None, options=None):
        seeds = list(seed) if isinstance(seed, (list, tuple)) else [seed]
        if len(seeds) != int(task.num_envs):
            raise ValueError(f"Expected {task.num_envs} seeds, got {len(seeds)}.")
        for index, item in enumerate(seeds):
            layout = layouts[int(item) % len(layouts)]
            task.scene_manager.layout_manager.set_saved_layout(
                index, manager.get_seed_scene_info(layout)
            )
        return original(seed=list(seeds), options=options)

    task.reset = reset
    return len(layouts)


def _controls_remain(task, env_idx_list) -> bool:
    empty = task.robot_manager.control_manager.get_empty(env_idx_list=env_idx_list)
    return len(empty) == 0


def _control_sequence(task, action: dict, env_idx: int, np):
    interval = int(task.online_obs.collect_interval)
    control = {}
    for robot in task.robot_manager.robot_list:
        if robot.type != "target":
            continue
        name = robot.arm_name.split("_")[0]
        pose_key = f"{name}_ee_pose"
        target_pose = action[pose_key]
        ik_result = task.robot_manager.solve_ik(target_pose=target_pose, env_idx=env_idx, robot=robot)
        obs_name = task.robot_manager.process_name(robot.arm_name)
        if ik_result["status"] == "Success":
            control[obs_name] = {"position": ik_result["joint_value"]}
        gripper_key = task.robot_manager.process_name(robot.gripper_name)
        if robot.ee_type == "gripper":
            val = float(np.clip(action[gripper_key][0], 0, 1))
            if robot.gripper_move["sign"] == 1:
                val = val * (robot.gripper_scale[1] - robot.gripper_scale[0]) + robot.gripper_scale[0]
            else:
                val = (1 - val) * (robot.gripper_scale[1] - robot.gripper_scale[0]) + robot.gripper_scale[0]
            control[gripper_key] = {
                "position": [
                    val,
                    val * robot.gripper_move["mimic"][1] + robot.gripper_move["mimic"][2],
                ]
            }
    if interval <= 0:
        return [deepcopy(control)]
    frames = [deepcopy(control) for _ in range(interval)]
    _interpolate(task, control, frames, env_idx, interval, np)
    return frames


def _interpolate(task, control, frames, env_idx: int, interval: int, np) -> None:
    interp_count = int(np.floor(interval * 0.8))
    for robot in task.robot_manager.robot_list:
        if robot.type != "target":
            continue
        arm_key = task.robot_manager.process_name(robot.arm_name)
        if arm_key in control:
            current = task.robot_manager.get_joint(robot, env_idx_list=[env_idx])[env_idx]
            target = np.array(control[arm_key]["position"])
            if current is not None:
                current_arr = np.array(current)
                for frame in range(interp_count):
                    alpha = (frame + 1) / (interp_count + 1)
                    frames[frame][arm_key]["position"] = ((1 - alpha) * current_arr + alpha * target).tolist()
                for frame in range(interp_count, interval):
                    frames[frame][arm_key]["position"] = target.tolist()
        gripper_key = task.robot_manager.process_name(robot.gripper_name)
        if gripper_key in control and robot.ee_type == "gripper":
            target_pos = control[gripper_key]["position"][0]
            current = task.robot_manager.get_end_effector_real_val(robot, env_idx_list=[env_idx])[env_idx][0]
            scale = robot.gripper_scale
            if current is not None:
                for frame in range(interp_count):
                    alpha = (frame + 1) / (interp_count + 1)
                    interp_pos = float(np.clip((1 - alpha) * current + alpha * target_pos, scale[0], scale[1]))
                    frames[frame][gripper_key]["position"] = [
                        interp_pos,
                        interp_pos * robot.gripper_move["mimic"][1] + robot.gripper_move["mimic"][2],
                    ]
                held = float(np.clip(target_pos, scale[0], scale[1]))
                held_vals = [held, held * robot.gripper_move["mimic"][1] + robot.gripper_move["mimic"][2]]
                for frame in range(interp_count, interval):
                    frames[frame][gripper_key]["position"] = list(held_vals)
