"""In-process adapter around a RoboDojo stack-blocks task.

The adapter does not edit the task reward. ``step`` returns the score produced
by ``reward_manager.get_score`` (0, 15, or 100). Block poses are available only
through ``get_privileged_state``.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from starter.interfaces import (
    BLOCK_LABELS,
    HEAD_CAMERA,
    POLICY_STATE_KEYS,
    SUCCESS_SCORE,
    STEP_LIMIT,
    assert_no_privileged,
    pack_ee_action,
    unpack_ee_action,
)


def _copy_image(value: Any) -> Any:
    if hasattr(value, "copy"):
        return value.copy()
    if isinstance(value, list):
        return [(_copy_image(item) if isinstance(item, list) else item) for item in value]
    return value


def _state_mapping(raw: Mapping) -> Mapping:
    state = raw.get("state", {})
    if not isinstance(state, Mapping):
        raise TypeError("Observation state must be a mapping.")
    return state


def _ensure_active_flags(task) -> None:
    """Give a plain task the per-scene flags that eval reward checks read.

    ``EvalEnv`` already maintains ``success`` and ``end_flag``. ``True`` means
    the scene is still scored. ``reward_manager`` sets an entry to ``False``
    after that scene fails a check, so each reset starts the list again.
    """

    if hasattr(task, "end_flag"):
        return
    num_envs = int(getattr(task, "num_envs", 1))
    task.success = [True] * num_envs


def _seed_gripper_state(task) -> None:
    """Restore the gripper command cache that ``task.reset`` clears.

    Official eval calls ``set_robot_init_state`` after reset. Gripper widths
    in the observation are read from that cache, so the first training frame
    otherwise has poses but no ``left_ee_joint_state`` or ``right_ee_joint_state``.
    """

    _ensure_active_flags(task)
    robot_manager = getattr(task, "robot_manager", None)
    if robot_manager is None or not hasattr(robot_manager, "set_robot_init_state"):
        return
    if hasattr(robot_manager, "set_origin_endpose"):
        robot_manager.set_origin_endpose()
    robot_manager.set_robot_init_state()
    reward_manager = getattr(task, "reward_manager", None)
    if reward_manager is None:
        return
    # EvalEnv.init_state reads env.success, which a plain task does not have.
    # The score check all_robot_back_to_origin still needs the end-pose cache
    # that init_state would have written for every active scene.
    if hasattr(task, "success") and hasattr(reward_manager, "init_state"):
        reward_manager.init_state()
        return
    _record_origin_endpose(task, reward_manager)


def _record_origin_endpose(task, reward_manager) -> None:
    parser = getattr(reward_manager, "func_parser", None)
    robot_manager = getattr(task, "robot_manager", None)
    origins = getattr(parser, "robot_origin_endpose", None)
    if parser is None or robot_manager is None or origins is None:
        return
    if not hasattr(robot_manager, "get_real_endpose"):
        return
    num_envs = int(getattr(task, "num_envs", len(origins)))
    for robot in getattr(robot_manager, "robot_list", []):
        arm_name = getattr(robot, "arm_name", None)
        if arm_name is None:
            continue
        real_endpose = robot_manager.get_real_endpose(robot)
        for env_idx in range(num_envs):
            origins[env_idx][arm_name] = deepcopy(real_endpose[env_idx])


class StackBlocksAdapter:
    """Policy-facing view of one RoboDojo environment.

    ``task`` is the eval task object (``take_action`` / ``get_obs`` /
    ``reward_manager``). Training calls ``take_action`` for IK and physics, and
    suspends ``is_episode_end`` for the duration of that call. The eval client's
    termination helper ends a non-success episode immediately; the adapter
    decides termination from the official score and ``step_lim`` instead.
    """

    def __init__(self, task: Any, env_index: int = 0):
        self.task = task
        self.env_index = env_index
        self._steps = 0

    @property
    def step_limit(self) -> int:
        return int(getattr(self.task, "step_lim", STEP_LIMIT))

    def reset(self, seed: int | None = None) -> tuple[dict, dict]:
        self.task.reset(seed=seed)
        _seed_gripper_state(self.task)
        self._arm_reward_checks()
        self._steps = 0
        observation = self.policy_observation(self._raw_obs())
        return observation, {"env_name": type(self.task).__name__, "seed": seed}

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        action_dict = unpack_ee_action(action)
        self._take_action(action_dict)
        self._steps += 1
        raw = self._raw_obs()
        score, success_reward = self._external_signals()
        terminated = float(score) == SUCCESS_SCORE or float(success_reward) >= 1.0 - 1e-3
        truncated = self._steps >= self.step_limit and not terminated
        info = {
            "success": terminated,
            "external_score": float(score),
            "success_reward": float(success_reward),
            "termination_reason": _reason(terminated, truncated),
            "step_id": self._steps,
        }
        return self.policy_observation(raw), float(score), terminated, truncated, info

    def get_privileged_state(self) -> dict:
        """Block poses for intrinsic rewards. This dict is not a policy input."""

        parser = self.task.reward_manager.func_parser
        poses = {}
        for label in BLOCK_LABELS:
            pos_list, rot_list = parser.get_label_pose(label)
            poses[label] = {
                "pos": _index(pos_list, self.env_index),
                "rot": _index(rot_list, self.env_index),
            }
        return {"blocks": poses}

    def policy_observation(self, raw: Mapping) -> dict:
        vision = raw.get("vision", {})
        if HEAD_CAMERA not in vision or "color" not in vision[HEAD_CAMERA]:
            raise KeyError("Policy observation requires vision.cam_head.color.")
        state = _state_mapping(raw)
        missing = [key for key in POLICY_STATE_KEYS if key not in state]
        if missing:
            raise KeyError(f"Policy proprio is missing {missing}.")
        proprio_state = {key: state[key] for key in POLICY_STATE_KEYS}
        observation = {
            "image": _copy_image(vision[HEAD_CAMERA]["color"]),
            "camera": HEAD_CAMERA,
            "proprio": pack_ee_action(proprio_state),
            "instruction": raw.get("instruction", ""),
        }
        assert_no_privileged(observation, "policy observation")
        return observation

    def _raw_obs(self) -> Mapping:
        if hasattr(self.task, "get_obs"):
            return self.task.get_obs()
        raise AttributeError("The task must provide get_obs().")

    def _take_action(self, action_dict: dict) -> None:
        original = getattr(self.task, "is_episode_end", None)
        if original is not None:
            self.task.is_episode_end = lambda: False
        try:
            self.task.take_action(action_dict)
        finally:
            if original is not None:
                self.task.is_episode_end = original

    def _arm_reward_checks(self) -> None:
        if hasattr(self.task, "run_reward"):
            self.task.run_reward()
        if hasattr(self.task, "get_score"):
            self.task.get_score()

    def _external_signals(self) -> tuple[float, float]:
        manager = self.task.reward_manager
        reward_list = manager.get_reward(final_check=False)
        score_list = manager.get_score(reward_lst=reward_list)
        return float(score_list[self.env_index]), float(reward_list[self.env_index])


def _reason(terminated: bool, truncated: bool) -> str:
    if terminated:
        return "success"
    if truncated:
        return "timeout"
    return "running"


def _index(values, index: int):
    if isinstance(values, (list, tuple)):
        return values[index]
    return values


class ParallelAdapter:
    """Several scenes that share one physics step.

    ``task`` exposes ``take_action_batch`` and ``get_obs_batch``. A one-scene
    task can keep the single-environment ``take_action`` / ``get_obs`` pair.
    """

    def __init__(self, task: Any):
        self.task = task
        self.num_envs = int(getattr(task, "num_envs", 1))
        self._slots = [StackBlocksAdapter(task, env_index=index) for index in range(self.num_envs)]

    def reset_batch(self, seeds) -> tuple[list[dict], list[dict]]:
        seeds = [None if seed is None else int(seed) for seed in seeds]
        if len(seeds) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} seeds, got {len(seeds)}.")
        self.task.reset(seed=seeds if self.num_envs > 1 else seeds[0])
        _seed_gripper_state(self.task)
        observations = []
        infos = []
        for index, seed in enumerate(seeds):
            slot = self._slots[index]
            slot._steps = 0
            slot._arm_reward_checks()
            observations.append(slot.policy_observation(self._raw(index)))
            infos.append({"env_name": type(self.task).__name__, "seed": seed, "env_index": index})
        return observations, infos

    def step_batch(self, actions, active) -> list[tuple[dict, float, bool, bool, dict]]:
        active = [int(index) for index in active]
        if len(actions) != len(active):
            raise ValueError(f"Got {len(actions)} actions for {len(active)} environments.")
        action_dicts = [unpack_ee_action(action) for action in actions]
        if hasattr(self.task, "take_action_batch"):
            self._call_batch(action_dicts, active)
        elif self.num_envs == 1 and active == [0]:
            self._slots[0]._take_action(action_dicts[0])
        else:
            raise RuntimeError("The task has no take_action_batch, so it cannot step several environments.")
        raws = self._raw_batch(active)
        results = []
        for raw, index in zip(raws, active):
            slot = self._slots[index]
            slot._steps += 1
            score, success_reward = slot._external_signals()
            terminated = float(score) == SUCCESS_SCORE or float(success_reward) >= 1.0 - 1e-3
            truncated = slot._steps >= slot.step_limit and not terminated
            info = {
                "success": terminated,
                "external_score": float(score),
                "success_reward": float(success_reward),
                "termination_reason": _reason(terminated, truncated),
                "step_id": slot._steps,
                "env_index": index,
            }
            results.append((slot.policy_observation(raw), float(score), terminated, truncated, info))
        return results

    def privileged_batch(self, active) -> list[dict]:
        return [self._slots[int(index)].get_privileged_state() for index in active]

    def _call_batch(self, action_dicts, active) -> None:
        original = getattr(self.task, "is_episode_end", None)
        if original is not None:
            self.task.is_episode_end = lambda: False
        try:
            self.task.take_action_batch(action_dicts, active)
        finally:
            if original is not None:
                self.task.is_episode_end = original

    def _raw(self, index: int) -> Mapping:
        if hasattr(self.task, "get_obs_batch"):
            return self.task.get_obs_batch([index])[0]
        if index != 0:
            raise RuntimeError("get_obs only covers environment 0.")
        return self._slots[0]._raw_obs()

    def _raw_batch(self, active) -> list[Mapping]:
        if hasattr(self.task, "get_obs_batch"):
            return list(self.task.get_obs_batch(list(active)))
        if list(active) == [0]:
            return [self._slots[0]._raw_obs()]
        raise RuntimeError("get_obs only covers environment 0.")
