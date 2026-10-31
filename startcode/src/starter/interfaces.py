"""Shared constants and the end-effector action packing used by XPolicyLab."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

# Dual-arm ARX X5, action_type=ee.
# Order matches XPolicyLab.utils.process_data.pack_robot_state for source_type='obs':
# [left_ee_pose(7), left_ee_joint_state(1), right_ee_pose(7), right_ee_joint_state(1)].
EE_POSE_DIM = 7
GRIPPER_DIM = 1
ACTION_DIM = 16
JOINT_ACTION_DIM = 14

EE_POSE_KEYS = ("left_ee_pose", "right_ee_pose")
GRIPPER_KEYS = ("left_ee_joint_state", "right_ee_joint_state")
ARM_KEYS = ("left_ee_pose", "left_ee_joint_state", "right_ee_pose", "right_ee_joint_state")
ARM_WIDTHS = (EE_POSE_DIM, GRIPPER_DIM, EE_POSE_DIM, GRIPPER_DIM)

JOINT_KEYS = ("left_arm_joint_state", "right_arm_joint_state")
POLICY_STATE_KEYS = ARM_KEYS
HEAD_CAMERA = "cam_head"
BLOCK_LABELS = ("block_0", "block_1", "block_2")
SUCCESS_SCORE = 100.0
PARTIAL_SCORE = 15.0
STEP_LIMIT = 550
ENV_CFG = "arx_x5"
EVAL_TASK = "stack_blocks_random"
TRAIN_TASK = "stack_blocks_random"
BENCH_NAME = "RoboDojo"

PRIVILEGED_KEYS = frozenset(
    {
        "block_0",
        "block_1",
        "block_2",
        "cube_pos",
        "object_pose",
        "privileged",
    }
)


def _as_float_list(value: Sequence, width: int, name: str) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    flat = [float(item) for item in value]
    if len(flat) != width:
        raise ValueError(f"{name} length {len(flat)} does not match {width}.")
    if not all(math.isfinite(item) for item in flat):
        raise ValueError(f"{name} contains a non-finite value.")
    return flat


def pack_ee_action(state: Mapping) -> list[float]:
    """Pack an ee state or action dict into the 16-D policy vector."""

    packed: list[float] = []
    for key, width in zip(ARM_KEYS, ARM_WIDTHS):
        if key not in state:
            raise KeyError(f"Missing '{key}' in the end-effector state.")
        packed.extend(_as_float_list(state[key], width, key))
    return packed


def unpack_ee_action(action: Sequence[float]) -> dict[str, list[float]]:
    """Unpack a 16-D vector into the dict RoboDojo's ee controller expects."""

    values = _as_float_list(action, ACTION_DIM, "action")
    unpacked: dict[str, list[float]] = {}
    offset = 0
    for key, width in zip(ARM_KEYS, ARM_WIDTHS):
        chunk = values[offset : offset + width]
        if key in GRIPPER_KEYS:
            chunk = [min(1.0, max(0.0, item)) for item in chunk]
        unpacked[key] = chunk
        offset += width
    return unpacked


def assert_no_privileged(mapping: Mapping, where: str) -> None:
    found = sorted(set(mapping) & PRIVILEGED_KEYS)
    if found:
        raise ValueError(f"{where} contains privileged keys {found}.")
