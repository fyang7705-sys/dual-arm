"""Diffusion Policy pipeline. Model, training, and checkpoint format are yours."""

from __future__ import annotations


def train_offline(config: dict) -> None:
    """Behavior-clone a diffusion policy on the standard stack_blocks demos.

    Read episodes with ``starter.data.demos``. The policy input is the head
    camera and the 16-D end-effector state. Save a checkpoint that
    ``train_online`` and your eval server can both load.
    """

    del config
    raise NotImplementedError("实现 Diffusion Policy 的离线训练，并保存 checkpoint。")


def train_online(config: dict) -> None:
    """Fine-tune the offline diffusion policy in ``stack_blocks``.

    ``scripts/train_dppo.sh`` has already started Isaac in the simulator image.
    Connect with ``starter.envs.remote.RemoteEnv`` using ``env_host`` and
    ``env_port``. Keep the environment score. Do not feed privileged block
    poses to the policy. The policy process and Isaac are on different GPUs.
    """

    del config
    raise NotImplementedError("实现扩散策略的在线微调。从离线 checkpoint 继续，不要随机初始化。")
