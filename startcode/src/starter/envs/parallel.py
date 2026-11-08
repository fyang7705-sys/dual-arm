"""How many Isaac scenes one online update samples."""

from __future__ import annotations

from starter.config import require


def parallel_plan(config: dict) -> tuple[int, int, int]:
    """Return ``(updates, episodes, num_envs)``.

    ``episodes`` is the number of episodes in one PPO update. They are sampled
    in waves of ``num_envs``, so it has to be a positive multiple of that width.
    """

    updates = int(require(config, "finetune_updates"))
    episodes = int(require(config, "finetune_episodes"))
    num_envs = int(require(config, "num_envs"))
    if updates < 1:
        raise ValueError(f"finetune_updates must be positive, got {updates}.")
    if num_envs < 1:
        raise ValueError(f"num_envs must be positive, got {num_envs}.")
    if episodes < 1 or episodes % num_envs != 0:
        raise ValueError(
            f"finetune_episodes ({episodes}) must be a positive multiple of num_envs ({num_envs})."
        )
    return updates, episodes, num_envs
