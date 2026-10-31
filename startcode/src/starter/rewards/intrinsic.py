"""Intrinsic reward added beside the environment score, never written back."""

from __future__ import annotations

from typing import Callable


def zero_intrinsic(privileged, action, public_score, terminated, truncated, info) -> float:
    """Default intrinsic reward. The external score is unchanged."""

    del privileged, action, public_score, terminated, truncated, info
    return 0.0


def combine_reward(public_score: float, intrinsic: float) -> float:
    """Training target. ``public_score`` stays the value returned by the env."""

    return float(public_score) + float(intrinsic)


IntrinsicFn = Callable[..., float]
