"""Training artifacts live under the repository ``outputs`` directory.

Each launch gets its own timestamp directory so a new run does not replace
the previous one:

``outputs/<pipeline>/<run name>/<YYYY.MM.DD-HH.MM.SS>/``
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from starter.config import require


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_ROOT = REPO_ROOT / "outputs"
STAMP_FORMAT = "%Y.%m.%d-%H.%M.%S"
_STAMP = re.compile(r"^\d{4}\.\d{2}\.\d{2}-\d{2}\.\d{2}\.\d{2}$")


def run_name(config: dict) -> str:
    return (
        f"{require(config, 'bench_name')}-"
        f"{require(config, 'data_task')}-"
        f"{require(config, 'env_cfg')}-"
        f"{require(config, 'action_type')}-"
        f"{require(config, 'seed')}"
    )


def run_stamp(config: dict) -> str:
    """Directory name for this launch. ``run_stamp`` pins it; otherwise use now."""

    chosen = config.get("run_stamp")
    if chosen:
        return str(chosen)
    return datetime.now().strftime(STAMP_FORMAT)


def dp_output_dir(config: dict) -> Path:
    """Hydra run directory: logs and the ``checkpoints`` subdirectory."""

    return OUTPUTS_ROOT / "dp" / run_name(config) / run_stamp(config)


def dp_checkpoint_dir(config: dict) -> Path:
    return dp_output_dir(config) / "checkpoints"


def act_output_dir(config: dict) -> Path:
    """ACT stats and ``policy_*.ckpt`` files share this directory."""

    return OUTPUTS_ROOT / "act" / run_name(config) / run_stamp(config)


def latest_stamped_dir(run_root: Path) -> Path | None:
    """Newest ``YYYY.MM.DD-HH.MM.SS`` child of a run directory."""

    if not run_root.is_dir():
        return None
    stamps = sorted(
        path.name for path in run_root.iterdir() if path.is_dir() and _STAMP.match(path.name)
    )
    if not stamps:
        return None
    return run_root / stamps[-1]


def resolve_dp_checkpoint_dir(run_name: str) -> Path | None:
    """Checkpoint directory of the newest DP run, or the unstamped legacy directory."""

    root = OUTPUTS_ROOT / "dp" / run_name
    stamped = latest_stamped_dir(root)
    if stamped is not None and (stamped / "checkpoints").is_dir():
        return stamped / "checkpoints"
    legacy = root / "checkpoints"
    if legacy.is_dir():
        return legacy
    return None


def resolve_act_checkpoint_dir(run_name: str) -> Path | None:
    """Directory of the newest ACT run that holds ``policy_*.ckpt``."""

    root = OUTPUTS_ROOT / "act" / run_name
    stamped = latest_stamped_dir(root)
    if stamped is not None:
        return stamped
    if (root / "policy_last.ckpt").is_file():
        return root
    return None
