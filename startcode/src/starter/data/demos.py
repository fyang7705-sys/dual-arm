"""Demonstration pairs for the offline stage.

Real episodes live under ``data/RoboDojo/stack_blocks/arx_x5``. The random
task has none. ``hold_demos`` is a stand-in used when that directory is absent,
so both training stages can be exercised without Isaac.
"""

from __future__ import annotations

import random
from pathlib import Path

from starter.interfaces import ACTION_DIM, ARM_KEYS, ARM_WIDTHS


def hold_demos(count: int = 64, seed: int = 0) -> list[tuple[list[float], list[float]]]:
    """Features are ``[image_mean, *proprio]``. The expert action holds the pose."""

    rng = random.Random(seed)
    demos = []
    for _ in range(count):
        proprio = [rng.uniform(-0.2, 0.2) for _ in range(ACTION_DIM)]
        proprio[7] = rng.uniform(0.0, 1.0)
        proprio[15] = rng.uniform(0.0, 1.0)
        obs = [rng.random(), *proprio]
        demos.append((obs, list(proprio)))
    return demos


def episode_files(root: Path) -> list[Path]:
    """``episode_*.hdf5`` under ``data/``, or ``root`` itself when it is a file."""

    files = sorted(Path(root).glob("data/episode_*.hdf5"))
    if not files and Path(root).is_file():
        return [Path(root)]
    return files


def read_episode(path: Path) -> dict:
    """State, action, and the stored head-camera buffers from one episode.

    ``state`` and ``action`` are lists of 16-D ee vectors. ``cam_head`` is the
    encoded ``vision/cam_head/colors`` array as stored in the file. Wrist
    cameras are in the same file and are not returned here.
    """

    import h5py

    with h5py.File(path, "r") as handle:
        return {
            "state": _pack_group(handle["state"]),
            "action": _pack_group(handle["action"]),
            "cam_head": handle["vision"]["cam_head"]["colors"][:],
        }


def load_demos(config: dict) -> list[tuple[list[float], list[float]]]:
    path = config.get("data_path")
    if path:
        loaded = load_ee_hdf5(Path(path))
        if loaded:
            return loaded
    return hold_demos(count=int(config.get("demo_count", 64)), seed=int(config.get("seed", 0)))


def load_ee_hdf5(root: Path) -> list[tuple[list[float], list[float]]]:
    try:
        import h5py
    except ImportError:
        return []
    files = sorted(root.glob("data/episode_*.hdf5"))
    if not files and root.is_file():
        files = [root]
    demos = []
    for episode in files:
        with h5py.File(episode, "r") as handle:
            state = _pack_group(handle["state"])
            action = _pack_group(handle["action"])
        for obs_row, act_row in zip(state, action):
            demos.append(([0.0, *obs_row], act_row))
    return demos


def _pack_group(group) -> list[list[float]]:
    columns = []
    for key, width in zip(ARM_KEYS, ARM_WIDTHS):
        name = key if key in group else key + "s"
        columns.append(group[name][:])
    length = len(columns[0])
    packed = []
    for index in range(length):
        row = []
        for column, width in zip(columns, ARM_WIDTHS):
            row.extend(float(value) for value in column[index][:width])
        packed.append(row)
    return packed
