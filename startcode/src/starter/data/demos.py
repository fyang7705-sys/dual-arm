"""Demonstration pairs for the offline stage.

Real episodes live under ``data/RoboDojo/stack_blocks/arx_x5``. Missing or
malformed training data raises an error. ``hold_demos`` is an explicit test fixture.
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

    root = Path(root)
    files = sorted(root.glob("data/episode_*.hdf5")) or sorted(root.glob("episode_*.hdf5"))
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
    import numpy as np

    with h5py.File(path, "r") as handle:
        episode = {
            "state": _pack_group(handle["state"]),
            "action": _pack_group(handle["action"]),
            "cam_head": handle["vision"]["cam_head"]["colors"][:],
        }
    lengths = [len(value) for value in episode.values()]
    if not lengths[0] or len(set(lengths)) != 1:
        raise ValueError(f"Empty or unaligned episode {path}: {lengths}")
    for key in ("state", "action"):
        if not np.isfinite(episode[key]).all():
            raise ValueError(f"Non-finite {key} in {path}")
    return episode


def load_demos(config: dict) -> list[tuple[list[float], list[float]]]:
    path = config.get("data_path")
    if path:
        loaded = load_ee_hdf5(Path(path))
        if loaded:
            return loaded
    raise FileNotFoundError(f"No real demonstrations found at {path!r}; download the dataset first")


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
    import numpy as np
    columns = []
    for key, width in zip(ARM_KEYS, ARM_WIDTHS):
        name = key if key in group else key + "s"
        column = np.asarray(group[name][:], dtype=np.float32)
        if column.ndim != 2 or column.shape[1] != width:
            raise ValueError(f"{group.name}/{name}: expected shape (T, {width}), got {column.shape}")
        columns.append(column)
    if len({len(column) for column in columns}) != 1:
        raise ValueError(f"Unaligned fields in {group.name}")
    length = len(columns[0])
    packed = []
    for index in range(length):
        row = []
        for column, width in zip(columns, ARM_WIDTHS):
            row.extend(float(value) for value in column[index][:width])
        packed.append(row)
    return packed
