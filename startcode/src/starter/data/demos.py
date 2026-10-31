"""Read real head-camera demonstrations; never silently train on dummy data."""
from __future__ import annotations

import bisect
import random
import sys
from pathlib import Path

import h5py
import numpy as np

from starter.interfaces import ACTION_DIM, ARM_KEYS, ARM_WIDTHS
from starter.paths import REPO_ROOT


def hold_demos(count=64, seed=0):
    """Explicit test fixture, never selected by a production data loader."""
    rng = random.Random(seed)
    return [([0.0, *([0.0] * ACTION_DIM)], [rng.random()] * ACTION_DIM) for _ in range(count)]


def episode_files(root: Path):
    root = Path(root)
    if root.is_file():
        return [root]
    files = sorted((root / "data").glob("episode_*.hdf5"))
    return files or sorted(root.glob("episode_*.hdf5"))


def _columns(group):
    return [group[key if key in group else key + "s"] for key in ARM_KEYS]


def _pack_group(group, selection=slice(None)):
    parts = []
    for column, width in zip(_columns(group), ARM_WIDTHS):
        value = np.asarray(column[selection], dtype=np.float32)
        if value.shape[-1] != width:
            raise ValueError(f"{column.name}: expected last dimension {width}, got {value.shape}")
        parts.append(value)
    result = np.concatenate(parts, axis=-1)
    if not np.isfinite(result).all():
        raise ValueError(f"Non-finite values in {group.name}")
    return result


def read_episode(path):
    with h5py.File(path, "r") as f:
        return {"state": _pack_group(f["state"]), "action": _pack_group(f["action"]),
                "cam_head": f["vision/cam_head/colors"][:]}


def decode_rgb(frame):
    """Raw RGB test frames, or official XPolicyLab legacy/marked JPEG decoder."""
    array = np.asarray(frame)
    if array.ndim == 3 and array.shape[-1] == 3 and array.dtype == np.uint8:
        return array
    third_party = str(REPO_ROOT / "third_party")
    if third_party not in sys.path:
        sys.path.insert(0, third_party)
    try:
        from XPolicyLab.utils.process_data import decode_image_bit
    except ImportError as exc:
        raise RuntimeError("Encoded demos require bootstrap_third_party.sh and XPolicyLab dependencies") from exc
    return decode_image_bit(frame)


def resize_rgb(frame, size):
    import cv2
    rgb = decode_rgb(frame)
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


def split_episodes(config):
    root = Path(config["data_path"]).expanduser()
    files = episode_files(root)
    limit = int(config.get("max_episodes", 0))
    if limit > 0:
        files = files[:limit]
    if len(files) < 2:
        raise ValueError(f"Need >=2 real episodes for a disjoint train/validation split: {root}")
    random.Random(int(config.get("seed", 0))).shuffle(files)
    fraction = float(config.get("val_fraction", 0.1))
    if not 0 < fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    count = min(len(files) - 1, max(1, round(len(files) * fraction)))
    return files[count:], files[:count]


def normalization_stats(files):
    """Accumulate only 16-D moments; no image decompression or corpus preload."""
    result = {}
    for kind in ("state", "action"):
        total = np.zeros(ACTION_DIM, dtype=np.float64)
        squares = total.copy()
        count = 0
        for path in files:
            with h5py.File(path, "r") as f:
                length = len(_columns(f[kind])[0])
                for start in range(0, length, 4096):
                    values = _pack_group(f[kind], slice(start, start + 4096)).astype(np.float64)
                    total += values.sum(0)
                    squares += (values ** 2).sum(0)
                    count += len(values)
        if not count:
            raise ValueError("Empty demonstration split")
        mean = total / count
        std = np.sqrt(np.maximum(squares / count - mean ** 2, 0)).clip(1e-3)
        result[kind + "_mean"] = mean.astype(np.float32).tolist()
        result[kind + "_std"] = std.astype(np.float32).tolist()
    return result


class ChunkDataset:
    """Map-style dataset: open one HDF5 per sample, read only one image/chunk.

    No persistent h5py handles cross DataLoader worker/fork boundaries. Padding
    stays inside its episode, and validation episodes never enter train stats.
    """
    def __init__(self, files, chunk, image_size, stats):
        self.files, self.chunk, self.image_size, self.stats = list(files), int(chunk), int(image_size), stats
        if self.chunk < 1 or self.image_size < 16:
            raise ValueError("chunk must be positive and image_size >=16")
        self.ends = []
        total = 0
        for path in self.files:
            with h5py.File(path, "r") as f:
                lengths = [len(c) for kind in ("state", "action") for c in _columns(f[kind])]
                lengths.append(len(f["vision/cam_head/colors"]))
                if min(lengths) < 1 or len(set(lengths)) != 1:
                    raise ValueError(f"Unaligned or empty episode {path}: {lengths}")
                total += lengths[0]
                self.ends.append(total)

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index):
        episode = bisect.bisect_right(self.ends, index)
        start = index - (self.ends[episode - 1] if episode else 0)
        with h5py.File(self.files[episode], "r") as f:
            state = _pack_group(f["state"], start)
            action = _pack_group(f["action"], slice(start, start + self.chunk))
            image = resize_rgb(f["vision/cam_head/colors"][start], self.image_size)
        valid = np.arange(self.chunk) < len(action)
        action = np.concatenate([action, np.repeat(action[-1:], self.chunk - len(action), axis=0)])
        state = (state - self.stats["state_mean"]) / self.stats["state_std"]
        action = (action - self.stats["action_mean"]) / self.stats["action_std"]
        return {"image": image.transpose(2, 0, 1).astype(np.float32) / 255,
                "state": state.astype(np.float32), "action": action.astype(np.float32), "valid": valid}


def load_demos(config):
    """Compatibility reader, strict on missing data; training uses ChunkDataset."""
    files = episode_files(Path(config["data_path"]))
    if not files:
        raise FileNotFoundError(config["data_path"])
    return [read_episode(path) for path in files]
