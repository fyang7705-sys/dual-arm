"""ACT pipeline. Model, training, and checkpoint format are yours.

This file currently holds a *minimal* baseline so that
``bash scripts/train_act.sh`` can be smoke-tested end to end:

* input: one head-camera frame (``cam_head``) + the 16-D end-effector state
* output: a short chunk of future 16-D end-effector actions
* loss: masked MSE on the demonstrated action chunk

It is deliberately tiny and untuned. It is not an ACT reproduction; replace or
extend :func:`train_offline` / :func:`load_policy` with your own model.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from starter.config import require
from starter.data.demos import episode_files, read_episode
from starter.interfaces import ACTION_DIM
from starter.paths import REPO_ROOT, act_output_dir

DEFAULT_IMAGE_SIZE = 96
DEFAULT_CHUNK = 4
DEFAULT_EPISODES = 4
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-3
VAL_FRACTION = 0.1


def train_offline(config: dict) -> None:
    """Behavior-clone a deterministic policy on the standard stack_blocks demos.

    Use the head camera and the 16-D end-effector state. The result has to be
    freezable: the online stage must not change these weights.
    """

    torch = _import_torch()

    image_size = int(config.get("act_image_size", DEFAULT_IMAGE_SIZE))
    chunk = int(config.get("act_chunk", DEFAULT_CHUNK))
    epochs = int(config.get("act_epochs", DEFAULT_EPOCHS))
    batch_size = int(config.get("act_batch_size", DEFAULT_BATCH_SIZE))
    lr = float(config.get("act_lr", DEFAULT_LR))
    seed = int(config.get("seed", 0))

    images, states, actions = _load_dataset(config, image_size)
    if len(images) < 2:
        raise RuntimeError("Not enough demonstration frames were loaded; check data_path.")

    targets, valid = _chunk_targets(actions, chunk)

    split = max(1, int(round(len(images) * (1.0 - VAL_FRACTION))))
    split = min(split, len(images))
    train_idx = np.arange(split)
    val_idx = np.arange(split, len(images))
    if len(val_idx) == 0:
        val_idx = train_idx

    state_mean, state_std = _stats(states[train_idx])
    action_mean, action_std = _stats(targets[train_idx])

    device = _pick_device(torch, config)
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = _build_model(torch, ACTION_DIM, chunk).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    image_t = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous().float() / 255.0
    state_t = _normalize(torch.from_numpy(states), state_mean, state_std)
    target_t = _normalize(torch.from_numpy(targets), action_mean, action_std)
    valid_t = torch.from_numpy(valid)
    train_rows = torch.from_numpy(train_idx)

    print(
        f"[act] frames={len(images)} train={len(train_idx)} val={len(val_idx)} "
        f"chunk={chunk} image={image_size} device={device}"
    )

    history: list[dict] = []
    start = time.time()
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train_idx))
        total = 0.0
        batches = 0
        for offset in range(0, len(order), batch_size):
            rows = train_rows[order[offset : offset + batch_size]]

            pred = model(image_t[rows].to(device), state_t[rows].to(device))
            loss = _masked_mse(pred, target_t[rows].to(device), valid_t[rows].to(device))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total += float(loss.detach())
            batches += 1

        train_loss = total / max(1, batches)
        val_loss = _evaluate(torch, model, image_t, state_t, target_t, valid_t, val_idx, device)
        history.append({"epoch": epoch, "train": train_loss, "val": val_loss})
        print(f"[act] epoch {epoch + 1}/{epochs} train={train_loss:.6f} val={val_loss:.6f}")

    out_dir = act_output_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": "starter.act.minimal.v1",
        "model": model.state_dict(),
        "action_dim": ACTION_DIM,
        "chunk": chunk,
        "image_size": image_size,
        "state_mean": state_mean,
        "state_std": state_std,
        "action_mean": action_mean,
        "action_std": action_std,
        "history": history,
        "config": {
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "data_path": str(require(config, "data_path")),
        },
    }
    torch.save(checkpoint, out_dir / "policy_last.ckpt")
    torch.save(checkpoint, out_dir / "policy_0000.ckpt")
    (out_dir / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"[act] saved checkpoint to {out_dir / 'policy_last.ckpt'}")
    print(f"[act] done in {time.time() - start:.1f}s")


def train_online(config: dict) -> None:
    """Learn a residual on the frozen policy in ``stack_blocks``.

    ``scripts/train_act_residual.sh`` has already started Isaac in the simulator
    image. Connect with ``starter.envs.remote.RemoteEnv``. The action is the
    frozen action plus a residual. Keep the environment score, and do not feed
    privileged block poses to the policy.
    """

    del config
    raise NotImplementedError("实现冻结基座上的残差在线训练。")


# --------------------------------------------------------------------------- #
# Minimal policy
# --------------------------------------------------------------------------- #


def _build_model(torch, action_dim: int, chunk: int, width: int = 64):
    """Small CNN + MLP mapping (head image, 16-D state) to ``chunk`` actions."""

    nn = torch.nn

    class TinyChunkPolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.image_encoder = nn.Sequential(
                nn.Conv2d(3, 32, 5, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, width, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
            )
            self.head = nn.Sequential(
                nn.Linear(width * 16 + ACTION_DIM, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, action_dim * chunk),
            )

        def forward(self, image, state):
            fused = torch.cat([self.image_encoder(image), state], dim=-1)
            return self.head(fused).view(-1, chunk, action_dim)

    return TinyChunkPolicy()


def load_policy(checkpoint_path: str | Path, device: str = "cpu"):
    """Rebuild the minimal policy from one of its checkpoints."""

    torch = _import_torch()
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    model = _build_model(torch, checkpoint["action_dim"], checkpoint["chunk"])
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, checkpoint


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def _load_dataset(config: dict, image_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = Path(str(require(config, "data_path")))
    files = episode_files(root)
    if not files:
        raise FileNotFoundError(
            f"No episode_*.hdf5 under {root}. Run 'bash scripts/download_stack_blocks.sh' first."
        )

    limit = int(config.get("act_max_episodes", DEFAULT_EPISODES))
    if limit > 0:
        files = files[:limit]

    decode_image_bit = _image_decoder()
    import cv2

    images: list[np.ndarray] = []
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for path in files:
        episode = read_episode(path)
        frames = np.asarray(episode["cam_head"])
        state = np.asarray(episode["state"], dtype=np.float32)
        action = np.asarray(episode["action"], dtype=np.float32)
        steps = min(len(frames), len(state), len(action))
        for index in range(steps):
            frame = decode_image_bit(frames[index])
            images.append(
                cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
            )
        states.append(state[:steps])
        actions.append(action[:steps])

    if not images:
        return (
            np.zeros((0, image_size, image_size, 3), np.uint8),
            np.zeros((0, ACTION_DIM), np.float32),
            np.zeros((0, ACTION_DIM), np.float32),
        )
    return (
        np.stack(images).astype(np.uint8),
        np.concatenate(states, axis=0).astype(np.float32),
        np.concatenate(actions, axis=0).astype(np.float32),
    )


def _image_decoder():
    """``decode_image_bit`` from the cloned XPolicyLab (channel-order safe)."""

    try:
        from XPolicyLab.utils.process_data import decode_image_bit
    except ImportError:
        third_party = str(REPO_ROOT / "third_party")
        if third_party not in sys.path:
            sys.path.append(third_party)
        from XPolicyLab.utils.process_data import decode_image_bit
    return decode_image_bit


def _chunk_targets(actions: np.ndarray, chunk: int) -> tuple[np.ndarray, np.ndarray]:
    """Future action chunk per timestep, padded by repeating the last action."""

    count = len(actions)
    width = actions.shape[1]
    targets = np.zeros((count, chunk, width), dtype=np.float32)
    valid = np.zeros((count,), dtype=np.float32)
    for index in range(count):
        available = min(chunk, count - index)
        target = actions[index : index + available]
        if available < chunk:
            padding = np.repeat(target[-1:], chunk - available, axis=0)
            target = np.concatenate([target, padding], axis=0)
        targets[index] = target
        valid[index] = available
    return targets.reshape(count, chunk * width), valid


def _stats(values: np.ndarray) -> tuple[list[float], list[float]]:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.tolist(), std.tolist()


# --------------------------------------------------------------------------- #
# Training helpers
# --------------------------------------------------------------------------- #


def _import_torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment problem
        raise RuntimeError(
            "PyTorch is not installed in this environment. Activate the policy "
            "environment (e.g. 'conda activate robodojo') before running train_act.sh."
        ) from error
    return torch


def _pick_device(torch, config: dict) -> str:
    if not torch.cuda.is_available():
        return "cpu"
    index = int(config.get("policy_gpu", 0))
    if index >= torch.cuda.device_count():
        index = 0
    return f"cuda:{index}"


def _normalize(tensor, mean: list[float], std: list[float]):
    import torch

    tensor = tensor.float()
    mean_t = torch.tensor(mean, dtype=torch.float32, device=tensor.device)
    std_t = torch.tensor(std, dtype=torch.float32, device=tensor.device)
    return (tensor - mean_t) / std_t


def _masked_mse(pred, target, valid):
    import torch

    chunk = pred.shape[1]
    mask = (torch.arange(chunk, device=valid.device)[None, :] < valid[:, None]).float()
    per_step = ((pred - target.view_as(pred)) ** 2).mean(dim=-1)
    return (per_step * mask).sum() / mask.sum().clamp_min(1.0)


def _evaluate(torch, model, images, states, targets, valid, indices, device) -> float:
    model.eval()
    rows = torch.from_numpy(indices)
    with torch.no_grad():
        pred = model(images[rows].to(device), states[rows].to(device))
        loss = _masked_mse(pred, targets[rows].to(device), valid[rows].to(device))
    return float(loss)
