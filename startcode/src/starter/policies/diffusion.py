"""Diffusion Policy + DPPO entry points.

``train_offline`` behavior-clones a diffusion policy on the stack_blocks
demonstrations.  ``train_online`` loads that checkpoint, starts from it (never
from random weights), and runs DPPO against the Isaac environment served by
``scripts/train_dppo.sh``.
"""

from __future__ import annotations

import csv
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from starter.envs.parallel import parallel_plan
from starter.envs.remote import RemoteEnv
from starter.paths import OUTPUTS_ROOT, dp_output_dir, run_name

from starter.policies.dp_agent import DPPOAgent
from starter.policies.dp_core import (
    ACTION_DIM,
    Critic,
    DiffusionPolicy,
    DiffusionRuntime,
    EMA,
    EpisodeDataset,
    Normalization,
    build_policy_from_payload,
    cfg_get,
    compute_normalization,
    count_parameters,
    load_checkpoint_payload,
    load_demo_episodes,
    resolve_device,
    save_checkpoint,
    set_seed,
)


DEFAULT_TRAIN_KEYS: Dict[str, Any] = {
    "action_dim": ACTION_DIM,
    "state_dim": ACTION_DIM,
    "chunk_size": 8,
    "image_size": 96,
    "diffusion_steps": 100,
    "beta_start": 1e-4,
    "beta_end": 0.02,
    "hidden_dim": 512,
    "context_dim": 256,
    "vision_dim": 128,
    "time_dim": 128,
    "epochs": 120,
    "batch_size": 64,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "warmup_steps": 500,
    "grad_clip": 1.0,
    "val_ratio": 0.1,
    "num_workers": 2,
    "ema_decay": 0.999,
    "save_every": 20,
    "log_every": 20,
}


def _merged_config(config: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(DEFAULT_TRAIN_KEYS)
    merged.update(config)
    return merged


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------------------
# offline behavior cloning
# --------------------------------------------------------------------------------------

def _split_episode_indices(count: int, val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    indices = list(range(count))
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    if count <= 1:
        return indices, indices
    val_count = int(round(count * max(0.0, min(val_ratio, 0.5))))
    val_count = max(1, min(val_count, count - 1))
    return indices[val_count:], indices[:val_count]


def _lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return float(step + 1) / float(max(warmup, 1))
    progress = float(step - warmup) / float(max(total - warmup, 1))
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def train_offline(config: Dict[str, Any]) -> None:
    config = _merged_config(config)
    seed = int(cfg_get(config, "seed", 0))
    set_seed(seed)
    device = resolve_device(str(cfg_get(config, "device", "cuda")))
    output_dir = dp_output_dir(config)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "train_config.json", config)

    print(f"[dp] loading demonstrations from {cfg_get(config, 'data_path')}", flush=True)
    episodes = load_demo_episodes(config)
    train_indices, val_indices = _split_episode_indices(len(episodes), float(cfg_get(config, "val_ratio", 0.1)), seed)
    normalization = compute_normalization([episodes[index] for index in train_indices])
    print(
        f"[dp] {len(episodes)} episodes: {len(train_indices)} train / {len(val_indices)} val; "
        f"chunk {cfg_get(config, 'chunk_size')}",
        flush=True,
    )

    train_dataset = EpisodeDataset(episodes, train_indices, normalization, int(cfg_get(config, "chunk_size", 8)))
    val_dataset = EpisodeDataset(episodes, val_indices, normalization, int(cfg_get(config, "chunk_size", 8)))
    num_workers = int(cfg_get(config, "num_workers", 2))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg_get(config, "batch_size", 64)),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg_get(config, "batch_size", 64)),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = DiffusionPolicy(config).to(device)
    ema = EMA(model, decay=float(cfg_get(config, "ema_decay", 0.999)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg_get(config, "learning_rate", 1e-4)),
        weight_decay=float(cfg_get(config, "weight_decay", 1e-4)),
    )
    epochs = int(cfg_get(config, "epochs", 120))
    total_steps = max(1, epochs * max(1, len(train_loader)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _lr_lambda(step, int(cfg_get(config, "warmup_steps", 500)), total_steps),
    )
    use_amp = bool(cfg_get(config, "amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"[dp] model parameters: {count_parameters(model) / 1e6:.2f}M", flush=True)

    best_val = float("inf")
    global_step = 0
    log_path = output_dir / "train_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "step", "train_loss", "val_loss", "lr", "elapsed_s"])
        started = time.time()

        for epoch in range(1, epochs + 1):
            model.train()
            running = 0.0
            batches = 0
            for batch in train_loader:
                images = batch["image"].to(device, non_blocking=True)
                states = batch["state"].to(device, non_blocking=True)
                actions = batch["action"].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                amp_context = (
                    torch.autocast(device_type="cuda", enabled=True) if use_amp else nullcontext()
                )
                with amp_context:
                    loss = model.training_loss(images, states, actions)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg_get(config, "grad_clip", 1.0)))
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                ema.update(model)

                running += float(loss.detach().cpu())
                batches += 1
                global_step += 1
                if global_step % int(cfg_get(config, "log_every", 20)) == 0:
                    print(
                        f"[dp] epoch {epoch}/{epochs} step {global_step} "
                        f"loss {running / max(batches, 1):.5f}",
                        flush=True,
                    )

            train_loss = running / max(batches, 1)
            model.eval()
            val_total = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(device)
                    states = batch["state"].to(device)
                    actions = batch["action"].to(device)
                    loss = model.training_loss(images, states, actions)
                    val_total += float(loss.detach().cpu())
                    val_batches += 1
            val_loss = val_total / max(val_batches, 1)
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - started
            print(
                f"[dp] epoch {epoch}: train {train_loss:.5f} val {val_loss:.5f} lr {lr:.2e}",
                flush=True,
            )
            writer.writerow([epoch, global_step, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{lr:.8f}", f"{elapsed:.1f}"])
            handle.flush()

            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    checkpoint_dir / "best.ckpt",
                    model,
                    ema,
                    config,
                    normalization,
                    kind="diffusion_policy",
                    extra={"epoch": epoch, "val_loss": best_val, "created": time.strftime("%Y-%m-%d %H:%M:%S")},
                )
            if epoch % int(cfg_get(config, "save_every", 20)) == 0 or epoch == epochs:
                save_checkpoint(
                    checkpoint_dir / f"epoch_{epoch:04d}.ckpt",
                    model,
                    ema,
                    config,
                    normalization,
                    kind="diffusion_policy",
                    extra={"epoch": epoch, "val_loss": val_loss},
                )

    save_checkpoint(
        checkpoint_dir / "last.ckpt",
        model,
        ema,
        config,
        normalization,
        kind="diffusion_policy",
        extra={"epoch": epochs, "val_loss": best_val, "created": time.strftime("%Y-%m-%d %H:%M:%S")},
    )
    save_checkpoint(
        checkpoint_dir / "offline_last.ckpt",
        model,
        ema,
        config,
        normalization,
        kind="diffusion_policy",
        extra={"epoch": epochs, "val_loss": best_val},
    )
    _write_json(checkpoint_dir / "latest.json", {"checkpoint": str(checkpoint_dir / "offline_last.ckpt")})
    print(f"[dp] checkpoints written to {checkpoint_dir}", flush=True)


# --------------------------------------------------------------------------------------
# checkpoint discovery
# --------------------------------------------------------------------------------------

def _candidate_checkpoint_dirs(config: Dict[str, Any]) -> List[Path]:
    root = OUTPUTS_ROOT / "dp" / run_name(config)
    dirs: List[Path] = []
    if root.is_dir():
        stamped = sorted([path for path in root.iterdir() if path.is_dir()], reverse=True)
        dirs.extend(stamped)
        dirs.append(root)
    return dirs


def find_offline_checkpoint(config: Dict[str, Any]) -> Optional[Path]:
    explicit = cfg_get(config, "offline_checkpoint")
    if explicit and explicit not in ("latest", "auto"):
        path = Path(str(explicit))
        if path.is_file():
            return path
        candidate = OUTPUTS_ROOT.parent / path
        if candidate.is_file():
            return candidate

    for directory in _candidate_checkpoint_dirs(config):
        for name in ("offline_last.ckpt", "last.ckpt", "best.ckpt"):
            path = directory / "checkpoints" / name if directory.name != "checkpoints" else directory / name
            if path.is_file():
                return path
        checkpoints = directory / "checkpoints"
        if checkpoints.is_dir():
            files = sorted(checkpoints.glob("*.ckpt"))
            if files:
                return files[-1]

    fallback = sorted((OUTPUTS_ROOT / "dp").glob(f"**/checkpoints/offline_last.ckpt"))
    if fallback:
        return fallback[-1]
    return None


def find_online_checkpoint() -> Optional[Path]:
    candidates = sorted((OUTPUTS_ROOT / "dp").glob("**/checkpoints/online_last.ckpt"))
    return candidates[-1] if candidates else None


def load_runtime(checkpoint: str | Path, device: Optional[torch.device] = None) -> DiffusionRuntime:
    path = Path(checkpoint)
    if path.is_dir():
        for directory in (path, path / "checkpoints"):
            for name in ("online_last.ckpt", "offline_last.ckpt", "last.ckpt", "best.ckpt"):
                if (directory / name).is_file():
                    path = directory / name
                    break
            if not path.is_dir():
                break
    payload = load_checkpoint_payload(path)
    chosen = device if device is not None else resolve_device("cuda")
    return DiffusionRuntime(payload, chosen, prefer_ema=True)


# --------------------------------------------------------------------------------------
# DPPO online fine-tuning
# --------------------------------------------------------------------------------------

def train_online(config: Dict[str, Any]) -> None:
    config = _merged_config(config)
    seed = int(cfg_get(config, "seed", 0))
    set_seed(seed)
    device = resolve_device(str(cfg_get(config, "device", "cuda")))
    checkpoint = find_offline_checkpoint(config)
    if checkpoint is None:
        raise FileNotFoundError(
            "No offline diffusion checkpoint found. Run `bash scripts/train_dp.sh` first, "
            "or set `offline_checkpoint` in configs/dppo.yaml."
        )
    payload = load_checkpoint_payload(checkpoint)
    model, architecture_config, normalization = build_policy_from_payload(payload, device, prefer_ema=True)
    model.train()
    print(f"[dppo] continuing from {checkpoint}", flush=True)
    print(f"[dppo] architecture: chunk={model.chunk_size} T={model.diffusion_steps}", flush=True)

    epoch = int(payload.get("epoch", 0))
    ema = EMA(model, decay=float(cfg_get(config, "ema_decay", 0.999)))
    critic = Critic(state_dim=model.state_dim).to(device)
    agent = DPPOAgent(model, ema, critic, normalization, config, device)

    updates, episodes_per_update, num_envs = parallel_plan(config)
    env = RemoteEnv(
        str(cfg_get(config, "env_host", "127.0.0.1")),
        int(cfg_get(config, "env_port", 5555)),
        timeout=float(cfg_get(config, "env_timeout", 2400.0)),
        reset_timeout=float(cfg_get(config, "env_reset_timeout", 10800.0)),
    )
    output_dir = dp_output_dir(config)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "train_config.json", config)

    log_path = output_dir / "online_log.jsonl"
    best_score = -1.0
    print(
        f"[dppo] {updates} updates x {episodes_per_update} episodes "
        f"({num_envs} parallel scenes)",
        flush=True,
    )
    started = time.time()
    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            for update_index in range(1, updates + 1):
                trajectories: List[List[Any]] = []
                episode_results: List[Any] = []
                waves = episodes_per_update // num_envs
                for wave in range(waves):
                    seeds = [
                        int(cfg_get(config, "seed", 0))
                        + (update_index - 1) * episodes_per_update
                        + wave * num_envs
                        + index
                        for index in range(num_envs)
                    ]
                    wave_trajectories, wave_episodes = agent.collect_wave(env, seeds)
                    trajectories.extend(wave_trajectories)
                    episode_results.extend(wave_episodes)

                metrics = agent.update(trajectories)
                scores = [episode.score for episode in episode_results]
                successes = [1.0 if episode.success else 0.0 for episode in episode_results]
                record = {
                    "update": update_index,
                    "episodes": len(episode_results),
                    "mean_score": float(np.mean(scores)) if scores else 0.0,
                    "success_rate": float(np.mean(successes)) if successes else 0.0,
                    "max_score": float(np.max(scores)) if scores else 0.0,
                    "mean_steps": float(np.mean([episode.steps for episode in episode_results]))
                    if episode_results
                    else 0.0,
                    "elapsed_s": time.time() - started,
                }
                record.update(metrics)
                log_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                log_handle.flush()
                print(
                    f"[dppo] update {update_index}/{updates} score {record['mean_score']:.1f} "
                    f"success {record['success_rate'] * 100:.1f}% "
                    f"pi {metrics['policy_loss']:.4f} v {metrics['value_loss']:.4f}",
                    flush=True,
                )

                save_checkpoint(
                    checkpoint_dir / "online_last.ckpt",
                    model,
                    ema,
                    config,
                    normalization,
                    kind="diffusion_policy_dppo",
                    extra={
                        "update": update_index,
                        "offline_checkpoint": str(checkpoint),
                        "episode_score": record["mean_score"],
                        "success_rate": record["success_rate"],
                        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                )
                if record["success_rate"] > best_score:
                    best_score = record["success_rate"]
                    save_checkpoint(
                        checkpoint_dir / "online_best.ckpt",
                        model,
                        ema,
                        config,
                        normalization,
                        kind="diffusion_policy_dppo",
                        extra={"update": update_index, "success_rate": best_score},
                    )
                every = int(cfg_get(config, "checkpoint_every", 10))
                if every > 0 and update_index % every == 0:
                    save_checkpoint(
                        checkpoint_dir / f"online_{update_index:04d}.ckpt",
                        model,
                        ema,
                        config,
                        normalization,
                        kind="diffusion_policy_dppo",
                        extra={"update": update_index},
                    )
    finally:
        env.close()
    _write_json(
        output_dir / "summary.json",
        {"best_success_rate": best_score, "updates": updates, "checkpoint": str(checkpoint)},
    )
    print(f"[dppo] done; checkpoints in {checkpoint_dir}", flush=True)
