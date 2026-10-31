"""ACT + residual PPO entry points.

Stage 1 (``train_offline``): CVAE/transformer behavior cloning on the
stack_blocks demonstrations.  The resulting network is frozen for the online
stage.

Stage 2 (``train_online``): keep the ACT base frozen and learn a small Gaussian
residual on top of it with PPO.  The base weights are fingerprinted before and
after training and must be identical.
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from starter.envs.parallel import parallel_plan
from starter.envs.remote import RemoteEnv
from starter.paths import OUTPUTS_ROOT, act_output_dir, run_name

from starter.policies.act_agent import ActResidualAgent, ResidualCritic
from starter.policies.act_core import (
    ACTION_DIM,
    ActPolicy,
    ActRuntime,
    EpisodeDataset,
    Normalization,
    ResidualHead,
    build_act_from_payload,
    build_act_payload,
    cfg_get,
    compute_normalization,
    count_parameters,
    default_residual_scales,
    load_demo_episodes,
    load_payload,
    resolve_device,
    save_checkpoint,
    set_seed,
)


DEFAULT_OFFLINE: Dict[str, Any] = {
    "action_dim": ACTION_DIM,
    "state_dim": ACTION_DIM,
    "chunk_size": 16,
    "image_size": 96,
    "latent_dim": 32,
    "d_model": 256,
    "nhead": 4,
    "dim_feedforward": 512,
    "decoder_layers": 2,
    "dropout": 0.1,
    "kl_weight": 10.0,
    "epochs": 200,
    "batch_size": 64,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "warmup_steps": 500,
    "grad_clip": 1.0,
    "val_ratio": 0.1,
    "augment": True,
    "num_workers": 2,
    "save_every": 25,
    "log_every": 20,
}


def _merged(config: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(DEFAULT_OFFLINE)
    merged.update(config)
    return merged


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _split(count: int, ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    indices = list(range(count))
    np.random.RandomState(seed).shuffle(indices)
    if count <= 1:
        return indices, indices
    val_count = max(1, min(int(round(count * max(0.0, min(ratio, 0.5)))), count - 1))
    return indices[val_count:], indices[:val_count]


def _lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return float(step + 1) / float(max(warmup, 1))
    progress = float(step - warmup) / float(max(total - warmup, 1))
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


# --------------------------------------------------------------------------------------
# offline ACT
# --------------------------------------------------------------------------------------

def train_offline(config: Dict[str, Any]) -> None:
    config = _merged(config)
    seed = int(cfg_get(config, "seed", 0))
    set_seed(seed)
    device = resolve_device(str(cfg_get(config, "device", "cuda")))
    output_dir = act_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "train_config.json", config)

    print(f"[act] loading demonstrations from {cfg_get(config, 'data_path')}", flush=True)
    episodes = load_demo_episodes(config)
    train_indices, val_indices = _split(len(episodes), float(cfg_get(config, "val_ratio", 0.1)), seed)
    normalization = compute_normalization([episodes[index] for index in train_indices])
    print(
        f"[act] {len(episodes)} episodes: {len(train_indices)} train / {len(val_indices)} val; "
        f"chunk {cfg_get(config, 'chunk_size')}",
        flush=True,
    )

    train_dataset = EpisodeDataset(
        episodes,
        train_indices,
        normalization,
        int(cfg_get(config, "chunk_size", 16)),
        augment=bool(cfg_get(config, "augment", True)),
    )
    val_dataset = EpisodeDataset(
        episodes, val_indices, normalization, int(cfg_get(config, "chunk_size", 16)), augment=False
    )
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
    )

    model = ActPolicy(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg_get(config, "learning_rate", 1e-4)),
        weight_decay=float(cfg_get(config, "weight_decay", 1e-4)),
    )
    epochs = int(cfg_get(config, "epochs", 200))
    total_steps = max(1, epochs * max(1, len(train_loader)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _lr_lambda(step, int(cfg_get(config, "warmup_steps", 500)), total_steps)
    )
    print(f"[act] model parameters: {count_parameters(model) / 1e6:.2f}M", flush=True)

    best_val = float("inf")
    global_step = 0
    log_path = output_dir / "train_log.csv"
    started = time.time()
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "step", "train_loss", "train_l1", "train_kl", "val_loss", "lr", "elapsed_s"])
        for epoch in range(1, epochs + 1):
            model.train()
            running = 0.0
            l1_running = 0.0
            kl_running = 0.0
            batches = 0
            for batch in train_loader:
                images = batch["image"].to(device, non_blocking=True)
                states = batch["state"].to(device, non_blocking=True)
                actions = batch["action"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss, stats = model.training_loss(images, states, actions, batch["valid"].to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg_get(config, "grad_clip", 1.0)))
                optimizer.step()
                scheduler.step()
                running += float(loss.detach().cpu())
                l1_running += stats["l1"]
                kl_running += stats["kl"]
                batches += 1
                global_step += 1
                if global_step % int(cfg_get(config, "log_every", 20)) == 0:
                    print(
                        f"[act] epoch {epoch}/{epochs} step {global_step} "
                        f"loss {running / max(batches, 1):.5f}",
                        flush=True,
                    )

            train_loss = running / max(batches, 1)
            model.eval()
            val_total = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    loss, _stats = model.training_loss(
                        batch["image"].to(device),
                        batch["state"].to(device),
                        batch["action"].to(device),
                        batch["valid"].to(device),
                    )
                    val_total += float(loss.detach().cpu())
                    val_batches += 1
            val_loss = val_total / max(val_batches, 1)
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - started
            print(f"[act] epoch {epoch}: train {train_loss:.5f} val {val_loss:.5f} lr {lr:.2e}", flush=True)
            writer.writerow(
                [
                    epoch,
                    global_step,
                    f"{train_loss:.6f}",
                    f"{l1_running / max(batches, 1):.6f}",
                    f"{kl_running / max(batches, 1):.6f}",
                    f"{val_loss:.6f}",
                    f"{lr:.8f}",
                    f"{elapsed:.1f}",
                ]
            )
            handle.flush()

            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    output_dir / "policy_best.ckpt",
                    build_act_payload(
                        model,
                        config,
                        normalization,
                        kind="act_policy",
                        extra={"epoch": epoch, "val_loss": best_val, "created": time.strftime("%Y-%m-%d %H:%M:%S")},
                    ),
                )
            if epoch % int(cfg_get(config, "save_every", 25)) == 0 or epoch == epochs:
                save_checkpoint(
                    output_dir / f"policy_epoch_{epoch:04d}.ckpt",
                    build_act_payload(
                        model, config, normalization, kind="act_policy", extra={"epoch": epoch, "val_loss": val_loss}
                    ),
                )

    save_checkpoint(
        output_dir / "policy_last.ckpt",
        build_act_payload(
            model,
            config,
            normalization,
            kind="act_policy",
            extra={"epoch": epochs, "val_loss": best_val, "created": time.strftime("%Y-%m-%d %H:%M:%S")},
        ),
    )
    save_checkpoint(
        output_dir / "policy_offline.ckpt",
        build_act_payload(model, config, normalization, kind="act_policy", extra={"epoch": epochs}),
    )
    _write_json(output_dir / "latest.json", {"checkpoint": str(output_dir / "policy_offline.ckpt")})
    print(f"[act] checkpoints written to {output_dir}", flush=True)


# --------------------------------------------------------------------------------------
# checkpoint discovery / runtime
# --------------------------------------------------------------------------------------

def _run_dirs(config: Dict[str, Any]) -> List[Path]:
    root = OUTPUTS_ROOT / "act" / run_name(config)
    if not root.is_dir():
        return []
    dirs = sorted([path for path in root.iterdir() if path.is_dir()], reverse=True)
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
    for directory in _run_dirs(config):
        for name in ("policy_offline.ckpt", "policy_last.ckpt", "policy_best.ckpt"):
            path = directory / name
            if path.is_file():
                return path
        files = sorted(directory.glob("policy_*.ckpt"))
        if files:
            return files[-1]
    fallback = sorted((OUTPUTS_ROOT / "act").glob("**/policy_offline.ckpt"))
    if fallback:
        return fallback[-1]
    fallback = sorted((OUTPUTS_ROOT / "act").glob("**/policy_last.ckpt"))
    return fallback[-1] if fallback else None


def find_residual_checkpoint() -> Optional[Path]:
    candidates = sorted((OUTPUTS_ROOT / "act").glob("**/residual_last.ckpt"))
    return candidates[-1] if candidates else None


def load_runtime(checkpoint: str | Path, device: Optional[torch.device] = None) -> ActRuntime:
    path = Path(checkpoint)
    if path.is_dir():
        for directory in (path, path / "checkpoints"):
            for name in ("residual_last.ckpt", "policy_offline.ckpt", "policy_last.ckpt"):
                if (directory / name).is_file():
                    path = directory / name
                    break
            if not path.is_dir():
                break
    payload = load_payload(path)
    chosen = device if device is not None else resolve_device("cuda")
    return ActRuntime(payload, chosen)


# --------------------------------------------------------------------------------------
# residual PPO
# --------------------------------------------------------------------------------------

def _residual_scales(config: Dict[str, Any]) -> List[float]:
    values = config.get("residual_scales")
    if values:
        return [float(item) for item in values]
    return default_residual_scales()


def train_online(config: Dict[str, Any]) -> None:
    config = _merged(config)
    seed = int(cfg_get(config, "seed", 0))
    set_seed(seed)
    device = resolve_device(str(cfg_get(config, "device", "cuda")))
    checkpoint = find_offline_checkpoint(config)
    if checkpoint is None:
        raise FileNotFoundError(
            "No offline ACT checkpoint found. Run `bash scripts/train_act.sh` first, "
            "or set `offline_checkpoint` in configs/act_residual.yaml."
        )
    payload = load_payload(checkpoint)
    if payload.get("kind") != "act_policy":
        raise ValueError(f"{checkpoint} is not an offline ACT checkpoint (kind={payload.get('kind')}).")

    base = build_act_from_payload(payload, device)
    base.freeze()
    fingerprint = base.weight_fingerprint()
    normalization = Normalization.from_dict(payload["normalization"])
    print(f"[act-rl] base checkpoint {checkpoint}", flush=True)
    print(f"[act-rl] base fingerprint {fingerprint[:16]}..., parameters frozen", flush=True)

    scales = _residual_scales(config)
    residual = ResidualHead(
        feature_dim=base.feature_dim,
        hidden_dim=int(cfg_get(config, "residual_hidden_dim", 256)),
        action_dim=base.action_dim,
        residual_scales=scales,
        initial_std=float(cfg_get(config, "residual_initial_std", 0.3)),
        use_base_action=bool(cfg_get(config, "residual_use_base_action", True)),
    ).to(device)
    critic = ResidualCritic(
        feature_dim=base.feature_dim,
        hidden_dim=int(cfg_get(config, "critic_hidden_dim", 256)),
    ).to(device)
    agent = ActResidualAgent(base, residual, critic, normalization, config, device)

    updates, episodes_per_update, num_envs = parallel_plan(config)
    env = RemoteEnv(
        str(cfg_get(config, "env_host", "127.0.0.1")),
        int(cfg_get(config, "env_port", 5555)),
        timeout=float(cfg_get(config, "env_timeout", 2400.0)),
        reset_timeout=float(cfg_get(config, "env_reset_timeout", 10800.0)),
    )
    output_dir = act_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "train_config.json", config)
    log_path = output_dir / "online_log.jsonl"
    best_success = -1.0
    print(f"[act-rl] {updates} updates x {episodes_per_update} episodes ({num_envs} scenes)", flush=True)

    def save(name: str, extra: Dict[str, Any]) -> None:
        save_checkpoint(
            output_dir / name,
            {
                "kind": "act_residual_ppo",
                "version": 1,
                "base_config": payload["config"],
                "base_state": base.state_dict(),
                "base_fingerprint": fingerprint,
                "residual": residual.state_dict(),
                "critic": critic.state_dict(),
                "normalization": normalization.to_dict(),
                "config": config,
                "feature_dim": base.feature_dim,
                "action_dim": base.action_dim,
                "residual_scales": scales,
                "residual_use_base_action": bool(cfg_get(config, "residual_use_base_action", True)),
                "image_size": base.image_size,
                "chunk_size": base.chunk_size,
                **extra,
            },
        )

    started = time.time()
    try:
        with log_path.open("w", encoding="utf-8") as handle:
            for update_index in range(1, updates + 1):
                trajectories: List[List[Any]] = []
                episodes: List[Any] = []
                for wave in range(episodes_per_update // num_envs):
                    seeds = [
                        int(cfg_get(config, "seed", 0))
                        + (update_index - 1) * episodes_per_update
                        + wave * num_envs
                        + index
                        for index in range(num_envs)
                    ]
                    wave_trajectories, wave_episodes = agent.collect_wave(env, seeds)
                    trajectories.extend(wave_trajectories)
                    episodes.extend(wave_episodes)

                metrics = agent.update(trajectories)
                if base.weight_fingerprint() != fingerprint:
                    raise RuntimeError("The frozen ACT base changed during residual PPO. Aborting.")
                scores = [episode.score for episode in episodes]
                successes = [1.0 if episode.success else 0.0 for episode in episodes]
                record = {
                    "update": update_index,
                    "episodes": len(episodes),
                    "mean_score": float(np.mean(scores)) if scores else 0.0,
                    "success_rate": float(np.mean(successes)) if successes else 0.0,
                    "max_score": float(np.max(scores)) if scores else 0.0,
                    "mean_steps": float(np.mean([episode.steps for episode in episodes])) if episodes else 0.0,
                    "elapsed_s": time.time() - started,
                }
                record.update(metrics)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"[act-rl] update {update_index}/{updates} score {record['mean_score']:.1f} "
                    f"success {record['success_rate'] * 100:.1f}% "
                    f"pi {metrics['policy_loss']:.4f} v {metrics['value_loss']:.4f}",
                    flush=True,
                )

                save(
                    "residual_last.ckpt",
                    {
                        "update": update_index,
                        "offline_checkpoint": str(checkpoint),
                        "episode_score": record["mean_score"],
                        "success_rate": record["success_rate"],
                        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                )
                if record["success_rate"] > best_success:
                    best_success = record["success_rate"]
                    save("residual_best.ckpt", {"update": update_index, "success_rate": best_success})
                every = int(cfg_get(config, "checkpoint_every", 10))
                if every > 0 and update_index % every == 0:
                    save(f"residual_{update_index:04d}.ckpt", {"update": update_index})
    finally:
        env.close()

    if base.weight_fingerprint() != fingerprint:
        raise RuntimeError("The frozen ACT base changed during residual PPO. Aborting.")
    _write_json(
        output_dir / "summary.json",
        {
            "best_success_rate": best_success,
            "updates": updates,
            "base_fingerprint": fingerprint,
            "checkpoint": str(checkpoint),
        },
    )
    print(f"[act-rl] done; checkpoints in {output_dir}", flush=True)
