"""Offline BC training, episode validation, EMA, optimizer resume and logs."""
from __future__ import annotations

import copy
import json

import torch
from torch.utils.data import DataLoader

from starter.data.demos import ChunkDataset, normalization_stats, split_episodes
from starter.interfaces import ACTION_DIM
from starter.policies.runtime import (FORMAT, append_metrics, build_model, clone_state, load_payload,
                                      model_config, pick_device, restore_rng, rng_state, run_directory,
                                      save_checkpoint, seed_everything)


def train(kind, config):
    seed_everything(int(config.get("seed", 0)))
    device, cfg = pick_device(config), model_config(config)
    resume = load_payload(config["resume"]) if config.get("resume") else None
    if resume and (resume["kind"] != kind or resume["stage"] != "offline"):
        raise ValueError("Offline resume must match policy kind and stage")
    if resume:
        cfg = resume["model_config"]
    train_files, val_files = split_episodes(config)
    split = {"train": [str(p.resolve()) for p in train_files], "validation": [str(p.resolve()) for p in val_files]}
    if resume and resume["split"] != split:
        raise ValueError("Resume requires the same train/validation files and split seed")
    stats = resume["stats"] if resume else normalization_stats(train_files)
    datasets = [ChunkDataset(paths, cfg["chunk"], cfg["image_size"], stats) for paths in (train_files, val_files)]
    batch_size, workers = int(config.get("batch_size", 32)), int(config.get("workers", 0))
    if batch_size < 1 or workers < 0:
        raise ValueError("batch_size must be positive and workers nonnegative")
    loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=(index == 0), num_workers=workers,
                          pin_memory=device.type == "cuda") for index, dataset in enumerate(datasets)]
    model = build_model(kind, cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 1e-4)),
                                  weight_decay=float(config.get("weight_decay", 1e-4)))
    ema = copy.deepcopy(model).eval() if kind == "diffusion" else None
    start, best = 0, float("inf")
    if resume:
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        if ema is not None:
            ema.load_state_dict(resume["ema"])
        start, best = int(resume["epoch"]) + 1, float(resume["best_validation"])
        restore_rng(resume.get("rng"))
    epochs = int(config.get("epochs", 100))
    if epochs <= start:
        raise ValueError("epochs must exceed completed resume epoch")
    directory = run_directory(config, kind, "offline")
    (directory / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    for epoch in range(start, epochs):
        totals = []
        for phase, loader in enumerate(loaders):
            model.train(phase == 0)
            evaluator = model if phase == 0 or ema is None else ema
            total, count = 0.0, 0
            with torch.set_grad_enabled(phase == 0):
                for batch in loader:
                    batch = {key: value.to(device) for key, value in batch.items()}
                    if kind == "act":
                        prediction, kl = evaluator(batch["image"], batch["state"],
                                                   batch["action"] if phase == 0 else None, batch["valid"])
                        error = (prediction - batch["action"]).abs().mean(-1)
                        reconstruction = (error * batch["valid"]).sum() / batch["valid"].sum().clamp_min(1)
                        loss = reconstruction + float(config.get("kl_weight", 0.01)) * kl
                    else:
                        loss = evaluator.training_loss(**batch)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite offline loss")
                    if phase == 0:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("max_grad_norm", 1.0)))
                        optimizer.step()
                        if ema is not None:
                            with torch.no_grad():
                                decay = float(config.get("ema_decay", 0.995))
                                for target, source in zip(ema.parameters(), model.parameters()):
                                    target.lerp_(source, 1 - decay)
                    total += float(loss.detach()) * len(batch["state"])
                    count += len(batch["state"])
            totals.append(total / count)
        improved = totals[1] < best
        best = min(best, totals[1])
        payload = {"format": FORMAT, "kind": kind, "stage": "offline", "action_dim": ACTION_DIM,
                   "model_config": cfg, "stats": stats, "model": clone_state(model),
                   "ema": clone_state(ema) if ema is not None else None, "optimizer": optimizer.state_dict(),
                   "epoch": epoch, "best_validation": best, "config": config, "split": split, "rng": rng_state()}
        save_checkpoint(directory / "policy_last.ckpt", payload)
        if improved:
            save_checkpoint(directory / "policy_best.ckpt", payload)
        frequency = int(config.get("save_every", 10))
        if frequency > 0 and (epoch + 1) % frequency == 0:
            save_checkpoint(directory / f"policy_epoch_{epoch + 1}.ckpt", payload)
        append_metrics(directory, {"epoch": epoch + 1, "train_loss": totals[0], "validation_loss": totals[1],
                                   "train_frames": len(datasets[0]), "validation_frames": len(datasets[1])})
    print(f"[{kind}] checkpoint: {directory / 'policy_last.ckpt'}", flush=True)
    return directory / "policy_last.ckpt"
