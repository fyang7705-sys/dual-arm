"""Resolve explicit checkpoints and record exactly which weights were evaluated."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def resolve_checkpoint(root: Path, kind: str, name: str) -> Path:
    root = Path(root)
    candidate = Path(name).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    if candidate.is_file():
        return candidate.resolve()
    offline = ("policy_offline.ckpt", "policy_last.ckpt") if kind == "act" else ("offline_last.ckpt", "last.ckpt")
    online = ("residual_last.ckpt",) if kind == "act" else ("online_last.ckpt",)
    if candidate.is_dir():
        choices = [p for directory in (candidate, candidate / "checkpoints")
                   for filename in online + offline for p in [directory / filename] if p.is_file()]
        if len(choices) == 1:
            return choices[0].resolve()
        raise ValueError(f"Pass a checkpoint file, not an ambiguous run directory: {candidate}")
    aliases = {
        "latest_offline": offline, "offline": offline,
        "latest_online": online, "online": online,
        "latest_residual": online, "residual": online,
        "latest": online + offline,
    }
    outputs = root / "outputs" / ("act" if kind == "act" else "dp")
    filenames = aliases.get(name, (name,))
    # Deduplicate aliases such as policy_offline/policy_last belonging to one run.
    runs = {}
    for filename in filenames:
        for path in sorted(outputs.rglob(filename)):
            if path.is_file():
                runs.setdefault(path.parent, path)
    if len(runs) == 1:
        return next(iter(runs.values())).resolve()
    if runs:
        choices = "\n".join(str(p) for p in runs.values())
        raise ValueError(f"Multiple runs match {name!r}; pass an explicit .ckpt path:\n{choices}")
    raise FileNotFoundError(f"Checkpoint {name!r} not found under {outputs}; train first or pass an explicit path")


def write_eval_manifest(checkpoint: Path, runtime, kind: str, seed: int) -> Path:
    checkpoint = checkpoint.resolve()
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    run_id = os.environ.get("ROBODOJO_RUN_ID", "manual")
    root = Path(os.environ.get("POLICY_MANIFEST_DIR", checkpoint.parents[0] / "eval_metadata"))
    root.mkdir(parents=True, exist_ok=True)
    payload = runtime.payload
    metadata = {
        "checkpoint": str(checkpoint), "sha256": digest.hexdigest(), "run_id": run_id,
        "kind": payload.get("kind"), "epoch": payload.get("epoch"), "seed": seed,
        "config": payload.get("config", {}), "eval_chunk_size": runtime.eval_chunk_size,
        "eval_sampler": getattr(runtime, "sampler", None),
    }
    path = root / f"{kind}_checkpoint.json"
    path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(f"[eval] checkpoint manifest: {path}\n{json.dumps(metadata, default=str)}", flush=True)
    return path
