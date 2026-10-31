#!/usr/bin/env python3
"""RoboDojo / XPolicyLab policy server for the Diffusion Policy checkpoint.

The eval client speaks XPolicyLab's WebSocket protocol.  We reuse XPolicyLab's
``PolicyServer`` (which handles framing, keep-alive, image decoding and
exactly-once retries) and only implement the model adapter methods:
``reset`` / ``update_obs`` / ``update_obs_batch`` / ``get_action`` /
``get_action_batch``.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _add_xpolicylab_paths() -> Path:
    candidates = []
    configured = os.environ.get("XPL_ROOT")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            ROOT / "third_party" / "RoboDojo" / "XPolicyLab",
            ROOT / "third_party" / "XPolicyLab",
        ]
    )
    for candidate in candidates:
        if (candidate / "client_server").is_dir() or (
            candidate / "XPolicyLab" / "utils" / "process_data.py"
        ).is_file():
            # `client_server` is top-level; the `XPolicyLab` package needs its
            # parent on sys.path as well.
            for entry in (candidate, candidate.parent):
                if str(entry) not in sys.path:
                    sys.path.insert(0, str(entry))
            return candidate
    raise ImportError(
        "Could not locate the XPolicyLab checkout. Run `bash scripts/bootstrap_third_party.sh` "
        "inside startcode, or export XPL_ROOT=/path/to/XPolicyLab."
    )


def _find_checkpoint(root: Path, name: str) -> Path:
    path = Path(name).expanduser()
    if path.is_file():
        return path
    if path.is_dir():
        for candidate in ("online_last.ckpt", "offline_last.ckpt", "last.ckpt", "best.ckpt"):
            if (path / candidate).is_file():
                return path / candidate
        nested = sorted((path / "checkpoints").glob("*.ckpt"))
        if nested:
            return nested[-1]

    outputs = root / "outputs" / "dp"
    patterns = {
        "latest_online": "**/checkpoints/online_last.ckpt",
        "online": "**/checkpoints/online_last.ckpt",
        "latest_offline": "**/checkpoints/offline_last.ckpt",
        "offline": "**/checkpoints/offline_last.ckpt",
        "latest": "**/checkpoints/online_last.ckpt",
    }
    pattern = patterns.get(name.lower())
    if pattern:
        matches = sorted(outputs.glob(pattern))
        if not matches and name.lower() in ("latest", "latest_offline", "offline"):
            matches = sorted(outputs.glob("**/checkpoints/offline_last.ckpt"))
        if not matches and name.lower() in ("latest", "latest_online", "online"):
            matches = sorted(outputs.glob("**/checkpoints/last.ckpt"))
        if matches:
            return matches[-1]

    # A bare filename: search the output tree.
    matches = sorted(outputs.glob(f"**/{name}"))
    if matches:
        return matches[-1]
    relative = root / name
    if relative.is_file():
        return relative
    raise FileNotFoundError(
        f"Checkpoint '{name}' not found under {outputs} or as a path. "
        "Train first, then set `ckpt` in configs/eval_dp.yaml."
    )


def _actions_to_dicts(chunk: np.ndarray) -> List[Dict[str, List[float]]]:
    from starter.interfaces import unpack_ee_action

    actions: List[Dict[str, List[float]]] = []
    for action in np.asarray(chunk, dtype=np.float64):
        actions.append(unpack_ee_action(action.tolist()))
    return actions


class DiffusionEvalModel:
    def __init__(self, runtime, chunk_override: Optional[int] = None) -> None:
        self.runtime = runtime
        if chunk_override:
            self.runtime.eval_chunk_size = int(chunk_override)
        self.observation: Optional[Dict[str, Any]] = None
        self.observations: List[Dict[str, Any]] = []

    # required hooks -----------------------------------------------------------
    def reset(self) -> None:
        self.observation = None
        self.observations = []

    def prepare_case(self, case_meta: Any = None) -> None:
        self.reset()

    def on_trial_end(self, result: Any = None) -> None:
        return None

    def update_obs(self, obs: Dict[str, Any]) -> None:
        self.observation = obs

    def update_obs_batch(self, obs_list: List[Dict[str, Any]]) -> None:
        self.observations = list(obs_list)

    def get_action(self) -> List[Dict[str, List[float]]]:
        if self.observation is None:
            raise RuntimeError("get_action called before update_obs")
        chunk = self.runtime.act_chunk_from_obs(self.observation)
        return _actions_to_dicts(chunk)

    def get_action_batch(self, env_idx_list: Any = None) -> List[List[Dict[str, List[float]]]]:
        if not self.observations:
            raise RuntimeError("get_action_batch called before update_obs_batch")
        return [_actions_to_dicts(self.runtime.act_chunk_from_obs(obs)) for obs in self.observations]


def main() -> None:
    parser = argparse.ArgumentParser(description="Diffusion Policy eval server.")
    parser.add_argument("--ckpt", required=True, help="checkpoint path or latest_online/latest_offline")
    parser.add_argument("--task", default="stack_blocks")
    parser.add_argument("--env-cfg", default="arx_x5")
    parser.add_argument("--action-type", default="ee")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk", type=int, default=0, help="override eval chunk length (0 = checkpoint default)")
    parser.add_argument("--ping-interval", type=float, default=20.0)
    parser.add_argument("--ping-timeout", type=float, default=20.0)
    args = parser.parse_args()

    _add_xpolicylab_paths()
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig

    from starter.policies.diffusion import load_runtime
    from starter.policies.dp_core import resolve_device

    checkpoint = _find_checkpoint(ROOT, args.ckpt)
    print(f"[serve-dp] loading {checkpoint}", flush=True)
    runtime = load_runtime(checkpoint, resolve_device(args.device))
    print(
        f"[serve-dp] checkpoint kind={runtime.payload.get('kind')} "
        f"chunk={runtime.chunk_size} eval_chunk={runtime.eval_chunk_size}",
        flush=True,
    )

    model = DiffusionEvalModel(runtime, args.chunk)
    server = PolicyServer(
        model,
        PolicyServerConfig(
            host=args.host,
            port=int(args.port),
            ws_ping_interval_s=args.ping_interval,
            ws_ping_timeout_s=args.ping_timeout,
        ),
    )
    print(f"[serve-dp] listening on ws://{args.host}:{args.port}", flush=True)
    try:
        asyncio.run(server.serve_forever())
    except KeyboardInterrupt:
        print("[serve-dp] interrupted", flush=True)


if __name__ == "__main__":
    main()
