"""Policy server shared by the ACT and diffusion evaluation entry points.

Loads a checkpoint produced by ``scripts/train_act.sh`` / ``scripts/train_dp.sh``
and answers the ``reset`` / ``update_obs`` / ``get_action`` RPCs that
``XPolicyLab.policy.<name>.deploy`` drives over the XPolicyLab websocket
protocol.

Started by ``policies/act/setup_eval_policy_server.sh`` or
``policies/diffusion/setup_eval_policy_server.sh``; it can also be run by hand:

    python policies/eval_server.py --kind act --ckpt latest --port 5555
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
XPL_ROOT = REPO_ROOT / "third_party" / "XPolicyLab"
ROBODOJO_ROOT = REPO_ROOT / "third_party" / "RoboDojo"
for entry in (REPO_ROOT / "src", XPL_ROOT, XPL_ROOT.parent):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from starter.policies.act import load_policy  # noqa: E402
from XPolicyLab.utils.process_data import unpack_robot_state  # noqa: E402
from client_server.ws.model_server import PolicyServer, PolicyServerConfig  # noqa: E402


def robot_action_dim_info(env_cfg: str) -> dict:
    """Read ``env_cfg/<name>.yml`` from RoboDojo, as its own client does.

    XPolicyLab's helper resolves ``../../env_cfg`` relative to itself, which only
    works when XPolicyLab sits inside RoboDojo (the container layout).
    """

    import json

    import yaml

    env_cfg_file = ROBODOJO_ROOT / "env_cfg" / f"{env_cfg}.yml"
    if not env_cfg_file.is_file():
        raise SystemExit(f"missing env cfg: {env_cfg_file}")
    cfg = yaml.safe_load(env_cfg_file.read_text(encoding="utf-8"))
    robots = json.loads(
        (ROBODOJO_ROOT / "env_cfg" / "robot" / "_robot_info.json").read_text(encoding="utf-8")
    )
    return robots[cfg["config"]["robot"]]


def resolve_checkpoint(kind: str, value: str) -> Path:
    """Accept a path, ``latest``, or a run name under ``outputs/<kind>``."""

    candidate = Path(value).expanduser()
    if candidate.is_file():
        return candidate
    if candidate.is_dir() and (candidate / "policy_last.ckpt").is_file():
        return candidate / "policy_last.ckpt"

    # Training writes outputs/<kind>/<stage>/seed<k>/<stamp>/, so search
    # recursively rather than assuming a fixed depth.
    root = REPO_ROOT / "outputs" / kind
    if value in {"", "latest", "last"}:
        matches = sorted(
            glob.glob(str(root / "**" / "policy_last.ckpt"), recursive=True),
            key=lambda item: Path(item).stat().st_mtime,
        )
    else:
        matches = sorted(
            glob.glob(str(root / "**" / f"policy_{value}.ckpt"), recursive=True)
            + glob.glob(str(root / "**" / f"*{value}*" / "policy_last.ckpt"), recursive=True)
        )
    if not matches:
        raise SystemExit(
            f"checkpoint '{value}' not found under {root}. Run the matching "
            f"training script first, or pass an explicit path to a policy_*.ckpt."
        )
    return Path(matches[-1])


class ServePolicy:
    """Adapter between RoboDojo observations and ``starter.policies.runtime``.

    All preprocessing (image resize, state normalization, quaternion/gripper
    projection) lives in ``starter.policies.runtime.Policy``, so this class only
    moves RPC payloads in and action dicts out.
    """

    def __init__(self, checkpoint: Path, env_cfg: str, action_type: str, device: str) -> None:
        try:
            self.policy = load_policy(checkpoint, device)
        except ValueError as error:
            raise SystemExit(
                f"{error}. This checkpoint predates the current policy format; "
                "retrain with scripts/train_act.sh or scripts/train_dp.sh."
            ) from error
        self.robot_info = robot_action_dim_info(env_cfg)
        self.action_type = action_type
        self.obs: dict | None = None
        self.batch_obs: dict[int, dict] = {}
        config = getattr(self.policy, "config", {}) or {}
        print(f"[policy-server] checkpoint={checkpoint}")
        print(
            f"[policy-server] kind={getattr(self.policy, 'kind', '?')} "
            f"execute_steps={config.get('execute_steps', '?')} device={device}"
        )

    # -- RPC surface (single env) -------------------------------------------
    def reset(self) -> None:
        self.obs = None
        self.batch_obs = {}

    def update_obs(self, obs) -> None:
        self.obs = obs

    def get_action(self):
        if self.obs is None:
            raise RuntimeError("get_action called before update_obs")
        return self._predict([self.obs])[0]

    # -- RPC surface (batched envs) -----------------------------------------
    def update_obs_batch(self, obs) -> None:
        self.batch_obs = {index: item for index, item in enumerate(obs)}

    def get_action_batch(self, env_idx_list=None):
        indices = list(env_idx_list) if env_idx_list is not None else sorted(self.batch_obs)
        observations = [self.batch_obs[index] for index in indices]
        return self._predict(observations)

    # -- inference -----------------------------------------------------------
    def _predict(self, observations: list[dict]) -> list[list[dict]]:
        actions = self.policy.predict(list(observations))
        return [
            unpack_robot_state(chunk, self.action_type, self.robot_info, source_type="obs")
            for chunk in actions
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description="RoboDojo policy server.")
    parser.add_argument("--kind", choices=("act", "diffusion"), default="act")
    parser.add_argument("--ckpt", default="latest")
    parser.add_argument("--env-cfg", default="arx_x5")
    parser.add_argument("--action-type", default="ee")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import torch

    device = args.device if torch.cuda.is_available() else "cpu"
    model = ServePolicy(
        resolve_checkpoint(args.kind, args.ckpt), args.env_cfg, args.action_type, device
    )
    if getattr(model.policy, "kind", args.kind) != args.kind:
        raise SystemExit(
            f"checkpoint kind '{getattr(model.policy, 'kind', '?')}' does not match --kind {args.kind}"
        )
    server = PolicyServer(model, PolicyServerConfig(host=args.host, port=args.port))
    print(f"[policy-server] listening on ws://{args.host}:{args.port}")
    try:
        asyncio.run(server.serve_forever())
    except KeyboardInterrupt:
        print("\n[policy-server] shutting down")


if __name__ == "__main__":
    main()
