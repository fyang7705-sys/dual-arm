"""Checkpoint-backed XPolicyLab model and server, using the official WS protocol."""
from __future__ import annotations

import argparse
import asyncio
import sys

from starter.interfaces import unpack_ee_action
from starter.paths import REPO_ROOT
from starter.policies.runtime import Policy, seed_everything


class Model:
    """Same actor/preprocessing/projection as training; no block pose features."""
    def __init__(self, model_cfg):
        self.policy = Policy(model_cfg.get("checkpoint") or model_cfg["ckpt_name"],
                             model_cfg.get("device", "cpu"))
        self.reset()

    def reset(self):
        self.observations = {}

    def update_obs(self, obs):
        self.observations[0] = obs

    def update_obs_batch(self, obs):
        for item in obs:
            self.observations[int(item["env_idx"])] = item

    def get_action(self):
        actions = self.policy.predict([self.observations[0]])[0]
        return [unpack_ee_action(action) for action in actions]

    def get_action_batch(self, env_idx_list=None, obs=None):
        indices = env_idx_list if env_idx_list is not None else obs
        if indices is None:
            indices = sorted(self.observations)
        actions = self.policy.predict([self.observations[int(i)] for i in indices])
        return [[unpack_ee_action(action) for action in chunk] for chunk in actions]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--kind", choices=("act", "diffusion"), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    third = REPO_ROOT / "third_party"
    sys.path[:0] = [str(third), str(third / "XPolicyLab")]
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig
    seed_everything(args.seed)
    model = Model({"checkpoint": args.checkpoint, "device": args.device})
    if model.policy.kind != args.kind:
        raise ValueError("Evaluation policy kind does not match checkpoint")
    server = PolicyServer(model, PolicyServerConfig(host=args.host, port=args.port))
    asyncio.run(server.serve_forever())


if __name__ == "__main__":
    main()
