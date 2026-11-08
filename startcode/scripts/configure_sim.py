"""Adjust only ignored runtime camera/parallel settings, never task rewards."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def configure(root, num_envs: int) -> None:
    if int(num_envs) < 1:
        raise ValueError("num_envs must be positive")
    root = Path(root).expanduser().resolve()
    sim_path = root / "env_cfg/sim/sim_config.yml"
    camera_path = root / "env_cfg/camera/camera_config.yml"
    if sim_path.is_file():
        sim = yaml.safe_load(sim_path.read_text(encoding="utf-8")) or {}
        sim.setdefault("scene", {})["num_envs"] = int(num_envs)
        sim_path.write_text(yaml.safe_dump(sim, sort_keys=False), encoding="utf-8")
    if camera_path.is_file():
        cameras = yaml.safe_load(camera_path.read_text(encoding="utf-8")) or {}
        for name, cfg in (cameras.get("annotator") or {}).items():
            if name != "common" and isinstance(cfg, dict):
                cfg["enabled"] = name == "cam_head"
        camera_path.write_text(yaml.safe_dump(cameras, sort_keys=False), encoding="utf-8")
    for name in ("StudentACT", "StudentDiffusion"):
        deploy = root / "XPolicyLab/policy" / name / "deploy.yml"
        if deploy.is_file():
            policy = yaml.safe_load(deploy.read_text(encoding="utf-8")) or {}
            policy["eval_batch"] = int(num_envs) > 1
            deploy.write_text(yaml.safe_dump(policy, sort_keys=False), encoding="utf-8")
    print(f"[configure_sim] num_envs={num_envs}, cameras: head only")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robodojo-root", required=True)
    parser.add_argument("--num-envs", type=int, default=1)
    args = parser.parse_args()
    configure(args.robodojo_root, args.num_envs)


if __name__ == "__main__":
    main()
