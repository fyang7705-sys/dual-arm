"""Adjust only ignored runtime camera/parallel settings, never task rewards."""
from pathlib import Path
import argparse
import yaml


def configure(root, num_envs):
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    root = Path(root)
    sim_path = root / "env_cfg/sim/sim_config.yml"
    camera_path = root / "env_cfg/camera/camera_config.yml"
    sim = yaml.safe_load(sim_path.read_text())
    sim["scene"]["num_envs"] = num_envs
    cameras = yaml.safe_load(camera_path.read_text())
    for name, cfg in cameras["annotator"].items():
        if name != "common":
            cfg["enabled"] = name == "cam_head"
    sim_path.write_text(yaml.safe_dump(sim, sort_keys=False), encoding="utf-8")
    camera_path.write_text(yaml.safe_dump(cameras, sort_keys=False), encoding="utf-8")
    for name in ("StudentACT", "StudentDiffusion"):
        deploy = root / "XPolicyLab/policy" / name / "deploy.yml"
        if deploy.exists():
            policy = yaml.safe_load(deploy.read_text())
            policy["eval_batch"] = num_envs > 1
            deploy.write_text(yaml.safe_dump(policy, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robodojo-root", required=True)
    parser.add_argument("--num-envs", type=int, default=1)
    args = parser.parse_args()
    configure(args.robodojo_root, args.num_envs)
