"""Identical configuration overrides for scripts and direct Python entry points."""
from __future__ import annotations

import argparse
from pathlib import Path

from starter.config import load_config, parse_simple_yaml


def arguments(kind, argv=None):
    parser = argparse.ArgumentParser(description=f"{kind}: offline cloning or online PPO")
    parser.add_argument("stage", choices=("offline", "online"))
    parser.add_argument("--config", default="configs/env.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    for item in args.set:
        key, separator, value = item.partition("=")
        if not separator or not key:
            parser.error("--set requires KEY=VALUE")
        config.update(parse_simple_yaml(f"{key}: {value}"))
    config["config_path"] = str(Path(args.config).resolve())
    return args.stage, config


def main(kind, argv=None):
    stage, config = arguments(kind, argv)
    if stage == "offline":
        from starter.policies.offline import train
    else:
        from starter.policies.online import train
    train(kind, config)
