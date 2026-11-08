"""Compatibility launcher; both routes use their checkpoint-aware policy servers."""
import argparse
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--kind", choices=("act", "diffusion"), default="act")
    args, remaining = parser.parse_known_args()
    target = Path(__file__).parent / args.kind / "serve_policy.py"
    sys.argv = [str(target), *remaining]
    runpy.run_path(str(target), run_name="__main__")
