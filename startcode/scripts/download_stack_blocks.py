"""Download only selected ARX X5 stack-blocks HDF5 episodes on the server."""
from pathlib import Path
import argparse


def main():
    from huggingface_hub import HfApi, hf_hub_download
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=0, help="0 means all episodes; start with 5 for a wiring run")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    if args.episodes < 0:
        parser.error("episodes must be nonnegative")
    root = Path(__file__).resolve().parents[1]
    prefix = "data/RoboDojo/stack_blocks/arx_x5/data/"
    repo = "RoboDojo-Benchmark/RoboDojo"
    files = sorted(name for name in HfApi().list_repo_files(repo, repo_type="dataset", revision=args.revision)
                   if name.startswith(prefix + "episode_") and name.endswith(".hdf5"))
    if args.episodes:
        files = files[:args.episodes]
    if len(files) < 2:
        raise RuntimeError("At least two episodes are needed for train/validation")
    for index, name in enumerate(files):
        print(f"[{index + 1}/{len(files)}] {name}", flush=True)
        hf_hub_download(repo, name, repo_type="dataset", revision=args.revision, local_dir=root)


if __name__ == "__main__":
    main()
