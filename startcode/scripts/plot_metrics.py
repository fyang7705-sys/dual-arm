"""Render loss/reward curves from a run's JSONL metrics without a simulator."""
from pathlib import Path
import argparse
import json


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    records = [json.loads(line) for line in (args.run_dir / "metrics.jsonl").read_text().splitlines() if line]
    if not records:
        raise ValueError("No metrics to plot")
    offline = "epoch" in records[0]
    keys = ["train_loss", "validation_loss"] if offline else ["return_mean", "success_rate", "benchmark_score", "policy_loss", "value_loss"]
    fig, axes = plt.subplots(len(keys), 1, figsize=(8, 2.5 * len(keys)), squeeze=False)
    for key, ax in zip(keys, axes[:, 0]):
        ax.plot([r["epoch" if offline else "update"] for r in records], [r[key] for r in records])
        ax.set_ylabel(key)
        ax.grid(alpha=.25)
    axes[-1, 0].set_xlabel("epoch" if offline else "PPO update")
    fig.tight_layout()
    fig.savefig(args.run_dir / "curves.png", dpi=150)


if __name__ == "__main__":
    main()
