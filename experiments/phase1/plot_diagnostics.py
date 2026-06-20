#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import TASKS, read_json, repo_path, write_json


VARIANTS = ("base", "success", "seed_balanced", "difficulty_weighted")


def success_rate(rows):
    if not rows:
        return np.nan
    return float(np.mean([row["success"] for row in rows]))


def rows_for_split(payload, split):
    return [row for row in payload["rows"] if row["split"] == split]


def load_eval(eval_dir, task, variant):
    return read_json(eval_dir / task / f"{variant}.json")


def plot_easy_seed_bias(task, rollout_dir, output_dir):
    stats = read_json(rollout_dir / task / "seed_stats.json")
    xs = []
    ys = []
    for seed, row in stats.items():
        xs.append(row["j_hat"])
        ys.append(row["successes"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(xs, ys, alpha=0.75)
    ax.set_xlabel("DP-Base per-seed success rate J_hat")
    ax.set_ylabel("success rollouts kept")
    ax.set_title(f"{task}: easy-seed bias")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_dir / task / "easy_seed_bias.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_mean_vs_coverage(task, eval_dir, output_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    points = {}
    for variant in VARIANTS:
        payload = load_eval(eval_dir, task, variant)
        split = payload["splits"]["id_heldout"]
        mean_sr = split.get("mean_sr", np.nan)
        coverage = split.get("solved_coverage", np.nan)
        points[variant] = {"mean_sr": mean_sr, "solved_coverage": coverage}
        ax.scatter([coverage], [mean_sr], label=variant)
        ax.annotate(variant, (coverage, mean_sr), fontsize=8)
    ax.set_xlabel("Solved coverage (ID-heldout)")
    ax.set_ylabel("Mean SR (ID-heldout)")
    ax.set_title(f"{task}: mean vs coverage")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_dir / task / "mean_vs_coverage.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path, points


def plot_per_bin(task, rollout_dir, eval_dir, output_dir, bins=5):
    stats = read_json(rollout_dir / task / "seed_stats.json")
    train_seeds = np.array(sorted(int(seed) for seed in stats))
    j_hat = np.array([stats[str(seed)]["j_hat"] for seed in train_seeds])
    quantiles = np.quantile(j_hat, np.linspace(0, 1, bins + 1))
    quantiles[0] -= 1e-6
    quantiles[-1] += 1e-6

    x = np.arange(bins)
    width = 0.18
    fig, ax = plt.subplots(figsize=(8, 4))
    values = {}
    for i, variant in enumerate(VARIANTS):
        payload = load_eval(eval_dir, task, variant)
        rows = rows_for_split(payload, "train_seen")
        vals = []
        for b in range(bins):
            seed_set = {
                int(seed)
                for seed, score in zip(train_seeds, j_hat)
                if quantiles[b] < score <= quantiles[b + 1]
            }
            bin_rows = [row for row in rows if row["env_seed"] in seed_set]
            vals.append(success_rate(bin_rows))
        values[variant] = vals
        ax.bar(x + (i - 1.5) * width, vals, width=width, label=variant)

    ax.set_xlabel("DP-Base J_hat bin (hard to easy)")
    ax.set_ylabel("Success rate")
    ax.set_title(f"{task}: per-bin improvement")
    ax.set_xticks(x)
    ax.set_xticklabels([f"bin {i + 1}" for i in range(bins)])
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = output_dir / task / "per_bin_sr.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path, values


def main():
    parser = argparse.ArgumentParser(description="Plot phase1 DP diagnostic figures.")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--rollout-dir", type=Path, default=repo_path("experiments", "phase1", "rollouts"))
    parser.add_argument("--eval-dir", type=Path, default=repo_path("experiments", "phase1", "eval_results"))
    parser.add_argument("--output-dir", type=Path, default=repo_path("experiments", "phase1", "figures"))
    args = parser.parse_args()

    summary = {}
    for task in args.tasks:
        task_summary = {}
        task_summary["easy_seed_bias"] = str(plot_easy_seed_bias(task, args.rollout_dir, args.output_dir))
        per_bin_path, per_bin_values = plot_per_bin(task, args.rollout_dir, args.eval_dir, args.output_dir)
        mean_path, mean_points = plot_mean_vs_coverage(task, args.eval_dir, args.output_dir)
        task_summary["per_bin_sr"] = str(per_bin_path)
        task_summary["mean_vs_coverage"] = str(mean_path)
        task_summary["per_bin_values"] = per_bin_values
        task_summary["mean_vs_coverage_points"] = mean_points
        summary[task] = task_summary

    out_path = args.output_dir / "summary.json"
    write_json(out_path, summary)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

