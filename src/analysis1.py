"""
analyze_experiments.py

Usage:
    python analyze_experiments.py --folder <path> --prefix <prefix>

Example:
    python analyze_experiments.py --folder ./results --prefix ex1

Outputs:
    - results.csv        : per-fold rows
    - auc_vs_macs.png    : boxplot of AUC per model, x-axis = MACs (log scale)
"""

import argparse
import csv
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np


# ── CSV ────────────────────────────────────────────────────────────────────────

FIELDS = ["model", "macs", "params", "run", "fold", "fold_time", "fold_acc", "fold_auc"]


def load_files(folder: str, prefix: str) -> list[dict]:
    pattern = os.path.join(folder, f"{prefix}*.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching '{pattern}'")
    records = []
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        for experiment in data["experiments"]:
            records.extend(parse_experiment(experiment))
    return records


def parse_experiment(experiment: dict) -> list[dict]:
    rows = []
    model_name = experiment["model"]
    exp_results = experiment["experiment_results"]
    macs = exp_results["macs"]
    params = exp_results["params"]

    for run_entry in exp_results["run_results"]:
        run_num = run_entry["run"]
        for fold_entry in run_entry["run_results"]["fold_results"]:
            fold_num = fold_entry["fold"]
            results = fold_entry["results"]
            fold_time = round(sum(results["epoch_times"]), 4)
            rows.append({
                "model":     model_name,
                "macs":      macs,
                "params":    params,
                "run":       run_num,
                "fold":      fold_num,
                "fold_time": fold_time,
                "fold_acc":  results["test_acc"],
                "fold_auc":  results["test_auc"],
            })
    return rows


def write_csv(records: list[dict], out_path: str):
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved CSV → {out_path}")


# ── Plot ───────────────────────────────────────────────────────────────────────

def make_boxplot(records: list[dict], out_path: str):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.DataFrame(records)

    # Sort by MACs (high → low)
    order = (df.groupby("model")["macs"].first()
               .sort_values(ascending=False)
               .index.tolist())

    macs_vals = df.groupby("model")["macs"].first()
    params_vals = df.groupby("model")["params"].first()

    def human_format(x):
        if x >= 1_000_000:
            return f"{x/1_000_000:.1f}M"
        if x >= 1_000:
            return f"{x/1_000:.1f}K"
        return str(x)

    # Combined labels: "302.6M / 12.4M"
    tick_labels = [
        f"{human_format(macs_vals[m])} / {human_format(params_vals[m])}"
        for m in order
    ]

    # Styling
    sns.set_theme(style="whitegrid", context="talk")
    palette = sns.color_palette("viridis", n_colors=len(order))

    fig, ax = plt.subplots(figsize=(max(8, len(order) * 1.2), 6))

    sns.boxplot(
        data=df,
        x="model",
        y="fold_auc",
        order=order,
        hue="model",
        hue_order=order,
        palette=palette,
        dodge=False,
        fliersize=0,
        linewidth=1.2,
        ax=ax
    )

    sns.stripplot(
        data=df,
        x="model",
        y="fold_auc",
        order=order,
        hue="model",
        hue_order=order,
        palette=palette,
        dodge=False,
        size=4,
        alpha=0.5,
        linewidth=0,
        ax=ax
    )

    # Apply combined tick labels
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(tick_labels, rotation=30, ha="right")

    ax.set_xlabel("MACs / Parameters")
    ax.set_ylabel("AUC")
    # ax.set_title("AUC distribution per model")

    sns.despine()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved plot → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyse contrastive-learning experiment results.")
    parser.add_argument("--folder", default=".", help="Folder containing JSON result files")
    parser.add_argument("--prefix", default="",  help="Filename prefix to filter (e.g. 'ex1')")
    parser.add_argument("--csv",    default="results.csv",      help="Output CSV path")
    parser.add_argument("--plot",   default="auc_vs_macs.png",  help="Output plot path")
    args = parser.parse_args()

    records = load_files(args.folder, args.prefix)
    print(f"Loaded {len(records)} fold records from {args.folder!r} (prefix={args.prefix!r})")

    write_csv(records, args.csv)
    make_boxplot(records, args.plot)


if __name__ == "__main__":
    main()