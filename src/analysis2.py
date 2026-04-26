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
import re

import matplotlib.pyplot as plt
import numpy as np


# ── CSV ────────────────────────────────────────────────────────────────────────

FIELDS = ["model", "emb_dim", "run", "fold", "fold_time", "fold_acc", "fold_auc"]


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
    emb_dim = experiment["embedding_dim"]   # ✔️ TOTO JE KLÍČ

    exp_results = experiment["experiment_results"]

    for run_entry in exp_results["run_results"]:
        run_num = run_entry["run"]

        for fold_entry in run_entry["run_results"]["fold_results"]:
            fold_num = fold_entry["fold"]
            results = fold_entry["results"]

            fold_time = round(sum(results["epoch_times"]), 4)

            rows.append({
                "model": model_name,
                "emb_dim": emb_dim,          # ✔️ přidáno
                "run": run_num,
                "fold": fold_num,
                "fold_time": fold_time,
                "fold_acc": results["test_acc"],
                "fold_auc": results["test_auc"],
            })

    return rows


def write_csv(records: list[dict], out_path: str):
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved CSV → {out_path}")


def extract_model_number(name: str) -> int:
    match = re.search(r"(\d+)$", name)
    return int(match.group(1)) if match else float("inf")

# ── Plot ───────────────────────────────────────────────────────────────────────

def make_boxplot(records: list[dict], out_path: str):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    df = pd.DataFrame(records)

    # ── pořadí modelů (jak přijdou) ───────────────────────────────────────────
    models = sorted(
        df["model"].unique(),
        key=extract_model_number
    )

    # ── embedding dim ────────────────────────────────────────────────────────
    emb_order = sorted(df["emb_dim"].unique())

    sns.set_theme(style="whitegrid", context="talk")

    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(6 * len(models), 6),
        sharey=False
    )

    # když je jen 1 model
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = df[df["model"] == model]

        sns.boxplot(
            data=sub,
            x="emb_dim",
            y="fold_auc",
            order=emb_order,
            hue='emb_dim',
            hue_order=emb_order,
            palette="viridis",
            legend=False,
            fliersize=0,
            linewidth=1.2,
            ax=ax
        )

        sns.stripplot(
            data=sub,
            x="emb_dim",
            y="fold_auc",
            order=emb_order,
            size=3,
            alpha=0.5,
            linewidth=0,
            ax=ax
        )

        ax.set_title(model, fontsize=14, fontweight="bold")
        ax.set_xlabel("Embedding dim")
        ax.set_ylabel("AUC")

        sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved plot → {out_path}")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyse contrastive-learning experiment results.")
    parser.add_argument("--folder", default=".", help="Folder containing JSON result files")
    parser.add_argument("--prefix", default="",  help="Filename prefix to filter (e.g. 'ex1')")
    parser.add_argument("--csv",    default="results.csv",      help="Output CSV path")
    parser.add_argument("--plot",   default="emb_dim_results.png",  help="Output plot path")
    args = parser.parse_args()

    records = load_files(args.folder, args.prefix)
    print(f"Loaded {len(records)} fold records from {args.folder!r} (prefix={args.prefix!r})")

    write_csv(records, args.csv)
    make_boxplot(records, args.plot)


if __name__ == "__main__":
    main()