
import argparse
import csv
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu


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


# ── Statistics ────────────────────────────────────────────────────────────────

def larger_is_better_tests(records: list[dict], alpha=0.05):
    import pandas as pd

    df = pd.DataFrame(records)

    # sort models by MACs (small → large)
    model_macs = df.groupby("model")["macs"].first().sort_values()
    models = list(model_macs.index)

    print("\n=== Larger model superiority tests (one-sided Mann–Whitney U) ===")

    results = []

    # pairwise tests: big vs small
    for i in range(len(models)):
        for j in range(i):
            big = models[i]
            small = models[j]

            big_auc = df[df["model"] == big]["fold_auc"]
            small_auc = df[df["model"] == small]["fold_auc"]

            # H1: big > small
            stat, p = mannwhitneyu(big_auc, small_auc, alternative="greater")
            results.append((big, small, p))

    # Bonferroni correction
    m = len(results)
    results_adj = [(b, s, min(p * m, 1.0)) for (b, s, p) in results]

    # summarize per big model
    for big in models:
        smaller = [s for s in models if model_macs[s] < model_macs[big]]

        if not smaller:
            continue

        pvals = [p for (b, s, p) in results_adj if b == big and s in smaller]

        all_significant = len(pvals) > 0 and all(p < alpha for p in pvals)

        print(f"\nModel: {big}")

        if all_significant:
            print("  ✔ statistically significantly better than all smaller models")
        else:
            print("  ✘ no full dominance over all smaller models")

        for (b, s, p) in results_adj:
            if b == big and s in smaller:
                print(f"    vs {s}: p_adj={p:.4g}")


# ── Plot ───────────────────────────────────────────────────────────────────────

def make_boxplot(records: list[dict], out_path: str):
    import pandas as pd
    import seaborn as sns

    df = pd.DataFrame(records)

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

    tick_labels = [
        f"{human_format(macs_vals[m])} / {human_format(params_vals[m])}"
        for m in order
    ]

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

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(tick_labels, rotation=30, ha="right")

    ax.set_xlabel("MACs / Počet parametrů")
    ax.set_ylabel("AUC")

    sns.despine()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved plot → {out_path}")

def compute_pareto_front(df):
    # aggregate per model
    agg = (df.groupby("model")
             .agg({
                 "fold_auc": "mean",
                 "macs": "first",
                 "params": "first"
             })
             .rename(columns={"fold_auc": "auc"})
             .reset_index())

    pareto = []

    for i, row_i in agg.iterrows():
        dominated = False
        for j, row_j in agg.iterrows():
            if i == j:
                continue
            if (
                (row_j["auc"] >= row_i["auc"]) and
                (row_j["macs"] <= row_i["macs"]) and
                ((row_j["auc"] > row_i["auc"]) or (row_j["macs"] < row_i["macs"]))
            ):
                dominated = True
                break
        if not dominated:
            pareto.append(row_i)

    return agg, pareto

def make_pareto_plot(records: list[dict], out_path: str):
    import pandas as pd
    import seaborn as sns

    df = pd.DataFrame(records)

    agg, pareto_rows = compute_pareto_front(df)
    pareto_df = pd.DataFrame(pareto_rows)

    # log MACs for plotting
    agg["log_macs"] = np.log10(agg["macs"])
    pareto_df["log_macs"] = np.log10(pareto_df["macs"])

    sns.set_theme(style="whitegrid", context="talk")

    plt.figure(figsize=(8, 6))

    # all models
    sns.scatterplot(
        data=agg,
        x="log_macs",
        y="auc",
        hue="model",
        palette="viridis",
        s=80
    )

    # pareto front (sorted by MACs)
    pareto_df = pareto_df.sort_values("macs")
    plt.plot(
        pareto_df["log_macs"],
        pareto_df["auc"],
        linestyle="--",
        color="black",
        linewidth=2,
        label="Pareto front"
    )

    # annotate points
    for _, row in agg.iterrows():
        plt.text(
            row["log_macs"],
            row["auc"],
            row["model"],
            fontsize=9,
            ha="left",
            va="bottom"
        )

    plt.gca().invert_xaxis()
    plt.xlabel("log10(MACs)")
    plt.ylabel("AUC")
    plt.title("Pareto fronta modelů (max AUC, min MACs)")

    sns.despine()
    plt.legend().remove()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved Pareto plot → {out_path}")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyse experiment results.")
    parser.add_argument("--folder", default=".", help="Folder with JSON files")
    parser.add_argument("--prefix", default="", help="Filename prefix filter")
    parser.add_argument("--csv", default="ex1-results.csv")
    parser.add_argument("--plot", default="auc_vs_macs.png")
    parser.add_argument("--pareto", default="pareto.png")
    args = parser.parse_args()

    records = load_files(args.folder, args.prefix)
    print(f"Loaded {len(records)} fold records")

    larger_is_better_tests(records)

    write_csv(records, args.csv)
    make_boxplot(records, args.plot)
    make_pareto_plot(records, args.pareto)


if __name__ == "__main__":
    main()