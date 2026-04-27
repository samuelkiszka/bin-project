import argparse
import csv
import glob
import json
import os
import re

from scipy.stats import mannwhitneyu, kruskal
import itertools


# ── CSV ────────────────────────────────────────────────────────────────────────

FIELDS = ["model", "pairs_per_epoch", "run", "fold", "fold_time", "fold_acc", "fold_auc"]


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
    pairs_per_epoch = experiment["pairs_per_epoch"]   # ✔️ TOTO JE KLÍČ

    exp_results = experiment["experiment_results"]

    for run_entry in exp_results["run_results"]:
        run_num = run_entry["run"]

        for fold_entry in run_entry["run_results"]["fold_results"]:
            fold_num = fold_entry["fold"]
            results = fold_entry["results"]

            fold_time = round(sum(results["epoch_times"]), 4)

            rows.append({
                "model": model_name,
                "pairs_per_epoch": pairs_per_epoch,
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

def embedding_dim_global_effect(records: list[dict], alpha=0.05):
    import pandas as pd

    df = pd.DataFrame(records)

    print("\n=== Embedding dimension effect (global + post-hoc) ===")

    for model in df["model"].unique():
        sub = df[df["model"] == model]

        dims = sorted(sub["pairs_per_epoch"].unique())

        print(f"\nModel: {model}")

        if len(dims) < 2:
            print("  Not enough embedding dimensions.")
            continue

        groups = [sub[sub["pairs_per_epoch"] == d]["fold_auc"].values for d in dims]

        # ── GLOBAL TEST ─────────────────────────────────────────────
        H, p_global = kruskal(*groups)

        if p_global < alpha:
            print(f"  ✔ Global effect present (Kruskal–Wallis p={p_global:.4g})")
        else:
            print(f"  ✘ No global effect (Kruskal–Wallis p={p_global:.4g})")
            continue  # když není efekt, post-hoc nemá smysl

        # ── POST-HOC TESTS ─────────────────────────────────────────
        results = []

        for d1, d2 in itertools.combinations(dims, 2):
            a = sub[sub["pairs_per_epoch"] == d1]["fold_auc"]
            b = sub[sub["pairs_per_epoch"] == d2]["fold_auc"]

            stat, p = mannwhitneyu(a, b, alternative="two-sided")
            results.append((d1, d2, p))

        # Bonferroni
        m = len(results)
        results_adj = [(d1, d2, min(p * m, 1.0)) for d1, d2, p in results]

        # print sorted
        for d1, d2, p in sorted(results_adj, key=lambda x: x[2]):
            if p < alpha:
                mean_a = sub[sub["pairs_per_epoch"] == d1]["fold_auc"].mean()
                mean_b = sub[sub["pairs_per_epoch"] == d2]["fold_auc"].mean()

                direction = "↑" if mean_b > mean_a else "↓"

                print(f"    {d1} vs {d2}: p_adj={p:.4g} {direction}")
            else:
                print(f"    {d1} vs {d2}: n.s. (p_adj={p:.4g})")

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
    emb_order = sorted(df["pairs_per_epoch"].unique())

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
            x="pairs_per_epoch",
            y="fold_auc",
            order=emb_order,
            hue='pairs_per_epoch',
            hue_order=emb_order,
            palette="viridis",
            legend=False,
            fliersize=0,
            linewidth=1.2,
            ax=ax
        )

        sns.stripplot(
            data=sub,
            x="pairs_per_epoch",
            y="fold_auc",
            order=emb_order,
            size=3,
            alpha=0.5,
            linewidth=0,
            ax=ax
        )

        ax.set_title(model, fontsize=14, fontweight="bold")
        ax.set_xlabel("Počet vzorků na epochu")
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
    parser.add_argument("--csv",    default="ex3-results.csv",      help="Output CSV path")
    parser.add_argument("--plot",   default="pairs_per_epoch_results.png",  help="Output plot path")
    args = parser.parse_args()

    records = load_files(args.folder, args.prefix)
    print(f"Loaded {len(records)} fold records from {args.folder!r} (prefix={args.prefix!r})")

    embedding_dim_global_effect(records)

    write_csv(records, args.csv)
    make_boxplot(records, args.plot)


if __name__ == "__main__":
    main()