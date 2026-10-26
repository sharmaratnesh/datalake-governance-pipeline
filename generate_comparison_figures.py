"""
generate_comparison_figures.py — Accuracy and time comparison charts for the paper

Reads evaluation/output/evaluation_summary_<ts>.csv (produced by datalake_evaluate_all.py)
and renders two figures for the Results section:
  1. comparison_accuracy_<ts>.png — grouped bar chart of all 4 accuracy metrics
     (category, security, linkage, overall) across every method.
  2. comparison_time_accuracy_<ts>.png — overall accuracy vs processing time
     scatter, showing the speed/accuracy trade-off across automated methods
     and the human/expert baseline.

Usage:
  python generate_comparison_figures.py evaluation/output/evaluation_summary_20260629_214431.csv
"""

import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR  = Path(__file__).parent
FIG_DIR   = BASE_DIR / "research_paper" / "figures"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Lightened Okabe-Ito colourblind-safe palette (Okabe & Ito, 2008), tinted ~25%
# toward white for a softer, print-friendly look. Pure black (ground truth)
# replaced with a neutral slate grey so no bar reads as "harsh"/near-black.
METHOD_COLORS = {
    "Human (avg)":      "#4D94C4",  # light blue
    "Expert Baseline":  "#6E6E6E",  # slate grey (ground truth reference)
    "Rule-Based":       "#33B08D",  # light bluish green
    "NLP":              "#EFB338",  # light orange
    "LLaMA 3.2 1B":     "#E17E33",  # light vermillion
    "Phi-3 Mini":       "#D696B7",  # light reddish purple
}

# Raw evaluation-summary method names -> standardized display names used in
# the paper text/tables, so figure legends/annotations match the prose
# exactly (previously these charts leaked the raw Ollama tag strings, e.g.
# "LLM-llama3.2-1b", into the rendered image).
DISPLAY_NAMES = {
    "LLM-llama3.2-1b": "LLaMA 3.2 1B",
    "LLM-phi3-mini":   "Phi-3 Mini",
}


def plot_accuracy_bars(df: pd.DataFrame, out_path: Path) -> None:
    metrics = ["category_accuracy", "security_accuracy", "linkage_accuracy", "overall_accuracy"]
    metric_labels = ["Category", "Security", "Linkage", "Overall"]
    methods = df["method"].tolist()

    x = np.arange(len(metrics))
    n = len(methods)
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, method in enumerate(methods):
        vals = df.loc[df["method"] == method, metrics].values.flatten()
        offset = (i - (n - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=method, color=METHOD_COLORS.get(method, "#888888"))

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_title("Accuracy by Method Across Four Evaluation Metrics (n = 55 files)", fontsize=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8.5, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_path}")


LABEL_OFFSETS = {
    "Rule-Based": (8, -14),
    "NLP":        (8, 10),
}


def plot_time_vs_accuracy(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for _, row in df.iterrows():
        ax.scatter(row["time_seconds"], row["overall_accuracy"], s=160,
                   color=METHOD_COLORS.get(row["method"], "#888888"), zorder=3,
                   edgecolors="white", linewidths=1.2)
        ax.annotate(row["method"], (row["time_seconds"], row["overall_accuracy"]),
                    textcoords="offset points", xytext=LABEL_OFFSETS.get(row["method"], (8, 6)),
                    fontsize=8.5)

    ax.set_xscale("log")
    ax.set_xlabel("Processing time (seconds, log scale)", fontsize=10)
    ax.set_ylabel("Overall accuracy", fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_title("Speed vs Accuracy Trade-off Across Methods (n = 55 files)", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate accuracy/time comparison figures from an evaluation summary CSV")
    parser.add_argument("summary_csv", type=Path, help="Path to evaluation_summary_<ts>.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.summary_csv)
    df["method"] = df["method"].replace(DISPLAY_NAMES)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plot_accuracy_bars(df, FIG_DIR / f"comparison_accuracy_{TIMESTAMP}.png")
    plot_time_vs_accuracy(df, FIG_DIR / f"comparison_time_accuracy_{TIMESTAMP}.png")


if __name__ == "__main__":
    main()
