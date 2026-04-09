from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import ensure_dir


plt.style.use("seaborn-v0_8-whitegrid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot pair-space reduction side by side for two datasets.")
    parser.add_argument("--left-summary-csv", required=True)
    parser.add_argument("--right-summary-csv", required=True)
    parser.add_argument("--left-title", required=True)
    parser.add_argument("--right-title", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", default="pair_space_reduction_side_by_side.png")
    return parser.parse_args()


def prepare_frame(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path).sort_values("subset_size").reset_index(drop=True)
    all_pairs = frame["naive_pairs_evaluated"].astype(float)
    frame["all_pairs_ratio"] = 1.0
    frame["nonzero_pair_ratio"] = frame["sparse_exact_nonzero_pairs"] / all_pairs
    frame["lsh_candidate_ratio_from_counts"] = frame["lsh_num_candidates"] / all_pairs
    return frame


def plot_dataset(ax, frame: pd.DataFrame, title: str) -> None:
    ax.plot(
        frame["subset_size"],
        frame["all_pairs_ratio"],
        marker="o",
        linewidth=2,
        color="#444444",
        label="All pairs (ratio = 1.0)",
    )
    ax.plot(
        frame["subset_size"],
        frame["nonzero_pair_ratio"],
        marker="o",
        linewidth=2,
        color="#2a6f97",
        label="Sparse-exact nonzero-pair ratio",
    )
    ax.plot(
        frame["subset_size"],
        frame["lsh_candidate_ratio_from_counts"],
        marker="o",
        linewidth=2,
        color="#bc4749",
        label="LSH candidate ratio",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Subset Size (max-items)")
    ax.set_title(title)

    full_row = frame.iloc[-1]
    ax.annotate(
        f"nonzero/all = {full_row['nonzero_pair_ratio']:.4f}",
        (full_row["subset_size"], full_row["nonzero_pair_ratio"]),
        textcoords="offset points",
        xytext=(-88, 8),
        fontsize=8,
        color="#2a6f97",
    )
    ax.annotate(
        f"candidate/all = {full_row['lsh_candidate_ratio_from_counts']:.4f}",
        (full_row["subset_size"], full_row["lsh_candidate_ratio_from_counts"]),
        textcoords="offset points",
        xytext=(-88, -18),
        fontsize=8,
        color="#bc4749",
    )


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    left_frame = prepare_frame(args.left_summary_csv)
    right_frame = prepare_frame(args.right_summary_csv)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), sharey=True)
    plot_dataset(axes[0], left_frame, args.left_title)
    plot_dataset(axes[1], right_frame, args.right_title)
    axes[0].set_ylabel("Ratio Relative to All Item Pairs (log scale)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Pair-Space Reduction Comparison: MovieLens 1M vs Amazon", fontsize=15, y=1.02)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output_dir / args.output_name, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
