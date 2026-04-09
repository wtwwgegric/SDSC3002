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
    parser = argparse.ArgumentParser(description="Plot all-pairs vs nonzero-pairs vs LSH candidates.")
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--title",
        default="Pair-Space Reduction Across Subset Size",
        help="Figure title",
    )
    parser.add_argument(
        "--output-name",
        default="pair_space_reduction.png",
        help="Output image filename",
    )
    return parser.parse_args()


def save_figure(fig, output_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    frame = pd.read_csv(args.summary_csv).sort_values("subset_size").reset_index(drop=True)

    all_pairs = frame["naive_pairs_evaluated"].astype(float)
    frame["all_pairs_ratio"] = 1.0
    frame["nonzero_pair_ratio"] = frame["sparse_exact_nonzero_pairs"] / all_pairs
    frame["lsh_candidate_ratio_from_counts"] = frame["lsh_num_candidates"] / all_pairs

    color_all = "#444444"
    color_nonzero = "#2a6f97"
    color_candidate = "#bc4749"

    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    ax.plot(
        frame["subset_size"],
        frame["all_pairs_ratio"],
        marker="o",
        linewidth=2,
        color=color_all,
        label="All pairs (ratio = 1.0)",
    )
    ax.plot(
        frame["subset_size"],
        frame["nonzero_pair_ratio"],
        marker="o",
        linewidth=2,
        color=color_nonzero,
        label="Sparse-exact nonzero-pair ratio",
    )
    ax.plot(
        frame["subset_size"],
        frame["lsh_candidate_ratio_from_counts"],
        marker="o",
        linewidth=2,
        color=color_candidate,
        label="LSH candidate ratio",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Subset Size (max-items)")
    ax.set_ylabel("Ratio Relative to All Item Pairs (log scale)")
    ax.set_title(args.title)
    ax.legend(loc="upper right")

    full_row = frame.iloc[-1]
    ax.annotate(
        f"nonzero/all = {full_row['nonzero_pair_ratio']:.4f}",
        (full_row["subset_size"], full_row["nonzero_pair_ratio"]),
        textcoords="offset points",
        xytext=(-85, 8),
        fontsize=8,
        color=color_nonzero,
    )
    ax.annotate(
        f"candidate/all = {full_row['lsh_candidate_ratio_from_counts']:.4f}",
        (full_row["subset_size"], full_row["lsh_candidate_ratio_from_counts"]),
        textcoords="offset points",
        xytext=(-85, -18),
        fontsize=8,
        color=color_candidate,
    )

    save_figure(fig, output_dir / args.output_name)


if __name__ == "__main__":
    main()
