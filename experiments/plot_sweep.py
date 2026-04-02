from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot LSH sweep results.")
    parser.add_argument("--summary-csv", required=True, help="Path to sweep_summary.csv")
    parser.add_argument("--output-dir", required=True, help="Directory to save PNG figures")
    return parser.parse_args()


def annotate_points(ax, frame: pd.DataFrame, x_col: str, y_col: str) -> None:
    for _, row in frame.iterrows():
        ax.annotate(
            f"b={int(row['bands'])}, r={int(row['rows_per_band'])}",
            (row[x_col], row[y_col]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )


def save_figure(fig, output_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    frame = pd.read_csv(args.summary_csv)
    frame = frame.sort_values(["num_hashes", "bands"]).reset_index(drop=True)

    hash_values = sorted(frame["num_hashes"].unique())
    colors = plt.cm.tab10(range(len(hash_values)))
    color_map = {hash_value: colors[index] for index, hash_value in enumerate(hash_values)}

    fig, ax = plt.subplots(figsize=(8, 5))
    for hash_value in hash_values:
        subset = frame.loc[frame["num_hashes"] == hash_value]
        ax.plot(subset["total_seconds"], subset["recall_at_k"], marker="o", label=f"hashes={hash_value}", color=color_map[hash_value])
        annotate_points(ax, subset, "total_seconds", "recall_at_k")
    ax.set_xlabel("Total Time (seconds)")
    ax.set_ylabel("Recall@k")
    ax.set_title("Recall vs Total Time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, output_dir / "recall_vs_time.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for hash_value in hash_values:
        subset = frame.loc[frame["num_hashes"] == hash_value]
        ax.plot(subset["candidate_ratio"], subset["recall_at_k"], marker="o", label=f"hashes={hash_value}", color=color_map[hash_value])
        annotate_points(ax, subset, "candidate_ratio", "recall_at_k")
    ax.set_xlabel("Candidate Ratio")
    ax.set_ylabel("Recall@k")
    ax.set_title("Recall vs Candidate Ratio")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, output_dir / "recall_vs_candidate_ratio.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for hash_value in hash_values:
        subset = frame.loc[frame["num_hashes"] == hash_value]
        ax.plot(subset["num_candidates"], subset["verification_seconds"], marker="o", label=f"hashes={hash_value}", color=color_map[hash_value])
        annotate_points(ax, subset, "num_candidates", "verification_seconds")
    ax.set_xlabel("Candidate Count")
    ax.set_ylabel("Verification Time (seconds)")
    ax.set_title("Verification Time vs Candidate Count")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, output_dir / "verification_vs_candidates.png")


if __name__ == "__main__":
    main()