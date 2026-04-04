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
    parser = argparse.ArgumentParser(description="Plot comparison scaling results.")
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def save_figure(fig, output_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def annotate_series(ax, x_values, y_values, fmt: str) -> None:
    for x_value, y_value in zip(x_values, y_values, strict=False):
        ax.annotate(
            fmt.format(y=y_value),
            (x_value, y_value),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    frame = pd.read_csv(args.summary_csv).sort_values("subset_size").reset_index(drop=True)

    crossover_rows = frame.loc[frame["speedup_lsh_vs_sparse_exact"] > 1.0]
    crossover_subset = int(crossover_rows.iloc[0]["subset_size"]) if not crossover_rows.empty else None

    color_naive = "#b23a48"
    color_exact = "#2a6f97"
    color_lsh = "#588157"
    accent = "#f4a259"

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(frame["subset_size"], frame["naive_time_seconds"], marker="o", linewidth=2, color=color_naive, label="Naive brute-force")
    ax.plot(frame["subset_size"], frame["sparse_exact_time_seconds"], marker="o", linewidth=2, color=color_exact, label="Sparse-exact")
    ax.plot(frame["subset_size"], frame["lsh_time_seconds"], marker="o", linewidth=2, color=color_lsh, label="LSH")
    if crossover_subset is not None:
        crossover_time = frame.loc[frame["subset_size"] == crossover_subset, "lsh_time_seconds"].iloc[0]
        ax.axvline(crossover_subset, color=accent, linestyle="--", linewidth=1.5, label=f"Crossover at {crossover_subset}")
        ax.annotate(
            f"LSH starts beating sparse-exact\nat subset={crossover_subset}",
            (crossover_subset, crossover_time),
            textcoords="offset points",
            xytext=(8, -28),
            fontsize=9,
            color=accent,
        )
    ax.set_xlabel("Subset Size (max-items)")
    ax.set_ylabel("Total Time (seconds)")
    ax.set_title("Three-tier Runtime Comparison Across Subset Size")
    ax.legend()
    save_figure(fig, output_dir / "runtime_vs_subset_size.png")

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(frame["subset_size"], frame["speedup_lsh_vs_naive"], marker="o", linewidth=2, color=color_naive, label="LSH speedup vs naive")
    ax.plot(frame["subset_size"], frame["speedup_lsh_vs_sparse_exact"], marker="o", linewidth=2, color=color_exact, label="LSH speedup vs sparse-exact")
    ax.axhline(1.0, color="#444444", linestyle="--", linewidth=1)
    if crossover_subset is not None:
        ax.axvline(crossover_subset, color=accent, linestyle="--", linewidth=1.5)
    ax.set_xlabel("Subset Size (max-items)")
    ax.set_ylabel("Speedup Ratio")
    ax.set_title("When LSH Becomes Faster Than Exact Methods")
    ax.legend()
    save_figure(fig, output_dir / "speedup_vs_subset_size.png")

    fig, ax1 = plt.subplots(figsize=(8.5, 5.2))
    ax1.plot(frame["subset_size"], frame["lsh_recall_at_k"], marker="o", linewidth=2, color=color_lsh, label="Recall@k")
    ax1.set_xlabel("Subset Size (max-items)")
    ax1.set_ylabel("Recall@k", color=color_lsh)
    ax1.tick_params(axis="y", labelcolor=color_lsh)
    ax1.set_ylim(0, min(1.02, max(1.0, frame["lsh_recall_at_k"].max() + 0.05)))

    ax2 = ax1.twinx()
    ax2.plot(frame["subset_size"], frame["lsh_candidate_ratio"], marker="s", linewidth=2, color=color_exact, label="Candidate ratio")
    ax2.set_ylabel("Candidate Ratio", color=color_exact)
    ax2.tick_params(axis="y", labelcolor=color_exact)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
    ax1.set_title("LSH Quality and Search Budget Across Subset Size")
    save_figure(fig, output_dir / "lsh_quality_vs_subset_size.png")


if __name__ == "__main__":
    main()
