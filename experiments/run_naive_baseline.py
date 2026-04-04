from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.baseline import export_neighbors, naive_topk_jaccard
from src.preprocess import load_preprocessed_artifacts
from src.utils import build_run_metadata, ensure_dir, save_json, timer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the naive O(N²) Jaccard baseline on a small item subset."
    )
    parser.add_argument("--artifacts-dir", required=True, help="Directory containing preprocessing artifacts")
    parser.add_argument("--results-dir", required=True, help="Directory to save outputs")
    parser.add_argument("--top-k", type=int, default=10, help="Neighbors per item")
    parser.add_argument(
        "--max-items",
        type=int,
        default=500,
        help="Number of items to sample for the subset (default: 500)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for item sampling")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.results_dir)

    matrix, metadata = load_preprocessed_artifacts(args.artifacts_dir)
    num_items = matrix.shape[0]

    rng = np.random.default_rng(args.seed)
    subset_size = min(args.max_items, num_items)
    item_subset = sorted(rng.choice(num_items, size=subset_size, replace=False).tolist())

    print(f"Running naive brute-force on {subset_size} / {num_items} items …")

    with timer() as timing:
        neighbors, naive_metrics = naive_topk_jaccard(matrix, top_k=args.top_k, item_subset=item_subset)

    # Export (only subset items have results; map via item_subset positions)
    subset_item_ids = [metadata["item_ids"][i] for i in item_subset]
    payload = export_neighbors(neighbors, item_ids=subset_item_ids, top_k=args.top_k)
    save_json(f"{args.results_dir}/naive_topk.json", payload)

    metrics = build_run_metadata(
        step="naive_baseline",
        artifacts_dir=args.artifacts_dir,
        results_dir=args.results_dir,
        elapsed_seconds=timing["elapsed_seconds"],
        **naive_metrics,
    )
    save_json(f"{args.results_dir}/naive_metrics.json", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
