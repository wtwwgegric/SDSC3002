from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.baseline import exact_topk_jaccard, export_neighbors
from src.preprocess import load_preprocessed_artifacts
from src.utils import build_run_metadata, ensure_dir, save_json, timer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the exact Jaccard baseline.")
    parser.add_argument("--artifacts-dir", required=True, help="Directory containing preprocessing artifacts")
    parser.add_argument("--results-dir", required=True, help="Directory to save exact top-k outputs")
    parser.add_argument("--top-k", type=int, default=10, help="Number of neighbors to keep per item")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.results_dir)
    matrix, metadata = load_preprocessed_artifacts(args.artifacts_dir)

    with timer() as timing:
        neighbors, baseline_metrics = exact_topk_jaccard(matrix, top_k=args.top_k)

    payload = export_neighbors(neighbors, item_ids=metadata["item_ids"], top_k=args.top_k)
    save_json(f"{args.results_dir}/exact_topk.json", payload)

    metrics = build_run_metadata(
        step="baseline",
        artifacts_dir=args.artifacts_dir,
        results_dir=args.results_dir,
        elapsed_seconds=timing["elapsed_seconds"],
        **baseline_metrics,
    )
    save_json(f"{args.results_dir}/baseline_metrics.json", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
