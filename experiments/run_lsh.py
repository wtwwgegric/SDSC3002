from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipeline import run_lsh_pipeline
from src.preprocess import load_preprocessed_artifacts
from src.utils import ensure_dir, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MinHash + LSH for approximate item similarity.")
    parser.add_argument("--artifacts-dir", required=True, help="Directory containing preprocessing artifacts")
    parser.add_argument("--results-dir", required=True, help="Directory to save LSH outputs")
    parser.add_argument("--num-hashes", type=int, required=True, help="Number of MinHash functions")
    parser.add_argument("--bands", type=int, required=True, help="Number of LSH bands")
    parser.add_argument("--rows-per-band", type=int, required=True, help="Rows in each band")
    parser.add_argument("--top-k", type=int, default=10, help="Number of neighbors to keep per item")
    parser.add_argument("--baseline", help="Path to exact_topk.json for recall evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for hash generation and sampling")
    parser.add_argument("--quality-samples", type=int, default=200, help="Sample size for MinHash quality check")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.results_dir)
    matrix, metadata = load_preprocessed_artifacts(args.artifacts_dir)
    baseline_payload = load_json(args.baseline) if args.baseline else None
    metrics, _ = run_lsh_pipeline(
        item_user_matrix=matrix,
        item_ids=metadata["item_ids"],
        num_hashes=args.num_hashes,
        bands=args.bands,
        rows_per_band=args.rows_per_band,
        top_k=args.top_k,
        seed=args.seed,
        quality_samples=args.quality_samples,
        baseline_payload=baseline_payload,
        results_dir=args.results_dir,
        save_neighbors=True,
    )
    metrics["artifacts_dir"] = args.artifacts_dir
    print(metrics)


if __name__ == "__main__":
    main()
