from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.baseline import export_neighbors
from src.lsh import (
    approximate_topk_from_candidates,
    compute_minhash_signatures,
    generate_lsh_candidates,
    recall_at_k,
    sample_signature_quality,
)
from src.preprocess import load_preprocessed_artifacts
from src.utils import build_run_metadata, ensure_dir, load_json, save_json, timer


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

    with timer() as signature_timing:
        signatures, signature_metrics = compute_minhash_signatures(
            matrix,
            num_hashes=args.num_hashes,
            seed=args.seed,
        )

    signature_quality = sample_signature_quality(
        matrix,
        signatures,
        num_samples=args.quality_samples,
        seed=args.seed,
    )

    with timer() as candidate_timing:
        candidates, candidate_metrics = generate_lsh_candidates(
            signatures,
            bands=args.bands,
            rows_per_band=args.rows_per_band,
        )

    with timer() as verification_timing:
        approx_neighbors, approx_metrics = approximate_topk_from_candidates(
            matrix,
            candidates,
            top_k=args.top_k,
        )

    approx_payload = export_neighbors(approx_neighbors, item_ids=metadata["item_ids"], top_k=args.top_k)
    save_json(f"{args.results_dir}/approx_topk.json", approx_payload)
    save_json(f"{args.results_dir}/signature_quality.json", signature_quality)

    metrics = build_run_metadata(
        step="lsh",
        artifacts_dir=args.artifacts_dir,
        results_dir=args.results_dir,
        signature_seconds=signature_timing["elapsed_seconds"],
        candidate_seconds=candidate_timing["elapsed_seconds"],
        verification_seconds=verification_timing["elapsed_seconds"],
        total_seconds=(
            signature_timing["elapsed_seconds"]
            + candidate_timing["elapsed_seconds"]
            + verification_timing["elapsed_seconds"]
        ),
        **signature_metrics,
        **candidate_metrics,
        **approx_metrics,
        signature_quality=signature_quality,
    )

    if args.baseline:
        baseline_payload = load_json(args.baseline)
        metrics["recall_at_k"] = recall_at_k(baseline_payload, approx_payload, k=args.top_k)

    save_json(f"{args.results_dir}/lsh_metrics.json", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
