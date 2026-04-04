"""Three-tier comparison: naive brute-force vs sparse-exact vs LSH.

All three methods run on the *same* small item subset so timings are directly
comparable.  LSH recall is measured against the sparse-exact result.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.baseline import exact_topk_jaccard, export_neighbors, naive_topk_jaccard
from src.pipeline import run_lsh_pipeline
from src.preprocess import load_preprocessed_artifacts
from src.utils import ensure_dir, save_json, timer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Three-tier comparison: naive / sparse-exact / LSH on a shared item subset."
    )
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--max-items", type=int, default=300, help="Subset size (default: 300)")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-hashes", type=int, default=100)
    parser.add_argument("--bands", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_comparison(
    matrix,
    item_ids: list,
    max_items: int,
    top_k: int,
    num_hashes: int,
    bands: int,
    seed: int,
    verbose: bool = True,
) -> dict:
    num_items = matrix.shape[0]

    rows_per_band, remainder = divmod(num_hashes, bands)
    if remainder != 0:
        raise ValueError(
            f"num_hashes ({num_hashes}) must be divisible by bands ({bands})."
        )

    rng = np.random.default_rng(seed)
    subset_size = min(max_items, num_items)
    item_subset = sorted(rng.choice(num_items, size=subset_size, replace=False).tolist())
    sub_matrix = matrix[item_subset, :]
    subset_item_ids = [item_ids[i] for i in item_subset]

    if verbose:
        print(
            f"\nSubset: {subset_size} items  |  "
            f"Total items: {num_items}  |  "
            f"LSH: {num_hashes} hashes, {bands} bands×{rows_per_band} rows\n"
        )

    results: dict = {
        "subset_size": subset_size,
        "num_items_total": num_items,
        "top_k": top_k,
        "num_hashes": num_hashes,
        "bands": bands,
        "rows_per_band": rows_per_band,
        "seed": seed,
    }

    if verbose:
        print("Running NAIVE brute-force ...")
    with timer() as t_naive:
        naive_neighbors, naive_metrics = naive_topk_jaccard(sub_matrix, top_k=top_k)
    naive_time = t_naive["elapsed_seconds"]
    naive_payload = export_neighbors(naive_neighbors, item_ids=subset_item_ids, top_k=top_k)
    results["naive"] = {
        "time_seconds": naive_time,
        "pairs_evaluated": naive_metrics["pairs_evaluated"],
    }
    if verbose:
        print(f"  Done in {naive_time:.3f}s  ({naive_metrics['pairs_evaluated']:,} pairs)")

    if verbose:
        print("Running SPARSE-EXACT baseline ...")
    with timer() as t_exact:
        exact_neighbors, exact_metrics = exact_topk_jaccard(sub_matrix, top_k=top_k)
    exact_time = t_exact["elapsed_seconds"]
    exact_payload = export_neighbors(exact_neighbors, item_ids=subset_item_ids, top_k=top_k)
    results["sparse_exact"] = {
        "time_seconds": exact_time,
        "nonzero_intersection_pairs": exact_metrics["nonzero_intersection_pairs"],
    }
    if verbose:
        print(f"  Done in {exact_time:.3f}s  ({exact_metrics['nonzero_intersection_pairs']:,} nonzero pairs)")

    if verbose:
        print("Running LSH ...")
    lsh_metrics, lsh_payload = run_lsh_pipeline(
        sub_matrix,
        item_ids=subset_item_ids,
        num_hashes=num_hashes,
        bands=bands,
        rows_per_band=rows_per_band,
        top_k=top_k,
        seed=seed,
        baseline_payload=exact_payload,
        results_dir=None,
        save_neighbors=True,
    )
    lsh_time = lsh_metrics["total_seconds"]
    recall = lsh_metrics.get("recall_at_k", None)
    results["lsh"] = {
        "time_seconds": lsh_time,
        "signature_seconds": lsh_metrics["signature_seconds"],
        "candidate_seconds": lsh_metrics["candidate_seconds"],
        "verification_seconds": lsh_metrics["verification_seconds"],
        "num_candidates": lsh_metrics["num_candidates"],
        "candidate_ratio": lsh_metrics["candidate_ratio"],
        "recall_at_k": recall,
    }
    if verbose:
        print(f"  Done in {lsh_time:.3f}s  |  recall@{top_k}: {recall:.3f}")

    speedup_vs_naive = naive_time / lsh_time if lsh_time > 0 else float("inf")
    speedup_vs_exact = exact_time / lsh_time if lsh_time > 0 else float("inf")
    results["speedup_lsh_vs_naive"] = speedup_vs_naive
    results["speedup_lsh_vs_sparse_exact"] = speedup_vs_exact

    if verbose:
        print("\n" + "=" * 65)
        print(f"{'Method':<22} {'Time (s)':>10} {'Speedup vs LSH':>16} {'Recall@K':>10}")
        print("-" * 65)
        print(f"{'Naive brute-force':<22} {naive_time:>10.3f} {speedup_vs_naive:>15.1f}x {'-':>10}")
        print(f"{'Sparse-exact':<22} {exact_time:>10.3f} {speedup_vs_exact:>15.1f}x {'1.000':>10}")
        print(f"{'LSH (approx)':<22} {lsh_time:>10.3f} {'1.0x':>16} {recall:>10.3f}")
        print("=" * 65)
        print(f"  Subset: {subset_size} items  |  Pairs in subset: {subset_size*(subset_size-1)//2:,}")
        print(
            f"  LSH candidates: {lsh_metrics['num_candidates']:,}  "
            f"({100 * lsh_metrics['candidate_ratio']:.1f}% of all pairs)"
        )
        print()

    return results


def main() -> None:
    args = parse_args()
    ensure_dir(args.results_dir)

    matrix, metadata = load_preprocessed_artifacts(args.artifacts_dir)
    results = run_comparison(
        matrix=matrix,
        item_ids=metadata["item_ids"],
        max_items=args.max_items,
        top_k=args.top_k,
        num_hashes=args.num_hashes,
        bands=args.bands,
        seed=args.seed,
        verbose=True,
    )

    save_json(f"{args.results_dir}/comparison_results.json", results)
    print(f"Saved to {args.results_dir}/comparison_results.json")


if __name__ == "__main__":
    main()
