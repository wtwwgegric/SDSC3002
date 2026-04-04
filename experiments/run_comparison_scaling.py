from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from experiments.run_comparison import run_comparison
from src.preprocess import load_preprocessed_artifacts
from src.utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep comparison results across multiple max-items values."
    )
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument(
        "--max-items-list",
        nargs="+",
        type=int,
        required=True,
        help="List of subset sizes to evaluate, e.g. 300 500 800 1000 1500 2000",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-hashes", type=int, default=100)
    parser.add_argument("--bands", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def flatten_result(result: dict) -> dict:
    return {
        "subset_size": result["subset_size"],
        "num_items_total": result["num_items_total"],
        "top_k": result["top_k"],
        "num_hashes": result["num_hashes"],
        "bands": result["bands"],
        "rows_per_band": result["rows_per_band"],
        "seed": result["seed"],
        "naive_time_seconds": result["naive"]["time_seconds"],
        "naive_pairs_evaluated": result["naive"]["pairs_evaluated"],
        "sparse_exact_time_seconds": result["sparse_exact"]["time_seconds"],
        "sparse_exact_nonzero_pairs": result["sparse_exact"]["nonzero_intersection_pairs"],
        "lsh_time_seconds": result["lsh"]["time_seconds"],
        "lsh_signature_seconds": result["lsh"]["signature_seconds"],
        "lsh_candidate_seconds": result["lsh"]["candidate_seconds"],
        "lsh_verification_seconds": result["lsh"]["verification_seconds"],
        "lsh_num_candidates": result["lsh"]["num_candidates"],
        "lsh_candidate_ratio": result["lsh"]["candidate_ratio"],
        "lsh_recall_at_k": result["lsh"]["recall_at_k"],
        "speedup_lsh_vs_naive": result["speedup_lsh_vs_naive"],
        "speedup_lsh_vs_sparse_exact": result["speedup_lsh_vs_sparse_exact"],
    }


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.results_dir)

    matrix, metadata = load_preprocessed_artifacts(args.artifacts_dir)

    flat_rows: list[dict] = []
    detailed_results: list[dict] = []

    for max_items in args.max_items_list:
        print(f"\n=== Running subset size: {max_items} ===")
        result = run_comparison(
            matrix=matrix,
            item_ids=metadata["item_ids"],
            max_items=max_items,
            top_k=args.top_k,
            num_hashes=args.num_hashes,
            bands=args.bands,
            seed=args.seed,
            verbose=True,
        )
        detailed_results.append(result)
        flat_rows.append(flatten_result(result))

    frame = pd.DataFrame(flat_rows).sort_values("subset_size").reset_index(drop=True)
    frame.to_csv(output_dir / "comparison_scaling_summary.csv", index=False)
    save_json(output_dir / "comparison_scaling_results.json", detailed_results)

    crossover_rows = frame.loc[frame["speedup_lsh_vs_sparse_exact"] > 1.0]
    crossover_subset = int(crossover_rows.iloc[0]["subset_size"]) if not crossover_rows.empty else None

    summary = {
        "artifacts_dir": args.artifacts_dir,
        "results_dir": str(output_dir),
        "max_items_list": sorted(args.max_items_list),
        "top_k": args.top_k,
        "num_hashes": args.num_hashes,
        "bands": args.bands,
        "seed": args.seed,
        "first_subset_where_lsh_beats_sparse_exact": crossover_subset,
    }
    save_json(output_dir / "comparison_scaling_metadata.json", summary)

    print("\nSaved:")
    print(output_dir / "comparison_scaling_summary.csv")
    print(output_dir / "comparison_scaling_results.json")
    print(output_dir / "comparison_scaling_metadata.json")
    if crossover_subset is not None:
        print(f"\nFirst measured subset where LSH beats sparse-exact: {crossover_subset}")
    else:
        print("\nLSH did not beat sparse-exact in the tested subset sizes.")


if __name__ == "__main__":
    main()
