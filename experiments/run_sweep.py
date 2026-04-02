from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipeline import run_lsh_pipeline
from src.preprocess import load_preprocessed_artifacts
from src.utils import ensure_dir, load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an automated LSH parameter sweep.")
    parser.add_argument("--artifacts-dir", required=True, help="Directory containing preprocessing artifacts")
    parser.add_argument("--results-dir", required=True, help="Directory to save sweep outputs")
    parser.add_argument("--baseline", required=True, help="Path to exact_topk.json for recall evaluation")
    parser.add_argument("--baseline-metrics", help="Path to baseline_metrics.json for speedup calculation")
    parser.add_argument(
        "--num-hashes",
        nargs="+",
        type=int,
        default=[100, 200],
        help="Hash counts to evaluate",
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        type=int,
        default=[10, 20, 30, 40, 50, 60],
        help="Band counts to evaluate",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Neighbors to keep per item")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quality-samples", type=int, default=200, help="Sample size for MinHash quality checks")
    parser.add_argument(
        "--save-neighbors",
        action="store_true",
        help="Save approx_topk.json for every sweep run. Off by default to reduce disk usage.",
    )
    return parser.parse_args()


def build_configs(hash_counts: list[int], band_counts: list[int]) -> tuple[list[dict], list[dict]]:
    valid_configs = []
    skipped_configs = []
    for num_hashes in hash_counts:
        for bands in band_counts:
            if bands <= 0 or num_hashes % bands != 0:
                skipped_configs.append(
                    {
                        "num_hashes": num_hashes,
                        "bands": bands,
                        "reason": "bands must divide num_hashes exactly",
                    }
                )
                continue
            rows_per_band = num_hashes // bands
            valid_configs.append(
                {
                    "num_hashes": num_hashes,
                    "bands": bands,
                    "rows_per_band": rows_per_band,
                }
            )
    valid_configs.sort(key=lambda config: (config["num_hashes"], config["bands"]))
    return valid_configs, skipped_configs


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    results_dir = ensure_dir(args.results_dir)
    run_root = ensure_dir(results_dir / "runs")

    matrix, metadata = load_preprocessed_artifacts(args.artifacts_dir)
    baseline_payload = load_json(args.baseline)
    baseline_metrics = load_json(args.baseline_metrics) if args.baseline_metrics else None

    valid_configs, skipped_configs = build_configs(args.num_hashes, args.bands)
    summary_rows = []

    for config in valid_configs:
        run_name = f"k{config['num_hashes']}_b{config['bands']}_r{config['rows_per_band']}"
        metrics, _ = run_lsh_pipeline(
            item_user_matrix=matrix,
            item_ids=metadata["item_ids"],
            num_hashes=config["num_hashes"],
            bands=config["bands"],
            rows_per_band=config["rows_per_band"],
            top_k=args.top_k,
            seed=args.seed,
            quality_samples=args.quality_samples,
            baseline_payload=baseline_payload,
            results_dir=run_root / run_name,
            save_neighbors=args.save_neighbors,
        )

        row = {
            "run_name": run_name,
            "num_hashes": metrics["num_hashes"],
            "bands": metrics["bands"],
            "rows_per_band": metrics["rows_per_band"],
            "top_k": metrics["top_k"],
            "num_candidates": metrics["num_candidates"],
            "candidate_ratio": metrics["candidate_ratio"],
            "signature_seconds": metrics["signature_seconds"],
            "candidate_seconds": metrics["candidate_seconds"],
            "verification_seconds": metrics["verification_seconds"],
            "total_seconds": metrics["total_seconds"],
            "verified_pairs": metrics["verified_pairs"],
            "recall_at_k": metrics.get("recall_at_k"),
            "signature_mae": metrics["signature_quality"]["mean_absolute_error"],
            "signature_corr": metrics["signature_quality"]["pearson_correlation"],
        }
        if baseline_metrics:
            baseline_seconds = baseline_metrics.get("elapsed_seconds")
            if baseline_seconds:
                row["speedup_vs_baseline"] = baseline_seconds / metrics["total_seconds"]
        summary_rows.append(row)
        print(row)

    summary_rows.sort(key=lambda row: (row["num_hashes"], row["bands"]))
    write_csv(results_dir / "sweep_summary.csv", summary_rows)
    save_json(
        results_dir / "sweep_summary.json",
        {
            "artifacts_dir": args.artifacts_dir,
            "baseline": args.baseline,
            "baseline_metrics": args.baseline_metrics,
            "configs": valid_configs,
            "skipped_configs": skipped_configs,
            "rows": summary_rows,
        },
    )
    print({"completed_runs": len(summary_rows), "skipped_runs": len(skipped_configs)})


if __name__ == "__main__":
    main()