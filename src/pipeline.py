from __future__ import annotations

from pathlib import Path

from scipy import sparse

from src.baseline import export_neighbors
from src.lsh import (
    approximate_topk_from_candidates,
    compute_minhash_signatures,
    generate_lsh_candidates,
    recall_at_k,
    sample_signature_quality,
)
from src.utils import build_run_metadata, ensure_dir, save_json, timer


def run_lsh_pipeline(
    item_user_matrix: sparse.csr_matrix,
    item_ids: list[int],
    num_hashes: int,
    bands: int,
    rows_per_band: int,
    top_k: int,
    seed: int = 42,
    quality_samples: int = 200,
    baseline_payload: dict | None = None,
    results_dir: str | Path | None = None,
    save_neighbors: bool = True,
) -> tuple[dict, dict | None]:
    if num_hashes != bands * rows_per_band:
        raise ValueError(
            f"Invalid LSH config: num_hashes={num_hashes}, bands={bands}, rows_per_band={rows_per_band}."
        )

    with timer() as signature_timing:
        signatures, signature_metrics = compute_minhash_signatures(
            item_user_matrix,
            num_hashes=num_hashes,
            seed=seed,
        )

    signature_quality = sample_signature_quality(
        item_user_matrix,
        signatures,
        num_samples=quality_samples,
        seed=seed,
    )

    with timer() as candidate_timing:
        candidates, candidate_metrics = generate_lsh_candidates(
            signatures,
            bands=bands,
            rows_per_band=rows_per_band,
        )

    with timer() as verification_timing:
        approx_neighbors, approx_metrics = approximate_topk_from_candidates(
            item_user_matrix,
            candidates,
            top_k=top_k,
        )

    approx_payload = export_neighbors(approx_neighbors, item_ids=item_ids, top_k=top_k)
    all_item_pairs = int(item_user_matrix.shape[0] * (item_user_matrix.shape[0] - 1) // 2)
    metrics = build_run_metadata(
        step="lsh",
        signature_seconds=signature_timing["elapsed_seconds"],
        candidate_seconds=candidate_timing["elapsed_seconds"],
        verification_seconds=verification_timing["elapsed_seconds"],
        total_seconds=(
            signature_timing["elapsed_seconds"]
            + candidate_timing["elapsed_seconds"]
            + verification_timing["elapsed_seconds"]
        ),
        all_item_pairs=all_item_pairs,
        candidate_ratio=(candidate_metrics["num_candidates"] / all_item_pairs) if all_item_pairs else 0.0,
        **signature_metrics,
        **candidate_metrics,
        **approx_metrics,
        signature_quality=signature_quality,
    )

    if baseline_payload:
        metrics["recall_at_k"] = recall_at_k(baseline_payload, approx_payload, k=top_k)

    if results_dir is not None:
        results_path = ensure_dir(results_dir)
        metrics["results_dir"] = str(results_path)
        save_json(results_path / "lsh_metrics.json", metrics)
        save_json(results_path / "signature_quality.json", signature_quality)
        if save_neighbors:
            save_json(results_path / "approx_topk.json", approx_payload)

    return metrics, approx_payload if save_neighbors else None