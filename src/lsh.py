from __future__ import annotations

import heapq
from collections import defaultdict

import numpy as np
from scipy import sparse


LARGE_PRIME = np.uint64(4_294_967_311)


def generate_hash_parameters(num_hashes: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.integers(1, LARGE_PRIME - 1, size=num_hashes, dtype=np.uint64)
    b = rng.integers(0, LARGE_PRIME - 1, size=num_hashes, dtype=np.uint64)
    return a, b


def compute_minhash_signatures(
    item_user_matrix: sparse.csr_matrix,
    num_hashes: int,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    a, b = generate_hash_parameters(num_hashes=num_hashes, seed=seed)
    signatures = np.full((item_user_matrix.shape[0], num_hashes), np.iinfo(np.uint64).max, dtype=np.uint64)

    for item_index in range(item_user_matrix.shape[0]):
        start = item_user_matrix.indptr[item_index]
        end = item_user_matrix.indptr[item_index + 1]
        user_indices = item_user_matrix.indices[start:end]
        if user_indices.size == 0:
            continue
        hashed = (a[:, None] * user_indices.astype(np.uint64)[None, :] + b[:, None]) % LARGE_PRIME
        signatures[item_index] = hashed.min(axis=1)

    metrics = {
        "num_items": int(item_user_matrix.shape[0]),
        "num_hashes": int(num_hashes),
        "seed": int(seed),
    }
    return signatures, metrics


def estimate_jaccard(signatures: np.ndarray, item_a: int, item_b: int) -> float:
    return float(np.mean(signatures[item_a] == signatures[item_b]))


def sample_signature_quality(
    item_user_matrix: sparse.csr_matrix,
    signatures: np.ndarray,
    num_samples: int = 200,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    num_items = item_user_matrix.shape[0]
    sample_count = min(num_samples, num_items * (num_items - 1) // 2)

    exact_scores = []
    estimated_scores = []
    seen_pairs: set[tuple[int, int]] = set()

    intersections = (item_user_matrix.astype(np.uint16) @ item_user_matrix.astype(np.uint16).T).tocoo()
    nonzero_pairs = [
        (int(i), int(j))
        for i, j in zip(intersections.row, intersections.col, strict=False)
        if i < j
    ]
    rng.shuffle(nonzero_pairs)

    def record_pair(i: int, j: int) -> None:
        row_i = item_user_matrix.indices[item_user_matrix.indptr[i] : item_user_matrix.indptr[i + 1]]
        row_j = item_user_matrix.indices[item_user_matrix.indptr[j] : item_user_matrix.indptr[j + 1]]
        intersection = np.intersect1d(row_i, row_j, assume_unique=True).size
        union = row_i.size + row_j.size - intersection
        exact = float(intersection / union) if union else 0.0
        estimated = estimate_jaccard(signatures, i, j)
        exact_scores.append(exact)
        estimated_scores.append(estimated)
        seen_pairs.add((i, j))

    for i, j in nonzero_pairs[:sample_count]:
        record_pair(i, j)

    while len(seen_pairs) < sample_count:
        i, j = sorted(rng.choice(num_items, size=2, replace=False).tolist())
        pair = (i, j)
        if pair in seen_pairs:
            continue
        record_pair(i, j)

    exact_array = np.asarray(exact_scores, dtype=float)
    estimated_array = np.asarray(estimated_scores, dtype=float)
    mae = float(np.mean(np.abs(exact_array - estimated_array))) if exact_array.size else 0.0
    if exact_array.size > 1 and np.std(exact_array) > 0 and np.std(estimated_array) > 0:
        correlation = float(np.corrcoef(exact_array, estimated_array)[0, 1])
    else:
        correlation = 0.0

    return {
        "samples": int(sample_count),
        "nonzero_pairs_available": int(len(nonzero_pairs)),
        "mean_absolute_error": mae,
        "pearson_correlation": correlation,
    }


def generate_lsh_candidates(
    signatures: np.ndarray,
    bands: int,
    rows_per_band: int,
) -> tuple[list[tuple[int, int]], dict]:
    expected_hashes = bands * rows_per_band
    if signatures.shape[1] != expected_hashes:
        raise ValueError(
            f"Signature width {signatures.shape[1]} does not match bands * rows_per_band = {expected_hashes}."
        )

    candidates: set[tuple[int, int]] = set()
    bucket_sizes: list[int] = []

    for band_index in range(bands):
        start = band_index * rows_per_band
        end = start + rows_per_band
        buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for item_index in range(signatures.shape[0]):
            key = tuple(int(value) for value in signatures[item_index, start:end])
            buckets[key].append(item_index)

        for bucket_items in buckets.values():
            if len(bucket_items) < 2:
                continue
            bucket_sizes.append(len(bucket_items))
            for left_index in range(len(bucket_items)):
                for right_index in range(left_index + 1, len(bucket_items)):
                    i = bucket_items[left_index]
                    j = bucket_items[right_index]
                    candidates.add((i, j) if i < j else (j, i))

    metrics = {
        "bands": int(bands),
        "rows_per_band": int(rows_per_band),
        "num_candidates": int(len(candidates)),
        "num_nontrivial_buckets": int(len(bucket_sizes)),
        "max_bucket_size": int(max(bucket_sizes, default=0)),
        "mean_bucket_size": float(np.mean(bucket_sizes)) if bucket_sizes else 0.0,
    }
    return sorted(candidates), metrics


def _push_topk(
    heaps: list[list[tuple[float, int]]],
    item_index: int,
    neighbor_index: int,
    score: float,
    top_k: int,
) -> None:
    heap = heaps[item_index]
    entry = (score, neighbor_index)
    if len(heap) < top_k:
        heapq.heappush(heap, entry)
    elif score > heap[0][0]:
        heapq.heapreplace(heap, entry)


def approximate_topk_from_candidates(
    item_user_matrix: sparse.csr_matrix,
    candidates: list[tuple[int, int]],
    top_k: int,
    verify_batch_size: int = 2048,
) -> tuple[list[list[dict]], dict]:
    binary_matrix = item_user_matrix.astype(np.uint16, copy=False)
    support = np.diff(binary_matrix.indptr).astype(np.int32)
    heaps: list[list[tuple[float, int]]] = [[] for _ in range(binary_matrix.shape[0])]
    grouped_candidates: dict[int, list[int]] = defaultdict(list)
    verified_pairs = 0
    batch_count = 0

    for item_a, item_b in candidates:
        grouped_candidates[item_a].append(item_b)

    for item_a, neighbor_indices in grouped_candidates.items():
        row_a = binary_matrix.getrow(item_a)
        for start in range(0, len(neighbor_indices), verify_batch_size):
            batch = neighbor_indices[start : start + verify_batch_size]
            batch_count += 1
            intersections = (binary_matrix[batch] @ row_a.T).toarray().ravel().astype(np.int32)
            if not intersections.size:
                continue
            valid_mask = intersections > 0
            if not np.any(valid_mask):
                continue

            valid_neighbors = np.asarray(batch, dtype=np.int32)[valid_mask]
            valid_intersections = intersections[valid_mask]
            unions = support[item_a] + support[valid_neighbors] - valid_intersections
            positive_union_mask = unions > 0
            if not np.any(positive_union_mask):
                continue

            final_neighbors = valid_neighbors[positive_union_mask]
            final_intersections = valid_intersections[positive_union_mask]
            final_unions = unions[positive_union_mask]
            scores = final_intersections / final_unions
            verified_pairs += int(final_neighbors.size)

            for neighbor_index, score in zip(final_neighbors.tolist(), scores.tolist(), strict=False):
                _push_topk(heaps, item_a, neighbor_index, float(score), top_k)
                _push_topk(heaps, neighbor_index, item_a, float(score), top_k)

    neighbors: list[list[dict]] = []
    for heap in heaps:
        ranked = sorted(heap, key=lambda entry: (-entry[0], entry[1]))
        neighbors.append(
            [
                {
                    "neighbor_index": int(neighbor_index),
                    "score": float(score),
                }
                for score, neighbor_index in ranked
            ]
        )

    metrics = {
        "verified_pairs": int(verified_pairs),
        "top_k": int(top_k),
        "verification_mode": "batched_sparse_dot",
        "verification_batches": int(batch_count),
        "verify_batch_size": int(verify_batch_size),
    }
    return neighbors, metrics


def recall_at_k(exact_payload: dict, approx_payload: dict, k: int) -> float:
    exact_items = exact_payload["items"]
    approx_items = approx_payload["items"]
    matched = 0
    total = 0

    for exact_item, approx_item in zip(exact_items, approx_items, strict=False):
        exact_neighbors = {
            entry["neighbor_item_id"] for entry in exact_item["neighbors"][:k]
        }
        approx_neighbors = {
            entry["neighbor_item_id"] for entry in approx_item["neighbors"][:k]
        }
        matched += len(exact_neighbors & approx_neighbors)
        total += len(exact_neighbors)

    return float(matched / total) if total else 0.0
