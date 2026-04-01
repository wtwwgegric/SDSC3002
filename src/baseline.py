from __future__ import annotations

import heapq
from typing import Iterable

import numpy as np
from scipy import sparse


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


def exact_topk_jaccard(
    item_user_matrix: sparse.csr_matrix,
    top_k: int,
) -> tuple[list[list[dict]], dict]:
    binary_matrix = item_user_matrix.astype(np.uint16)
    support = np.diff(binary_matrix.indptr).astype(np.int32)
    intersections = (binary_matrix @ binary_matrix.T).tocoo()

    heaps: list[list[tuple[float, int]]] = [[] for _ in range(binary_matrix.shape[0])]
    nonzero_pairs = 0

    for row, col, intersection in zip(intersections.row, intersections.col, intersections.data, strict=False):
        if row >= col:
            continue
        union = int(support[row] + support[col] - intersection)
        if union <= 0:
            continue
        score = float(intersection / union)
        nonzero_pairs += 1
        _push_topk(heaps, row, col, score, top_k)
        _push_topk(heaps, col, row, score, top_k)

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
        "num_items": int(binary_matrix.shape[0]),
        "num_users": int(binary_matrix.shape[1]),
        "top_k": int(top_k),
        "nonzero_intersection_pairs": int(nonzero_pairs),
        "all_item_pairs": int(binary_matrix.shape[0] * (binary_matrix.shape[0] - 1) // 2),
    }
    return neighbors, metrics


def export_neighbors(neighbors: Iterable[Iterable[dict]], item_ids: list[int], top_k: int) -> dict:
    exported = []
    for item_index, neighbor_list in enumerate(neighbors):
        exported.append(
            {
                "item_index": item_index,
                "item_id": int(item_ids[item_index]),
                "neighbors": [
                    {
                        "neighbor_index": int(entry["neighbor_index"]),
                        "neighbor_item_id": int(item_ids[entry["neighbor_index"]]),
                        "score": float(entry["score"]),
                    }
                    for entry in neighbor_list
                ],
            }
        )
    return {
        "top_k": int(top_k),
        "items": exported,
    }
