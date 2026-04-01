from __future__ import annotations

"""Small end-to-end smoke runner for the SDSC3002 project.

This script is intended for quick local verification on the full MovieLens 1M
artifacts path using a single parameter setting.
"""

from pathlib import Path

from src.baseline import exact_topk_jaccard, export_neighbors
from src.lsh import approximate_topk_from_candidates, compute_minhash_signatures, generate_lsh_candidates, recall_at_k
from src.preprocess import build_item_user_matrix, load_ratings


def main() -> None:
	root = Path(__file__).resolve().parent
	ratings_path = root / "data" / "ml-1m" / "ratings.dat"
	ratings = load_ratings(ratings_path)
	matrix, metadata = build_item_user_matrix(ratings, min_rating=4.0)

	exact_neighbors, _ = exact_topk_jaccard(matrix, top_k=10)
	exact_payload = export_neighbors(exact_neighbors, item_ids=metadata["item_ids"], top_k=10)

	signatures, _ = compute_minhash_signatures(matrix, num_hashes=120, seed=42)
	candidates, _ = generate_lsh_candidates(signatures, bands=20, rows_per_band=6)
	approx_neighbors, _ = approximate_topk_from_candidates(matrix, candidates, top_k=10)
	approx_payload = export_neighbors(approx_neighbors, item_ids=metadata["item_ids"], top_k=10)

	print(
		{
			"num_items": metadata["num_items"],
			"num_users": metadata["num_users"],
			"positive_interactions": metadata["positive_interactions"],
			"candidates": len(candidates),
			"recall_at_10": recall_at_k(exact_payload, approx_payload, k=10),
		}
	)


if __name__ == "__main__":
	main()
