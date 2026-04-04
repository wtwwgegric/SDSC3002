from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.preprocess import build_item_user_matrix, load_ratings, save_preprocessed_artifacts
from src.utils import build_run_metadata, save_json, timer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess rating data into a binary item-user matrix.")
    parser.add_argument("--ratings", required=True, help="Path to ratings.dat")
    parser.add_argument("--artifacts-dir", required=True, help="Directory to save sparse matrix and metadata")
    parser.add_argument(
        "--dataset",
        choices=["movielens", "amazon"],
        default="movielens",
        help="Input dataset format",
    )
    parser.add_argument("--min-rating", type=float, default=4.0, help="Threshold for positive interactions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with timer() as timing:
        ratings = load_ratings(args.ratings, dataset=args.dataset)
        item_column = "movie_id" if args.dataset == "movielens" else "item_id"
        matrix, metadata = build_item_user_matrix(
            ratings,
            min_rating=args.min_rating,
            user_column="user_id",
            item_column=item_column,
            rating_column="rating",
        )
        save_preprocessed_artifacts(matrix, metadata, args.artifacts_dir)

    summary_metadata = {
        key: metadata[key]
        for key in (
            "min_rating",
            "original_ratings",
            "positive_interactions",
            "num_users",
            "num_items",
            "density",
        )
    }
    preprocess_metrics = build_run_metadata(
        step="preprocess",
        dataset=args.dataset,
        ratings_path=args.ratings,
        artifacts_dir=args.artifacts_dir,
        elapsed_seconds=timing["elapsed_seconds"],
        **summary_metadata,
    )
    save_json(f"{args.artifacts_dir}/preprocess_metrics.json", preprocess_metrics)
    print(preprocess_metrics)


if __name__ == "__main__":
    main()
