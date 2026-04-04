from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from src.utils import ensure_dir, load_json, save_json


MOVIELENS_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]


def load_movielens_ratings(ratings_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=MOVIELENS_COLUMNS,
        encoding="latin-1",
    )


def load_amazon_reviews(ratings_path: str | Path) -> pd.DataFrame:
    rows: list[dict] = []
    with gzip.open(ratings_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            rows.append(
                {
                    "user_id": record["reviewerID"],
                    "item_id": record["asin"],
                    "rating": record["overall"],
                    "timestamp": record.get("unixReviewTime"),
                }
            )
    return pd.DataFrame.from_records(rows)


def load_ratings(ratings_path: str | Path, dataset: str = "movielens") -> pd.DataFrame:
    if dataset == "movielens":
        return load_movielens_ratings(ratings_path)
    if dataset == "amazon":
        return load_amazon_reviews(ratings_path)
    raise ValueError(f"Unsupported dataset type: {dataset}")


def build_item_user_matrix(
    ratings: pd.DataFrame,
    min_rating: float = 4.0,
    user_column: str = "user_id",
    item_column: str = "movie_id",
    rating_column: str = "rating",
) -> tuple[sparse.csr_matrix, dict]:
    positive = ratings.loc[ratings[rating_column] >= min_rating, [user_column, item_column]].copy()
    positive = positive.drop_duplicates()

    user_ids = np.sort(positive[user_column].unique())
    item_ids = np.sort(positive[item_column].unique())

    user_to_index = {user_id: index for index, user_id in enumerate(user_ids.tolist())}
    item_to_index = {item_id: index for index, item_id in enumerate(item_ids.tolist())}

    row_indices = positive[item_column].map(item_to_index).to_numpy(dtype=np.int32)
    col_indices = positive[user_column].map(user_to_index).to_numpy(dtype=np.int32)
    values = np.ones(len(positive), dtype=np.uint8)

    matrix = sparse.csr_matrix(
        (values, (row_indices, col_indices)),
        shape=(len(item_ids), len(user_ids)),
        dtype=np.uint8,
    )
    matrix.sum_duplicates()
    matrix.data[:] = 1
    matrix.eliminate_zeros()

    metadata = {
        "min_rating": min_rating,
        "original_ratings": int(len(ratings)),
        "positive_interactions": int(matrix.nnz),
        "num_users": int(matrix.shape[1]),
        "num_items": int(matrix.shape[0]),
        "density": float(matrix.nnz / (matrix.shape[0] * matrix.shape[1])),
        "user_column": user_column,
        "item_column": item_column,
        "rating_column": rating_column,
        "user_ids": user_ids.tolist(),
        "item_ids": item_ids.tolist(),
        "item_support": np.diff(matrix.indptr).astype(int).tolist(),
    }
    return matrix, metadata


def save_preprocessed_artifacts(
    matrix: sparse.csr_matrix,
    metadata: dict,
    artifacts_dir: str | Path,
) -> None:
    artifacts_path = ensure_dir(artifacts_dir)
    sparse.save_npz(artifacts_path / "item_user_matrix.npz", matrix)
    save_json(artifacts_path / "metadata.json", metadata)


def load_preprocessed_artifacts(
    artifacts_dir: str | Path,
) -> tuple[sparse.csr_matrix, dict]:
    artifacts_path = Path(artifacts_dir)
    matrix = sparse.load_npz(artifacts_path / "item_user_matrix.npz").tocsr()
    metadata = load_json(artifacts_path / "metadata.json")
    return matrix, metadata
