from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from src.utils import ensure_dir, load_json, save_json


RATINGS_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]


def load_ratings(ratings_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=RATINGS_COLUMNS,
        encoding="latin-1",
    )


def build_item_user_matrix(
    ratings: pd.DataFrame,
    min_rating: float = 4.0,
) -> tuple[sparse.csr_matrix, dict]:
    positive = ratings.loc[ratings["rating"] >= min_rating, ["user_id", "movie_id"]].copy()
    positive = positive.drop_duplicates()

    user_ids = np.sort(positive["user_id"].unique())
    item_ids = np.sort(positive["movie_id"].unique())

    user_to_index = {user_id: index for index, user_id in enumerate(user_ids.tolist())}
    item_to_index = {item_id: index for index, item_id in enumerate(item_ids.tolist())}

    row_indices = positive["movie_id"].map(item_to_index).to_numpy(dtype=np.int32)
    col_indices = positive["user_id"].map(user_to_index).to_numpy(dtype=np.int32)
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
