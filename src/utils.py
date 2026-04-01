from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, payload) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    metrics: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield metrics
    finally:
        metrics["elapsed_seconds"] = time.perf_counter() - start


def build_run_metadata(**kwargs):
    payload = dict(kwargs)
    payload["created_at_epoch"] = time.time()
    return payload
