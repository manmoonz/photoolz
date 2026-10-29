from __future__ import annotations

import sqlite3
from pathlib import Path

from photoolz.config import Config
from photoolz.db import (
    bulk_update_quality_scores,
    get_all_photo_ids_and_paths,
    get_photos_missing_quality,
)
from photoolz.quality.blur import laplacian_variance, blur_score_from_variance
from photoolz.quality.exposure import exposure_score
from photoolz.utils.console import console, make_progress


def compute_quality_score(blur: float, exposure: float,
                           weights: tuple[float, float] = (0.6, 0.4)) -> float:
    return blur * weights[0] + exposure * weights[1]


def score_photo(path: Path, blur_threshold: float = 100.0) -> tuple[float, float, float] | None:
    variance = laplacian_variance(path)
    exp = exposure_score(path)
    if variance is None or exp is None:
        return None
    blur = blur_score_from_variance(variance, blur_threshold)
    quality = compute_quality_score(blur, exp)
    return blur, exp, quality


def score_all_unscored(conn: sqlite3.Connection, config: Config,
                        blur_threshold: float = 100.0) -> int:
    return score_photos(conn, get_photos_missing_quality(conn), blur_threshold)


def rescore_all(conn: sqlite3.Connection, blur_threshold: float = 100.0) -> int:
    return score_photos(conn, get_all_photo_ids_and_paths(conn), blur_threshold)


def score_photos(conn: sqlite3.Connection, rows: list,
                 blur_threshold: float = 100.0) -> int:
    """Score each (id, file_path) row and store the results. Returns the number scored."""
    if not rows:
        return 0

    updates: list[tuple[float, float, float, int]] = []

    with make_progress("Scoring quality", len(rows)) as progress:
        task = progress.add_task("Scoring quality", total=len(rows))
        for row in rows:
            photo_id, file_path = row[0], row[1]
            result = score_photo(Path(file_path), blur_threshold)
            if result:
                blur, exp, quality = result
                updates.append((blur, exp, quality, photo_id))
            progress.advance(task)

    if updates:
        bulk_update_quality_scores(conn, updates)

    return len(updates)
