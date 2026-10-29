from __future__ import annotations

import sqlite3
from pathlib import Path

import click
import faiss
import numpy as np

from photoolz.config import Config
from photoolz.db import get_all_embeddings
from photoolz.indexer.embedder import deserialize_embedding


class ClipModelMismatchError(click.ClickException):
    """Stored embeddings come from a different CLIP model than the configured one."""


def check_clip_model(conn: sqlite3.Connection, config: Config) -> None:
    """Raise if any stored embedding was made with a model other than config.clip_model.

    Embeddings from different models aren't comparable, so mixing them gives meaningless
    search results (or fails outright when the embedding sizes differ).
    """
    models = {r[0] for r in conn.execute(
        "SELECT DISTINCT clip_model FROM photos WHERE clip_embedding IS NOT NULL AND is_deleted=0"
    )}
    stale = sorted(m for m in models if m and m != config.clip_model)
    if stale:
        raise ClipModelMismatchError(
            f"The index has CLIP embeddings from {', '.join(stale)}, but PHOTOOLZ_CLIP_MODEL is "
            f"{config.clip_model}. Re-index every directory listed by 'photoolz dirs' with "
            f"'photoolz index <path> --force', or set PHOTOOLZ_CLIP_MODEL back."
        )


def build_index(embeddings: np.ndarray, photo_ids: list[int]) -> tuple[faiss.Index, np.ndarray]:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, np.array(photo_ids, dtype=np.int64)


def save_index(index: faiss.Index, id_map: np.ndarray,
               index_path: Path, id_map_path: Path) -> None:
    faiss.write_index(index, str(index_path))
    np.save(str(id_map_path), id_map)


def load_index(index_path: Path, id_map_path: Path) -> tuple[faiss.Index, np.ndarray]:
    index = faiss.read_index(str(index_path))
    id_map = np.load(str(id_map_path))
    return index, id_map


def build_or_update_faiss_index(conn: sqlite3.Connection, config: Config) -> None:
    check_clip_model(conn, config)
    rows = get_all_embeddings(conn)
    if not rows:
        return

    photo_ids = []
    vecs = []
    for row in rows:
        try:
            vec = deserialize_embedding(row["clip_embedding"])
            vecs.append(vec)
            photo_ids.append(row["id"])
        except Exception:
            continue

    if not vecs:
        return

    matrix = np.vstack(vecs).astype(np.float32)
    index, id_map = build_index(matrix, photo_ids)
    save_index(index, id_map, config.faiss_index_path, config.faiss_id_map_path)


def query_index(index: faiss.Index, id_map: np.ndarray,
                query_vec: np.ndarray, top_k: int = 20) -> list[tuple[int, float]]:
    q = query_vec.reshape(1, -1).astype(np.float32)
    scores, indices = index.search(q, min(top_k, index.ntotal))
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        results.append((int(id_map[idx]), float(score)))
    return results


def load_or_build_index(conn: sqlite3.Connection,
                         config: Config) -> tuple[faiss.Index, np.ndarray] | tuple[None, None]:
    check_clip_model(conn, config)
    if config.faiss_index_path.exists() and config.faiss_id_map_path.exists():
        try:
            return load_index(config.faiss_index_path, config.faiss_id_map_path)
        except Exception:
            pass
    if conn is not None:
        build_or_update_faiss_index(conn, config)
        if config.faiss_index_path.exists():
            return load_index(config.faiss_index_path, config.faiss_id_map_path)
    return None, None
