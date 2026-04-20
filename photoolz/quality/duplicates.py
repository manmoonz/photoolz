from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path

from photoolz.config import Config
from photoolz.db import get_all_phashes, get_photos_by_ids
from photoolz.search.faiss_index import load_or_build_index, query_index
from photoolz.indexer.embedder import deserialize_embedding
from photoolz.utils.hash import hamming_distance
from photoolz.utils.console import console


def _phash_groups(rows: list, hamming_threshold: int) -> list[list[int]]:
    """Group photo IDs by pHash Hamming distance."""
    # Only compare photos taken on the same day to keep complexity manageable
    by_day: dict[str, list] = defaultdict(list)
    for row in rows:
        day = (row["taken_at"] or "")[:10]
        by_day[day].append(row)

    visited = set()
    groups = []
    for day_rows in by_day.values():
        for i, a in enumerate(day_rows):
            if a["id"] in visited:
                continue
            if not a["phash"]:
                continue
            group = [a["id"]]
            for b in day_rows[i + 1:]:
                if b["id"] in visited or not b["phash"]:
                    continue
                if hamming_distance(a["phash"], b["phash"]) <= hamming_threshold:
                    group.append(b["id"])
                    visited.add(b["id"])
            if len(group) > 1:
                visited.add(a["id"])
                groups.append(group)

    return groups


def find_near_duplicates(
    conn: sqlite3.Connection,
    config: Config,
    similarity_threshold: float = 0.97,
    hamming_threshold: int = 8,
) -> list[list[dict]]:
    all_phash_rows = get_all_phashes(conn)

    # Pass 1: pHash Hamming
    phash_group_ids = _phash_groups(all_phash_rows, hamming_threshold)

    # Pass 2: FAISS cosine similarity
    index, id_map = load_or_build_index(conn, config)
    faiss_group_ids: list[list[int]] = []

    if index is not None:
        from photoolz.db import get_all_embeddings
        emb_rows = {r["id"]: r["clip_embedding"] for r in get_all_embeddings(conn)}

        visited = set()
        all_ids = list(emb_rows.keys())
        for photo_id in all_ids:
            if photo_id in visited:
                continue
            blob = emb_rows.get(photo_id)
            if blob is None:
                continue
            vec = deserialize_embedding(blob)
            candidates = query_index(index, id_map, vec, top_k=20)
            group = [photo_id]
            for cid, score in candidates:
                if cid == photo_id or cid in visited:
                    continue
                if score >= similarity_threshold:
                    group.append(cid)
                    visited.add(cid)
            if len(group) > 1:
                visited.add(photo_id)
                faiss_group_ids.append(group)

    # Merge groups
    all_groups: list[list[int]] = phash_group_ids + faiss_group_ids

    # Deduplicate groups (a photo may appear in both passes)
    seen_pairs: set[frozenset] = set()
    unique_groups: list[list[int]] = []
    for group in all_groups:
        key = frozenset(group)
        if key not in seen_pairs:
            seen_pairs.add(key)
            unique_groups.append(group)

    # Fetch full photo records and sort each group by quality (best first)
    result_groups = []
    for group_ids in unique_groups:
        photos = get_photos_by_ids(conn, group_ids)
        photo_dicts = [dict(p) for p in photos]
        photo_dicts.sort(key=lambda x: x.get("quality_score") or 0.0, reverse=True)
        result_groups.append(photo_dicts)

    return result_groups
