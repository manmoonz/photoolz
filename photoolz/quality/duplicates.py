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


def _phash_pairs(rows: list, hamming_threshold: int) -> list[list[int]]:
    """Pairs of photo IDs within the pHash Hamming threshold."""
    # Only compare photos taken on the same day to keep complexity manageable
    by_day: dict[str, list] = defaultdict(list)
    for row in rows:
        if row["phash"]:
            by_day[(row["taken_at"] or "")[:10]].append(row)

    pairs = []
    for day_rows in by_day.values():
        for i, a in enumerate(day_rows):
            for b in day_rows[i + 1:]:
                if hamming_distance(a["phash"], b["phash"]) <= hamming_threshold:
                    pairs.append([a["id"], b["id"]])
    return pairs


def _merge_groups(groups: list[list[int]]) -> list[list[int]]:
    """Union overlapping groups, so A~B and B~C end up in one group even if A and C differ."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for group in groups:
        for other in group[1:]:
            parent[find(other)] = find(group[0])

    merged: dict[int, list[int]] = defaultdict(list)
    for x in parent:
        merged[find(x)].append(x)
    return [g for g in merged.values() if len(g) > 1]


def find_near_duplicates(
    conn: sqlite3.Connection,
    config: Config,
    similarity_threshold: float = 0.97,
    hamming_threshold: int = 8,
) -> list[list[dict]]:
    # Pass 1: pHash Hamming
    groups = _phash_pairs(get_all_phashes(conn), hamming_threshold)

    # Pass 2: FAISS cosine similarity
    index, id_map = load_or_build_index(conn, config)
    if index is not None:
        from photoolz.db import get_all_embeddings
        # Only photos not flagged as deleted; the FAISS index may still contain flagged ones
        emb_rows = {r["id"]: r["clip_embedding"] for r in get_all_embeddings(conn)}
        for photo_id, blob in emb_rows.items():
            candidates = query_index(index, id_map, deserialize_embedding(blob), top_k=20)
            group = [photo_id] + [cid for cid, score in candidates
                                  if cid != photo_id and cid in emb_rows
                                  and score >= similarity_threshold]
            if len(group) > 1:
                groups.append(group)

    unique_groups = _merge_groups(groups)

    # Fetch full photo records and sort each group by quality (best first)
    result_groups = []
    for group_ids in unique_groups:
        photos = get_photos_by_ids(conn, group_ids)
        photo_dicts = [dict(p) for p in photos]
        photo_dicts.sort(key=lambda x: x.get("quality_score") or 0.0, reverse=True)
        result_groups.append(photo_dicts)

    return result_groups
