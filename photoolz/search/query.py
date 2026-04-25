from __future__ import annotations

import sqlite3

from photoolz.config import Config
from photoolz.db import get_photos_by_ids
from photoolz.indexer.embedder import embed_text, load_clip_model
from photoolz.search.faiss_index import load_or_build_index, query_index
from photoolz.search.filters import apply_sql_filters


def semantic_search(
    query_text: str,
    conn: sqlite3.Connection,
    config: Config,
    top_k: int = 20,
    since: str | None = None,
    until: str | None = None,
    person_name: str | None = None,
    person_id: int | None = None,
    location_label: str | None = None,
) -> list[dict]:
    index, id_map = load_or_build_index(conn, config)
    if index is None:
        return []

    model, _, tokenizer = load_clip_model(config.clip_model, config.clip_pretrained, config.device)
    query_vec = embed_text(query_text, model, tokenizer, config.device)

    # Retrieve more candidates than needed so filters have room to work
    candidate_k = max(top_k * 10, 200)
    raw_results = query_index(index, id_map, query_vec, candidate_k)

    if not raw_results:
        return []

    candidate_ids = [pid for pid, _ in raw_results]
    score_map = {pid: score for pid, score in raw_results}

    filtered_ids = apply_sql_filters(conn, candidate_ids, since, until, person_name, person_id, location_label)
    filtered_ids = filtered_ids[:top_k]

    photos = get_photos_by_ids(conn, filtered_ids)
    photo_dict_map = {p["id"]: dict(p) for p in photos}

    results = []
    for pid in filtered_ids:
        if pid in photo_dict_map:
            entry = photo_dict_map[pid]
            entry["similarity_score"] = score_map.get(pid, 0.0)
            results.append(entry)

    results.sort(key=lambda x: x["similarity_score"], reverse=True)
    return results
