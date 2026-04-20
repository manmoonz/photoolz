from __future__ import annotations

import sqlite3

from photoolz.config import Config
from photoolz.db import get_photos_by_ids
from photoolz.search.query import semantic_search


def retrieve_candidates(
    query_text: str,
    conn: sqlite3.Connection,
    config: Config,
    n_candidates: int = 120,
    since: str | None = None,
    until: str | None = None,
    location_label: str | None = None,
) -> list[dict]:
    results = semantic_search(
        query_text,
        conn,
        config,
        top_k=n_candidates,
        since=since,
        until=until,
        location_label=location_label,
    )

    # Enrich with geo cluster label
    cluster_labels: dict[int, str] = {}
    cluster_ids = {r.get("geo_cluster_id") for r in results if r.get("geo_cluster_id")}
    if cluster_ids:
        placeholders = ",".join("?" * len(cluster_ids))
        rows = conn.execute(
            f"SELECT id, label FROM geo_clusters WHERE id IN ({placeholders})",
            list(cluster_ids),
        ).fetchall()
        cluster_labels = {row["id"]: (row["label"] or f"Location {row['id']}") for row in rows}

    enriched = []
    for r in results:
        r["geo_cluster_label"] = cluster_labels.get(r.get("geo_cluster_id"), "")
        enriched.append(r)

    return enriched
