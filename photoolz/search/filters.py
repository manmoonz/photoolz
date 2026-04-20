from __future__ import annotations

import sqlite3


def apply_sql_filters(
    conn: sqlite3.Connection,
    photo_ids: list[int],
    since: str | None = None,
    until: str | None = None,
    person_name: str | None = None,
    location_label: str | None = None,
) -> list[int]:
    if not photo_ids:
        return []

    placeholders = ",".join("?" * len(photo_ids))
    params: list = list(photo_ids)
    conditions = [f"p.id IN ({placeholders})"]

    if since:
        conditions.append("p.taken_at >= ?")
        params.append(since)

    if until:
        conditions.append("p.taken_at <= ?")
        params.append(until + "T23:59:59" if len(until) == 10 else until)

    if person_name:
        conditions.append(
            "p.id IN (SELECT DISTINCT f.photo_id FROM faces f "
            "JOIN people pe ON pe.id = f.person_id WHERE pe.name = ?)"
        )
        params.append(person_name)

    if location_label:
        conditions.append(
            "p.geo_cluster_id IN (SELECT id FROM geo_clusters WHERE label = ?)"
        )
        params.append(location_label)

    where_clause = " AND ".join(conditions)
    sql = f"SELECT p.id FROM photos p WHERE {where_clause} AND p.is_deleted=0"

    rows = conn.execute(sql, params).fetchall()
    filtered_ids = {row[0] for row in rows}
    return [pid for pid in photo_ids if pid in filtered_ids]
