from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from photoolz.db import get_photos_ordered_by_time, assign_burst_group
from photoolz.utils.console import console


def detect_bursts(conn: sqlite3.Connection, gap_seconds: int = 3) -> dict:
    rows = get_photos_ordered_by_time(conn)
    if not rows:
        console.print("[yellow]No photos with timestamp data found.[/yellow]")
        return {"burst_groups_found": 0, "photos_in_bursts": 0}

    groups: list[list[int]] = []
    current_group: list[int] = []
    prev_dt: datetime | None = None

    for row in rows:
        try:
            dt = datetime.fromisoformat(row["taken_at"].replace("Z", "+00:00"))
        except Exception:
            prev_dt = None
            if len(current_group) >= 3:
                groups.append(current_group)
            current_group = []
            continue

        if prev_dt is not None and (dt - prev_dt).total_seconds() <= gap_seconds:
            current_group.append(row["id"])
        else:
            if len(current_group) >= 3:
                groups.append(current_group)
            current_group = [row["id"]]

        prev_dt = dt

    if len(current_group) >= 3:
        groups.append(current_group)

    assigned = 0
    for group_id, group_ids in enumerate(groups, start=1):
        assign_burst_group(conn, group_ids, group_id)
        assigned += len(group_ids)

    console.print(
        f"Found [cyan]{len(groups)}[/cyan] burst groups, "
        f"[cyan]{assigned}[/cyan] photos in bursts."
    )
    return {"burst_groups_found": len(groups), "photos_in_bursts": assigned}


def get_burst_best_frames(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT burst_group_id, id, file_path, quality_score "
        "FROM photos WHERE burst_group_id IS NOT NULL AND is_deleted=0 "
        "ORDER BY burst_group_id, quality_score DESC"
    ).fetchall()

    groups: dict[int, list] = {}
    for row in rows:
        gid = row["burst_group_id"]
        groups.setdefault(gid, []).append(dict(row))

    results = []
    for gid, photos in groups.items():
        best = photos[0]
        best["burst_size"] = len(photos)
        best["other_ids"] = [p["id"] for p in photos[1:]]
        results.append(best)

    return results
