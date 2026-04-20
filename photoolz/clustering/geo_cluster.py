from __future__ import annotations

import math
import sqlite3

import numpy as np

from photoolz.db import (
    get_photos_with_gps,
    upsert_geo_cluster,
    assign_geo_cluster,
)
from photoolz.utils.console import console

EARTH_RADIUS_KM = 6371.0


def cluster_geolocations(conn: sqlite3.Connection, eps_km: float = 1.0,
                          min_samples: int = 5) -> dict:
    from sklearn.cluster import DBSCAN

    rows = get_photos_with_gps(conn)
    if not rows:
        console.print("[yellow]No photos with GPS data found.[/yellow]")
        return {"clusters_found": 0, "photos_assigned": 0}

    photo_ids = [r["id"] for r in rows]
    coords = np.radians([[r["gps_lat"], r["gps_lon"]] for r in rows])

    eps_rad = eps_km / EARTH_RADIUS_KM
    labels = DBSCAN(
        eps=eps_rad, min_samples=min_samples, algorithm="ball_tree", metric="haversine"
    ).fit_predict(coords)

    cluster_map: dict[int, list[int]] = {}
    for photo_id, label in zip(photo_ids, labels):
        if label == -1:
            continue
        cluster_map.setdefault(label, []).append(photo_id)

    lat_lon = {r["id"]: (r["gps_lat"], r["gps_lon"]) for r in rows}

    assigned = 0
    for label, group_ids in cluster_map.items():
        lats = [lat_lon[pid][0] for pid in group_ids]
        lons = [lat_lon[pid][1] for pid in group_ids]
        center_lat = float(np.mean(lats))
        center_lon = float(np.mean(lons))
        cluster_id = upsert_geo_cluster(
            conn, label=None, center_lat=center_lat, center_lon=center_lon,
            photo_count=len(group_ids),
        )
        assign_geo_cluster(conn, group_ids, cluster_id)
        assigned += len(group_ids)

    clusters_found = len(cluster_map)
    console.print(
        f"Found [cyan]{clusters_found}[/cyan] geo clusters, "
        f"[cyan]{assigned}[/cyan] photos assigned."
    )
    return {"clusters_found": clusters_found, "photos_assigned": assigned}
