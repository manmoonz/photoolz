from __future__ import annotations

import sqlite3

import numpy as np

from photoolz.db import (
    get_all_face_encodings,
    upsert_person,
    assign_person_to_faces,
    get_photos_by_ids,
)
from photoolz.utils.console import console


def cluster_faces(conn: sqlite3.Connection, eps: float = 0.5,
                  min_samples: int = 3) -> dict:
    from sklearn.cluster import DBSCAN

    rows = get_all_face_encodings(conn)
    if not rows:
        console.print("[yellow]No unassigned faces found. Run 'photoolz index' first.[/yellow]")
        return {"clusters_found": 0, "faces_assigned": 0, "noise_faces": 0}

    face_ids = [r["id"] for r in rows]
    photo_ids = [r["photo_id"] for r in rows]
    encodings = []
    for r in rows:
        try:
            enc = np.frombuffer(r["encoding"], dtype=np.float64).copy()
            encodings.append(enc)
        except Exception:
            encodings.append(np.zeros(128))

    matrix = np.vstack(encodings)
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit_predict(matrix)

    cluster_map: dict[int, list[int]] = {}
    noise = 0
    for face_id, label in zip(face_ids, labels):
        if label == -1:
            noise += 1
            continue
        cluster_map.setdefault(label, []).append(face_id)

    assigned = 0
    for label, group_face_ids in cluster_map.items():
        group_photo_ids = [photo_ids[face_ids.index(fid)] for fid in group_face_ids]
        rep_photo_id = group_photo_ids[0] if group_photo_ids else None
        person_id = upsert_person(conn, name=None, face_count=len(group_face_ids),
                                   rep_photo_id=rep_photo_id)
        assign_person_to_faces(conn, group_face_ids, person_id)
        assigned += len(group_face_ids)

    clusters_found = len(cluster_map)
    console.print(
        f"Found [cyan]{clusters_found}[/cyan] face clusters, "
        f"[cyan]{assigned}[/cyan] faces assigned, "
        f"[dim]{noise}[/dim] noise faces."
    )
    return {"clusters_found": clusters_found, "faces_assigned": assigned, "noise_faces": noise}
