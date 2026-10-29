from __future__ import annotations

import numpy as np

from photoolz import db

ENC = np.zeros(128).tobytes()


def _setup(tmp_path):
    conn = db.get_connection(tmp_path / "t.db")
    db.init_schema(conn)
    photos = [db.upsert_photo(conn, {"file_path": f"/p/{i}.jpg", "file_hash": str(i),
                                     "indexed_at": "2026-01-01T00:00:00+00:00"}) for i in range(5)]
    return conn, photos


def _add_faces(conn, photo_ids, person_id):
    face_ids = [db.upsert_face(conn, {"photo_id": pid, "encoding": ENC, "detected_at": "x"})
                for pid in photo_ids]
    db.assign_person_to_faces(conn, face_ids, person_id)


def _counts(conn):
    return {r["id"]: r["face_count"] for r in db.get_people(conn)}


def test_face_counts_follow_faces_when_redetected(tmp_path):
    conn, photos = _setup(tmp_path)
    alice = db.upsert_person(conn, name="Alice", rep_photo_id=photos[0])
    _add_faces(conn, photos[:3], alice)
    assert _counts(conn) == {alice: 3}

    # Re-indexing a changed photo replaces its faces with new, unassigned ones
    db.delete_faces_for_photo(conn, photos[0])
    assert _counts(conn) == {alice: 2}


def test_merge_keeps_largest_by_actual_faces_and_totals_them(tmp_path):
    conn, photos = _setup(tmp_path)
    # Stored counts are deliberately wrong; the merge must use the real ones
    small = db.upsert_person(conn, name=None, rep_photo_id=None)
    large = db.upsert_person(conn, name=None, rep_photo_id=None)
    _add_faces(conn, photos[:1], small)
    _add_faces(conn, photos[1:4], large)

    assert db.merge_people(conn, [small, large]) == large
    assert _counts(conn) == {large: 4}


def test_cluster_faces_uses_largest_face_as_representative(tmp_path):
    from photoolz.clustering.faces_cluster import cluster_faces

    conn, photos = _setup(tmp_path)
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(2, 128))
    sizes = [10, 50, 20, 30, 40]  # photos[1] has the largest face of person A
    for i, pid in enumerate(photos):
        enc = centers[0 if i < 3 else 1] + rng.normal(scale=0.01, size=128)
        s = sizes[i]
        db.upsert_face(conn, {"photo_id": pid, "encoding": enc.astype(np.float64).tobytes(),
                              "bbox_top": 0, "bbox_left": 0, "bbox_bottom": s, "bbox_right": s,
                              "detected_at": "x"})

    result = cluster_faces(conn, eps=0.5, min_samples=2)
    assert result == {"clusters_found": 2, "faces_assigned": 5, "noise_faces": 0}
    reps = sorted((r["face_count"], r["representative_photo_id"]) for r in db.get_people(conn))
    assert reps == [(2, photos[4]), (3, photos[1])]
