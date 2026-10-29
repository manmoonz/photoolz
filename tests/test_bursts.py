from __future__ import annotations

from photoolz import db
from photoolz.clustering.burst import detect_bursts, get_burst_best_frames


def _add(conn, name, taken_at, quality=0.5):
    pid = db.upsert_photo(conn, {"file_path": f"/p/{name}.jpg", "file_hash": name,
                                 "taken_at": taken_at, "indexed_at": "2026-01-01T00:00:00+00:00"})
    db.bulk_update_quality_scores(conn, [(quality, quality, quality, pid)])
    return pid


def _burst_ids(conn):
    return [r[0] for r in conn.execute("SELECT burst_group_id FROM photos ORDER BY id")]


def test_bursts_group_close_frames_and_pick_sharpest(tmp_path):
    conn = db.get_connection(tmp_path / "t.db")
    db.init_schema(conn)
    burst = [_add(conn, f"b{i}", f"2024-01-01T10:00:0{i * 2}+00:00", quality=q)
             for i, q in enumerate([0.2, 0.9, 0.4])]
    _add(conn, "alone", "2024-01-01T11:00:00+00:00")

    assert detect_bursts(conn, gap_seconds=3) == {"burst_groups_found": 1, "photos_in_bursts": 3}
    (best,) = get_burst_best_frames(conn)
    assert best["id"] == burst[1] and best["burst_size"] == 3


def test_rerun_clears_bursts_from_previous_run(tmp_path):
    conn = db.get_connection(tmp_path / "t.db")
    db.init_schema(conn)
    for i in range(3):
        _add(conn, f"b{i}", f"2024-01-01T10:00:0{i * 2}+00:00")

    detect_bursts(conn, gap_seconds=3)
    assert None not in _burst_ids(conn)

    # Frames are 2s apart, so a 1s gap finds no bursts and must clear the old IDs
    assert detect_bursts(conn, gap_seconds=1)["burst_groups_found"] == 0
    assert _burst_ids(conn) == [None, None, None]
    assert get_burst_best_frames(conn) == []
