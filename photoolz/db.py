from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

_local = threading.local()
_write_lock = threading.Lock()

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS photos (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path       TEXT    NOT NULL UNIQUE,
    file_hash       TEXT    NOT NULL,
    phash           TEXT,
    width           INTEGER,
    height          INTEGER,
    format          TEXT,
    file_size_bytes INTEGER,
    taken_at        TEXT,
    taken_at_source TEXT,
    gps_lat         REAL,
    gps_lon         REAL,
    gps_alt         REAL,
    camera_make     TEXT,
    camera_model    TEXT,
    blur_score      REAL,
    exposure_score  REAL,
    quality_score   REAL,
    clip_embedding  BLOB,
    clip_model      TEXT,
    geo_cluster_id  INTEGER,
    burst_group_id  INTEGER,
    indexed_at      TEXT    NOT NULL,
    is_deleted      INTEGER NOT NULL DEFAULT 0,
    deletion_reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_photos_taken_at    ON photos(taken_at);
CREATE INDEX IF NOT EXISTS idx_photos_geo_cluster ON photos(geo_cluster_id);
CREATE INDEX IF NOT EXISTS idx_photos_burst_group ON photos(burst_group_id);
CREATE INDEX IF NOT EXISTS idx_photos_quality     ON photos(quality_score);
CREATE INDEX IF NOT EXISTS idx_photos_phash       ON photos(phash);

CREATE TABLE IF NOT EXISTS faces (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id    INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    person_id   INTEGER,
    bbox_top    INTEGER,
    bbox_right  INTEGER,
    bbox_bottom INTEGER,
    bbox_left   INTEGER,
    encoding    BLOB    NOT NULL,
    detected_at TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_faces_photo_id  ON faces(photo_id);
CREATE INDEX IF NOT EXISTS idx_faces_person_id ON faces(person_id);

CREATE TABLE IF NOT EXISTS people (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    name                    TEXT,
    face_count              INTEGER,
    representative_photo_id INTEGER REFERENCES photos(id),
    clustered_at            TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS albums (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT    NOT NULL,
    description TEXT,
    query       TEXT,
    created_at  TEXT    NOT NULL,
    source      TEXT    NOT NULL DEFAULT 'claude'
);

CREATE TABLE IF NOT EXISTS album_photos (
    album_id  INTEGER NOT NULL REFERENCES albums(id) ON DELETE CASCADE,
    photo_id  INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    rank      INTEGER,
    PRIMARY KEY (album_id, photo_id)
);

CREATE TABLE IF NOT EXISTS geo_clusters (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    label       TEXT,
    center_lat  REAL,
    center_lon  REAL,
    photo_count INTEGER,
    clustered_at TEXT NOT NULL
);
"""


def get_connection(db_path: Path) -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None or _local.db_path != db_path:
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _local.conn = conn
        _local.db_path = db_path
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    with _write_lock:
        conn.executescript(SCHEMA)
        conn.commit()


def upsert_photo(conn: sqlite3.Connection, record: dict) -> int:
    cols = list(record.keys())
    placeholders = ", ".join("?" * len(cols))
    col_names = ", ".join(cols)
    updates = ", ".join(f"{c}=excluded.{c}" for c in cols if c != "file_path")
    sql = f"""
        INSERT INTO photos ({col_names}) VALUES ({placeholders})
        ON CONFLICT(file_path) DO UPDATE SET {updates}
    """
    with _write_lock:
        cur = conn.execute(sql, list(record.values()))
        conn.commit()
        if cur.lastrowid:
            return cur.lastrowid
        row = conn.execute("SELECT id FROM photos WHERE file_path=?", (record["file_path"],)).fetchone()
        return row["id"]


def get_photo_by_path(conn: sqlite3.Connection, file_path: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM photos WHERE file_path=?", (file_path,)).fetchone()


def get_photos_missing_embedding(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT id, file_path FROM photos WHERE clip_embedding IS NULL AND is_deleted=0"
    ).fetchall()


def get_all_embeddings(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT id, clip_embedding FROM photos WHERE clip_embedding IS NOT NULL AND is_deleted=0"
    ).fetchall()


def get_all_photo_ids_and_paths(conn: sqlite3.Connection) -> list[tuple[int, str]]:
    rows = conn.execute("SELECT id, file_path FROM photos WHERE is_deleted=0").fetchall()
    return [(r["id"], r["file_path"]) for r in rows]


def update_clip_embedding(conn: sqlite3.Connection, photo_id: int, embedding: bytes, model: str) -> None:
    with _write_lock:
        conn.execute(
            "UPDATE photos SET clip_embedding=?, clip_model=? WHERE id=?",
            (embedding, model, photo_id),
        )
        conn.commit()


def update_quality_scores(conn: sqlite3.Connection, photo_id: int,
                           blur: float, exposure: float, quality: float) -> None:
    with _write_lock:
        conn.execute(
            "UPDATE photos SET blur_score=?, exposure_score=?, quality_score=? WHERE id=?",
            (blur, exposure, quality, photo_id),
        )
        conn.commit()


def bulk_update_quality_scores(conn: sqlite3.Connection,
                                rows: list[tuple[float, float, float, int]]) -> None:
    with _write_lock:
        conn.executemany(
            "UPDATE photos SET blur_score=?, exposure_score=?, quality_score=? WHERE id=?",
            rows,
        )
        conn.commit()


def mark_deleted(conn: sqlite3.Connection, photo_id: int, reason: str) -> None:
    with _write_lock:
        conn.execute(
            "UPDATE photos SET is_deleted=1, deletion_reason=? WHERE id=?",
            (reason, photo_id),
        )
        conn.commit()


def get_photos_missing_quality(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT id, file_path FROM photos WHERE quality_score IS NULL AND is_deleted=0"
    ).fetchall()


def get_worst_quality_photos(conn: sqlite3.Connection, limit: int = 50,
                              min_quality: float | None = None) -> list[sqlite3.Row]:
    if min_quality is not None:
        return conn.execute(
            "SELECT * FROM photos WHERE quality_score IS NOT NULL AND quality_score <= ? "
            "AND is_deleted=0 ORDER BY quality_score ASC LIMIT ?",
            (min_quality, limit),
        ).fetchall()
    return conn.execute(
        "SELECT * FROM photos WHERE quality_score IS NOT NULL AND is_deleted=0 "
        "ORDER BY quality_score ASC LIMIT ?",
        (limit,),
    ).fetchall()


def get_all_phashes(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT id, file_path, phash, taken_at, camera_model, quality_score "
        "FROM photos WHERE phash IS NOT NULL AND is_deleted=0"
    ).fetchall()


def save_album(conn: sqlite3.Connection, title: str, description: str,
               query: str, photo_ids: list[int], source: str = "claude") -> int:
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    with _write_lock:
        cur = conn.execute(
            "INSERT INTO albums (title, description, query, created_at, source) VALUES (?,?,?,?,?)",
            (title, description, query, now, source),
        )
        album_id = cur.lastrowid
        conn.executemany(
            "INSERT INTO album_photos (album_id, photo_id, rank) VALUES (?,?,?)",
            [(album_id, pid, rank) for rank, pid in enumerate(photo_ids)],
        )
        conn.commit()
    return album_id


def get_album(conn: sqlite3.Connection, album_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM albums WHERE id=?", (album_id,)).fetchone()


def get_album_photos(conn: sqlite3.Connection, album_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT p.* FROM photos p "
        "JOIN album_photos ap ON ap.photo_id = p.id "
        "WHERE ap.album_id=? ORDER BY ap.rank",
        (album_id,),
    ).fetchall()


def list_albums(conn: sqlite3.Connection, limit: int = 20) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT a.*, COUNT(ap.photo_id) as photo_count FROM albums a "
        "LEFT JOIN album_photos ap ON ap.album_id = a.id "
        "GROUP BY a.id ORDER BY a.created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()


def upsert_face(conn: sqlite3.Connection, record: dict) -> int:
    cols = list(record.keys())
    placeholders = ", ".join("?" * len(cols))
    col_names = ", ".join(cols)
    sql = f"INSERT INTO faces ({col_names}) VALUES ({placeholders})"
    with _write_lock:
        cur = conn.execute(sql, list(record.values()))
        conn.commit()
        return cur.lastrowid


def get_all_face_encodings(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT id, photo_id, encoding FROM faces WHERE person_id IS NULL").fetchall()


def upsert_person(conn: sqlite3.Connection, name: str | None, face_count: int,
                   rep_photo_id: int | None) -> int:
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    with _write_lock:
        cur = conn.execute(
            "INSERT INTO people (name, face_count, representative_photo_id, clustered_at) VALUES (?,?,?,?)",
            (name, face_count, rep_photo_id, now),
        )
        conn.commit()
        return cur.lastrowid


def assign_person_to_faces(conn: sqlite3.Connection, face_ids: list[int], person_id: int) -> None:
    with _write_lock:
        conn.executemany(
            "UPDATE faces SET person_id=? WHERE id=?",
            [(person_id, fid) for fid in face_ids],
        )
        conn.commit()


def label_person(conn: sqlite3.Connection, person_id: int, name: str) -> None:
    with _write_lock:
        conn.execute("UPDATE people SET name=? WHERE id=?", (name, person_id))
        conn.commit()


def get_people(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT p.*, ph.file_path as rep_path FROM people p "
        "LEFT JOIN photos ph ON ph.id = p.representative_photo_id "
        "ORDER BY p.face_count DESC"
    ).fetchall()


def get_photos_by_person(conn: sqlite3.Connection, person_name: str) -> list[int]:
    rows = conn.execute(
        "SELECT DISTINCT f.photo_id FROM faces f "
        "JOIN people pe ON pe.id = f.person_id "
        "WHERE pe.name=?",
        (person_name,),
    ).fetchall()
    return [r["photo_id"] for r in rows]


def upsert_geo_cluster(conn: sqlite3.Connection, label: str | None,
                        center_lat: float, center_lon: float, photo_count: int) -> int:
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    with _write_lock:
        cur = conn.execute(
            "INSERT INTO geo_clusters (label, center_lat, center_lon, photo_count, clustered_at) "
            "VALUES (?,?,?,?,?)",
            (label, center_lat, center_lon, photo_count, now),
        )
        conn.commit()
        return cur.lastrowid


def assign_geo_cluster(conn: sqlite3.Connection, photo_ids: list[int], cluster_id: int) -> None:
    with _write_lock:
        conn.executemany(
            "UPDATE photos SET geo_cluster_id=? WHERE id=?",
            [(cluster_id, pid) for pid in photo_ids],
        )
        conn.commit()


def assign_burst_group(conn: sqlite3.Connection, photo_ids: list[int], group_id: int) -> None:
    with _write_lock:
        conn.executemany(
            "UPDATE photos SET burst_group_id=? WHERE id=?",
            [(group_id, pid) for pid in photo_ids],
        )
        conn.commit()


def get_photos_with_gps(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT id, gps_lat, gps_lon FROM photos WHERE gps_lat IS NOT NULL AND is_deleted=0"
    ).fetchall()


def get_photos_ordered_by_time(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT id, taken_at, quality_score FROM photos "
        "WHERE taken_at IS NOT NULL AND is_deleted=0 ORDER BY taken_at ASC"
    ).fetchall()


def get_photos_by_ids(conn: sqlite3.Connection, photo_ids: list[int]) -> list[sqlite3.Row]:
    if not photo_ids:
        return []
    placeholders = ",".join("?" * len(photo_ids))
    return conn.execute(
        f"SELECT * FROM photos WHERE id IN ({placeholders})", photo_ids
    ).fetchall()


def get_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    total = conn.execute("SELECT COUNT(*) FROM photos WHERE is_deleted=0").fetchone()[0]
    indexed = conn.execute(
        "SELECT COUNT(*) FROM photos WHERE clip_embedding IS NOT NULL AND is_deleted=0"
    ).fetchone()[0]
    flagged = conn.execute("SELECT COUNT(*) FROM photos WHERE is_deleted=1").fetchone()[0]
    albums = conn.execute("SELECT COUNT(*) FROM albums").fetchone()[0]
    people = conn.execute("SELECT COUNT(*) FROM people").fetchone()[0]
    faces = conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
    geo = conn.execute("SELECT COUNT(*) FROM geo_clusters").fetchone()[0]
    return {
        "total_photos": total,
        "indexed": indexed,
        "flagged_deleted": flagged,
        "albums": albums,
        "people": people,
        "faces": faces,
        "geo_clusters": geo,
    }
