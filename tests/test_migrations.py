from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest
from click.testing import CliRunner
from PIL import Image

from photoolz import db
from photoolz.cli import cli
from photoolz.indexer.metadata import extract_metadata


@pytest.fixture
def new_york_time(monkeypatch):
    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


@pytest.fixture
def v0_db(tmp_path, monkeypatch):
    """A database as created before migrations existed."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("PHOTOOLZ_DATA_DIR", str(data_dir))
    path = data_dir / "index.db"
    conn = sqlite3.connect(path)
    conn.executescript(db.SCHEMA)
    conn.execute("ALTER TABLE people ADD COLUMN face_count INTEGER")
    conn.execute("PRAGMA user_version = 0")
    conn.executemany(
        "INSERT INTO photos (file_path, file_hash, taken_at, taken_at_source, indexed_at) "
        "VALUES (?, 'h', ?, ?, 'x')",
        [("/p/exif.jpg", "2024-06-01T10:00:00+00:00", "exif"),
         ("/p/name.jpg", "2024-06-01T00:00:00+00:00", "filename"),
         ("/p/mtime.jpg", "2024-06-01T10:00:00.123456+00:00", "filesystem"),
         ("/p/none.jpg", None, None)],
    )
    conn.execute("INSERT INTO people (name, face_count, clustered_at) VALUES ('Alice', 3, 'x')")
    conn.commit()
    conn.close()
    return path


def _columns(conn, table):
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]


def test_new_database_starts_at_latest_version(tmp_path):
    conn = db.get_connection(tmp_path / "new.db")
    db.init_schema(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
    assert db.pending_migrations(conn) == []
    assert "face_count" not in _columns(conn, "people")


def test_commands_refuse_outdated_database_without_changing_it(v0_db):
    result = CliRunner().invoke(cli, ["stats"])
    assert result.exit_code == 1
    assert "photoolz db init" in result.output and "people.face_count" in result.output

    conn = sqlite3.connect(v0_db)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 0
    assert "face_count" in _columns(conn, "people")


def test_db_init_backs_up_then_migrates(v0_db, new_york_time):
    runner = CliRunner()
    result = runner.invoke(cli, ["db", "init"])
    assert result.exit_code == 0, result.output
    assert "Upgraded the database (2 migration(s))" in result.output

    (backup,) = v0_db.parent.glob("index.db.bak-v0-*")
    old = sqlite3.connect(backup)
    assert old.execute("PRAGMA user_version").fetchone()[0] == 0
    assert "face_count" in _columns(old, "people")

    conn = sqlite3.connect(v0_db)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
    assert "face_count" not in _columns(conn, "people")
    assert conn.execute("SELECT name FROM people").fetchone()[0] == "Alice"
    taken = dict(conn.execute("SELECT file_path, taken_at FROM photos"))
    assert taken == {
        "/p/exif.jpg": "2024-06-01T10:00:00",      # local time; only the label was wrong
        "/p/name.jpg": "2024-06-01T00:00:00",
        "/p/mtime.jpg": "2024-06-01T06:00:00.123456",  # real UTC, converted to New York (EDT)
        "/p/none.jpg": None,
    }

    # Upgraded databases work, and running init again changes nothing
    assert runner.invoke(cli, ["stats"]).exit_code == 0
    again = runner.invoke(cli, ["db", "init"])
    assert "up to date" in again.output
    assert len(list(v0_db.parent.glob("index.db.bak-*"))) == 1


def test_failed_migration_rolls_back_to_last_completed_version(v0_db, monkeypatch):
    def broken(conn):
        conn.execute("UPDATE photos SET taken_at = 'clobbered'")
        raise RuntimeError("boom")

    monkeypatch.setattr(db, "MIGRATIONS", [db.MIGRATIONS[0], (2, "Broken", broken)])
    conn = db.get_connection(v0_db)
    with pytest.raises(RuntimeError, match="boom"):
        db.migrate_schema(conn, v0_db)

    assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
    assert "face_count" not in _columns(conn, "people")
    assert "clobbered" not in {r[0] for r in conn.execute("SELECT taken_at FROM photos")}


def test_new_photos_get_local_times_without_a_time_zone(tmp_path, new_york_time):
    named = tmp_path / "2024-06-01_10-00-00.png"
    Image.new("RGB", (4, 4)).save(named)
    meta = extract_metadata(named)
    assert (meta["taken_at"], meta["taken_at_source"]) == ("2024-06-01T10:00:00", "filename")

    undated = tmp_path / "photo.png"
    Image.new("RGB", (4, 4)).save(undated)
    meta = extract_metadata(undated)
    assert meta["taken_at_source"] == "filesystem"
    assert "+" not in meta["taken_at"]
    expected = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(undated.stat().st_mtime))
    assert meta["taken_at"].startswith(expected)
