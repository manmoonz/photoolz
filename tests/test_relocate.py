from __future__ import annotations

import sqlite3

import pytest
from click.testing import CliRunner

from photoolz import db
from photoolz.cli import cli

BASE = {"file_hash": "h", "indexed_at": "2026-01-01T00:00:00+00:00"}


@pytest.fixture
def conn(tmp_path):
    conn = db.get_connection(tmp_path / "t.db")
    db.init_schema(conn)
    return conn


def _paths(conn):
    return {r["id"]: r["file_path"] for r in conn.execute("SELECT id, file_path FROM photos")}


def test_relocate_rewrites_prefix_and_keeps_ids(conn):
    a = db.upsert_photo(conn, {"file_path": "/media/DATA/pics/a.jpg", **BASE})
    b = db.upsert_photo(conn, {"file_path": "/media/DATA/pics/2014/b.jpg", **BASE})
    db.upsert_indexed_dir(conn, "/media/DATA/pics", 2)
    db.upsert_indexed_dir(conn, "/media/DATA/pics/2014", 1)

    assert db.relocate_paths(conn, "/media/DATA/", "/run/media/DATA") == (2, 2)
    assert _paths(conn) == {a: "/run/media/DATA/pics/a.jpg", b: "/run/media/DATA/pics/2014/b.jpg"}
    assert {r["path"] for r in db.list_indexed_dirs(conn)} == {
        "/run/media/DATA/pics", "/run/media/DATA/pics/2014",
    }


def test_relocate_only_matches_whole_path_components(conn):
    sibling = db.upsert_photo(conn, {"file_path": "/media/DATA2/a.jpg", **BASE})
    # '_' would be a wildcard under LIKE and match 'bxc'
    wildcard = db.upsert_photo(conn, {"file_path": "/media/bxc/a.jpg", **BASE})
    moved = db.upsert_photo(conn, {"file_path": "/media/b_c/a.jpg", **BASE})

    db.relocate_paths(conn, "/media/DATA", "/new")
    db.relocate_paths(conn, "/media/b_c", "/new2")

    assert _paths(conn) == {
        sibling: "/media/DATA2/a.jpg", wildcard: "/media/bxc/a.jpg", moved: "/new2/a.jpg",
    }


def test_relocate_conflict_changes_nothing(conn):
    db.upsert_photo(conn, {"file_path": "/old/a.jpg", **BASE})
    db.upsert_photo(conn, {"file_path": "/old/b.jpg", **BASE})
    db.upsert_photo(conn, {"file_path": "/new/b.jpg", **BASE})
    before = _paths(conn)

    with pytest.raises(sqlite3.IntegrityError):
        db.relocate_paths(conn, "/old", "/new")
    assert _paths(conn) == before


def test_cli_relocate_reports_found_files_and_confirms(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("PHOTOOLZ_DATA_DIR", str(data_dir))
    new_root = tmp_path / "mnt" / "DATA"
    (new_root / "pics").mkdir(parents=True)
    (new_root / "pics" / "a.jpg").write_bytes(b"x")

    data_dir.mkdir()
    conn = db.get_connection(data_dir / "index.db")
    db.init_schema(conn)
    db.upsert_photo(conn, {"file_path": "/gone/DATA/pics/a.jpg", **BASE})
    db.upsert_photo(conn, {"file_path": "/gone/DATA/pics/missing.jpg", **BASE})

    runner = CliRunner()
    declined = runner.invoke(cli, ["db", "relocate", "/gone/DATA", str(new_root)], input="n\n")
    assert "1 of 2 photos exist" in declined.output
    assert declined.exit_code != 0
    assert set(_paths(conn).values()) == {"/gone/DATA/pics/a.jpg", "/gone/DATA/pics/missing.jpg"}

    result = runner.invoke(cli, ["db", "relocate", "/gone/DATA", str(new_root), "--force"])
    assert result.exit_code == 0, result.output
    assert set(_paths(conn).values()) == {
        f"{new_root}/pics/a.jpg", f"{new_root}/pics/missing.jpg",
    }
