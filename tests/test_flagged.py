from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from photoolz import db
from photoolz.cli import cli


@pytest.fixture
def lib(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("PHOTOOLZ_DATA_DIR", str(data_dir))
    conn = db.get_connection(data_dir / "index.db")
    db.init_schema(conn)

    library = tmp_path / "lib"
    (library / "sub").mkdir(parents=True)
    db.upsert_indexed_dir(conn, str(library), 0)
    ids = {}
    for name, reason in [("blurry", "quality"), ("sub/copy", "duplicate"), ("keep", None)]:
        path = library / f"{name}.jpg"
        path.write_bytes(name.encode())
        ids[name] = db.upsert_photo(conn, {"file_path": str(path), "file_hash": name,
                                           "indexed_at": "x"})
        if reason:
            db.mark_deleted(conn, ids[name], reason)
    return conn, library, ids


def _flagged_names(conn):
    return sorted(Path(r["file_path"]).stem for r in db.get_flagged_photos(conn))


def test_list_filters_by_reason(lib):
    conn, library, _ = lib
    runner = CliRunner()
    result = runner.invoke(cli, ["flagged", "list", "--output", "paths"])
    assert result.stdout.split() == [str(library / "blurry.jpg"), str(library / "sub/copy.jpg")]
    result = runner.invoke(cli, ["flagged", "list", "--reason", "duplicate", "--output", "paths"])
    assert result.stdout.split() == [str(library / "sub/copy.jpg")]


def test_restore_by_id_and_all(lib):
    conn, _, ids = lib
    runner = CliRunner()
    assert runner.invoke(cli, ["flagged", "restore"]).exit_code == 1  # needs IDs or --all

    result = runner.invoke(cli, ["flagged", "restore", str(ids["blurry"]), str(ids["keep"])])
    assert "Restored 1 photo" in result.output and "1 ID(s) weren't flagged" in result.output
    assert _flagged_names(conn) == ["copy"]

    runner.invoke(cli, ["flagged", "restore", "--all", "--reason", "duplicate"])
    assert _flagged_names(conn) == []


def test_move_mirrors_paths_and_removes_moved_photos_from_index(lib, tmp_path):
    conn, library, ids = lib
    trash = tmp_path / "trash"
    result = CliRunner().invoke(cli, ["flagged", "move", str(trash), "--force"])
    assert result.exit_code == 0, result.output

    for name in ["blurry", "sub/copy"]:
        src = library / f"{name}.jpg"
        assert not src.exists()
        assert (trash / src.relative_to("/")).read_bytes() == name.encode()
    assert (library / "keep.jpg").exists()
    assert [r["id"] for r in conn.execute("SELECT id FROM photos")] == [ids["keep"]]


def test_move_skips_conflicts_and_missing_files(lib, tmp_path):
    conn, library, _ = lib
    trash = tmp_path / "trash"
    taken = trash / (library / "blurry.jpg").relative_to("/")
    taken.parent.mkdir(parents=True)
    taken.write_bytes(b"something else")
    (library / "sub/copy.jpg").unlink()

    result = CliRunner().invoke(cli, ["flagged", "move", str(trash), "--force"])
    assert "Moved 0 photo(s)" in result.output and "Skipped 2" in result.output
    assert (library / "blurry.jpg").exists() and taken.read_bytes() == b"something else"
    assert _flagged_names(conn) == ["blurry", "copy"]


def test_move_refuses_destination_inside_indexed_dir(lib):
    conn, library, _ = lib
    result = CliRunner().invoke(cli, ["flagged", "move", str(library / "trash"), "--force"])
    assert result.exit_code == 1 and "inside indexed directory" in result.output
    assert (library / "blurry.jpg").exists()
    assert _flagged_names(conn) == ["blurry", "copy"]
