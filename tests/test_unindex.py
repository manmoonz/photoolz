from __future__ import annotations

from click.testing import CliRunner

from photoolz import db
from photoolz.cli import cli

BASE = {"file_hash": "h", "indexed_at": "2026-01-01T00:00:00+00:00"}


def test_path_matching_treats_underscore_and_percent_literally(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("PHOTOOLZ_DATA_DIR", str(data_dir))
    conn = db.get_connection(data_dir / "index.db")
    db.init_schema(conn)
    for path in ["/a/b_c/1.jpg", "/a/bxc/1.jpg", "/a/100%/1.jpg", "/a/100abc/1.jpg"]:
        db.upsert_photo(conn, {"file_path": path, **BASE})

    # Under LIKE, 'b_c' also matched 'bxc' and '100%' also matched '100abc'
    assert db.get_photo_count_under_path(conn, "/a/b_c") == 1
    assert db.get_photo_count_under_path(conn, "/a/100%") == 1

    result = CliRunner().invoke(cli, ["unindex", "/a/b_c", "--force"])
    assert result.exit_code == 0, result.output
    remaining = {r[0] for r in conn.execute("SELECT file_path FROM photos")}
    assert remaining == {"/a/bxc/1.jpg", "/a/100%/1.jpg", "/a/100abc/1.jpg"}
