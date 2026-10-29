from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from photoolz import db
from photoolz.cli import cli


@pytest.fixture
def conn(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("PHOTOOLZ_DATA_DIR", str(data_dir))
    conn = db.get_connection(data_dir / "index.db")
    db.init_schema(conn)
    return conn


def _add_photo(conn, name, quality=0.1):
    pid = db.upsert_photo(conn, {"file_path": f"/p/{name}.jpg", "file_hash": name,
                                 "indexed_at": "2026-01-01T00:00:00+00:00"})
    db.bulk_update_quality_scores(conn, [(quality, quality, quality, pid)])
    return pid


def test_album_show_json_is_valid_json(conn):
    pid = _add_photo(conn, "a")
    album_id = db.save_album(conn, "Trip", "A trip", "q", [pid])

    result = CliRunner().invoke(cli, ["album", "show", str(album_id), "--output", "json"])
    assert result.exit_code == 0, result.output
    assert [p["id"] for p in json.loads(result.stdout)] == [pid]


def test_status_messages_stay_off_stdout(conn, monkeypatch):
    _add_photo(conn, "a")
    monkeypatch.setattr("photoolz.utils.viewer.subprocess.Popen", lambda *a, **k: None)
    runner = CliRunner()

    flagged = runner.invoke(cli, ["quality", "--output", "paths", "--mark-deleted",
                                  "--open", "--viewer", "feh"])
    assert flagged.stdout.split() == ["/p/a.jpg"]
    assert "Flagged 1 photos" in flagged.stderr and "Opened 1 photo" in flagged.stderr

    # Everything is now flagged, so there's nothing to report
    empty = runner.invoke(cli, ["quality", "--output", "paths"])
    assert empty.stdout == "" and "No quality issues" in empty.stderr

    missing = runner.invoke(cli, ["people", "photos", "999", "--output", "csv"])
    assert missing.stdout == ""
