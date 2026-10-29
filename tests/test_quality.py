from __future__ import annotations

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image

from photoolz import db
from photoolz.cli import cli

pytest.importorskip("cv2")


def test_db_reindex_rescores_every_photo_with_given_threshold(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("PHOTOOLZ_DATA_DIR", str(data_dir))
    conn = db.get_connection(data_dir / "index.db")
    db.init_schema(conn)

    rng = np.random.default_rng(0)
    ids = []
    for i in range(2):
        path = tmp_path / f"noise{i}.png"
        Image.fromarray(rng.integers(0, 256, (64, 64), dtype=np.uint8)).save(path)
        ids.append(db.upsert_photo(conn, {"file_path": str(path), "file_hash": str(i),
                                          "indexed_at": "2026-01-01T00:00:00+00:00"}))
    # Pre-existing scores must be overwritten, not skipped
    db.bulk_update_quality_scores(conn, [(0.5, 0.5, 0.5, pid) for pid in ids])

    def blur_scores():
        return [r["blur_score"] for r in conn.execute("SELECT blur_score FROM photos ORDER BY id")]

    runner = CliRunner()
    result = runner.invoke(cli, ["db", "reindex", "--table", "photos", "--blur-threshold", "1e9"])
    assert result.exit_code == 0, result.output
    assert "Scored 2 photos" in result.output
    assert all(b < 0.01 for b in blur_scores())

    runner.invoke(cli, ["db", "reindex", "--table", "photos"])
    assert all(b == 1.0 for b in blur_scores())  # random noise is very sharp at the default


def test_quality_no_longer_accepts_blur_threshold():
    result = CliRunner().invoke(cli, ["quality", "--blur-threshold", "50"])
    assert result.exit_code == 2
    assert "No such option" in result.output
