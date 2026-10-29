from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from photoolz import db
from photoolz.config import Config
from photoolz.indexer import embedder, faces, pipeline

DIM = 512


def _write_image(path: Path, red: int) -> None:
    Image.new("RGB", (32, 32), (red, 0, 0)).save(path)


def _fake_embed_images(paths, model, preprocess, device="cpu", batch_size=64):
    # One-hot vector keyed by the N in "imgN.png", so each photo's embedding is identifiable
    vecs = np.zeros((len(paths), DIM), dtype=np.float32)
    for i, p in enumerate(paths):
        vecs[i, int(Path(p).stem.removeprefix("img"))] = 1.0
    return vecs, list(range(len(paths)))


def _fake_detect_faces(path, model="hog"):
    # One face per image; bbox_top records the image's red value so stale faces are detectable
    red = Image.open(path).getpixel((0, 0))[0]
    return [{
        "bbox_top": red, "bbox_right": 10, "bbox_bottom": 10, "bbox_left": 0,
        "encoding": np.zeros(128, dtype=np.float64).tobytes(),
    }]


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(embedder, "load_clip_model", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(embedder, "embed_images", _fake_embed_images)
    monkeypatch.setattr(faces, "is_face_recognition_available", lambda: True)
    monkeypatch.setattr(pipeline, "detect_and_encode_faces", _fake_detect_faces)

    library = tmp_path / "library"
    library.mkdir()
    config = Config(library_path=library, data_dir=tmp_path / "data", device="cpu")

    def run(path: Path = library, **kwargs):
        return pipeline.run_index_pipeline(path, config, workers=1, **kwargs)

    def query(sql: str, params=()):
        conn = sqlite3.connect(config.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(sql, params).fetchall()
            conn.commit()
            return rows
        finally:
            conn.close()

    return library, run, query


def _embedding_slot(blob: bytes) -> int:
    return int(np.argmax(np.frombuffer(blob, dtype=np.float32)))


def test_upsert_photo_returns_id_of_updated_row(tmp_path):
    conn = db.get_connection(tmp_path / "t.db")
    db.init_schema(conn)
    base = {"file_hash": "h", "indexed_at": "2026-01-01T00:00:00+00:00"}

    a = db.upsert_photo(conn, {"file_path": "/a.jpg", **base})
    b = db.upsert_photo(conn, {"file_path": "/b.jpg", **base})
    assert a != b
    # The UPDATE branch must return a's id, not the last inserted rowid (b)
    assert db.upsert_photo(conn, {"file_path": "/a.jpg", **base, "file_hash": "h2"}) == a


def test_mixed_new_and_changed_files_keep_their_own_embeddings(env):
    library, run, query = env
    for i in range(3):
        _write_image(library / f"img{i}.png", red=10 + i)
    run()

    # Change two existing files and add two new ones in the same run
    _write_image(library / "img0.png", red=200)
    _write_image(library / "img2.png", red=201)
    _write_image(library / "img3.png", red=30)
    _write_image(library / "img4.png", red=31)
    run()

    rows = query("SELECT file_path, clip_embedding FROM photos")
    assert len(rows) == 5
    for r in rows:
        assert _embedding_slot(r["clip_embedding"]) == int(Path(r["file_path"]).stem[3:])

    face_rows = query("SELECT p.file_path, f.bbox_top FROM faces f JOIN photos p ON p.id = f.photo_id")
    assert {Path(r["file_path"]).name: r["bbox_top"] for r in face_rows} == {
        "img0.png": 200, "img1.png": 11, "img2.png": 201, "img3.png": 30, "img4.png": 31,
    }


def test_force_reindex_does_not_duplicate_faces_or_drop_people(env):
    library, run, query = env
    for i in range(3):
        _write_image(library / f"img{i}.png", red=10 + i)
    run()

    query("INSERT INTO people (name, face_count, clustered_at) VALUES ('Alice', 3, 'x')")
    query("UPDATE faces SET person_id = 1")

    run(force=True)

    rows = query("SELECT photo_id, person_id FROM faces")
    assert len(rows) == 3
    assert all(r["person_id"] == 1 for r in rows)


def test_changed_file_replaces_its_old_faces(env):
    library, run, query = env
    _write_image(library / "img0.png", red=10)
    run()

    _write_image(library / "img0.png", red=99)
    run()

    rows = query("SELECT bbox_top FROM faces")
    assert [r["bbox_top"] for r in rows] == [99]


def test_relative_library_path_is_stored_absolute(env, monkeypatch):
    library, run, query = env
    _write_image(library / "img0.png", red=10)
    monkeypatch.chdir(library.parent)
    run(Path("library"))

    (photo,) = query("SELECT file_path FROM photos")
    assert photo["file_path"] == str(library.resolve() / "img0.png")
    (indexed,) = query("SELECT path FROM indexed_dirs")
    assert indexed["path"] == str(library.resolve())

    # Indexing again via the absolute path must not create a second row
    run(library)
    assert len(query("SELECT id FROM photos")) == 1
