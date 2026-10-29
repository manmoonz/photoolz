from __future__ import annotations

import numpy as np
import pytest

from photoolz import db
from photoolz.config import Config
from photoolz.indexer.embedder import serialize_embedding
from photoolz.search import query
from photoolz.search.faiss_index import build_or_update_faiss_index
from photoolz.search.filters import apply_sql_filters

DIM = 512
ENC = np.zeros(128).tobytes()


def _vec(*weights: float) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[:len(weights)] = weights
    return v / np.linalg.norm(v)


@pytest.fixture
def lib(tmp_path, monkeypatch):
    config = Config(library_path=tmp_path, data_dir=tmp_path, clip_model="ViT-B-32")
    conn = db.get_connection(config.db_path)
    db.init_schema(conn)

    # The fake text embedding points along axis 0, so photos score by their axis-0 weight
    monkeypatch.setattr(query, "load_clip_model", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(query, "embed_text", lambda text, *a, **k: _vec(1.0))

    alice = db.upsert_person(conn, name="Alice", rep_photo_id=None)
    bob = db.upsert_person(conn, name="Bob", rep_photo_id=None)
    place = db.upsert_geo_cluster(conn, label="Beach", center_lat=0, center_lon=0, photo_count=0)

    ids = {}

    def add(name, taken_at, relevance, person=None, geo=None):
        pid = db.upsert_photo(conn, {"file_path": f"/p/{name}.jpg", "file_hash": name,
                                     "taken_at": taken_at, "indexed_at": "x"})
        db.update_clip_embedding(conn, pid, serialize_embedding(_vec(relevance, 1 - relevance)),
                                 "ViT-B-32")
        if person:
            face = db.upsert_face(conn, {"photo_id": pid, "encoding": ENC, "detected_at": "x"})
            db.assign_person_to_faces(conn, [face], person)
        if geo:
            db.assign_geo_cluster(conn, [pid], geo)
        ids[name] = pid

    add("best", "2024-06-15T10:00:00+00:00", 0.9, person=alice, geo=place)
    add("good", "2024-07-01T23:30:00+00:00", 0.7, person=bob)
    add("ok", "2024-08-01T09:00:00+00:00", 0.5, person=alice)
    add("poor", "2023-01-01T09:00:00+00:00", 0.1)
    build_or_update_faiss_index(conn, config)
    return conn, config, ids, alice


def _names(results, ids):
    by_id = {v: k for k, v in ids.items()}
    return [by_id[r["id"]] for r in results]


def test_results_ranked_by_similarity(lib):
    conn, config, ids, _ = lib
    results = query.semantic_search("anything", conn, config, top_k=3)
    assert _names(results, ids) == ["best", "good", "ok"]
    scores = [r["similarity_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("kwargs, expected", [
    ({"since": "2024-07-01"}, ["good", "ok"]),
    # A bare --until date includes that whole day
    ({"until": "2024-07-01"}, ["best", "good", "poor"]),
    ({"since": "2024-06-01", "until": "2024-07-31"}, ["best", "good"]),
    ({"person_name": "Alice"}, ["best", "ok"]),
    ({"person_name": "Nobody"}, []),
    ({"location_label": "Beach"}, ["best"]),
    ({"person_name": "Alice", "since": "2024-07-01"}, ["ok"]),
])
def test_filters(lib, kwargs, expected):
    conn, config, ids, _ = lib
    assert _names(query.semantic_search("anything", conn, config, **kwargs), ids) == expected


def test_person_id_filter_and_deleted_photos_excluded(lib):
    conn, config, ids, alice = lib
    db.mark_deleted(conn, ids["best"], "quality")
    results = query.semantic_search("anything", conn, config, person_id=alice)
    assert _names(results, ids) == ["ok"]


def test_apply_sql_filters_keeps_input_order(lib):
    conn, _, ids, _ = lib
    order = [ids["poor"], ids["ok"], ids["best"]]
    assert apply_sql_filters(conn, order) == order
    assert apply_sql_filters(conn, []) == []
