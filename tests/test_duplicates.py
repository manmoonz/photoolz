from __future__ import annotations

import math

import numpy as np

from photoolz import db
from photoolz.config import Config
from photoolz.indexer.embedder import serialize_embedding
from photoolz.quality.duplicates import find_near_duplicates
from photoolz.search.faiss_index import build_or_update_faiss_index


def _unit(angle_deg: float) -> np.ndarray:
    v = np.zeros(512, dtype=np.float32)
    v[0], v[1] = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    return v


def _setup(tmp_path):
    config = Config(library_path=tmp_path, data_dir=tmp_path, clip_model="ViT-B-32")
    conn = db.get_connection(config.db_path)
    db.init_schema(conn)

    def add(name, angle, quality, phash=None):
        pid = db.upsert_photo(conn, {"file_path": f"/p/{name}.jpg", "file_hash": name, "phash": phash,
                                     "taken_at": "2024-01-01T10:00:00", "indexed_at": "x"})
        db.update_clip_embedding(conn, pid, serialize_embedding(_unit(angle)), "ViT-B-32")
        db.bulk_update_quality_scores(conn, [(quality, quality, quality, pid)])
        return pid

    return config, conn, add


def _id_groups(groups):
    return sorted(sorted(p["id"] for p in g) for g in groups)


def test_chained_near_duplicates_form_one_group(tmp_path):
    config, conn, add = _setup(tmp_path)
    # cos(11°) ≈ 0.98 passes 0.97, cos(22°) ≈ 0.93 doesn't: a~b and b~c, but not a~c
    a, b, c = add("a", 0, 0.5), add("b", 11, 0.9), add("c", 22, 0.4)
    add("far", 90, 0.5)
    build_or_update_faiss_index(conn, config)

    groups = find_near_duplicates(conn, config, similarity_threshold=0.97)
    assert _id_groups(groups) == [sorted([a, b, c])]
    assert groups[0][0]["id"] == b  # best quality first


def test_phash_and_clip_matches_merge(tmp_path):
    config, conn, add = _setup(tmp_path)
    a = add("a", 0, 0.5, phash="ffffffffffffffff")
    b = add("b", 60, 0.5, phash="fffffffffffffffe")  # pHash match only
    c = add("c", 5, 0.5)                              # CLIP match with a only
    build_or_update_faiss_index(conn, config)

    assert _id_groups(find_near_duplicates(conn, config)) == [sorted([a, b, c])]


def test_flagged_photos_still_in_faiss_are_excluded(tmp_path):
    config, conn, add = _setup(tmp_path)
    a, b = add("a", 0, 0.5), add("b", 1, 0.5)
    flagged = add("flagged", 0.5, 0.5)
    build_or_update_faiss_index(conn, config)
    db.mark_deleted(conn, flagged, "duplicate")  # FAISS index isn't rebuilt by marking

    assert _id_groups(find_near_duplicates(conn, config)) == [sorted([a, b])]
