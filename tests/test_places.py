from __future__ import annotations

import pytest
from click.testing import CliRunner

from photoolz import db
from photoolz.cli import cli
from photoolz.clustering.geo_cluster import cluster_geolocations
from photoolz.search.filters import apply_sql_filters


@pytest.fixture
def env(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("PHOTOOLZ_DATA_DIR", str(data_dir))
    conn = db.get_connection(data_dir / "index.db")
    db.init_schema(conn)

    def add(name, lat, lon, taken):
        return db.upsert_photo(conn, {
            "file_path": f"/p/{name}.jpg", "file_hash": name, "gps_lat": lat, "gps_lon": lon,
            "taken_at": taken, "indexed_at": "2026-01-01T00:00:00+00:00",
        })

    # Five photos within a few metres in Paris, five in Tokyo
    paris = [add(f"paris{i}", 48.8584 + i * 1e-5, 2.2945, f"2024-06-0{i + 1}T12:00:00") for i in range(5)]
    tokyo = [add(f"tokyo{i}", 35.6586 + i * 1e-5, 139.7454, "2025-03-01T09:00:00") for i in range(5)]
    cluster_geolocations(conn, eps_km=1.0, min_samples=5)
    return conn, paris, tokyo


def _cluster_of(conn, photo_id):
    return conn.execute("SELECT geo_cluster_id FROM photos WHERE id=?", (photo_id,)).fetchone()[0]


def test_label_place_makes_location_filter_match(env):
    conn, paris, tokyo = env
    paris_cluster = _cluster_of(conn, paris[0])

    runner = CliRunner()
    result = runner.invoke(cli, ["places", "label", str(paris_cluster), "Paris"])
    assert result.exit_code == 0, result.output

    assert apply_sql_filters(conn, paris + tokyo, location_label="Paris") == paris
    assert apply_sql_filters(conn, paris + tokyo, location_label="Tokyo") == []


def test_places_list_shows_counts_and_dates(env):
    conn, paris, _ = env
    db.label_geo_cluster(conn, _cluster_of(conn, paris[0]), "Paris")

    result = CliRunner().invoke(cli, ["places", "list"])
    assert result.exit_code == 0, result.output
    assert "Paris" in result.output and "unlabeled" in result.output
    assert "2024-06-01 – 2024-06-05" in result.output

    (paris_row,) = [r for r in db.get_geo_clusters(conn) if r["label"] == "Paris"]
    assert paris_row["photo_count"] == 5


def test_label_unknown_place_fails(env):
    result = CliRunner().invoke(cli, ["places", "label", "999", "Nowhere"])
    assert result.exit_code == 1
    assert "No place with ID 999" in result.output
