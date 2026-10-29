from __future__ import annotations

from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from photoolz import db
from photoolz.albums import candidates as candidates_mod
from photoolz.albums import curator
from photoolz.cli import cli

CANDIDATES = [{"id": i, "taken_at": None, "similarity_score": 0.3, "quality_score": 0.5}
              for i in (10, 11, 12, 13)]


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.requests: list[dict] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return self.response


def _tool_response(selected, stop_reason="tool_use"):
    block = SimpleNamespace(type="tool_use", name="create_album", input={
        "title": "T", "description": "D", "selected_photo_ids": selected, "rationale": "R",
    })
    return SimpleNamespace(stop_reason=stop_reason, stop_details=None, content=[block])


def test_request_uses_strict_tool_without_forced_choice():
    client = FakeClient(_tool_response([10]))
    curator.propose_album("q", CANDIDATES, client=client)

    (req,) = client.requests
    assert req["model"] == "claude-opus-5-5"
    assert "tool_choice" not in req  # forced tool_choice is a 400 on current models
    assert req["tools"][0]["strict"] is True
    assert req["tools"][0]["input_schema"]["additionalProperties"] is False
    assert req["fallbacks"] == "default"
    assert req["betas"] == ["server-side-fallback-2026-07-01"]


def test_no_fallbacks_for_models_without_server_side_fallback():
    client = FakeClient(_tool_response([10]))
    curator.propose_album("q", CANDIDATES, model="claude-haiku-5-5", client=client)
    assert "fallbacks" not in client.requests[0] and "betas" not in client.requests[0]


def test_selection_keeps_order_and_drops_unknown_and_repeated_ids():
    client = FakeClient(_tool_response([12, 999, 10, 12, 13]))
    result = curator.propose_album("q", CANDIDATES, client=client)
    assert result["photo_ids"] == [12, 10, 13]
    assert result["dropped_ids"] == [999, 12]


def test_refusal_raises():
    response = SimpleNamespace(stop_reason="refusal", content=[],
                               stop_details=SimpleNamespace(category="cyber"))
    with pytest.raises(RuntimeError, match="declined.*cyber"):
        curator.propose_album("q", CANDIDATES, client=FakeClient(response))


def test_missing_tool_call_raises():
    response = SimpleNamespace(stop_reason="end_turn", stop_details=None,
                               content=[SimpleNamespace(type="text", text="hi")])
    with pytest.raises(RuntimeError, match="did not call create_album"):
        curator.propose_album("q", CANDIDATES, client=FakeClient(response))


def test_cli_shows_photos_in_ranked_order_with_configured_model(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("PHOTOOLZ_DATA_DIR", str(data_dir))
    monkeypatch.setenv("PHOTOOLZ_ALBUM_MODEL", "claude-sonnet-5-5")

    conn = db.get_connection(data_dir / "index.db")
    db.init_schema(conn)
    ids = [db.upsert_photo(conn, {"file_path": f"/p/{n}.jpg", "file_hash": "h",
                                  "indexed_at": "2026-01-01T00:00:00+00:00"}) for n in "abc"]

    monkeypatch.setattr(candidates_mod, "retrieve_candidates",
                        lambda *a, **k: [{"id": i} for i in ids])
    seen = {}

    def fake_propose(query, cands, target_count, model):
        seen["model"] = model
        return {"title": "T", "description": "D", "rationale": "R",
                "photo_ids": [ids[2], ids[0], ids[1]], "dropped_ids": []}

    monkeypatch.setattr(curator, "propose_album", fake_propose)

    result = CliRunner().invoke(cli, ["album", "propose", "q", "--output", "paths"])
    assert result.exit_code == 0, result.output
    assert result.stdout.split() == ["/p/c.jpg", "/p/a.jpg", "/p/b.jpg"]
    assert seen["model"] == "claude-sonnet-5-5"
