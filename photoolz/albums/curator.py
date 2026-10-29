from __future__ import annotations

import os

import anthropic

from photoolz.config import DEFAULT_ALBUM_MODEL

ALBUM_TOOL = {
    "name": "create_album",
    "description": "Create a curated photo album from the provided candidate list.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Album title, max 60 characters",
            },
            "description": {
                "type": "string",
                "description": "2-3 sentence album description",
            },
            "selected_photo_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Photo IDs selected from candidates, in album order",
            },
            "rationale": {
                "type": "string",
                "description": "Brief explanation of selection strategy",
            },
        },
        "required": ["title", "description", "selected_photo_ids", "rationale"],
        "additionalProperties": False,
    },
}

# Models that accept server-side refusal fallbacks (fallbacks="default")
_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}

SYSTEM_PROMPT = """You are a photo curator. Given a user's album request and a list of candidate \
photos with metadata, select the best photos for the album and submit them by calling the \
create_album tool once.

Selection priorities (in order):
1. Temporal and geographic coherence with the query (date range, location)
2. Photo quality (prefer higher quality_score)
3. Visual diversity (avoid near-identical shots; prefer varied moments)
4. Narrative arc (chronological progression, variety of subjects)
5. Face diversity (include different people if faces are present)

Each candidate row: ID | Date | Location | Similarity | Quality
Higher similarity = more relevant to the query.
Higher quality = sharper, better exposed.
Only select IDs from the candidate list, and list them in the order they should appear.
"""


def _build_prompt(query: str, candidates: list[dict], target_count: int) -> str:
    header = (
        f"Album request: \"{query}\"\n"
        f"Target count: {target_count}\n\n"
        f"Candidates ({len(candidates)} photos):\n"
        f"{'ID':<8} | {'Date':<20} | {'Location':<25} | {'Similarity':>10} | {'Quality':>8}\n"
        f"{'-'*8}-+-{'-'*20}-+-{'-'*25}-+-{'-'*10}-+-{'-'*8}\n"
    )
    rows = []
    for c in candidates:
        taken = (c.get("taken_at") or "")[:19]
        loc = (c.get("geo_cluster_label") or "")[:25]
        sim = f"{c.get('similarity_score', 0.0):.3f}"
        qual = f"{c.get('quality_score', 0.0):.3f}" if c.get("quality_score") is not None else "N/A"
        rows.append(f"{c['id']:<8} | {taken:<20} | {loc:<25} | {sim:>10} | {qual:>8}")
    return header + "\n".join(rows)


def _validate_selection(selected: list, candidates: list[dict]) -> tuple[list[int], list]:
    """Keep candidate IDs in Claude's order, dropping unknown IDs and repeats."""
    candidate_ids = {c["id"] for c in candidates}
    kept: list[int] = []
    dropped: list = []
    for pid in selected:
        if pid in candidate_ids and pid not in kept:
            kept.append(pid)
        else:
            dropped.append(pid)
    return kept, dropped


def propose_album(
    query: str,
    candidates: list[dict],
    target_count: int = 40,
    model: str = DEFAULT_ALBUM_MODEL,
    client: anthropic.Anthropic | None = None,
) -> dict:
    if client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set.")
        client = anthropic.Anthropic(api_key=api_key)

    # Forced tool_choice is rejected by current models, so the system prompt asks for the
    # tool call and strict=True keeps its input schema-valid.
    request: dict = {
        "model": model,
        "max_tokens": 16000,
        "system": SYSTEM_PROMPT,
        "tools": [ALBUM_TOOL],
        "output_config": {"effort": "high"},
        "messages": [{"role": "user", "content": _build_prompt(query, candidates, target_count)}],
    }
    if model in _FALLBACK_MODELS:
        # Re-run on Anthropic's recommended model if a safety classifier declines
        request["betas"] = ["server-side-fallback-2026-07-01"]
        request["fallbacks"] = "default"

    response = client.beta.messages.create(**request)

    if response.stop_reason == "refusal":
        category = response.stop_details.category if response.stop_details else None
        raise RuntimeError(f"Claude declined the request (category: {category}).")

    tool_use = next(
        (block for block in response.content
         if block.type == "tool_use" and block.name == "create_album"),
        None,
    )
    if tool_use is None:
        raise RuntimeError(f"Claude did not call create_album (stop reason: {response.stop_reason}).")

    inp = tool_use.input
    photo_ids, dropped = _validate_selection(inp.get("selected_photo_ids", []), candidates)
    return {
        "title": inp.get("title", "Untitled Album"),
        "description": inp.get("description", ""),
        "photo_ids": photo_ids,
        "dropped_ids": dropped,
        "rationale": inp.get("rationale", ""),
    }
