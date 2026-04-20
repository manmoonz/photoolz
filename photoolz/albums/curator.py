from __future__ import annotations

import os

import anthropic

ALBUM_TOOL = {
    "name": "create_album",
    "description": "Create a curated photo album from the provided candidate list.",
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
                "description": "Photo IDs selected from candidates",
            },
            "rationale": {
                "type": "string",
                "description": "Brief explanation of selection strategy",
            },
        },
        "required": ["title", "description", "selected_photo_ids", "rationale"],
    },
}

SYSTEM_PROMPT = """You are a photo curator. Given a user's album request and a list of candidate \
photos with metadata, select the best photos for the album.

Selection priorities (in order):
1. Temporal and geographic coherence with the query (date range, location)
2. Photo quality (prefer higher quality_score)
3. Visual diversity (avoid near-identical shots; prefer varied moments)
4. Narrative arc (chronological progression, variety of subjects)
5. Face diversity (include different people if faces are present)

Each candidate row: ID | Date | Location | Similarity | Quality
Higher similarity = more relevant to the query.
Higher quality = sharper, better exposed.
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


def propose_album(
    query: str,
    candidates: list[dict],
    target_count: int = 40,
    model: str = "claude-opus-4-5",
) -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set.")

    client = anthropic.Anthropic(api_key=api_key)
    user_message = _build_prompt(query, candidates, target_count)

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        tools=[ALBUM_TOOL],
        tool_choice={"type": "tool", "name": "create_album"},
        messages=[{"role": "user", "content": user_message}],
    )

    tool_use = next(
        (block for block in response.content if block.type == "tool_use"),
        None,
    )
    if tool_use is None:
        raise RuntimeError("Claude did not return a tool_use block.")

    inp = tool_use.input
    return {
        "title": inp.get("title", "Untitled Album"),
        "description": inp.get("description", ""),
        "photo_ids": inp.get("selected_photo_ids", []),
        "rationale": inp.get("rationale", ""),
    }
