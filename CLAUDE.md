# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (order matters — dlib must come first)
python3 -m venv ~/.venvs/photoolz
source ~/.venvs/photoolz/bin/activate
pip install dlib
pip install -e .
pip install face_recognition  # optional, enables face clustering

# Run the CLI
photoolz --help

# No test suite or lint tooling exists yet
```

## Architecture

Photoolz is a local photo library CLI. It indexes photos once (storing metadata, CLIP embeddings, and face encodings in SQLite + a FAISS flat index), then supports semantic search, quality analysis, duplicate detection, face/geo/burst clustering, and Claude-powered album curation — all offline except `album propose`.

**Data flow: index → query/cluster/curate**

1. `photoolz index <path>` runs a multi-stage pipeline (`indexer/pipeline.py`):
   - SHA256 dedup → EXIF extraction → CLIP embedding (batched) → face detection → quality scoring → FAISS index rebuild
2. All subsequent commands read from SQLite (`~/.photoolz/photoolz.db`) and the FAISS index (`~/.photoolz/faiss.index`).

**Module responsibilities:**

| Module | Role |
|---|---|
| `config.py` | Loads `.env`; exposes `PHOTOOLZ_DATA_DIR`, `CLIP_MODEL`, `DEVICE` |
| `db.py` | Thread-safe SQLite; all schema migrations via `photoolz db init` |
| `indexer/` | `scanner.py` finds files → `metadata.py` reads EXIF → `embedder.py` runs CLIP → `faces.py` runs face_recognition → `pipeline.py` orchestrates |
| `search/` | `faiss_index.py` manages FAISS flat index; `query.py` embeds a text query and retrieves top-k, then `filters.py` applies SQL-based date/person/location filters |
| `quality/` | `blur.py` (Laplacian variance) + `exposure.py` (histogram) → `scorer.py` aggregates; `duplicates.py` does two-stage pHash + CLIP matching |
| `clustering/` | `faces_cluster.py` (DBSCAN on 128-dim face encodings), `geo_cluster.py` (DBSCAN on haversine), `burst.py` (timestamp window grouping) |
| `albums/` | `candidates.py` retrieves candidates via semantic search; `curator.py` calls Claude with tool_use to select the final photo set |
| `utils/` | `console.py` (Rich output, CSV/JSON), `hash.py` (SHA256 + pHash), `viewer.py` (auto-detects feh/eog/xdg-open) |

**SQLite schema key points:**
- `photos` is the central table: holds file path, hashes, EXIF, CLIP embedding blob, quality scores, cluster IDs, and soft-delete flag.
- `faces` → `people` is a one-to-many: each detected face links to a `people` row after clustering.
- `albums` / `album_photos` store Claude-curated or manually created collections.
- `indexed_dirs` tracks which directories have been indexed.

**Search over-retrieves** 10× top-k from FAISS before applying SQL filters so that date/person/location narrowing doesn't starve results.

**CLIP model** is loaded once per process in `embedder.py` and cached on the module. Default: `ViT-B-32` via `open_clip`.

**`album propose`** is the only command requiring `ANTHROPIC_API_KEY`. It uses `tool_use` so Claude can return a structured photo selection with ranked ordering.

## Configuration

Copy `.env.example` to `.env`. Key variables:

```
PHOTOOLZ_DATA_DIR=~/.photoolz/   # SQLite DB, FAISS index, thumbnails
PHOTOOLZ_CLIP_MODEL=ViT-B-32     # Any open_clip model name
PHOTOOLZ_DEVICE=auto             # cpu | cuda | mps
ANTHROPIC_API_KEY=...            # Required for album propose only
```
