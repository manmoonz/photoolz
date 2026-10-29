# Outstanding work

Known issues from the 2026-10-28 audit, in the order to tackle them. Tick an item (`[x]`) in the
same commit that fixes it, and add anything new you find. Each fix should come with tests, and
`pytest` must pass. CLIP and face detection are faked in tests, so they run without torch or dlib.

Already fixed (for context): stale id from `upsert_photo` on re-index, duplicate face rows on
`--force`, relative library paths, the `face_recognition` `quit()` crash, and `db relocate`.

**Environment note:** the venv needs `setuptools<81`. `face_recognition_models` imports
`pkg_resources`, which setuptools 81 removed. `photoolz index` now warns about this instead of
crashing.

---

## 1. High priority

- [x] **1.1 Album curation uses an outdated model and trusts Claude's output** — `albums/curator.py`, `cli.py:album_propose`
  - The default model `claude-opus-4-5` is hard-coded in `curator.py`, the `--model` default in `cli.py`, and a README example. Use a current model, and make it configurable with `PHOTOOLZ_ALBUM_MODEL` in `.env`.
  - The IDs in `selected_photo_ids` aren't checked against the candidates. Drop unknown IDs and duplicates, and warn when any are dropped. Otherwise a made-up ID becomes a foreign-key error on `--save`.
  - Results are displayed in ID order, because `get_photos_by_ids` uses `IN (...)`. Show them in Claude's ranked order. The saved album already uses rank order.
  - *Done when:* there are tests for ID validation and order, using a faked Anthropic client.

- [x] **1.2 Packaging: make face recognition optional and document the setuptools pin** — `pyproject.toml`, `requirements.txt`, README, CLAUDE.md
  - Move `face_recognition` out of the required dependencies into a `faces` extra, along with `setuptools<81`. The README already says it's optional.
  - Delete `requirements.txt`, which duplicates `pyproject.toml`. Drop `piexif` and `tqdm`, which are never imported.
  - Update the README and CLAUDE.md install steps (`pip install dlib && pip install -e '.[faces,dev]'`).
  - Correct CLAUDE.md:
    - the data files are `index.db`, `clip.faiss` and `clip_ids.npy`
    - `PHOTOOLZ_DEVICE=auto` is invalid (leave it unset to auto-detect)
    - `mps` is never auto-detected
    - there are no schema migrations

## 2. Features that look like they work but don't

- [x] **2.1 Location clusters can't be named, so `--location` never matches** — `clustering/geo_cluster.py`, `db.py`, `cli.py`
  - Add `photoolz places list` (ID, label, photo count, center, date range) and `photoolz places label <id> <name>`, mirroring `people`. Reverse geocoding is optional later; it would need a network call or an offline dataset.
  - *Done when:* `search --location <name>` works after `places label`, and there's a test for it.

- [x] **2.2 Quality re-scoring is a no-op** — `quality/scorer.py`, `db.py`, `cli.py:quality`, `cli.py:db_reindex`, `indexer/pipeline.py`
  - `quality --blur-threshold` is accepted but ignored, because `quality` only reads stored scores. Either drop the option, or move it to `db reindex --table photos` and re-score everything.
  - `db reindex --table photos` says "re-score all" but only scores photos with no score. Make it re-score all photos.
  - When a photo is edited and re-indexed, its old quality score is kept. Clear the scores (and `clip_embedding`, for safety) in the upsert for re-indexed photos so they're recomputed.

## 3. Correctness on re-runs

- [x] **3.1 `cluster geo` piles up clusters** — `clustering/geo_cluster.py`
  - Each run adds new `geo_clusters` rows without removing the old ones, so `stats` overcounts. Clear old clusters first, and keep labels from 2.1 where a new cluster lines up with a labeled old one (e.g. nearest center within the clustering radius).
- [x] **3.2 `cluster bursts` keeps old burst IDs** — `clustering/burst.py`
  - Reset `burst_group_id` to NULL before assigning new groups.
- [x] **3.3 `LIKE` wildcards in path matching** — `db.py:get_photo_count_under_path`, `db.py:delete_photos_under_path`
  - `_` and `%` in folder names act as wildcards. Switch to the `substr()` prefix match that `get_photo_paths_under` / `relocate_paths` already use.
- [x] **3.4 A changed CLIP model isn't detected** — `indexer/pipeline.py`, `search/faiss_index.py`
  - `photos.clip_model` is stored but never checked. When `PHOTOOLZ_CLIP_MODEL` differs from the stored model, refuse to search or rebuild with a clear "re-index with --force" message. Today it fails when building the index if the embedding sizes differ, or quietly gives bad results if they match.
- [x] **3.5 People face counts drift** — `db.py`, `clustering/faces_cluster.py`
  - `people.face_count` is a stored count that goes stale when faces are replaced on re-index. Recompute it from `faces` after clustering and re-indexing, or derive it in `get_people`.

## 4. Smaller fixes

- [x] **4.1 Wrong `index` summary.** It always reports `updated: 0` and counts changed files as new. Count them separately (`indexer/pipeline.py`).
- [x] **4.2 Duplicate detection.** Merge overlapping groups (union-find) instead of only dropping exact repeats, and leave out photos flagged as deleted that the search index still contains (`quality/duplicates.py`).
- [x] **4.3 Face clustering lookup.** Replace the O(n²) `face_ids.index()` with a dict, and pick a sharper face as each person's representative photo (`clustering/faces_cluster.py`).
- [ ] **4.4 EXIF time zones.** *Do after 5.2:* fixing this means rewriting existing `taken_at` values, and a mix of time-zone-aware and naive timestamps would crash burst detection (`dt - prev_dt`), so it needs a migration. EXIF times are local but stored as UTC. Store them without a time zone, or use `OffsetTimeOriginal` when present (`indexer/metadata.py`). Check that date filters still behave.
- [x] **4.5 Unused code.** Remove most of `utils/image_io.py` (keep `SUPPORTED_EXTENSIONS`), plus `update_quality_scores`, `get_photo_by_path` and `get_photos_missing_embedding` in `db.py`, and the unused `lru_cache` import in `embedder.py`.
- [x] **4.6 Status messages on stdout.** Warnings and "no results" messages went through the stdout `console`, so they ended up in `--output paths/json/csv` pipes. They now go to `err_console` (stderr), and `album show --output json` no longer prints the title before the JSON.

## 5. Workflow gaps (features)

- [ ] **5.1 Acting on flagged photos.** Add `photoolz flagged list` / `flagged restore <id...>` / `flagged move <dest>` (move the files to a trash folder, then drop them from the index). Never delete files outright.
- [ ] **5.2 Schema migrations.** Add a `schema_version` table (or `PRAGMA user_version`) and ordered migrations in `db.init_schema`, before any change that adds a column. The first migration can drop the unused `people.face_count` column (counts are now derived from `faces`).
- [ ] **5.3 Test coverage.** Add tests for search filters (`search/filters.py`), burst detection, geo clustering and duplicate grouping.
