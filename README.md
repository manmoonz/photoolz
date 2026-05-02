# photoolz

A modular Python CLI suite for managing a large photo library. It indexes your photos into a local SQLite database with CLIP embeddings, then lets you search by content, find deletion candidates, detect duplicates, cluster by face/location/burst, and generate AI-curated albums.

Supports JPEG and PNG. Designed for libraries of thousands of photos.

---

## How it works

Running `photoolz index` scans your photo directory and builds a local index. Everything it computes is stored in `~/.photoolz/`:

| File | Contents |
|---|---|
| `~/.photoolz/index.db` | SQLite database: photo metadata, quality scores, face encodings, GPS coords, album records |
| `~/.photoolz/clip.faiss` | FAISS vector index for fast semantic similarity search |
| `~/.photoolz/clip_ids.npy` | Mapping from FAISS positions to database photo IDs |

The index is incremental — re-running `photoolz index` only processes files whose content has changed (detected via SHA-256 hash).

You can override the data directory with the `PHOTOOLZ_DATA_DIR` environment variable.

---

## Typical workflow

```bash
# 1. First-time setup: index your library
photoolz index ~/Pictures/

# 2. Check what you have
photoolz stats

# 3. Find and review low-quality photos
photoolz quality --worst 50

# 4. Find near-duplicates, flag the worse copies
photoolz duplicates --mark-deleted

# 5. Cluster faces and name people
photoolz cluster faces
photoolz people list
photoolz people label 1 "Alice"

# 6. Cluster by GPS location
photoolz cluster geo

# 7. Search
photoolz search "beach vacation" --since 2024-06-01 --until 2024-08-31
photoolz search "photos with Alice at the beach" --person Alice

# 8. Generate an album (requires ANTHROPIC_API_KEY)
photoolz album propose "best beach photos summer 2024" --count 30 --save
photoolz album list
photoolz album show 1 --output paths
```

---

## Setup

### Prerequisites

Python 3.11+ is required. Face detection depends on `dlib`, which must be compiled from source on most systems.

**Ubuntu/Debian:**
```bash
sudo apt-get install -y cmake libdlib-dev
```

**macOS (Homebrew):**
```bash
brew install cmake dlib
```

### Install

```bash
# Create a virtual environment
python3 -m venv ~/.venvs/photoolz

# Activate it — you need to do this in every new shell session before using photoolz
source ~/.venvs/photoolz/bin/activate

# Install dlib before anything else — it compiles from source and takes ~5 minutes
pip install dlib

# Install photoolz and all remaining dependencies
# Note: torch alone is ~2 GB, so this download will take a while
pip install -e /path/to/photoolz

# Install face_recognition as a separate step after dlib is confirmed working
pip install face_recognition
```

**Why the order matters:** `face_recognition` depends on `dlib`. If you let `pip` install them together it will try to build `dlib` without the system libraries in place and fail. Installing `dlib` first lets you confirm it compiles cleanly before proceeding.

**face_recognition is optional.** If it is not installed, `photoolz index` will still run but will print a warning and skip face detection. The `photoolz cluster faces` and `photoolz people` commands will have nothing to work with. You can install it later and re-run `photoolz index --force` to backfill face data.

Every time you open a new terminal, activate the venv before running any `photoolz` commands:

```bash
source ~/.venvs/photoolz/bin/activate
photoolz --help
```

### Configure

Copy `.env.example` to `.env` in the project directory and fill in your values:

```bash
cp .env.example .env
```

```ini
# Required only for 'photoolz album propose'
ANTHROPIC_API_KEY=your-api-key-here

# Optional overrides
PHOTOOLZ_DATA_DIR=/path/to/data    # default: ~/.photoolz/
PHOTOOLZ_CLIP_MODEL=ViT-B-32       # default: ViT-B-32
PHOTOOLZ_DEVICE=cpu                 # default: auto-detect (cuda if available)
```

---

## Commands

### `photoolz index` — build the index

Scans a photo directory and computes metadata, CLIP embeddings, quality scores, and face encodings for every new or changed file. Run this first before using any other command.

```bash
# Basic index
photoolz index ~/Pictures/

# Skip face detection (faster, useful for a quick first pass)
photoolz index ~/Pictures/ --skip-faces

# Force re-index everything, even unchanged files
photoolz index ~/Pictures/ --force

# Use more parallel workers for large libraries
photoolz index ~/Pictures/ --workers 8

# Larger CLIP batch size if you have lots of VRAM/RAM
photoolz index ~/Pictures/ --clip-batch-size 128
```

---

### `photoolz search` — find photos by content and date

Uses CLIP to embed your query as a vector and finds the most semantically similar photos. Combine with date and person filters for precise results.

```bash
# Semantic content search
photoolz search "sunset at the beach"

# With a date range
photoolz search "family dinner" --since 2024-12-01 --until 2024-12-31

# Filter by a named person (requires face clustering first)
photoolz search "birthday party" --person Alice

# Filter by a named location (requires geo clustering first)
photoolz search "hiking" --location "Yosemite"

# Return more results
photoolz search "snow" --top-k 50

# Output just the file paths (useful for piping to other tools)
photoolz search "dog playing" --output paths

# Output as JSON
photoolz search "sunset" --output json
```

---

### `photoolz quality` — find low-quality photos

Scores photos by sharpness (Laplacian variance) and exposure (histogram analysis) and surfaces the worst ones. Scores are on a 0–1 scale where 1 is best.

```bash
# Show the 50 worst photos
photoolz quality

# Show more results
photoolz quality --worst 100

# Use a stricter blur threshold (lower = flag more as blurry)
photoolz quality --blur-threshold 50.0

# Output as paths for manual review
photoolz quality --output paths

# Flag the worst photos as deleted in the index (does NOT delete files on disk)
photoolz quality --worst 20 --mark-deleted
```

---

### `photoolz duplicates` — find near-duplicate photos

Uses two methods in combination: perceptual hashing (pHash) for visually identical images and CLIP cosine similarity for semantically near-identical ones. Results are grouped with the highest-quality photo listed first in each group.

```bash
# Find near-duplicates with default thresholds
photoolz duplicates

# Stricter similarity (fewer results, higher confidence)
photoolz duplicates --similarity 0.99

# Looser pHash threshold (flags more visually similar pairs)
photoolz duplicates --hamming-dist 12

# Flag all but the best photo in each group as deleted in the index
photoolz duplicates --mark-deleted

# Output as JSON for scripted processing
photoolz duplicates --output json
```

---

### `photoolz album propose` — AI-curated album  *(requires Anthropic API key)*

Uses CLIP to retrieve the most relevant candidate photos for your query, then sends them to Claude to select a final set that is temporally coherent, visually diverse, and high quality.

```bash
# Propose a 40-photo album (default)
photoolz album propose "our trip to Curacao in December 2025"

# Specify the target count
photoolz album propose "summer vacation 2024" --count 25

# Narrow candidates to a date range before sending to Claude
photoolz album propose "Christmas morning" --since 2024-12-25 --until 2024-12-25 --count 20

# Save the album to the database for later retrieval
photoolz album propose "beach photos" --save

# Output just the selected file paths
photoolz album propose "hiking trip" --output paths

# Use a specific Claude model
photoolz album propose "family reunion" --model claude-opus-4-5
```

### `photoolz album list` — list saved albums

```bash
photoolz album list
photoolz album list --limit 50
```

### `photoolz album show` — show photos in a saved album

```bash
photoolz album show 3
photoolz album show 3 --output paths
```

---

### `photoolz cluster faces` — group photos by person

Runs DBSCAN clustering on the 128-dimensional face encodings stored during indexing. Creates unlabeled people clusters that you can then name with `photoolz people label`.

```bash
# Cluster with default settings
photoolz cluster faces

# Tighter clustering (fewer, more confident groups)
photoolz cluster faces --eps 0.4 --min-samples 5

# Clear existing clusters and re-run from scratch
photoolz cluster faces --reset
photoolz cluster faces --reset --eps 0.4
```

### `photoolz cluster geo` — group photos by location

Clusters photos by GPS coordinates using haversine distance. Photos without GPS data in their EXIF are skipped.

```bash
# Cluster locations within 1 km of each other (default)
photoolz cluster geo

# Larger radius — useful for city-level grouping
photoolz cluster geo --eps-km 5.0
```

### `photoolz cluster bursts` — detect burst sequences

Groups photos taken within a short time window (default: 3 seconds) and identifies the sharpest frame in each burst.

```bash
photoolz cluster bursts

# Use a wider window for cameras with slower burst rates
photoolz cluster bursts --gap-seconds 5
```

---

### `photoolz people` — manage named people

After running `photoolz cluster faces`, assign names to the clusters so they can be used in `photoolz search --person`.

```bash
# List all face clusters (ID, name, face count, representative photo)
photoolz people list

# Assign a name to a cluster
photoolz people label 1 "Alice"
photoolz people label 2 "Bob"

# Now search works with names
photoolz search "birthday" --person Alice
```

The same person may appear as multiple clusters if their appearance changed significantly over time (e.g. photos spanning many years). Use `people merge` to consolidate them:

```bash
# Merge clusters 2, 7, and 12 into one
photoolz people merge 2 7 12

# Then label the merged cluster if needed
photoolz people label 2 "Sophie"
```

The largest cluster survives the merge. If any of the clusters already have a name, that name is preserved on the surviving record.

---

### `photoolz stats` — library overview

```bash
photoolz stats
```

Example output:
```
           Library Statistics
 Metric               Value
 Total Photos         12,847
 Indexed              12,847
 Flagged Deleted         231
 Albums                    4
 People                   12
 Faces                 8,403
 Geo Clusters             17
```

---

### `photoolz db` — database management

```bash
# Initialize or migrate the schema (safe to re-run)
photoolz db init

# Rebuild the FAISS vector index from the database (e.g. after manual DB edits)
photoolz db reindex --table faiss

# Re-score all photos for quality (blur + exposure)
photoolz db reindex --table photos

# Remove index entries for photos that have been deleted from disk
photoolz db prune
```

---

### `photoolz dirs` — list indexed directories

```bash
photoolz dirs
```

Shows all directories that have been indexed, along with the photo count and first/last indexed dates.

---

### `photoolz unindex` — remove a directory from the index

Deletes all photos under the given path from the index and rebuilds the FAISS index. Files on disk are not touched.

```bash
# Remove with confirmation prompt
photoolz unindex ~/Pictures/OldTrip/

# Skip confirmation
photoolz unindex ~/Pictures/OldTrip/ --force
```

---

## Opening results in an image viewer

Any command that outputs a photo list supports `--open` to launch the results directly in an image viewer:

```bash
# Open quality candidates in feh
photoolz quality --open

# Open search results in a specific viewer
photoolz search "beach sunset" --open --viewer feh

# Open duplicate groups for visual comparison
photoolz duplicates --open

# Open an album
photoolz album show 3 --open

# Open burst best frames
photoolz cluster bursts --open
```

Viewer auto-detection tries: `feh`, `eog`, `xviewer`, `shotwell`, `gthumb`, `xdg-open` (Linux), `open` (macOS). You can also set a default in `.env`:

```ini
PHOTOOLZ_VIEWER=feh
```

`feh` is recommended on Linux — install it with `sudo apt-get install feh`. The `--open` flag fires after terminal output, so `--output table --open` gives you the table and the viewer simultaneously.

---

## Output formats

Most commands support `--output` with these choices:

| Value | Description |
|---|---|
| `table` | Rich-formatted table in the terminal (default for most commands) |
| `paths` | One absolute file path per line — pipe to `cp`, `open`, `feh`, etc. |
| `json` | Full record as JSON — useful for scripting or further processing |
| `csv` | Comma-separated with a header row — pipe to `awk`, `cut`, `sort`, etc. |

```bash
# Copy search results to a folder
photoolz search "fireworks" --output paths | xargs -I{} cp {} ~/Desktop/fireworks/

# Open quality candidates in an image viewer
photoolz quality --output paths | head -20 | xargs feh

# Get file paths of the 20 worst photos sorted by quality score
photoolz quality --worst 20 --output csv | tail -n +2 | sort -t, -k6 -n | cut -d, -f2

# Show only the file paths from a search result
photoolz search "beach" --output csv | tail -n +2 | cut -d, -f2

# Show only duplicate group 1
photoolz duplicates --output csv | awk -F, 'NR==1 || $1=="1"'

# Interactive photo selection with fzf
photoolz search "hiking" --output csv | fzf --header-lines=1 | cut -d, -f2
```

The `duplicates` CSV adds a leading `group` column (integer, 1-based) so you can filter or group by duplicate set. All other commands use the same columns shown in `--output table`.

---

## Commands requiring Anthropic API access

Only one command calls the Anthropic API:

| Command | Why |
|---|---|
| `photoolz album propose` | Claude selects the final photo set from CLIP candidates, applying quality, diversity, and narrative judgement |

All other commands run entirely locally with no external API calls.
