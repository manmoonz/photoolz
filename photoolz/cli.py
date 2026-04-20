from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from photoolz.utils.console import console


def _get_conn_and_config(library_path: str | None = None):
    from photoolz.config import load_config
    from photoolz.db import get_connection, init_schema

    config = load_config(library_path)
    config.ensure_data_dir()
    conn = get_connection(config.db_path)
    init_schema(conn)
    return conn, config


@click.group()
@click.version_option("0.1.0")
def cli():
    """photoolz — modular photo library management tools."""


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("library_path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--workers", default=4, show_default=True, help="Parallel worker threads.")
@click.option("--force", is_flag=True, help="Re-index even unchanged files.")
@click.option("--skip-faces", is_flag=True, help="Skip face detection.")
@click.option("--clip-batch-size", default=64, show_default=True, help="Images per CLIP batch.")
def index(library_path: Path, workers: int, force: bool,
          skip_faces: bool, clip_batch_size: int):
    """Scan and index a photo library."""
    from photoolz.indexer.pipeline import run_index_pipeline
    from photoolz.config import load_config

    config = load_config(library_path)
    config.ensure_data_dir()

    summary = run_index_pipeline(
        library_path, config,
        workers=workers, force=force,
        skip_faces=skip_faces, clip_batch_size=clip_batch_size,
    )
    console.print("\n[bold green]Indexing complete.[/bold green]")
    for key, val in summary.items():
        console.print(f"  {key.replace('_', ' ').title()}: [cyan]{val}[/cyan]")


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("query")
@click.option("--top-k", default=20, show_default=True)
@click.option("--since", default=None, help="Start date filter YYYY-MM-DD.")
@click.option("--until", default=None, help="End date filter YYYY-MM-DD.")
@click.option("--person", default=None, help="Filter by person name.")
@click.option("--location", default=None, help="Filter by geo cluster label.")
@click.option("--output", default="table",
              type=click.Choice(["table", "paths", "json"]), show_default=True)
@click.option("--library", default=None, help="Library path (if not already indexed from here).")
def search(query: str, top_k: int, since: str | None, until: str | None,
           person: str | None, location: str | None, output: str, library: str | None):
    """Search photos by content and/or date."""
    from photoolz.search.query import semantic_search
    from photoolz.utils.console import print_photo_table

    conn, config = _get_conn_and_config(library)
    results = semantic_search(query, conn, config, top_k=top_k,
                               since=since, until=until,
                               person_name=person, location_label=location)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    if output == "paths":
        for r in results:
            click.echo(r["file_path"])
    elif output == "json":
        click.echo(json.dumps(results, default=str, indent=2))
    else:
        print_photo_table(
            results,
            columns=["id", "file_path", "taken_at", "similarity_score", "quality_score"],
        )


# ---------------------------------------------------------------------------
# quality
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--blur-threshold", default=100.0, show_default=True,
              help="Laplacian variance threshold for sharpness.")
@click.option("--worst", default=50, show_default=True, help="Show N worst photos.")
@click.option("--output", default="table",
              type=click.Choice(["table", "paths", "json"]), show_default=True)
@click.option("--mark-deleted", is_flag=True,
              help="Flag worst photos as deleted in the index (does NOT delete files).")
@click.option("--library", default=None)
def quality(blur_threshold: float, worst: int, output: str,
            mark_deleted: bool, library: str | None):
    """Find low-quality photos (blur, exposure)."""
    from photoolz.db import get_worst_quality_photos, mark_deleted as db_mark_deleted
    from photoolz.utils.console import print_photo_table

    conn, config = _get_conn_and_config(library)
    rows = get_worst_quality_photos(conn, limit=worst)
    photos = [dict(r) for r in rows]

    if not photos:
        console.print("[green]No quality issues found.[/green]")
        return

    if output == "paths":
        for p in photos:
            click.echo(p["file_path"])
    elif output == "json":
        click.echo(json.dumps(photos, default=str, indent=2))
    else:
        print_photo_table(
            photos,
            columns=["id", "file_path", "taken_at", "blur_score", "exposure_score", "quality_score"],
        )

    if mark_deleted:
        for p in photos:
            db_mark_deleted(conn, p["id"], "quality")
        console.print(f"[yellow]Flagged {len(photos)} photos as deleted (quality).[/yellow]")


# ---------------------------------------------------------------------------
# duplicates
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--similarity", default=0.97, show_default=True,
              help="FAISS cosine similarity threshold (0–1).")
@click.option("--hamming-dist", default=8, show_default=True,
              help="pHash Hamming distance threshold (0–64).")
@click.option("--output", default="table",
              type=click.Choice(["table", "paths", "json"]), show_default=True)
@click.option("--mark-deleted", is_flag=True,
              help="Flag the lower-quality duplicate in each group as deleted.")
@click.option("--library", default=None)
def duplicates(similarity: float, hamming_dist: int, output: str,
               mark_deleted: bool, library: str | None):
    """Find near-duplicate photos."""
    from photoolz.quality.duplicates import find_near_duplicates
    from photoolz.db import mark_deleted as db_mark_deleted
    from photoolz.utils.console import print_duplicate_groups

    conn, config = _get_conn_and_config(library)
    groups = find_near_duplicates(conn, config,
                                   similarity_threshold=similarity,
                                   hamming_threshold=hamming_dist)

    if output == "paths":
        for group in groups:
            for p in group:
                click.echo(p["file_path"])
            click.echo("")
    elif output == "json":
        click.echo(json.dumps(groups, default=str, indent=2))
    else:
        print_duplicate_groups(groups)

    if mark_deleted and groups:
        flagged = 0
        for group in groups:
            for p in group[1:]:  # keep the best (first), flag the rest
                db_mark_deleted(conn, p["id"], "duplicate")
                flagged += 1
        console.print(f"[yellow]Flagged {flagged} duplicates as deleted.[/yellow]")


# ---------------------------------------------------------------------------
# album
# ---------------------------------------------------------------------------

@cli.group()
def album():
    """Photo album commands."""


@album.command("propose")
@click.argument("query")
@click.option("--count", default=40, show_default=True, help="Target number of photos.")
@click.option("--candidates", default=120, show_default=True,
              help="Number of CLIP candidates to send to Claude.")
@click.option("--model", default="claude-opus-4-5", show_default=True)
@click.option("--since", default=None, help="Start date filter YYYY-MM-DD.")
@click.option("--until", default=None, help="End date filter YYYY-MM-DD.")
@click.option("--location", default=None, help="Filter by geo cluster label.")
@click.option("--save", is_flag=True, help="Save album to the index database.")
@click.option("--output", default="rich",
              type=click.Choice(["rich", "json", "paths"]), show_default=True)
@click.option("--library", default=None)
def album_propose(query: str, count: int, candidates: int, model: str,
                  since: str | None, until: str | None, location: str | None,
                  save: bool, output: str, library: str | None):
    """Propose a photo album using Claude AI."""
    from photoolz.albums.candidates import retrieve_candidates
    from photoolz.albums.curator import propose_album
    from photoolz.db import save_album, get_photos_by_ids
    from photoolz.utils.console import print_album, print_photo_table

    conn, config = _get_conn_and_config(library)

    console.print(f"[bold]Retrieving {candidates} candidates...[/bold]")
    cands = retrieve_candidates(query, conn, config, n_candidates=candidates,
                                 since=since, until=until, location_label=location)
    if not cands:
        console.print("[red]No candidate photos found. Make sure the library is indexed.[/red]")
        return

    console.print(f"[bold]Asking Claude to curate {count} photos...[/bold]")
    try:
        result = propose_album(query, cands, target_count=count, model=model)
    except Exception as e:
        console.print(f"[red]Album proposal failed: {e}[/red]")
        raise SystemExit(1)

    photos = get_photos_by_ids(conn, result["photo_ids"])
    photo_dicts = [dict(p) for p in photos]

    if output == "paths":
        for p in photo_dicts:
            click.echo(p["file_path"])
    elif output == "json":
        click.echo(json.dumps({**result, "photos": photo_dicts}, default=str, indent=2))
    else:
        print_album(result)
        print_photo_table(
            photo_dicts,
            columns=["id", "file_path", "taken_at", "quality_score"],
        )

    if save:
        album_id = save_album(conn, result["title"], result["description"],
                               query, result["photo_ids"])
        console.print(f"\n[green]Album saved with ID {album_id}.[/green]")


@album.command("list")
@click.option("--limit", default=20, show_default=True)
@click.option("--library", default=None)
def album_list(limit: int, library: str | None):
    """List saved albums."""
    from photoolz.db import list_albums

    conn, _ = _get_conn_and_config(library)
    albums = list_albums(conn, limit)
    if not albums:
        console.print("[yellow]No albums saved yet.[/yellow]")
        return

    from rich.table import Table
    table = Table(title="Saved Albums", show_header=True, header_style="bold cyan")
    for col in ["ID", "Title", "Photos", "Query", "Created"]:
        table.add_column(col)
    for a in albums:
        table.add_row(
            str(a["id"]), a["title"] or "", str(a["photo_count"]),
            (a["query"] or "")[:40], (a["created_at"] or "")[:10],
        )
    console.print(table)


@album.command("show")
@click.argument("album_id", type=int)
@click.option("--output", default="table",
              type=click.Choice(["table", "paths", "json"]), show_default=True)
@click.option("--library", default=None)
def album_show(album_id: int, output: str, library: str | None):
    """Show photos in a saved album."""
    from photoolz.db import get_album, get_album_photos
    from photoolz.utils.console import print_photo_table

    conn, _ = _get_conn_and_config(library)
    album_row = get_album(conn, album_id)
    if not album_row:
        console.print(f"[red]Album {album_id} not found.[/red]")
        return

    photos = get_album_photos(conn, album_id)
    photo_dicts = [dict(p) for p in photos]

    console.print(f"\n[bold]{album_row['title']}[/bold]")
    if album_row["description"]:
        console.print(f"[italic]{album_row['description']}[/italic]\n")

    if output == "paths":
        for p in photo_dicts:
            click.echo(p["file_path"])
    elif output == "json":
        click.echo(json.dumps(photo_dicts, default=str, indent=2))
    else:
        print_photo_table(photo_dicts, columns=["id", "file_path", "taken_at", "quality_score"])


# ---------------------------------------------------------------------------
# cluster
# ---------------------------------------------------------------------------

@cli.group()
def cluster():
    """Clustering commands (faces, geo, bursts)."""


@cluster.command("faces")
@click.option("--eps", default=0.5, show_default=True,
              help="DBSCAN epsilon for face encoding distance.")
@click.option("--min-samples", default=3, show_default=True)
@click.option("--library", default=None)
def cluster_faces(eps: float, min_samples: int, library: str | None):
    """Cluster faces into people groups."""
    from photoolz.clustering.faces_cluster import cluster_faces as _cluster_faces

    conn, _ = _get_conn_and_config(library)
    _cluster_faces(conn, eps=eps, min_samples=min_samples)


@cluster.command("geo")
@click.option("--eps-km", default=1.0, show_default=True,
              help="DBSCAN epsilon in kilometres.")
@click.option("--min-samples", default=5, show_default=True)
@click.option("--library", default=None)
def cluster_geo(eps_km: float, min_samples: int, library: str | None):
    """Cluster photos by GPS location."""
    from photoolz.clustering.geo_cluster import cluster_geolocations

    conn, _ = _get_conn_and_config(library)
    cluster_geolocations(conn, eps_km=eps_km, min_samples=min_samples)


@cluster.command("bursts")
@click.option("--gap-seconds", default=3, show_default=True,
              help="Max seconds between burst frames.")
@click.option("--library", default=None)
def cluster_bursts(gap_seconds: int, library: str | None):
    """Detect burst photo sequences."""
    from photoolz.clustering.burst import detect_bursts, get_burst_best_frames
    from photoolz.utils.console import print_photo_table

    conn, _ = _get_conn_and_config(library)
    summary = detect_bursts(conn, gap_seconds=gap_seconds)

    if summary["burst_groups_found"] > 0:
        console.print("\n[bold]Best frames per burst:[/bold]")
        best_frames = get_burst_best_frames(conn)
        print_photo_table(
            best_frames,
            columns=["id", "file_path", "taken_at", "quality_score", "burst_size"],
        )


# ---------------------------------------------------------------------------
# people
# ---------------------------------------------------------------------------

@cli.group()
def people():
    """People (face cluster) commands."""


@people.command("label")
@click.argument("person_id", type=int)
@click.argument("name")
@click.option("--library", default=None)
def people_label(person_id: int, name: str, library: str | None):
    """Assign a name to a face cluster."""
    from photoolz.db import label_person

    conn, _ = _get_conn_and_config(library)
    label_person(conn, person_id, name)
    console.print(f"[green]Person {person_id} labeled as '{name}'.[/green]")


@people.command("list")
@click.option("--library", default=None)
def people_list(library: str | None):
    """List all identified people."""
    from photoolz.db import get_people
    from rich.table import Table

    conn, _ = _get_conn_and_config(library)
    rows = get_people(conn)
    if not rows:
        console.print("[yellow]No people clusters found. Run 'photoolz cluster faces' first.[/yellow]")
        return

    table = Table(title="People", show_header=True, header_style="bold cyan")
    for col in ["ID", "Name", "Faces", "Representative Photo"]:
        table.add_column(col)
    for r in rows:
        table.add_row(
            str(r["id"]),
            r["name"] or "[dim]unlabeled[/dim]",
            str(r["face_count"]),
            (r["rep_path"] or "")[:60],
        )
    console.print(table)


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--library", default=None)
def stats(library: str | None):
    """Show library statistics."""
    from photoolz.db import get_stats
    from photoolz.utils.console import print_stats

    conn, _ = _get_conn_and_config(library)
    print_stats(get_stats(conn))


# ---------------------------------------------------------------------------
# db
# ---------------------------------------------------------------------------

@cli.group()
def db():
    """Database management commands."""


@db.command("init")
@click.option("--library", default=None)
def db_init(library: str | None):
    """Initialize or migrate the SQLite schema."""
    conn, _ = _get_conn_and_config(library)
    console.print("[green]Database schema initialized.[/green]")


@db.command("reindex")
@click.option("--table", default="faiss",
              type=click.Choice(["photos", "faces", "faiss"]), show_default=True,
              help="Which component to rebuild.")
@click.option("--library", default=None)
def db_reindex(table: str, library: str | None):
    """Rebuild a specific index component."""
    conn, config = _get_conn_and_config(library)

    if table == "faiss":
        from photoolz.search.faiss_index import build_or_update_faiss_index
        console.print("[bold]Rebuilding FAISS index...[/bold]")
        build_or_update_faiss_index(conn, config)
        console.print("[green]FAISS index rebuilt.[/green]")

    elif table == "photos":
        from photoolz.quality.scorer import score_all_unscored
        console.print("[bold]Re-scoring all photos...[/bold]")
        count = score_all_unscored(conn, config)
        console.print(f"[green]Scored {count} photos.[/green]")

    elif table == "faces":
        console.print("[yellow]Face re-detection requires re-running 'photoolz index'.[/yellow]")
