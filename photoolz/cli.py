from __future__ import annotations

import json
from pathlib import Path

import click

from photoolz.utils.console import console, err_console, print_csv


def _get_conn_and_config():
    from photoolz.config import load_config
    from photoolz.db import get_connection, init_schema

    config = load_config()
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
@click.option("--person-id", "person_id", default=None, type=int, help="Filter by person ID.")
@click.option("--location", default=None, help="Filter by geo cluster label.")
@click.option("--output", default="table",
              type=click.Choice(["table", "paths", "json", "csv"]), show_default=True)
@click.option("--open", "open_viewer", is_flag=True, help="Open results in an image viewer.")
@click.option("--viewer", default=None, help="Viewer command (default: auto-detect or PHOTOOLZ_VIEWER).")
def search(query: str, top_k: int, since: str | None, until: str | None,
           person: str | None, person_id: int | None, location: str | None, output: str,
           open_viewer: bool, viewer: str | None):
    """Search photos by content and/or date."""
    from photoolz.search.query import semantic_search
    from photoolz.utils.console import print_photo_table

    _COLS = ["id", "file_path", "taken_at", "similarity_score", "quality_score"]
    conn, config = _get_conn_and_config()
    results = semantic_search(query, conn, config, top_k=top_k,
                               since=since, until=until,
                               person_name=person, person_id=person_id,
                               location_label=location)

    if not results:
        err_console.print("[yellow]No results found.[/yellow]")
        return

    if output == "paths":
        for r in results:
            click.echo(r["file_path"])
    elif output == "json":
        click.echo(json.dumps(results, default=str, indent=2))
    elif output == "csv":
        print_csv(results, _COLS)
    else:
        print_photo_table(results, columns=_COLS)

    if open_viewer:
        from photoolz.utils.viewer import open_in_viewer
        open_in_viewer([r["file_path"] for r in results], viewer)


# ---------------------------------------------------------------------------
# quality
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--worst", default=50, show_default=True, help="Show N worst photos.")
@click.option("--output", default="table",
              type=click.Choice(["table", "paths", "json", "csv"]), show_default=True)
@click.option("--mark-deleted", is_flag=True,
              help="Flag worst photos as deleted in the index (does NOT delete files).")
@click.option("--open", "open_viewer", is_flag=True, help="Open results in an image viewer.")
@click.option("--viewer", default=None, help="Viewer command (default: auto-detect or PHOTOOLZ_VIEWER).")
def quality(worst: int, output: str,
            mark_deleted: bool, open_viewer: bool, viewer: str | None):
    """Find low-quality photos (blur, exposure).

    Uses the scores stored at index time. To re-score with a different blur
    threshold, run 'photoolz db reindex --table photos --blur-threshold N'.
    """
    from photoolz.db import get_worst_quality_photos, mark_deleted as db_mark_deleted
    from photoolz.utils.console import print_photo_table

    _COLS = ["id", "file_path", "taken_at", "blur_score", "exposure_score", "quality_score"]
    conn, config = _get_conn_and_config()
    rows = get_worst_quality_photos(conn, limit=worst)
    photos = [dict(r) for r in rows]

    if not photos:
        err_console.print("[green]No quality issues found.[/green]")
        return

    if output == "paths":
        for p in photos:
            click.echo(p["file_path"])
    elif output == "json":
        click.echo(json.dumps(photos, default=str, indent=2))
    elif output == "csv":
        print_csv(photos, _COLS)
    else:
        print_photo_table(photos, columns=_COLS)

    if open_viewer:
        from photoolz.utils.viewer import open_in_viewer
        open_in_viewer([p["file_path"] for p in photos], viewer)

    if mark_deleted:
        for p in photos:
            db_mark_deleted(conn, p["id"], "quality")
        err_console.print(f"[yellow]Flagged {len(photos)} photos as deleted (quality).[/yellow]")


# ---------------------------------------------------------------------------
# duplicates
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--similarity", default=0.97, show_default=True,
              help="FAISS cosine similarity threshold (0–1).")
@click.option("--hamming-dist", default=8, show_default=True,
              help="pHash Hamming distance threshold (0–64).")
@click.option("--output", default="table",
              type=click.Choice(["table", "paths", "json", "csv"]), show_default=True)
@click.option("--min-group-size", default=2, show_default=True,
              help="Only show groups with at least this many duplicates.")
@click.option("--mark-deleted", is_flag=True,
              help="Flag the lower-quality duplicate in each group as deleted.")
@click.option("--open", "open_viewer", is_flag=True, help="Open results in an image viewer.")
@click.option("--viewer", default=None, help="Viewer command (default: auto-detect or PHOTOOLZ_VIEWER).")
def duplicates(similarity: float, hamming_dist: int, output: str, min_group_size: int,
               mark_deleted: bool, open_viewer: bool, viewer: str | None):
    """Find near-duplicate photos."""
    from photoolz.quality.duplicates import find_near_duplicates
    from photoolz.db import mark_deleted as db_mark_deleted
    from photoolz.utils.console import print_duplicate_groups

    _COLS = ["group", "id", "file_path", "taken_at", "quality_score"]
    conn, config = _get_conn_and_config()
    groups = find_near_duplicates(conn, config,
                                   similarity_threshold=similarity,
                                   hamming_threshold=hamming_dist)
    groups = [g for g in groups if len(g) >= min_group_size]

    if output == "paths":
        for group in groups:
            for p in group:
                click.echo(p["file_path"])
            click.echo("")
    elif output == "json":
        click.echo(json.dumps(groups, default=str, indent=2))
    elif output == "csv":
        flat = [{**p, "group": i} for i, group in enumerate(groups, 1) for p in group]
        print_csv(flat, _COLS)
    else:
        print_duplicate_groups(groups)

    if open_viewer and groups:
        from photoolz.utils.viewer import open_in_viewer
        open_in_viewer([p["file_path"] for g in groups for p in g], viewer)

    if mark_deleted and groups:
        flagged = 0
        for group in groups:
            for p in group[1:]:  # keep the best (first), flag the rest
                db_mark_deleted(conn, p["id"], "duplicate")
                flagged += 1
        err_console.print(f"[yellow]Flagged {flagged} duplicates as deleted.[/yellow]")


# ---------------------------------------------------------------------------
# flagged
# ---------------------------------------------------------------------------

_REASONS = click.Choice(["quality", "duplicate"])


@cli.group()
def flagged():
    """Photos flagged by --mark-deleted: review, restore, or move them out."""


@flagged.command("list")
@click.option("--reason", type=_REASONS, default=None, help="Only photos flagged for this reason.")
@click.option("--output", default="table",
              type=click.Choice(["table", "paths", "json", "csv"]), show_default=True)
@click.option("--open", "open_viewer", is_flag=True, help="Open results in an image viewer.")
@click.option("--viewer", default=None, help="Viewer command (default: auto-detect or PHOTOOLZ_VIEWER).")
def flagged_list(reason: str | None, output: str, open_viewer: bool, viewer: str | None):
    """List photos flagged as deleted."""
    from photoolz.db import get_flagged_photos
    from photoolz.utils.console import print_photo_table

    _COLS = ["id", "file_path", "deletion_reason", "taken_at", "quality_score"]
    conn, _ = _get_conn_and_config()
    photos = [dict(r) for r in get_flagged_photos(conn, reason)]
    if not photos:
        err_console.print("[green]No flagged photos.[/green]")
        return

    if output == "paths":
        for p in photos:
            click.echo(p["file_path"])
    elif output == "json":
        click.echo(json.dumps(photos, default=str, indent=2))
    elif output == "csv":
        print_csv(photos, _COLS)
    else:
        print_photo_table(photos, columns=_COLS)

    if open_viewer:
        from photoolz.utils.viewer import open_in_viewer
        open_in_viewer([p["file_path"] for p in photos], viewer)


@flagged.command("restore")
@click.argument("photo_ids", nargs=-1, type=int)
@click.option("--all", "restore_all", is_flag=True, help="Restore every flagged photo.")
@click.option("--reason", type=_REASONS, default=None, help="With --all: only this reason.")
def flagged_restore(photo_ids: tuple[int, ...], restore_all: bool, reason: str | None):
    """Un-flag photos so they show up in search and other commands again."""
    from photoolz.db import get_flagged_photos, restore_flagged

    if bool(photo_ids) == restore_all:
        console.print("[red]Give photo IDs or --all (not both).[/red]")
        raise SystemExit(1)

    conn, _ = _get_conn_and_config()
    ids = list(photo_ids) or [r["id"] for r in get_flagged_photos(conn, reason)]
    restored = restore_flagged(conn, ids)
    console.print(f"[green]Restored {restored} photo(s).[/green]")
    if photo_ids and restored < len(set(photo_ids)):
        console.print(f"[yellow]{len(set(photo_ids)) - restored} ID(s) weren't flagged.[/yellow]")


@flagged.command("move")
@click.argument("dest", type=click.Path(file_okay=False, path_type=Path))
@click.option("--reason", type=_REASONS, default=None, help="Only photos flagged for this reason.")
@click.option("--force", is_flag=True, help="Skip confirmation prompt.")
def flagged_move(dest: Path, reason: str | None, force: bool):
    """Move flagged photos' files into DEST and remove them from the index.

    Each file keeps its full original path under DEST (e.g.
    DEST/home/me/Pictures/IMG_1.jpg), so nothing collides and it's clear where
    files came from. Files are moved, never deleted.
    """
    import shutil
    from photoolz.db import delete_photos_by_ids, get_flagged_photos, list_indexed_dirs
    from photoolz.search.faiss_index import build_or_update_faiss_index

    dest_root = dest.resolve()
    conn, config = _get_conn_and_config()
    for d in list_indexed_dirs(conn):
        indexed = Path(d["path"])
        if dest_root == indexed or indexed in dest_root.parents:
            console.print(f"[red]{dest_root} is inside indexed directory {indexed}; "
                          f"re-indexing would pick the files up again. Choose another DEST.[/red]")
            raise SystemExit(1)

    photos = get_flagged_photos(conn, reason)
    if not photos:
        console.print("[green]No flagged photos.[/green]")
        return
    if not force:
        click.confirm(f"Move {len(photos)} flagged photo(s) into {dest_root} "
                      f"and remove them from the index?", abort=True)

    moved: list[int] = []
    skipped: list[str] = []
    for p in photos:
        src = Path(p["file_path"])
        target = dest_root / src.relative_to(src.anchor)
        if not src.exists():
            skipped.append(f"{src} (missing)")
        elif target.exists():
            skipped.append(f"{src} (already exists at {target})")
        else:
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(target))
                moved.append(p["id"])
            except OSError as e:
                skipped.append(f"{src} ({e})")

    delete_photos_by_ids(conn, moved)
    if moved:
        build_or_update_faiss_index(conn, config)
    console.print(f"[green]Moved {len(moved)} photo(s) to {dest_root}.[/green]")
    if skipped:
        console.print(f"[yellow]Skipped {len(skipped)}; they stay flagged in the index:[/yellow]")
        for line in skipped[:20]:
            console.print(f"  {line}")
        if len(skipped) > 20:
            console.print(f"  ... and {len(skipped) - 20} more")


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
@click.option("--model", default=None,
              help="Claude model (default: PHOTOOLZ_ALBUM_MODEL, or claude-opus-5-5).")
@click.option("--since", default=None, help="Start date filter YYYY-MM-DD.")
@click.option("--until", default=None, help="End date filter YYYY-MM-DD.")
@click.option("--location", default=None, help="Filter by geo cluster label.")
@click.option("--save", is_flag=True, help="Save album to the index database.")
@click.option("--output", default="table",
              type=click.Choice(["table", "paths", "json", "csv"]), show_default=True)
@click.option("--open", "open_viewer", is_flag=True, help="Open results in an image viewer.")
@click.option("--viewer", default=None, help="Viewer command (default: auto-detect or PHOTOOLZ_VIEWER).")
def album_propose(query: str, count: int, candidates: int, model: str | None,
                  since: str | None, until: str | None, location: str | None,
                  save: bool, output: str, open_viewer: bool, viewer: str | None):
    """Propose a photo album using Claude AI."""
    from photoolz.albums.candidates import retrieve_candidates
    from photoolz.albums.curator import propose_album
    from photoolz.db import save_album, get_photos_by_ids
    from photoolz.utils.console import print_album, print_photo_table

    _COLS = ["id", "file_path", "taken_at", "quality_score"]
    conn, config = _get_conn_and_config()

    err_console.print(f"[bold]Retrieving {candidates} candidates...[/bold]")
    cands = retrieve_candidates(query, conn, config, n_candidates=candidates,
                                 since=since, until=until, location_label=location)
    if not cands:
        err_console.print("[red]No candidate photos found. Make sure the library is indexed.[/red]")
        return

    err_console.print(f"[bold]Asking Claude to curate {count} photos...[/bold]")
    try:
        result = propose_album(query, cands, target_count=count, model=model or config.album_model)
    except Exception as e:
        err_console.print(f"[red]Album proposal failed: {e}[/red]")
        raise SystemExit(1)

    if result["dropped_ids"]:
        err_console.print(
            f"[yellow]Ignored {len(result['dropped_ids'])} selected ID(s) that were not "
            f"candidates or were repeated: {result['dropped_ids']}[/yellow]"
        )

    # get_photos_by_ids doesn't preserve order; show photos in Claude's ranked order
    by_id = {p["id"]: dict(p) for p in get_photos_by_ids(conn, result["photo_ids"])}
    photo_dicts = [by_id[pid] for pid in result["photo_ids"] if pid in by_id]

    if output == "paths":
        for p in photo_dicts:
            click.echo(p["file_path"])
    elif output == "json":
        click.echo(json.dumps({**result, "photos": photo_dicts}, default=str, indent=2))
    elif output == "csv":
        print_csv(photo_dicts, _COLS)
    else:
        print_album(result)
        print_photo_table(photo_dicts, columns=_COLS)

    if open_viewer:
        from photoolz.utils.viewer import open_in_viewer
        open_in_viewer([p["file_path"] for p in photo_dicts], viewer)

    if save:
        album_id = save_album(conn, result["title"], result["description"],
                               query, result["photo_ids"])
        err_console.print(f"\n[green]Album saved with ID {album_id}.[/green]")


@album.command("list")
@click.option("--limit", default=20, show_default=True)
def album_list(limit: int):
    """List saved albums."""
    from photoolz.db import list_albums

    conn, _ = _get_conn_and_config()
    albums = list_albums(conn, limit)
    if not albums:
        console.print("[yellow]No albums saved yet.[/yellow]")
        return

    from rich.table import Table
    table = Table(title="Saved Albums", show_header=True, header_style="bold cyan")
    table.add_column("ID", min_width=4, no_wrap=True)
    table.add_column("Title")
    table.add_column("Photos", min_width=6, no_wrap=True)
    table.add_column("Query", overflow="fold")
    table.add_column("Created", min_width=10, no_wrap=True)
    for a in albums:
        table.add_row(
            str(a["id"]), a["title"] or "", str(a["photo_count"]),
            (a["query"] or "")[:40], (a["created_at"] or "")[:10],
        )
    console.print(table)


@album.command("show")
@click.argument("album_id", type=int)
@click.option("--output", default="table",
              type=click.Choice(["table", "paths", "json", "csv"]), show_default=True)
@click.option("--open", "open_viewer", is_flag=True, help="Open results in an image viewer.")
@click.option("--viewer", default=None, help="Viewer command (default: auto-detect or PHOTOOLZ_VIEWER).")
def album_show(album_id: int, output: str, open_viewer: bool, viewer: str | None):
    """Show photos in a saved album."""
    from photoolz.db import get_album, get_album_photos
    from photoolz.utils.console import print_photo_table

    _COLS = ["id", "file_path", "taken_at", "quality_score"]
    conn, _ = _get_conn_and_config()
    album_row = get_album(conn, album_id)
    if not album_row:
        err_console.print(f"[red]Album {album_id} not found.[/red]")
        return

    photos = get_album_photos(conn, album_id)
    photo_dicts = [dict(p) for p in photos]

    if output == "table":
        console.print(f"\n[bold]{album_row['title']}[/bold]")
        if album_row["description"]:
            console.print(f"[italic]{album_row['description']}[/italic]\n")

    if output == "paths":
        for p in photo_dicts:
            click.echo(p["file_path"])
    elif output == "json":
        click.echo(json.dumps(photo_dicts, default=str, indent=2))
    elif output == "csv":
        print_csv(photo_dicts, _COLS)
    else:
        print_photo_table(photo_dicts, columns=_COLS)

    if open_viewer:
        from photoolz.utils.viewer import open_in_viewer
        open_in_viewer([p["file_path"] for p in photo_dicts], viewer)


# ---------------------------------------------------------------------------
# cluster
# ---------------------------------------------------------------------------

@cli.group()
def cluster():
    """Clustering commands (faces, geo, bursts)."""


@cluster.command("faces")
@click.option("--eps", default=0.3, show_default=True,
              help="DBSCAN epsilon for face encoding distance.")
@click.option("--min-samples", default=3, show_default=True)
@click.option("--reset", is_flag=True, help="Clear existing face clusters before re-clustering.")
def cluster_faces(eps: float, min_samples: int, reset: bool):
    """Cluster faces into people groups."""
    from photoolz.clustering.faces_cluster import cluster_faces as _cluster_faces

    conn, _ = _get_conn_and_config()
    if reset:
        from photoolz.db import reset_face_clusters
        reset_face_clusters(conn)
        console.print("[dim]Cleared existing face clusters.[/dim]")
    _cluster_faces(conn, eps=eps, min_samples=min_samples)


@cluster.command("geo")
@click.option("--eps-km", default=1.0, show_default=True,
              help="DBSCAN epsilon in kilometres.")
@click.option("--min-samples", default=5, show_default=True)
def cluster_geo(eps_km: float, min_samples: int):
    """Cluster photos by GPS location."""
    from photoolz.clustering.geo_cluster import cluster_geolocations

    conn, _ = _get_conn_and_config()
    cluster_geolocations(conn, eps_km=eps_km, min_samples=min_samples)


@cluster.command("bursts")
@click.option("--gap-seconds", default=3, show_default=True,
              help="Max seconds between burst frames.")
@click.option("--open", "open_viewer", is_flag=True, help="Open best frames in an image viewer.")
@click.option("--viewer", default=None, help="Viewer command (default: auto-detect or PHOTOOLZ_VIEWER).")
def cluster_bursts(gap_seconds: int, open_viewer: bool, viewer: str | None):
    """Detect burst photo sequences."""
    from photoolz.clustering.burst import detect_bursts, get_burst_best_frames
    from photoolz.utils.console import print_photo_table

    conn, _ = _get_conn_and_config()
    summary = detect_bursts(conn, gap_seconds=gap_seconds)

    if summary["burst_groups_found"] > 0:
        console.print("\n[bold]Best frames per burst:[/bold]")
        best_frames = get_burst_best_frames(conn)
        print_photo_table(
            best_frames,
            columns=["id", "file_path", "taken_at", "quality_score", "burst_size"],
        )
        if open_viewer:
            from photoolz.utils.viewer import open_in_viewer
            open_in_viewer([f["file_path"] for f in best_frames], viewer)


# ---------------------------------------------------------------------------
# people
# ---------------------------------------------------------------------------

@cli.group()
def people():
    """People (face cluster) commands."""


@people.command("label")
@click.argument("person_id", type=int)
@click.argument("name")
def people_label(person_id: int, name: str):
    """Assign a name to a face cluster."""
    from photoolz.db import label_person

    conn, _ = _get_conn_and_config()
    label_person(conn, person_id, name)
    console.print(f"[green]Person {person_id} labeled as '{name}'.[/green]")


@people.command("merge")
@click.argument("person_ids", nargs=-1, type=int, required=True)
def people_merge(person_ids: tuple[int, ...]):
    """Merge multiple face clusters into one person.

    Example: photoolz people merge 2 7 12
    """
    from photoolz.db import merge_people

    if len(person_ids) < 2:
        console.print("[red]Provide at least two person IDs to merge.[/red]")
        raise SystemExit(1)

    conn, _ = _get_conn_and_config()
    try:
        survivor_id = merge_people(conn, list(person_ids))
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise SystemExit(1)

    console.print(
        f"[green]Merged {len(person_ids)} clusters into person ID {survivor_id}.[/green]"
    )


@people.command("list")
def people_list():
    """List all identified people."""
    from photoolz.db import get_people
    from rich.table import Table

    conn, _ = _get_conn_and_config()
    rows = get_people(conn)
    if not rows:
        console.print("[yellow]No people clusters found. Run 'photoolz cluster faces' first.[/yellow]")
        return

    table = Table(title="People", show_header=True, header_style="bold cyan")
    table.add_column("ID", min_width=4, no_wrap=True)
    table.add_column("Name")
    table.add_column("Faces", min_width=5, no_wrap=True)
    table.add_column("Representative Photo", overflow="fold")
    for r in rows:
        table.add_row(
            str(r["id"]),
            r["name"] or "[dim]unlabeled[/dim]",
            str(r["face_count"]),
            r["rep_path"] or "",
        )
    console.print(table)


@people.command("photos")
@click.argument("person_id", type=int)
@click.option("--output", default="table",
              type=click.Choice(["table", "paths", "json", "csv"]), show_default=True)
@click.option("--open", "open_viewer", is_flag=True, help="Open results in an image viewer.")
@click.option("--viewer", default=None, help="Viewer command (default: auto-detect or PHOTOOLZ_VIEWER).")
def people_photos(person_id: int, output: str, open_viewer: bool, viewer: str | None):
    """List photos containing a person by their ID."""
    from photoolz.db import get_photos_by_person_id
    from photoolz.utils.console import print_photo_table

    _COLS = ["id", "file_path", "taken_at", "quality_score"]
    conn, _ = _get_conn_and_config()
    rows = get_photos_by_person_id(conn, person_id)
    if not rows:
        err_console.print(f"[yellow]No photos found for person ID {person_id}.[/yellow]")
        return

    results = [dict(r) for r in rows]
    if output == "paths":
        for r in results:
            click.echo(r["file_path"])
    elif output == "json":
        click.echo(json.dumps(results, default=str, indent=2))
    elif output == "csv":
        print_csv(results, _COLS)
    else:
        print_photo_table(results, columns=_COLS)

    if open_viewer:
        from photoolz.utils.viewer import open_in_viewer
        open_in_viewer([r["file_path"] for r in results], viewer)


# ---------------------------------------------------------------------------
# places
# ---------------------------------------------------------------------------

@cli.group()
def places():
    """Places (geo cluster) commands."""


@places.command("list")
def places_list():
    """List location clusters, so they can be named with 'places label'."""
    from photoolz.db import get_geo_clusters
    from rich.table import Table

    conn, _ = _get_conn_and_config()
    rows = get_geo_clusters(conn)
    if not rows:
        console.print("[yellow]No places found. Run 'photoolz cluster geo' first.[/yellow]")
        return

    table = Table(title="Places", show_header=True, header_style="bold cyan")
    table.add_column("ID", min_width=4, no_wrap=True)
    table.add_column("Name")
    table.add_column("Photos", min_width=6, no_wrap=True)
    table.add_column("Center", no_wrap=True)
    table.add_column("Dates", no_wrap=True)
    for r in rows:
        first, last = (r["first_taken"] or "")[:10], (r["last_taken"] or "")[:10]
        table.add_row(
            str(r["id"]),
            r["label"] or "[dim]unlabeled[/dim]",
            str(r["photo_count"]),
            f"{r['center_lat']:.4f}, {r['center_lon']:.4f}",
            first if first == last else f"{first} – {last}",
        )
    console.print(table)


@places.command("label")
@click.argument("place_id", type=int)
@click.argument("name")
def places_label(place_id: int, name: str):
    """Name a location cluster, for use with 'search --location'."""
    from photoolz.db import label_geo_cluster

    conn, _ = _get_conn_and_config()
    if not label_geo_cluster(conn, place_id, name):
        console.print(f"[red]No place with ID {place_id}. See 'photoolz places list'.[/red]")
        raise SystemExit(1)
    console.print(f"[green]Place {place_id} labeled as '{name}'.[/green]")


# ---------------------------------------------------------------------------
# dirs
# ---------------------------------------------------------------------------

@cli.command()
def dirs():
    """List all previously indexed directories."""
    from photoolz.db import list_indexed_dirs
    from rich.table import Table

    conn, _ = _get_conn_and_config()
    rows = list_indexed_dirs(conn)
    if not rows:
        console.print("[yellow]No directories have been indexed yet. Run 'photoolz index <path>' first.[/yellow]")
        return

    table = Table(title="Indexed Directories", show_header=True, header_style="bold cyan")
    table.add_column("Path", overflow="fold")
    table.add_column("Photos", min_width=6, no_wrap=True)
    table.add_column("First Indexed", min_width=10, no_wrap=True)
    table.add_column("Last Indexed", min_width=10, no_wrap=True)
    for r in rows:
        table.add_row(
            r["path"],
            str(r["photo_count"]),
            (r["first_indexed_at"] or "")[:10],
            (r["last_indexed_at"] or "")[:10],
        )
    console.print(table)


# ---------------------------------------------------------------------------
# unindex
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("path", type=click.Path(path_type=Path))
@click.option("--force", is_flag=True, help="Skip confirmation prompt.")
def unindex(path: Path, force: bool):
    """Remove all photos under PATH from the index."""
    from photoolz.db import (
        get_photo_count_under_path,
        delete_photos_under_path,
        fix_people_representatives,
        remove_indexed_dir,
    )
    from photoolz.search.faiss_index import build_or_update_faiss_index

    abs_path = str(path.resolve())
    conn, config = _get_conn_and_config()

    count = get_photo_count_under_path(conn, abs_path)
    if count == 0:
        console.print(f"[yellow]No indexed photos found under {abs_path}[/yellow]")
        return

    if not force:
        click.confirm(f"Remove {count} photos under {abs_path} from the index?", abort=True)

    deleted = delete_photos_under_path(conn, abs_path)
    fix_people_representatives(conn)
    remove_indexed_dir(conn, abs_path)

    console.print("[bold]Rebuilding FAISS index...[/bold]")
    build_or_update_faiss_index(conn, config)

    console.print(f"[green]Removed {deleted} photos. FAISS index rebuilt.[/green]")


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

@cli.command()
def stats():
    """Show library statistics."""
    from photoolz.db import get_stats
    from photoolz.utils.console import print_stats

    conn, _ = _get_conn_and_config()
    print_stats(get_stats(conn))


# ---------------------------------------------------------------------------
# db
# ---------------------------------------------------------------------------

@cli.group()
def db():
    """Database management commands."""


@db.command("init")
def db_init():
    """Initialize or migrate the SQLite schema."""
    conn, _ = _get_conn_and_config()
    console.print("[green]Database schema initialized.[/green]")


@db.command("reindex")
@click.option("--table", default="faiss",
              type=click.Choice(["photos", "faces", "faiss"]), show_default=True,
              help="Which component to rebuild.")
@click.option("--blur-threshold", default=100.0, show_default=True,
              help="With --table photos: Laplacian variance at which a photo counts as fully sharp "
                   "(lower = more lenient). Newly indexed photos always use the default.")
def db_reindex(table: str, blur_threshold: float):
    """Rebuild a specific index component."""
    conn, config = _get_conn_and_config()

    if table == "faiss":
        from photoolz.search.faiss_index import build_or_update_faiss_index
        console.print("[bold]Rebuilding FAISS index...[/bold]")
        build_or_update_faiss_index(conn, config)
        console.print("[green]FAISS index rebuilt.[/green]")

    elif table == "photos":
        from photoolz.quality.scorer import rescore_all
        console.print("[bold]Re-scoring all photos...[/bold]")
        count = rescore_all(conn, blur_threshold)
        console.print(f"[green]Scored {count} photos.[/green]")

    elif table == "faces":
        console.print("[yellow]Face re-detection requires re-running 'photoolz index'.[/yellow]")


@db.command("relocate")
@click.argument("old_path")
@click.argument("new_path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--force", is_flag=True, help="Skip confirmation prompt.")
def db_relocate(old_path: str, new_path: Path, force: bool):
    """Point index entries under OLD_PATH at NEW_PATH after moving or remounting photos.

    Example: photoolz db relocate /media/me/DATA /run/media/me/DATA
    """
    import os
    import sqlite3
    from photoolz.db import get_photo_paths_under, relocate_paths

    # OLD_PATH usually no longer exists, so normalize without resolving symlinks
    old_root = os.path.abspath(old_path)
    new_root = str(new_path.resolve())
    conn, _ = _get_conn_and_config()

    paths = get_photo_paths_under(conn, old_root)
    if not paths:
        console.print(f"[yellow]No indexed photos found under {old_root}[/yellow]")
        return

    found = sum(os.path.exists(new_root + p[len(old_root):]) for p in paths)
    color = "green" if found == len(paths) else "yellow"
    console.print(f"[{color}]{found} of {len(paths)} photos exist under {new_root}.[/{color}]")

    if not force:
        click.confirm(f"Rewrite {len(paths)} paths from {old_root} to {new_root}?", abort=True)

    try:
        photos, dirs = relocate_paths(conn, old_root, new_root)
    except sqlite3.IntegrityError:
        console.print(
            f"[red]Some photos are already indexed under {new_root}; nothing was changed.[/red]\n"
            f"[dim]Run 'photoolz unindex {new_root}' first, or use 'photoolz db prune'.[/dim]"
        )
        raise SystemExit(1)

    console.print(f"[green]Relocated {photos} photo(s) and {dirs} indexed director(ies).[/green]")


@db.command("prune")
def db_prune():
    """Remove indexed photos whose files no longer exist on disk."""
    from photoolz.db import prune_missing_photos, fix_people_representatives
    from photoolz.search.faiss_index import build_or_update_faiss_index

    conn, config = _get_conn_and_config()
    removed = prune_missing_photos(conn)
    if removed == 0:
        console.print("[green]No missing files found. Index is clean.[/green]")
        return
    fix_people_representatives(conn)
    console.print(f"[bold]Rebuilding FAISS index...[/bold]")
    build_or_update_faiss_index(conn, config)
    console.print(f"[green]Removed {removed} missing photo(s). FAISS index rebuilt.[/green]")
