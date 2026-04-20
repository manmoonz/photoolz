from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from photoolz.config import Config
from photoolz.db import (
    get_connection,
    init_schema,
    upsert_photo,
    update_clip_embedding,
    upsert_face,
)
from photoolz.indexer.scanner import compute_file_hash, needs_reindex, scan_library
from photoolz.indexer.metadata import extract_metadata
from photoolz.indexer.faces import detect_and_encode_faces
from photoolz.utils.console import console, make_progress
from photoolz.utils.hash import phash_image


def run_index_pipeline(
    library_path: Path,
    config: Config,
    workers: int = 4,
    force: bool = False,
    skip_faces: bool = False,
    clip_batch_size: int = 64,
) -> dict:
    config.ensure_data_dir()
    conn = get_connection(config.db_path)
    init_schema(conn)

    console.print(f"[bold]Scanning[/bold] {library_path} ...")
    all_paths = list(scan_library(library_path))
    console.print(f"Found [cyan]{len(all_paths)}[/cyan] image files.")

    # Phase 1: metadata + hash filtering
    to_index: list[tuple[Path, str]] = []
    skipped = 0

    with make_progress("Hashing files", len(all_paths)) as progress:
        task = progress.add_task("Hashing files", total=len(all_paths))
        for path in all_paths:
            try:
                file_hash = compute_file_hash(path)
                if needs_reindex(conn, path, file_hash, force):
                    to_index.append((path, file_hash))
                else:
                    skipped += 1
            except Exception as e:
                console.print(f"[red]Hash error {path}: {e}[/red]")
            progress.advance(task)

    console.print(f"[cyan]{len(to_index)}[/cyan] files need indexing, [dim]{skipped}[/dim] unchanged.")

    if not to_index:
        return {"scanned": len(all_paths), "new": 0, "updated": 0, "skipped": skipped, "errors": 0}

    # Phase 2: metadata extraction (parallel)
    now = datetime.now(timezone.utc).isoformat()
    errors = 0
    photo_id_map: dict[str, int] = {}  # path → db id

    def _process_meta(item: tuple[Path, str]) -> tuple[int | None, str | None]:
        path, file_hash = item
        try:
            meta = extract_metadata(path)
            meta["file_hash"] = file_hash
            meta["phash"] = phash_image(path)
            meta["indexed_at"] = now
            photo_id = upsert_photo(conn, meta)
            return photo_id, str(path)
        except Exception as e:
            console.print(f"[red]Metadata error {path}: {e}[/red]")
            return None, None

    with make_progress("Extracting metadata", len(to_index)) as progress:
        task = progress.add_task("Extracting metadata", total=len(to_index))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process_meta, item): item for item in to_index}
            for future in as_completed(futures):
                photo_id, path_str = future.result()
                if photo_id and path_str:
                    photo_id_map[path_str] = photo_id
                else:
                    errors += 1
                progress.advance(task)

    # Phase 3: CLIP embeddings (batched on main thread for GPU efficiency)
    console.print("[bold]Computing CLIP embeddings...[/bold]")
    from photoolz.indexer.embedder import (
        load_clip_model, embed_images, serialize_embedding
    )
    model, preprocess, tokenizer = load_clip_model(
        config.clip_model, config.clip_pretrained, config.device
    )

    indexed_paths = [Path(p) for p in photo_id_map]
    with make_progress("Embedding images", len(indexed_paths)) as progress:
        task = progress.add_task("Embedding images", total=len(indexed_paths))
        for start in range(0, len(indexed_paths), clip_batch_size):
            batch_paths = indexed_paths[start:start + clip_batch_size]
            embeddings, valid_idx = embed_images(batch_paths, model, preprocess,
                                                  config.device, clip_batch_size)
            for i, emb in zip(valid_idx, embeddings):
                path_str = str(batch_paths[i])
                photo_id = photo_id_map.get(path_str)
                if photo_id:
                    update_clip_embedding(conn, photo_id, serialize_embedding(emb), config.clip_model)
            progress.advance(task, len(batch_paths))

    # Phase 4: quality scores
    console.print("[bold]Computing quality scores...[/bold]")
    from photoolz.quality.scorer import score_all_unscored
    scored = score_all_unscored(conn, config)
    console.print(f"Scored [cyan]{scored}[/cyan] photos.")

    # Phase 5: face detection (optional, parallel)
    if not skip_faces:
        console.print("[bold]Detecting faces...[/bold]")
        face_paths = [(path, photo_id_map[str(path)])
                      for path, _ in to_index if str(path) in photo_id_map]

        def _detect_faces(item: tuple[Path, int]) -> int:
            path, photo_id = item
            faces = detect_and_encode_faces(path)
            for face in faces:
                face["photo_id"] = photo_id
                face["detected_at"] = now
                try:
                    upsert_face(conn, face)
                except Exception:
                    pass
            return len(faces)

        total_faces = 0
        with make_progress("Detecting faces", len(face_paths)) as progress:
            task = progress.add_task("Detecting faces", total=len(face_paths))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_detect_faces, item) for item in face_paths]
                for future in as_completed(futures):
                    total_faces += future.result()
                    progress.advance(task)

        console.print(f"Found [cyan]{total_faces}[/cyan] faces in {len(face_paths)} photos.")

    # Phase 6: rebuild FAISS index
    console.print("[bold]Building FAISS index...[/bold]")
    from photoolz.search.faiss_index import build_or_update_faiss_index
    build_or_update_faiss_index(conn, config)
    console.print("[green]FAISS index updated.[/green]")

    new_count = len(photo_id_map)
    return {
        "scanned": len(all_paths),
        "new": new_count,
        "updated": 0,
        "skipped": skipped,
        "errors": errors,
    }
