from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

from photoolz.utils.hash import sha256_file
from photoolz.utils.image_io import SUPPORTED_EXTENSIONS


def scan_library(library_path: Path) -> Generator[Path, None, None]:
    for root, _dirs, files in library_path.walk() if hasattr(library_path, "walk") else _os_walk(library_path):
        for fname in files:
            p = Path(root) / fname
            if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield p


def _os_walk(path: Path):
    import os
    for root, dirs, files in os.walk(path):
        yield root, dirs, files


def compute_file_hash(path: Path) -> str:
    return sha256_file(path)
