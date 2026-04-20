from __future__ import annotations

import hashlib
from pathlib import Path

import imagehash
from PIL import Image


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def phash_image(path: Path) -> str | None:
    try:
        img = Image.open(path)
        return str(imagehash.phash(img))
    except Exception:
        return None


def hamming_distance(hash1: str, hash2: str) -> int:
    try:
        return bin(int(hash1, 16) ^ int(hash2, 16)).count("1")
    except (ValueError, TypeError):
        return 64
