from __future__ import annotations

from pathlib import Path

from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def open_image(path: Path) -> Image.Image | None:
    try:
        img = Image.open(path)
        img.verify()
        img = Image.open(path)
        return img
    except Exception:
        return None


def open_image_rgb(path: Path) -> Image.Image | None:
    img = open_image(path)
    if img is None:
        return None
    try:
        return img.convert("RGB")
    except Exception:
        return None


def resize_for_clip(img: Image.Image, size: int = 224) -> Image.Image:
    return img.resize((size, size), Image.LANCZOS)


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS
