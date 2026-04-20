from __future__ import annotations

from pathlib import Path


def laplacian_variance(path: Path) -> float | None:
    try:
        import cv2
        import numpy as np
        from PIL import Image

        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.uint8)
        return float(cv2.Laplacian(arr, cv2.CV_64F).var())
    except Exception:
        return None


def blur_score_from_variance(variance: float, threshold: float = 100.0) -> float:
    return min(variance / threshold, 1.0)
