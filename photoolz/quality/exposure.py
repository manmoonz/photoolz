from __future__ import annotations

from pathlib import Path


def exposure_score(path: Path, low_clip: float = 0.05, high_clip: float = 0.95) -> float | None:
    try:
        from PIL import Image
        import numpy as np

        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.float32)
        total = arr.size
        if total == 0:
            return None
        low_thresh = low_clip * 255.0
        high_thresh = high_clip * 255.0
        good = np.sum((arr >= low_thresh) & (arr <= high_thresh))
        return float(good / total)
    except Exception:
        return None
