from __future__ import annotations

from pathlib import Path


def detect_and_encode_faces(path: Path, model: str = "hog") -> list[dict]:
    try:
        import face_recognition
        import numpy as np
        from PIL import Image

        img = Image.open(path).convert("RGB")
        img_array = np.array(img)

        locations = face_recognition.face_locations(img_array, model=model)
        if not locations:
            return []

        encodings = face_recognition.face_encodings(img_array, locations)

        results = []
        for (top, right, bottom, left), encoding in zip(locations, encodings):
            results.append({
                "bbox_top": top,
                "bbox_right": right,
                "bbox_bottom": bottom,
                "bbox_left": left,
                "encoding": encoding.astype(np.float64).tobytes(),
            })
        return results
    except ImportError:
        return []
    except Exception:
        return []
