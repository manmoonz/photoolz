from __future__ import annotations

from pathlib import Path


def face_recognition_unavailable_reason() -> str | None:
    """Return None if face_recognition is usable, otherwise a hint on how to fix it."""
    import contextlib
    import io

    try:
        # face_recognition prints install instructions to stdout and calls quit()
        # when face_recognition_models fails to import, so silence it and catch SystemExit
        with contextlib.redirect_stdout(io.StringIO()):
            import face_recognition  # noqa: F401
        return None
    except ImportError:
        return "pip install face_recognition"
    except SystemExit:
        pass

    try:
        import face_recognition_models  # noqa: F401
    except ModuleNotFoundError as e:
        if e.name == "pkg_resources":
            # setuptools 81 removed pkg_resources, which face_recognition_models still imports
            return "pip install 'setuptools<81'"
        return "pip install face_recognition_models"
    except Exception as e:
        return f"fix face_recognition_models, which failed to import: {e}"
    return "reinstall face_recognition and face_recognition_models"


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
