from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import exifread
from PIL import Image


_DATE_FORMATS = [
    "%Y:%m:%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y%m%d_%H%M%S",
]

_FILENAME_PATTERNS = [
    re.compile(r"(\d{4})[-_](\d{2})[-_](\d{2})[T _-](\d{2})[-:](\d{2})[-:](\d{2})"),
    re.compile(r"(\d{4})[-_](\d{2})[-_](\d{2})"),
    re.compile(r"IMG_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})"),
]


def parse_exif_datetime(raw: str) -> datetime | None:
    if not raw:
        return None
    raw = raw.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _parse_dms(dms_values, ref: str) -> float | None:
    try:
        d = float(dms_values[0].num) / float(dms_values[0].den)
        m = float(dms_values[1].num) / float(dms_values[1].den)
        s = float(dms_values[2].num) / float(dms_values[2].den)
        val = d + m / 60 + s / 3600
        if ref in ("S", "W"):
            val = -val
        return val
    except Exception:
        return None


def parse_gps(tags: dict) -> tuple[float | None, float | None, float | None]:
    try:
        lat_vals = tags.get("GPS GPSLatitude")
        lat_ref = tags.get("GPS GPSLatitudeRef")
        lon_vals = tags.get("GPS GPSLongitude")
        lon_ref = tags.get("GPS GPSLongitudeRef")
        alt_val = tags.get("GPS GPSAltitude")

        lat = None
        if lat_vals and lat_ref:
            lat = _parse_dms(lat_vals.values, str(lat_ref))

        lon = None
        if lon_vals and lon_ref:
            lon = _parse_dms(lon_vals.values, str(lon_ref))

        alt = None
        if alt_val:
            try:
                alt = float(alt_val.values[0].num) / float(alt_val.values[0].den)
            except Exception:
                pass

        return lat, lon, alt
    except Exception:
        return None, None, None


def _parse_filename_date(path: Path) -> datetime | None:
    stem = path.stem
    for pattern in _FILENAME_PATTERNS:
        m = pattern.search(stem)
        if m:
            groups = m.groups()
            try:
                if len(groups) >= 6:
                    return datetime(
                        int(groups[0]), int(groups[1]), int(groups[2]),
                        int(groups[3]), int(groups[4]), int(groups[5]),
                        tzinfo=timezone.utc,
                    )
                elif len(groups) >= 3:
                    return datetime(
                        int(groups[0]), int(groups[1]), int(groups[2]),
                        tzinfo=timezone.utc,
                    )
            except ValueError:
                continue
    return None


def extract_metadata(path: Path) -> dict:
    result: dict = {
        "file_path": str(path),
        "file_size_bytes": path.stat().st_size,
        "width": None,
        "height": None,
        "format": None,
        "taken_at": None,
        "taken_at_source": None,
        "gps_lat": None,
        "gps_lon": None,
        "gps_alt": None,
        "camera_make": None,
        "camera_model": None,
    }

    try:
        with Image.open(path) as img:
            result["width"] = img.width
            result["height"] = img.height
            result["format"] = img.format or path.suffix.lstrip(".").upper()
    except Exception:
        result["format"] = path.suffix.lstrip(".").upper()

    try:
        with open(path, "rb") as f:
            tags = exifread.process_file(f, stop_tag="GPS GPSAltitude", details=False)

        for tag_key in ("EXIF DateTimeOriginal", "EXIF DateTimeDigitized", "Image DateTime"):
            val = tags.get(tag_key)
            if val:
                dt = parse_exif_datetime(str(val))
                if dt:
                    result["taken_at"] = dt.isoformat()
                    result["taken_at_source"] = "exif"
                    break

        lat, lon, alt = parse_gps(tags)
        result["gps_lat"] = lat
        result["gps_lon"] = lon
        result["gps_alt"] = alt

        make = tags.get("Image Make")
        model = tags.get("Image Model")
        if make:
            result["camera_make"] = str(make).strip()
        if model:
            result["camera_model"] = str(model).strip()

    except Exception:
        pass

    if result["taken_at"] is None:
        dt = _parse_filename_date(path)
        if dt:
            result["taken_at"] = dt.isoformat()
            result["taken_at_source"] = "filename"

    if result["taken_at"] is None:
        try:
            mtime = path.stat().st_mtime
            dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
            result["taken_at"] = dt.isoformat()
            result["taken_at_source"] = "filesystem"
        except Exception:
            pass

    return result
