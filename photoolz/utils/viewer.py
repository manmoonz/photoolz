from __future__ import annotations

import os
import shutil
import subprocess

_KNOWN_VIEWERS = ["feh", "eog", "xviewer", "shotwell", "gthumb"]


def detect_viewer() -> str | None:
    for v in _KNOWN_VIEWERS:
        if shutil.which(v):
            return v
    if shutil.which("xdg-open"):
        return "xdg-open"
    if shutil.which("open"):
        return "open"
    return None


def open_in_viewer(paths: list[str], viewer: str | None = None) -> None:
    from photoolz.utils.console import console

    if not paths:
        return

    resolved = viewer or os.environ.get("PHOTOOLZ_VIEWER") or detect_viewer()
    if not resolved:
        console.print(
            "[red]No image viewer found. Install feh or set PHOTOOLZ_VIEWER in .env.[/red]"
        )
        return

    if resolved in ("xdg-open", "open"):
        for p in paths:
            subprocess.Popen([resolved, p])
    elif resolved == "feh":
        subprocess.Popen(["feh", "--scale-down"] + paths)
    else:
        subprocess.Popen([resolved] + paths)

    console.print(f"[dim]Opened {len(paths)} photo(s) in {resolved}.[/dim]")
