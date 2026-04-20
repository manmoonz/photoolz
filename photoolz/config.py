from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _default_data_dir() -> Path:
    override = os.environ.get("PHOTOOLZ_DATA_DIR")
    if override:
        return Path(override)
    return Path.home() / ".photoolz"


@dataclass
class Config:
    library_path: Path
    data_dir: Path = field(default_factory=_default_data_dir)
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    device: str = "cpu"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "index.db"

    @property
    def faiss_index_path(self) -> Path:
        return self.data_dir / "clip.faiss"

    @property
    def faiss_id_map_path(self) -> Path:
        return self.data_dir / "clip_ids.npy"

    def ensure_data_dir(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)


def load_config(library_path: str | Path | None = None) -> Config:
    data_dir = _default_data_dir()

    clip_model = os.environ.get("PHOTOOLZ_CLIP_MODEL", "ViT-B-32")
    device_env = os.environ.get("PHOTOOLZ_DEVICE")

    if device_env:
        device = device_env
    else:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    lib = Path(library_path) if library_path else Path(".")

    return Config(
        library_path=lib,
        data_dir=data_dir,
        clip_model=clip_model,
        device=device,
    )
