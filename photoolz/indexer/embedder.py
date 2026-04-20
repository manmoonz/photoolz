from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

# Lazy imports to avoid loading torch at CLI startup
_clip_cache: dict[tuple[str, str, str], Any] = {}


def _load_clip(model_name: str, pretrained: str, device: str):
    key = (model_name, pretrained, device)
    if key not in _clip_cache:
        import os
        import open_clip

        prev = os.environ.get("HF_HUB_OFFLINE")
        try:
            os.environ["HF_HUB_OFFLINE"] = "1"
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=device
            )
        except Exception:
            # Model not in cache yet — go online for the initial download
            if prev is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = prev
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=device
            )
        else:
            if prev is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = prev

        tokenizer = open_clip.get_tokenizer(model_name)
        model.eval()
        _clip_cache[key] = (model, preprocess, tokenizer)
    return _clip_cache[key]


def load_clip_model(model_name: str = "ViT-B-32", pretrained: str = "openai",
                    device: str = "cpu"):
    return _load_clip(model_name, pretrained, device)


def embed_images(paths: list[Path], model, preprocess, device: str = "cpu",
                 batch_size: int = 64) -> tuple[np.ndarray, list[int]]:
    import torch

    valid_indices: list[int] = []
    tensors = []

    for i, path in enumerate(paths):
        try:
            from PIL import Image
            img = Image.open(path).convert("RGB")
            tensor = preprocess(img)
            tensors.append(tensor)
            valid_indices.append(i)
        except Exception:
            continue

    if not tensors:
        return np.zeros((0, 512), dtype=np.float32), valid_indices

    all_embeddings = []
    with torch.no_grad():
        for start in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[start:start + batch_size]).to(device)
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_embeddings.append(feats.cpu().numpy().astype(np.float32))

    return np.vstack(all_embeddings), valid_indices


def embed_text(text: str, model, tokenizer, device: str = "cpu") -> np.ndarray:
    import torch
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype(np.float32)[0]


def serialize_embedding(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def deserialize_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()
