"""Deterministic image preprocessing: bytes -> normalized CHW float32 tensor."""
from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image

IMAGE_SIZE = 64
_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def preprocess(image_bytes: bytes) -> torch.Tensor:
    """Decode an image and return a (3, 64, 64) float32 tensor.

    Raises ValueError if the bytes are not a decodable image.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    except Exception as exc:  # noqa: BLE001 - any decode failure is a bad request
        raise ValueError(f"cannot decode image: {exc}") from exc
    arr = (arr - _MEAN) / _STD
    arr = np.ascontiguousarray(arr.transpose(2, 0, 1))  # HWC -> CHW
    return torch.from_numpy(arr)
