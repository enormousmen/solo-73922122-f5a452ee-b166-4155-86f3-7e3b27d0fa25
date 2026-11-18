"""Load and preprocess the bundled sample images for capacity probes.

POST /probes must exercise the model with the same sample images the service
ships, preprocessed exactly like user uploads (see :mod:`app.preprocessing`).
"""
from __future__ import annotations

from pathlib import Path

import torch

from .preprocessing import preprocess

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class SampleLoadError(RuntimeError):
    """Raised when the sample directory cannot provide enough valid images."""


def load_sample_tensors(samples_dir: str | Path, min_images: int) -> list[torch.Tensor]:
    """Read every image in ``samples_dir`` and return preprocessed tensors.

    Files are processed in sorted filename order so trials are deterministic.
    Raises SampleLoadError when the directory is missing, contains no
    decodable image, or provides fewer than ``min_images`` images.
    """
    base = Path(samples_dir)
    if not base.is_dir():
        raise SampleLoadError(f"samples directory not found: {base}")

    paths = sorted(p for p in base.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)
    tensors: list[torch.Tensor] = []
    errors: list[str] = []
    for path in paths:
        try:
            tensors.append(preprocess(path.read_bytes()))
        except ValueError as exc:
            errors.append(f"{path.name}: {exc}")

    if not tensors:
        detail = f"; {'; '.join(errors)}" if errors else ""
        raise SampleLoadError(f"no decodable sample images in {base}{detail}")
    if len(tensors) < min_images:
        raise SampleLoadError(
            f"need at least {min_images} sample images for batch-size probes, "
            f"found {len(tensors)} in {base}"
        )
    return tensors
