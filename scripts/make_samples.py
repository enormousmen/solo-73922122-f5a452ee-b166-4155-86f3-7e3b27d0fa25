"""Generate deterministic sample images for manual testing of the API.

Usage: python scripts/make_samples.py [--out samples]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 96  # the service resizes to 64x64 internally


def make_images() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    y, x = np.mgrid[0:SIZE, 0:SIZE]
    images: dict[str, np.ndarray] = {}

    # 1. horizontal color gradient
    grad = np.stack(
        [
            np.tile(np.linspace(0, 255, SIZE), (SIZE, 1)),
            np.tile(np.linspace(255, 0, SIZE), (SIZE, 1)),
            np.full((SIZE, SIZE), 128.0),
        ],
        axis=-1,
    )
    images["sample_gradient.png"] = grad

    # 2. circles
    rr = np.sqrt((x - SIZE / 2) ** 2 + (y - SIZE / 2) ** 2)
    circ = np.zeros((SIZE, SIZE, 3))
    circ[rr < 20] = [230, 40, 40]
    circ[(rr >= 20) & (rr < 34)] = [40, 200, 80]
    circ[rr >= 34] = [30, 30, 60]
    images["sample_circles.png"] = circ

    # 3. diagonal stripes
    stripes = (((x + y) // 8) % 2 * 220 + 20).astype(float)
    images["sample_stripes.png"] = np.stack([stripes, 255 - stripes, stripes / 2], axis=-1)

    # 4. seeded noise
    images["sample_noise.png"] = rng.integers(0, 256, (SIZE, SIZE, 3)).astype(float)

    return images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="samples")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in make_images().items():
        Image.fromarray(arr.astype(np.uint8), "RGB").save(out_dir / name)
        print("wrote", out_dir / name)


if __name__ == "__main__":
    main()
