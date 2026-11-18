"""Generate the fixed-weight TorchScript model shipped with the service.

Weights are initialized with a fixed seed, so the produced model.pt is
byte-stable in behavior (same architecture + same seed => same weights).
The script prints a SHA-256 fingerprint of the weights for verification.

Usage: python scripts/make_model.py [--out app/model.pt]
"""
from __future__ import annotations

import argparse
import hashlib
import io

import torch
from torch import nn

SEED = 20240521
NUM_CLASSES = 10


class TinyClassifier(nn.Module):
    """Small CNN for 64x64 RGB images -> 10-class logits."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Linear(16 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def weights_fingerprint(model: nn.Module) -> str:
    buf = io.BytesIO()
    for p in model.state_dict().values():
        buf.write(p.detach().numpy().tobytes())
    return hashlib.sha256(buf.getvalue()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="app/model.pt")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    model = TinyClassifier()
    model.eval()

    scripted = torch.jit.script(model)
    scripted.save(args.out)

    fp = weights_fingerprint(model)
    print(f"saved TorchScript model -> {args.out}")
    print(f"seed={SEED}  classes={NUM_CLASSES}  weights_sha256={fp}")

    # sanity: reload and run one forward pass
    loaded = torch.jit.load(args.out, map_location="cpu")
    loaded.eval()
    with torch.inference_mode():
        out = loaded(torch.zeros(1, 3, 64, 64))
    assert out.shape == (1, NUM_CLASSES), out.shape
    print("reload + forward OK, output shape", tuple(out.shape))


if __name__ == "__main__":
    main()
