"""Runtime configuration, overridable via environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
_DEFAULT_MODEL_PATH = str(_APP_DIR / "model.pt")
_DEFAULT_SAMPLES_DIR = str(_APP_DIR.parent / "samples")


@dataclass
class Settings:
    # Hard limits from the spec (can be lowered in tests).
    max_capacity: int = 16        # max images queued OR in-flight
    batch_size: int = 4           # max images per inference batch
    batch_wait_ms: int = 50       # idle model waits at most this long to fill a batch
    # Demo knob: artificial delay inside the inference worker to simulate a slow model.
    inference_delay_ms: int = 0
    model_path: str = _DEFAULT_MODEL_PATH
    # Bundled sample images used by POST /probes for batch-size trials.
    samples_dir: str = _DEFAULT_SAMPLES_DIR

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            max_capacity=int(os.getenv("MAX_CAPACITY", "16")),
            batch_size=int(os.getenv("BATCH_SIZE", "4")),
            batch_wait_ms=int(os.getenv("BATCH_WAIT_MS", "50")),
            inference_delay_ms=int(os.getenv("INFERENCE_DELAY_MS", "0")),
            model_path=os.getenv("MODEL_PATH", _DEFAULT_MODEL_PATH),
            samples_dir=os.getenv("SAMPLES_DIR", _DEFAULT_SAMPLES_DIR),
        )
