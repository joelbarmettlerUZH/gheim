"""Model registry — lazy-loads one chunking detector per model id."""
from __future__ import annotations

import os
import threading
from typing import Any

DEFAULT_MODEL = os.getenv("GHEIM_DEFAULT_MODEL", "openai/privacy-filter")
ALLOWED_MODELS = {
    m.strip() for m in os.getenv("GHEIM_ALLOWED_MODELS", DEFAULT_MODEL).split(",") if m.strip()
}


class ModelRegistry:
    """One detector per model id, loaded on first request, cached forever after.

    The detector is ``gheim.LocalDetector``, which splits inputs longer than the
    model's 512-token limit into overlapping windows and merges the spans.
    A bare token-classification pipeline would silently ignore everything past
    the first ~512 tokens (about 1,700 characters).
    """

    def __init__(self) -> None:
        self._pipelines: dict[str, Any] = {}
        self._lock = threading.Lock()

    def get(self, model_id: str) -> Any:
        if model_id not in ALLOWED_MODELS:
            raise ValueError(f"model {model_id!r} is not in GHEIM_ALLOWED_MODELS")
        existing = self._pipelines.get(model_id)
        if existing is not None:
            return existing
        with self._lock:
            existing = self._pipelines.get(model_id)
            if existing is not None:
                return existing
            self._pipelines[model_id] = self._load(model_id)
            return self._pipelines[model_id]

    @staticmethod
    def _load(model_id: str) -> Any:
        from gheim import LocalDetector

        # Weights load on the first detect() call (thread-safe); device "auto"
        # picks CUDA when available, else CPU.
        return LocalDetector(model_id=model_id, device="auto")

    def warmup(self) -> None:
        """Pre-load the default model so the first request is fast."""
        self.get(DEFAULT_MODEL).detect("Warm-up.")


registry = ModelRegistry()
