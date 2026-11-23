"""Model registry — lazy-loads one chunking detector per model id."""
from __future__ import annotations

import contextlib
import logging
import os
import threading
from typing import Any

DEFAULT_MODEL = os.getenv("GHEIM_DEFAULT_MODEL", "joelbarmettler/gheim-ch-560m")
ALLOWED_MODELS = {
    m.strip() for m in os.getenv("GHEIM_ALLOWED_MODELS", DEFAULT_MODEL).split(",") if m.strip()
}
# Torch device: "cuda", "mps" or "cpu". Unset keeps the original choice (cuda when
# available, else cpu). A requested device that is not available falls back to
# cpu with a warning rather than refusing to start: a detector that does not come
# up leaves its callers with nothing to ask.
DEVICE = os.getenv("GHEIM_DEVICE", "").strip().lower()
# Weight precision: "float32"/"fp32", "float16"/"fp16" or "bfloat16"/"bf16". Unset
# loads the model's own dtype, as before. An unknown value does the same, with a
# warning, for the same reason as the device.
DTYPE = os.getenv("GHEIM_DTYPE", "").strip().lower()
_DTYPE_NAMES = {"float32": "float32", "fp32": "float32", "float16": "float16",
                "fp16": "float16", "bfloat16": "bfloat16", "bf16": "bfloat16"}

log = logging.getLogger("gheim_server.registry")


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
        # One inference at a time on MPS. FastAPI runs the sync /v1/detect in a
        # threadpool, and torch's MPS backend is not safe to drive from several
        # threads at once. CPU and CUDA keep running requests concurrently.
        self.inference_guard: Any = (
            threading.Lock() if DEVICE == "mps" else contextlib.nullcontext()
        )

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
        import torch
        from gheim import LocalDetector

        device = DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
        if device == "mps" and not torch.backends.mps.is_available():
            log.warning("GHEIM_DEVICE=mps but MPS is not available; using cpu")
            device = "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            log.warning("GHEIM_DEVICE=cuda but CUDA is not available; using cpu")
            device = "cpu"
        dtype_name = _DTYPE_NAMES.get(DTYPE)
        if DTYPE and dtype_name is None:
            log.warning("GHEIM_DTYPE=%r is not a known dtype; using the model's own", DTYPE)
        log.warning("loading %s on device=%s dtype=%s", model_id, device, dtype_name or "model default")
        # Weights load on the first detect() call (thread-safe). LocalDetector
        # hands any non-cpu device to the pipeline as index 0, which transformers
        # resolves to mps:0 on a machine without CUDA.
        dtype = getattr(torch, dtype_name) if dtype_name else None
        return LocalDetector(model_id=model_id, device=device, dtype=dtype)

    @staticmethod
    def release_cache() -> None:
        """Hand the MPS allocator's cached blocks back to the system.

        Without this the allocator keeps every buffer a long request needed, so
        the idle footprint settles at the high-water mark -- and on Apple
        silicon that memory is wired, so nothing else on the machine can use it.
        Call it under ``inference_guard``: MPS is not safe across threads.
        """
        if DEVICE == "mps":
            import torch

            torch.mps.empty_cache()

    def warmup(self) -> None:
        """Pre-load the default model so the first request is fast."""
        self.get(DEFAULT_MODEL).detect("Warm-up.")


registry = ModelRegistry()
