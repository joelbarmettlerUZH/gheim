"""Model registry — lazy-loads HuggingFace token-classification pipelines per model id."""
from __future__ import annotations

import os
import threading
from typing import Any

DEFAULT_MODEL = os.getenv("GHEIM_DEFAULT_MODEL", "openai/privacy-filter")
ALLOWED_MODELS = {
    m.strip() for m in os.getenv("GHEIM_ALLOWED_MODELS", DEFAULT_MODEL).split(",") if m.strip()
}


class ModelRegistry:
    """One pipeline per model id, loaded on first request, cached forever after."""

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
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForTokenClassification.from_pretrained(model_id)
        model.to(device)
        model.eval()
        return pipeline(
            task="token-classification",
            model=model,
            tokenizer=tokenizer,
            device=0 if device == "cuda" else -1,
            aggregation_strategy="simple",
        )

    def warmup(self) -> None:
        """Pre-load the default model so the first request is fast."""
        self.get(DEFAULT_MODEL)


registry = ModelRegistry()
