"""OpenAI SDK drop-in wrapper.

Public surface (unchanged from the previous single-file module):

    from gheim.openai import OpenAI, AsyncOpenAI

    client = OpenAI()                  # subclass of openai.OpenAI
    response = client.chat.completions.create(...)

Coverage:
    - Wrapped (round-trip): chat.completions, completions (legacy), responses,
      audio.transcriptions, audio.translations
    - Wrapped (anonymize input only): embeddings, moderations, audio.speech,
      images.generate / images.edit
    - Strict-blocked (raise RuntimeError by default): beta.assistants.*,
      beta.threads.*, batches, files, uploads, fine_tuning, vector_stores
    - Always pass-through: models

Construct with ``gheim_strict=False`` to downgrade strict-blocks to one-time
warnings + pass-through. Use ``client.raw`` to call any underlying endpoint
without gheim wrapping.

The ``openai`` package is an *optional* peer dependency — only required when
you actually access ``OpenAI`` / ``AsyncOpenAI`` from this module.
"""
from __future__ import annotations

from typing import Any

from ._client import _import_openai, make_async_client_class, make_sync_client_class


def __getattr__(name: str) -> Any:
    """Lazy: importing ``gheim.openai`` doesn't import ``openai`` until needed."""
    if name == "OpenAI":
        return make_sync_client_class(_import_openai().OpenAI)
    if name == "AsyncOpenAI":
        return make_async_client_class(_import_openai().AsyncOpenAI)
    raise AttributeError(f"module 'gheim.openai' has no attribute {name!r}")


__all__ = ["OpenAI", "AsyncOpenAI"]
