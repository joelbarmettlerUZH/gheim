"""Endpoints that send text but receive no text — embeddings, moderations, audio.speech, images."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

from ..core.session import Session
from ..detectors import Detector
from ._base import anonymize_value, resolve_call_session


def _anonymize_keys(kwargs: dict[str, Any], keys: list[str], sess: Session, det: Detector) -> None:
    for k in keys:
        if k in kwargs:
            kwargs[k] = anonymize_value(kwargs[k], sess, det)


# ---------- Embeddings ----------

@dataclass
class EmbeddingsProxy:
    inner: Annotated[Any, "openai client's embeddings object."]
    detector: Annotated[Detector, "Default detector for this client."]

    def create(self, *args: Any, **kwargs: Any) -> Any:
        sess, detector = resolve_call_session(kwargs, self.detector)
        _anonymize_keys(kwargs, ["input"], sess, detector)
        return self.inner.create(*args, **kwargs)


@dataclass
class AsyncEmbeddingsProxy:
    inner: Annotated[Any, "openai async client's embeddings object."]
    detector: Annotated[Detector, "Default detector for this client."]

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        sess, detector = resolve_call_session(kwargs, self.detector)
        _anonymize_keys(kwargs, ["input"], sess, detector)
        return await self.inner.create(*args, **kwargs)


# ---------- Moderations ----------

@dataclass
class ModerationsProxy:
    inner: Annotated[Any, "openai client's moderations object."]
    detector: Annotated[Detector, "Default detector for this client."]

    def create(self, *args: Any, **kwargs: Any) -> Any:
        sess, detector = resolve_call_session(kwargs, self.detector)
        _anonymize_keys(kwargs, ["input"], sess, detector)
        return self.inner.create(*args, **kwargs)


@dataclass
class AsyncModerationsProxy:
    inner: Annotated[Any, "openai async client's moderations object."]
    detector: Annotated[Detector, "Default detector for this client."]

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        sess, detector = resolve_call_session(kwargs, self.detector)
        _anonymize_keys(kwargs, ["input"], sess, detector)
        return await self.inner.create(*args, **kwargs)


# ---------- Images (generate / edit) ----------

@dataclass
class ImagesProxy:
    inner: Annotated[Any, "openai client's images object."]
    detector: Annotated[Detector, "Default detector for this client."]

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        sess, detector = resolve_call_session(kwargs, self.detector)
        _anonymize_keys(kwargs, ["prompt"], sess, detector)
        return self.inner.generate(*args, **kwargs)

    def edit(self, *args: Any, **kwargs: Any) -> Any:
        sess, detector = resolve_call_session(kwargs, self.detector)
        _anonymize_keys(kwargs, ["prompt"], sess, detector)
        return self.inner.edit(*args, **kwargs)

    # create_variation has no text input — pass through.
    def create_variation(self, *args: Any, **kwargs: Any) -> Any:
        return self.inner.create_variation(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


@dataclass
class AsyncImagesProxy:
    inner: Annotated[Any, "openai async client's images object."]
    detector: Annotated[Detector, "Default detector for this client."]

    async def generate(self, *args: Any, **kwargs: Any) -> Any:
        sess, detector = resolve_call_session(kwargs, self.detector)
        _anonymize_keys(kwargs, ["prompt"], sess, detector)
        return await self.inner.generate(*args, **kwargs)

    async def edit(self, *args: Any, **kwargs: Any) -> Any:
        sess, detector = resolve_call_session(kwargs, self.detector)
        _anonymize_keys(kwargs, ["prompt"], sess, detector)
        return await self.inner.edit(*args, **kwargs)

    async def create_variation(self, *args: Any, **kwargs: Any) -> Any:
        return await self.inner.create_variation(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)
