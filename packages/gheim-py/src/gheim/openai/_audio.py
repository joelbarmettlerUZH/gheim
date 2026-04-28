"""Audio endpoints — speech (TTS) is anonymize-only; transcriptions/translations
deanonymize the returned text (and SSE deltas if streaming)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

from ..core.session import Session
from ..core.stream import StreamDeanonymizer
from ..detectors import Detector
from ._base import (
    AsyncStreamingResponse,
    StreamingResponse,
    anonymize_value,
    resolve_call_session,
)


# ---------- audio.speech (TTS) — anonymize-only ----------

@dataclass
class SpeechProxy:
    inner: Annotated[Any, "openai client's audio.speech object."]
    detector: Annotated[Detector, "Default detector for this client."]

    def create(self, *args: Any, **kwargs: Any) -> Any:
        sess, detector = resolve_call_session(kwargs, self.detector)
        for k in ("input", "instructions"):
            if k in kwargs:
                kwargs[k] = anonymize_value(kwargs[k], sess, detector)
        return self.inner.create(*args, **kwargs)


@dataclass
class AsyncSpeechProxy:
    inner: Annotated[Any, "openai async client's audio.speech object."]
    detector: Annotated[Detector, "Default detector for this client."]

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        sess, detector = resolve_call_session(kwargs, self.detector)
        for k in ("input", "instructions"):
            if k in kwargs:
                kwargs[k] = anonymize_value(kwargs[k], sess, detector)
        return await self.inner.create(*args, **kwargs)


# ---------- audio.transcriptions / audio.translations — deanonymize text out ----------

def _restore_text_field(response: Any, session: Session) -> None:
    text = getattr(response, "text", None)
    if isinstance(text, str):
        response.text = session.restore(text)


def _patch_transcription_chunk(chunk: Any, buf: StreamDeanonymizer) -> None:
    # transformers.js / openai SSE shapes for transcription deltas can vary; handle
    # the documented ones. Skip events with no text payload.
    delta = getattr(chunk, "delta", None)
    if isinstance(delta, str):
        chunk.delta = buf.feed(delta)
        return
    text = getattr(chunk, "text", None)
    if isinstance(text, str):
        chunk.text = buf.feed(text)


@dataclass
class TranscriptionsProxy:
    inner: Annotated[Any, "openai client's audio.transcriptions object."]
    detector: Annotated[Detector, "Default detector for this client."]

    def create(self, *args: Any, **kwargs: Any) -> Any:
        sess, detector = resolve_call_session(kwargs, self.detector)
        if "prompt" in kwargs and isinstance(kwargs["prompt"], str):
            kwargs["prompt"] = anonymize_value(kwargs["prompt"], sess, detector)
        stream = bool(kwargs.get("stream"))
        response = self.inner.create(*args, **kwargs)
        if stream:
            return StreamingResponse(
                stream=response,
                session=sess,
                mutate_chunk=_patch_transcription_chunk,
            )
        _restore_text_field(response, sess)
        return response


@dataclass
class AsyncTranscriptionsProxy:
    inner: Annotated[Any, "openai async client's audio.transcriptions object."]
    detector: Annotated[Detector, "Default detector for this client."]

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        sess, detector = resolve_call_session(kwargs, self.detector)
        if "prompt" in kwargs and isinstance(kwargs["prompt"], str):
            kwargs["prompt"] = anonymize_value(kwargs["prompt"], sess, detector)
        stream = bool(kwargs.get("stream"))
        response = await self.inner.create(*args, **kwargs)
        if stream:
            return AsyncStreamingResponse(
                stream=response,
                session=sess,
                mutate_chunk=_patch_transcription_chunk,
            )
        _restore_text_field(response, sess)
        return response


@dataclass
class TranslationsProxy:
    """Same shape as TranscriptionsProxy but no streaming support upstream."""

    inner: Annotated[Any, "openai client's audio.translations object."]
    detector: Annotated[Detector, "Default detector for this client."]

    def create(self, *args: Any, **kwargs: Any) -> Any:
        sess, detector = resolve_call_session(kwargs, self.detector)
        if "prompt" in kwargs and isinstance(kwargs["prompt"], str):
            kwargs["prompt"] = anonymize_value(kwargs["prompt"], sess, detector)
        response = self.inner.create(*args, **kwargs)
        _restore_text_field(response, sess)
        return response


@dataclass
class AsyncTranslationsProxy:
    inner: Annotated[Any, "openai async client's audio.translations object."]
    detector: Annotated[Detector, "Default detector for this client."]

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        sess, detector = resolve_call_session(kwargs, self.detector)
        if "prompt" in kwargs and isinstance(kwargs["prompt"], str):
            kwargs["prompt"] = anonymize_value(kwargs["prompt"], sess, detector)
        response = await self.inner.create(*args, **kwargs)
        _restore_text_field(response, sess)
        return response


# ---------- AudioProxy: aggregator that exposes .speech, .transcriptions, .translations ----------

@dataclass
class AudioProxy:
    inner: Annotated[Any, "openai client's audio object."]
    detector: Annotated[Detector, "Default detector for this client."]

    def __post_init__(self) -> None:
        self.speech = SpeechProxy(inner=self.inner.speech, detector=self.detector)
        self.transcriptions = TranscriptionsProxy(
            inner=self.inner.transcriptions, detector=self.detector
        )
        self.translations = TranslationsProxy(
            inner=self.inner.translations, detector=self.detector
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


@dataclass
class AsyncAudioProxy:
    inner: Annotated[Any, "openai async client's audio object."]
    detector: Annotated[Detector, "Default detector for this client."]

    def __post_init__(self) -> None:
        self.speech = AsyncSpeechProxy(inner=self.inner.speech, detector=self.detector)
        self.transcriptions = AsyncTranscriptionsProxy(
            inner=self.inner.transcriptions, detector=self.detector
        )
        self.translations = AsyncTranslationsProxy(
            inner=self.inner.translations, detector=self.detector
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)
