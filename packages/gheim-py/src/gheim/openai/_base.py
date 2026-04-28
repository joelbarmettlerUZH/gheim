"""Shared helpers for all endpoint proxies — session resolution, in-place text mutation."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any

from ..core.session import Session
from ..core.stream import StreamDeanonymizer
from ..detectors import Detector


def resolve_call_session(
    kwargs: dict[str, Any], default_det: Detector
) -> tuple[Session, Detector]:
    """Pop ``gheim_session`` / ``gheim_detector`` from kwargs and resolve them."""
    session: Session | None = kwargs.pop("gheim_session", None)
    detector: Detector = kwargs.pop("gheim_detector", None) or default_det
    sess = session if session is not None else Session(detector=detector)
    return sess, detector


def anonymize_value(
    value: Any,
    session: Session,
    detector: Detector,
    label: Annotated[str, "Detector label override; only used when value already has a known surface."] | None = None,
) -> Any:
    """Anonymize ``value`` if it's a string; recurse into list[str]; pass everything else through.

    Pre-tokenized inputs (``list[int]``, nested int lists) are passed through untouched —
    we cannot redact token ids. Mixed lists keep their non-string members untouched.
    """
    if isinstance(value, str):
        if not value:
            return value
        spans = detector.detect(value)
        return session.apply_spans(value, spans)
    if isinstance(value, list):
        # If every element is an int, it's pre-tokenized — passthrough.
        if all(isinstance(x, int) for x in value):
            return value
        return [anonymize_value(x, session, detector, label) for x in value]
    return value


def restore_str(value: Any, session: Session) -> Any:
    """Restore sentinels in a string in-place; non-strings pass through."""
    if isinstance(value, str):
        return session.restore(value)
    return value


@dataclass
class StreamingResponse:
    """Generic sync streaming wrapper.

    ``mutate_chunk`` receives each chunk and the streaming buffer; it should call
    ``buf.feed(text)`` and write the returned string back to the chunk's text field.
    Returns nothing — mutation is in-place.
    """

    stream: Annotated[Any, "Underlying openai stream."]
    session: Annotated[Session, "Session for restoration."]
    mutate_chunk: Annotated[
        Callable[[Any, StreamDeanonymizer], None],
        "Per-chunk mutator; calls buf.feed(...) and writes back.",
    ]
    make_tail_chunk: Annotated[
        Callable[[Any, str], Any] | None,
        "Optional: build a synthetic final chunk carrying the held-back tail.",
    ] = None
    _buf: StreamDeanonymizer = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._buf = StreamDeanonymizer(self.session.mapping)

    def __iter__(self):
        last: Any = None
        for chunk in self.stream:
            last = chunk
            self.mutate_chunk(chunk, self._buf)
            yield chunk
        tail = self._buf.flush()
        if tail and last is not None and self.make_tail_chunk is not None:
            yield self.make_tail_chunk(last, tail)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.stream, name)

    def close(self) -> None:
        if hasattr(self.stream, "close"):
            self.stream.close()


@dataclass
class AsyncStreamingResponse:
    """Async counterpart of ``StreamingResponse``."""

    stream: Annotated[Any, "Underlying openai async stream."]
    session: Annotated[Session, "Session for restoration."]
    mutate_chunk: Annotated[
        Callable[[Any, StreamDeanonymizer], None],
        "Per-chunk mutator (sync function — only mutates state).",
    ]
    make_tail_chunk: Annotated[Callable[[Any, str], Any] | None, "Tail chunk builder."] = None
    _buf: StreamDeanonymizer = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._buf = StreamDeanonymizer(self.session.mapping)

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        last: Any = None
        async for chunk in self.stream:
            last = chunk
            self.mutate_chunk(chunk, self._buf)
            yield chunk
        tail = self._buf.flush()
        if tail and last is not None and self.make_tail_chunk is not None:
            yield self.make_tail_chunk(last, tail)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.stream, name)

    async def close(self) -> None:
        closer = getattr(self.stream, "close", None)
        if closer is None:
            return
        result = closer()
        if hasattr(result, "__await__"):
            await result


def make_chat_tail_chunk(template: Any, tail: str) -> Any:
    """Best-effort clone of a chat chunk shape, swapping in the tail content."""
    try:
        cloned = template.model_copy(deep=True)  # pydantic
        if cloned.choices:
            cloned.choices[0].delta.content = tail
            cloned.choices[0].delta.role = None
        return cloned
    except Exception:
        return _SimpleTailChunk(content=tail)


@dataclass
class _SimpleTailDelta:
    content: str
    role: None = None


@dataclass
class _SimpleTailChoice:
    delta: _SimpleTailDelta
    index: int = 0
    finish_reason: None = None


@dataclass
class _SimpleTailChunk:
    """Fallback shim used when the upstream chunk shape can't be cloned."""

    content: str
    choices: list[_SimpleTailChoice] = field(init=False)

    def __post_init__(self) -> None:
        self.choices = [_SimpleTailChoice(delta=_SimpleTailDelta(content=self.content))]
