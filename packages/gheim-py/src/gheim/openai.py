"""OpenAI SDK drop-in wrapper — sync (``OpenAI``) and async (``AsyncOpenAI``).

Usage:

    from gheim.openai import OpenAI         # sync
    from gheim.openai import AsyncOpenAI    # async

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hi, my name is Joel"}],
    )

OpenAI's servers see ``<PERSON_1>``; ``response`` contains ``Joel``. Streaming
works the same — chunks are de-anonymized as they arrive.

Each ``create()`` call gets its own ``Session`` by default. Pass
``gheim_session=...`` to reuse one across calls (e.g. multi-turn chat where
``Joel`` should keep mapping to ``<PERSON_1>``).
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Annotated, Any

from .core.session import Session
from .core.stream import StreamDeanonymizer
from .detectors import Detector, default_detector
from .plain import anonymize_messages


def _import_openai() -> Any:
    try:
        import openai
    except ImportError as e:  # pragma: no cover - import guard
        raise ImportError(
            "gheim.openai requires the `openai` package. Install with: uv add 'gheim[openai]'"
        ) from e
    return openai


def _restore_response(response: Any, session: Session) -> Any:
    """In-place restore for non-streaming chat completions."""
    try:
        for choice in response.choices:
            msg = getattr(choice, "message", None)
            if msg is None:
                continue
            content = getattr(msg, "content", None)
            if isinstance(content, str):
                msg.content = session.restore(content)
            tool_calls = getattr(msg, "tool_calls", None) or []
            for tc in tool_calls:
                fn = getattr(tc, "function", None)
                if fn is not None and isinstance(getattr(fn, "arguments", None), str):
                    fn.arguments = session.restore(fn.arguments)
    except AttributeError:
        pass
    return response


def _patch_chunk(chunk: Any, buf: StreamDeanonymizer) -> None:
    choices = getattr(chunk, "choices", None) or []
    for choice in choices:
        delta = getattr(choice, "delta", None)
        if delta is None:
            continue
        content = getattr(delta, "content", None)
        if isinstance(content, str):
            delta.content = buf.feed(content)


def _make_tail_chunk(template: Any, tail: str) -> Any:
    try:
        cloned = template.model_copy(deep=True)  # pydantic
        if cloned.choices:
            cloned.choices[0].delta.content = tail
            cloned.choices[0].delta.role = None
        return cloned
    except Exception:
        return _TailChunk(tail)


@dataclass
class _TailDelta:
    content: str
    role: None = None


@dataclass
class _TailChoice:
    delta: _TailDelta
    index: int = 0
    finish_reason: None = None


@dataclass
class _TailChunk:
    """Minimal shim used when the upstream chunk shape can't be cloned."""

    content: str
    choices: list[_TailChoice] = field(init=False)

    def __post_init__(self) -> None:
        self.choices = [_TailChoice(delta=_TailDelta(content=self.content))]


@dataclass
class _StreamingResponse:
    """Wraps a sync openai Stream so delta chunks come out de-anonymized."""

    stream: Annotated[Any, "Underlying openai Stream[ChatCompletionChunk]."]
    session: Annotated[Session, "Session whose mapping is used for restoration."]
    _buf: StreamDeanonymizer = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._buf = StreamDeanonymizer(self.session.mapping)

    def __iter__(self) -> Iterator[Any]:
        last: Any = None
        for chunk in self.stream:
            last = chunk
            _patch_chunk(chunk, self._buf)
            yield chunk
        tail = self._buf.flush()
        if tail and last is not None:
            yield _make_tail_chunk(last, tail)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.stream, name)

    def close(self) -> None:
        if hasattr(self.stream, "close"):
            self.stream.close()


@dataclass
class _AsyncStreamingResponse:
    """Wraps an async openai AsyncStream so delta chunks come out de-anonymized."""

    stream: Annotated[Any, "Underlying openai AsyncStream[ChatCompletionChunk]."]
    session: Annotated[Session, "Session whose mapping is used for restoration."]
    _buf: StreamDeanonymizer = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._buf = StreamDeanonymizer(self.session.mapping)

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[Any]:
        last: Any = None
        async for chunk in self.stream:
            last = chunk
            _patch_chunk(chunk, self._buf)
            yield chunk
        tail = self._buf.flush()
        if tail and last is not None:
            yield _make_tail_chunk(last, tail)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.stream, name)

    async def close(self) -> None:
        closer = getattr(self.stream, "close", None)
        if closer is None:
            return
        result = closer()
        if hasattr(result, "__await__"):
            await result


@dataclass
class _ChatCompletionsProxy:
    inner: Annotated[Any, "openai client's chat.completions object."]
    detector: Annotated[Detector, "Default detector for this client."]

    def create(self, *args: Any, **kwargs: Any) -> Any:
        sess, detector = _resolve_call_session(kwargs, self.detector)
        _maybe_anonymize_messages(kwargs, sess, detector)
        stream = bool(kwargs.get("stream"))
        response = self.inner.create(*args, **kwargs)
        if stream:
            return _StreamingResponse(stream=response, session=sess)
        return _restore_response(response, sess)


@dataclass
class _AsyncChatCompletionsProxy:
    inner: Annotated[Any, "openai async client's chat.completions object."]
    detector: Annotated[Detector, "Default detector for this client."]

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        sess, detector = _resolve_call_session(kwargs, self.detector)
        _maybe_anonymize_messages(kwargs, sess, detector)
        stream = bool(kwargs.get("stream"))
        response = await self.inner.create(*args, **kwargs)
        if stream:
            return _AsyncStreamingResponse(stream=response, session=sess)
        return _restore_response(response, sess)


def _resolve_call_session(
    kwargs: dict[str, Any], default_det: Detector
) -> tuple[Session, Detector]:
    session: Session | None = kwargs.pop("gheim_session", None)
    detector: Detector = kwargs.pop("gheim_detector", None) or default_det
    sess = session if session is not None else Session(detector=detector)
    return sess, detector


def _maybe_anonymize_messages(
    kwargs: dict[str, Any], sess: Session, detector: Detector
) -> None:
    messages = kwargs.get("messages")
    if messages is None:
        return
    kwargs["messages"] = anonymize_messages(messages, sess, detector=detector)


@dataclass
class _ChatProxy:
    inner: Annotated[Any, "Underlying openai client's chat object."]
    detector: Annotated[Detector, "Default detector for this client."]
    completions: _ChatCompletionsProxy = field(init=False)

    def __post_init__(self) -> None:
        self.completions = _ChatCompletionsProxy(
            inner=self.inner.completions, detector=self.detector
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


@dataclass
class _AsyncChatProxy:
    inner: Annotated[Any, "Underlying openai async client's chat object."]
    detector: Annotated[Detector, "Default detector for this client."]
    completions: _AsyncChatCompletionsProxy = field(init=False)

    def __post_init__(self) -> None:
        self.completions = _AsyncChatCompletionsProxy(
            inner=self.inner.completions, detector=self.detector
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def _make_sync_client_class(base_cls: Any) -> Any:
    class _GheimOpenAI(base_cls):  # type: ignore[misc, valid-type]
        """Drop-in replacement for ``openai.OpenAI`` with PII round-trip."""

        def __init__(
            self,
            *args: Any,
            gheim_detector: Detector | None = None,
            **kwargs: Any,
        ):
            super().__init__(*args, **kwargs)
            self._gheim_detector: Detector = gheim_detector or default_detector()
            self._gheim_chat = _ChatProxy(inner=super().chat, detector=self._gheim_detector)

        @property
        def chat(self) -> _ChatProxy:  # type: ignore[override]
            return self._gheim_chat

    _GheimOpenAI.__name__ = "OpenAI"
    _GheimOpenAI.__qualname__ = "OpenAI"
    return _GheimOpenAI


def _make_async_client_class(base_cls: Any) -> Any:
    class _GheimAsyncOpenAI(base_cls):  # type: ignore[misc, valid-type]
        """Drop-in replacement for ``openai.AsyncOpenAI`` with PII round-trip."""

        def __init__(
            self,
            *args: Any,
            gheim_detector: Detector | None = None,
            **kwargs: Any,
        ):
            super().__init__(*args, **kwargs)
            self._gheim_detector: Detector = gheim_detector or default_detector()
            self._gheim_chat = _AsyncChatProxy(
                inner=super().chat, detector=self._gheim_detector
            )

        @property
        def chat(self) -> _AsyncChatProxy:  # type: ignore[override]
            return self._gheim_chat

    _GheimAsyncOpenAI.__name__ = "AsyncOpenAI"
    _GheimAsyncOpenAI.__qualname__ = "AsyncOpenAI"
    return _GheimAsyncOpenAI


def __getattr__(name: str) -> Any:
    """Lazy: only import openai if the user actually accesses ``OpenAI`` / ``AsyncOpenAI``."""
    if name == "OpenAI":
        return _make_sync_client_class(_import_openai().OpenAI)
    if name == "AsyncOpenAI":
        return _make_async_client_class(_import_openai().AsyncOpenAI)
    raise AttributeError(f"module 'gheim.openai' has no attribute {name!r}")
