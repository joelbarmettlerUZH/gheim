"""Responses API — text-in / text-out with event-based streaming.

Streaming events come typed (``event.type``); we route a few well-known event
types to the streaming buffer and pass everything else through. One
``StreamDeanonymizer`` per logical text channel (output text vs reasoning text
vs audio transcript) so independent streams don't interfere.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any

from ..core.session import Session
from ..core.stream import StreamDeanonymizer
from ..detectors import Detector
from ._base import anonymize_value, resolve_call_session


# Event types whose ``delta`` field carries text we should restore.
_TEXT_DELTA_EVENTS: dict[str, str] = {
    "response.output_text.delta": "text",
    "response.reasoning_text.delta": "reasoning",
    "response.audio_transcript.delta": "audio",
    "response.function_call_arguments.delta": "function_args",
    "response.custom_tool_call_input.delta": "custom_tool_input",
}


def _anonymize_responses_kwargs(
    kwargs: dict[str, Any], session: Session, detector: Detector
) -> None:
    for key in ("input", "instructions"):
        if key in kwargs:
            kwargs[key] = anonymize_value(kwargs[key], session, detector)
    # Tool descriptions, same as chat.
    tools = kwargs.get("tools")
    if tools:
        new_tools: list[Any] = []
        for tool in tools:
            if isinstance(tool, dict):
                fn = tool.get("function")
                if isinstance(fn, dict) and isinstance(fn.get("description"), str):
                    new_fn = {**fn, "description": anonymize_value(fn["description"], session, detector)}
                    new_tools.append({**tool, "function": new_fn})
                    continue
            new_tools.append(tool)
        kwargs["tools"] = new_tools


def _restore_responses_response(response: Any, session: Session) -> None:
    """Walk ``response.output[*].content[*].text`` and restore each text field."""
    try:
        for item in getattr(response, "output", None) or []:
            content = getattr(item, "content", None) or []
            for part in content:
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    part.text = session.restore(text)
    except AttributeError:
        pass


@dataclass
class _ResponsesStream:
    stream: Any
    session: Session
    _bufs: dict[str, StreamDeanonymizer] = field(default_factory=dict, init=False)

    def _buf_for(self, channel: str) -> StreamDeanonymizer:
        if channel not in self._bufs:
            self._bufs[channel] = StreamDeanonymizer(self.session.mapping)
        return self._bufs[channel]

    def _patch(self, event: Any) -> None:
        et = getattr(event, "type", None)
        if not isinstance(et, str):
            return
        channel = _TEXT_DELTA_EVENTS.get(et)
        if channel is None:
            return
        delta = getattr(event, "delta", None)
        if isinstance(delta, str):
            event.delta = self._buf_for(channel).feed(delta)


@dataclass
class ResponsesStreamingResponse(_ResponsesStream):
    def __iter__(self):
        for event in self.stream:
            self._patch(event)
            yield event
        # Tail flush: synthesize a final delta-shaped event per channel that has
        # a held-back tail. We don't have a great template event for each channel,
        # so emit a minimal shim object.
        for channel, buf in self._bufs.items():
            tail = buf.flush()
            if tail:
                yield _TailEvent(type=f"response.{channel}.delta.tail", delta=tail)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.stream, name)

    def close(self) -> None:
        if hasattr(self.stream, "close"):
            self.stream.close()


@dataclass
class AsyncResponsesStreamingResponse(_ResponsesStream):
    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        async for event in self.stream:
            self._patch(event)
            yield event
        for channel, buf in self._bufs.items():
            tail = buf.flush()
            if tail:
                yield _TailEvent(type=f"response.{channel}.delta.tail", delta=tail)

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
class _TailEvent:
    """Synthetic end-of-stream event carrying any held-back tail per channel."""

    type: str
    delta: str


@dataclass
class ResponsesProxy:
    inner: Annotated[Any, "openai client's responses object."]
    detector: Annotated[Detector, "Default detector for this client."]

    def create(self, *args: Any, **kwargs: Any) -> Any:
        sess, detector = resolve_call_session(kwargs, self.detector)
        _anonymize_responses_kwargs(kwargs, sess, detector)
        stream = bool(kwargs.get("stream"))
        response = self.inner.create(*args, **kwargs)
        if stream:
            return ResponsesStreamingResponse(stream=response, session=sess)
        _restore_responses_response(response, sess)
        return response


@dataclass
class AsyncResponsesProxy:
    inner: Annotated[Any, "openai async client's responses object."]
    detector: Annotated[Detector, "Default detector for this client."]

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        sess, detector = resolve_call_session(kwargs, self.detector)
        _anonymize_responses_kwargs(kwargs, sess, detector)
        stream = bool(kwargs.get("stream"))
        response = await self.inner.create(*args, **kwargs)
        if stream:
            return AsyncResponsesStreamingResponse(stream=response, session=sess)
        _restore_responses_response(response, sess)
        return response
