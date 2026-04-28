"""Chat completions proxy — sync + async. Wraps messages, tool definitions, and
restores message content + tool-call arguments on the response."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any

from ..core.session import Session
from ..core.stream import StreamDeanonymizer
from ..detectors import Detector
from ..plain import anonymize_messages
from ._base import (
    AsyncStreamingResponse,
    StreamingResponse,
    anonymize_value,
    make_chat_tail_chunk,
    resolve_call_session,
)


def _anonymize_chat_kwargs(
    kwargs: dict[str, Any], session: Session, detector: Detector
) -> None:
    """Mutate kwargs in place: anonymize messages, message names, tool descriptions."""
    messages = kwargs.get("messages")
    if messages is not None:
        kwargs["messages"] = anonymize_messages(messages, session, detector=detector)
        # Also redact the optional ``name`` field on each message — it's the speaker
        # identifier and is literally a name (e.g. "joel-barmettler").
        for msg in kwargs["messages"]:
            if isinstance(msg, dict) and isinstance(msg.get("name"), str):
                msg["name"] = anonymize_value(msg["name"], session, detector)

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


def _restore_chat_response(response: Any, session: Session) -> None:
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


def _patch_chat_chunk(chunk: Any, buf: StreamDeanonymizer) -> None:
    choices = getattr(chunk, "choices", None) or []
    for choice in choices:
        delta = getattr(choice, "delta", None)
        if delta is None:
            continue
        content = getattr(delta, "content", None)
        if isinstance(content, str):
            delta.content = buf.feed(content)


@dataclass
class ChatCompletionsProxy:
    inner: Annotated[Any, "openai client's chat.completions object."]
    detector: Annotated[Detector, "Default detector for this client."]

    def create(self, *args: Any, **kwargs: Any) -> Any:
        sess, detector = resolve_call_session(kwargs, self.detector)
        _anonymize_chat_kwargs(kwargs, sess, detector)
        stream = bool(kwargs.get("stream"))
        response = self.inner.create(*args, **kwargs)
        if stream:
            return StreamingResponse(
                stream=response,
                session=sess,
                mutate_chunk=_patch_chat_chunk,
                make_tail_chunk=make_chat_tail_chunk,
            )
        _restore_chat_response(response, sess)
        return response


@dataclass
class AsyncChatCompletionsProxy:
    inner: Annotated[Any, "openai async client's chat.completions object."]
    detector: Annotated[Detector, "Default detector for this client."]

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        sess, detector = resolve_call_session(kwargs, self.detector)
        _anonymize_chat_kwargs(kwargs, sess, detector)
        stream = bool(kwargs.get("stream"))
        response = await self.inner.create(*args, **kwargs)
        if stream:
            return AsyncStreamingResponse(
                stream=response,
                session=sess,
                mutate_chunk=_patch_chat_chunk,
                make_tail_chunk=make_chat_tail_chunk,
            )
        _restore_chat_response(response, sess)
        return response


@dataclass
class ChatProxy:
    inner: Annotated[Any, "Underlying openai client's chat object."]
    detector: Annotated[Detector, "Default detector for this client."]
    completions: ChatCompletionsProxy = field(init=False)

    def __post_init__(self) -> None:
        self.completions = ChatCompletionsProxy(inner=self.inner.completions, detector=self.detector)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


@dataclass
class AsyncChatProxy:
    inner: Annotated[Any, "Underlying openai async client's chat object."]
    detector: Annotated[Detector, "Default detector for this client."]
    completions: AsyncChatCompletionsProxy = field(init=False)

    def __post_init__(self) -> None:
        self.completions = AsyncChatCompletionsProxy(
            inner=self.inner.completions, detector=self.detector
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)
