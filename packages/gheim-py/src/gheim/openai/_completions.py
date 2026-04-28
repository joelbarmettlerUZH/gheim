"""Legacy completions API — single ``prompt`` field in, ``choices[*].text`` out.

Pre-tokenized prompts (``list[int]``) are passed through unchanged.
"""
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


def _anonymize_completions_kwargs(
    kwargs: dict[str, Any], session: Session, detector: Detector
) -> None:
    if "prompt" in kwargs:
        kwargs["prompt"] = anonymize_value(kwargs["prompt"], session, detector)


def _restore_completions_response(response: Any, session: Session) -> None:
    try:
        for choice in response.choices:
            text = getattr(choice, "text", None)
            if isinstance(text, str):
                choice.text = session.restore(text)
    except AttributeError:
        pass


def _patch_completions_chunk(chunk: Any, buf: StreamDeanonymizer) -> None:
    choices = getattr(chunk, "choices", None) or []
    for choice in choices:
        text = getattr(choice, "text", None)
        if isinstance(text, str):
            choice.text = buf.feed(text)


def _make_completions_tail_chunk(template: Any, tail: str) -> Any:
    try:
        cloned = template.model_copy(deep=True)
        if cloned.choices:
            cloned.choices[0].text = tail
        return cloned
    except Exception:  # pragma: no cover — pure fallback path
        return None


@dataclass
class CompletionsProxy:
    inner: Annotated[Any, "openai client's completions object (legacy)."]
    detector: Annotated[Detector, "Default detector for this client."]

    def create(self, *args: Any, **kwargs: Any) -> Any:
        sess, detector = resolve_call_session(kwargs, self.detector)
        _anonymize_completions_kwargs(kwargs, sess, detector)
        stream = bool(kwargs.get("stream"))
        response = self.inner.create(*args, **kwargs)
        if stream:
            return StreamingResponse(
                stream=response,
                session=sess,
                mutate_chunk=_patch_completions_chunk,
                make_tail_chunk=_make_completions_tail_chunk,
            )
        _restore_completions_response(response, sess)
        return response


@dataclass
class AsyncCompletionsProxy:
    inner: Annotated[Any, "openai async client's completions object."]
    detector: Annotated[Detector, "Default detector for this client."]

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        sess, detector = resolve_call_session(kwargs, self.detector)
        _anonymize_completions_kwargs(kwargs, sess, detector)
        stream = bool(kwargs.get("stream"))
        response = await self.inner.create(*args, **kwargs)
        if stream:
            return AsyncStreamingResponse(
                stream=response,
                session=sess,
                mutate_chunk=_patch_completions_chunk,
                make_tail_chunk=_make_completions_tail_chunk,
            )
        _restore_completions_response(response, sess)
        return response
