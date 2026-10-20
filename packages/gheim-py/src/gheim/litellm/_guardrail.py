"""LiteLLM ``CustomGuardrail`` that anonymizes requests and restores responses.

Request path (``async_pre_call_hook``): every chat message is anonymized into
sentinels (``<PERSON_1>``) before the proxy forwards it to the provider.

Response path:

- non-streaming: sentinels in ``content`` / reasoning / tool-call arguments
  are restored in place (``async_post_call_success_hook``);
- streaming: restored incrementally per chunk with :class:`StreamDeanonymizer`
  (``async_post_call_streaming_iterator_hook``). Only a possibly-incomplete
  sentinel (a few characters) is ever held back, so time-to-first-token and
  chunk cadence are preserved. Presidio's ``output_parse_pii`` instead buffers
  the whole stream and re-emits one chunk.

The sentinel mapping is PII, so it lives only in this process, keyed by an
opaque per-request id; it is never written into request metadata, which
LiteLLM forwards to spend logs and callbacks.
"""
from __future__ import annotations

import asyncio
import copy
import json
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterable
from typing import Any

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    Function,
    ModelResponseStream,
)

from ..core.sentinels import SENTINEL_RE
from ..core.session import Session
from ..core.stream import StreamDeanonymizer
from ..detectors import Detector
from ..plain import anonymize_messages
from ._config import build_detector

_CALL_KEY = "gheim_call_key"
_TEXT_FIELDS = ("content", "reasoning_content")


def _json_escape(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)[1:-1]


class _StreamState:
    """Per-request streaming state: one deanonymizer per (choice, field[, tool])."""

    def __init__(self, session: Session) -> None:
        self._text_map = dict(session.mapping)
        self._args_map = {k: _json_escape(v) for k, v in session.mapping.items()}
        self._bufs: dict[tuple, StreamDeanonymizer] = {}

    def feed(self, key: tuple, text: str) -> str:
        buf = self._bufs.get(key)
        if buf is None:
            mapping = self._args_map if key[1] == "tool" else self._text_map
            buf = self._bufs[key] = StreamDeanonymizer(mapping)
        return buf.feed(text)

    def flush(self, choice_index: int | None = None) -> list[tuple[tuple, str]]:
        out = []
        for key, buf in self._bufs.items():
            if choice_index is None or key[0] == choice_index:
                held = buf.flush()
                if held:
                    out.append((key, held))
        return out


def _tail_chunk(
    template: ModelResponseStream, held: list[tuple[tuple, str]]
) -> ModelResponseStream:
    """Synthesize a chunk carrying text that was held back until stream end."""
    tail = copy.deepcopy(template)
    tail.usage = None  # type: ignore[attr-defined]
    keep = []
    for choice in tail.choices:
        items = [(k, t) for k, t in held if k[0] == choice.index]
        if not items:
            continue
        choice.finish_reason = None
        delta = choice.delta
        delta.role = None
        delta.content = None
        delta.tool_calls = None
        for key, text in items:
            if key[1] == "tool":
                delta.tool_calls = [
                    ChatCompletionDeltaToolCall(
                        index=key[2], function=Function(arguments=text), type=None, id=None
                    )
                ]
            else:
                setattr(delta, key[1], text)
        keep.append(choice)
    tail.choices = keep
    return tail


class GheimGuardrail(CustomGuardrail):
    """PII anonymization with streaming-safe restoration for the LiteLLM proxy.

    Extra ``litellm_params`` accepted in ``config.yaml`` (all optional; see
    :mod:`gheim.litellm._config` for the detector knobs and an example):

    - ``detector``: ``composite`` (default; ``remote`` if an ``api_key`` /
      ``GHEIM_API_KEY`` is set) | ``calibrated`` | ``local`` | ``remote`` |
      ``module:factory``;
    - ``detector_model``, ``o_bias``, ``device``, ``dtype``, ``chunk_tokens``,
      ``chunk_overlap_tokens``, ``batch_size``, ``trim_whitespace``: model knobs;
    - ``model_detector`` (``calibrated`` | ``local`` | ``remote``: what the
      composite runs after its regexes), ``custom_patterns``,
      ``propagate_names``, ``propagate_surnames``: composite-detector knobs;
    - ``api_key``, ``api_base``, ``timeout``: remote detector;
    - ``normalizers``: per-label canonical forms, e.g.
      ``{private_phone: e164, private_date: iso_date}``;
    - ``restore_output``: restore sentinels in responses (default ``true``);
      ``false`` returns the model's anonymized output as-is;
    - ``session_ttl_seconds``: drop per-request state of calls that never
      reach a response hook (default 3600).

    A detector knob the chosen detector would ignore is a startup error.
    """

    def __init__(
        self,
        guardrail_name: str | None = None,
        *,
        detector: Detector | str | None = None,
        detector_model: str | None = None,
        o_bias: float | None = None,
        device: str | None = None,
        dtype: str | None = None,
        chunk_tokens: int | None = None,
        chunk_overlap_tokens: int | None = None,
        batch_size: int | None = None,
        trim_whitespace: bool | None = None,
        custom_patterns: list[dict[str, Any]] | None = None,
        model_detector: str | None = None,
        propagate_names: bool | None = None,
        propagate_surnames: bool | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        timeout: float | None = None,
        normalizers: dict[str, str] | None = None,
        restore_output: bool = True,
        session_ttl_seconds: float = 3600.0,
        **kwargs: Any,
    ) -> None:
        # LiteLLM only calls the response hooks of a guardrail whose mode
        # includes post_call. Anonymize-without-restore is never what a user
        # who wrote ``mode: pre_call`` wants, so always register both.
        kwargs["event_hook"] = [GuardrailEventHooks.pre_call, GuardrailEventHooks.post_call]
        super().__init__(guardrail_name=guardrail_name, **kwargs)
        self.detector = build_detector(
            detector,
            detector_model=detector_model, o_bias=o_bias, device=device, dtype=dtype,
            chunk_tokens=chunk_tokens, chunk_overlap_tokens=chunk_overlap_tokens,
            batch_size=batch_size, trim_whitespace=trim_whitespace,
            custom_patterns=custom_patterns, model_detector=model_detector,
            propagate_names=propagate_names,
            propagate_surnames=propagate_surnames,
            api_key=api_key, api_base=api_base, timeout=timeout,
        )
        self.normalizers = normalizers
        self.restore_output = restore_output
        self.session_ttl_seconds = session_ttl_seconds
        self._sessions: dict[str, tuple[Session, float]] = {}

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks] | None:
        return [GuardrailEventHooks.pre_call, GuardrailEventHooks.post_call]

    # -- session bookkeeping ------------------------------------------------

    def _sweep(self) -> None:
        cutoff = time.monotonic() - self.session_ttl_seconds
        for key in [k for k, (_, ts) in self._sessions.items() if ts < cutoff]:
            self._sessions.pop(key, None)

    @staticmethod
    def _key(data: dict | None) -> str | None:
        return ((data or {}).get("metadata") or {}).get(_CALL_KEY)

    def _session_for(self, data: dict | None) -> Session | None:
        key = self._key(data)
        entry = self._sessions.get(key) if key else None
        return entry[0] if entry else None

    def _drop(self, data: dict | None) -> None:
        key = self._key(data)
        if key:
            self._sessions.pop(key, None)

    # -- request ------------------------------------------------------------

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):  # type: ignore[override]
        messages = data.get("messages")
        if not isinstance(messages, list) or not messages:
            return data
        if not self.should_run_guardrail(data=data, event_type=GuardrailEventHooks.pre_call):
            return data

        self._sweep()
        session = Session(detector=self.detector, normalizers=self.normalizers)
        # Detection is blocking (model inference / HTTP): keep the event loop free.
        data["messages"] = await asyncio.to_thread(
            anonymize_messages, messages, session, detector=self.detector
        )
        if self.restore_output:
            key = uuid.uuid4().hex
            data.setdefault("metadata", {})[_CALL_KEY] = key
            self._sessions[key] = (session, time.monotonic())
        return data

    # -- response, non-streaming ---------------------------------------------

    async def async_post_call_success_hook(self, data, user_api_key_dict, response):  # type: ignore[override]
        session = self._session_for(data)
        if session is None:
            return response
        self._drop(data)

        def restore_args(s: str) -> str:
            return SENTINEL_RE.sub(
                lambda m: _json_escape(session.mapping.get(m.group(0), m.group(0))), s
            )

        for choice in getattr(response, "choices", None) or []:
            msg = getattr(choice, "message", None)
            if msg is None:
                continue
            for name in _TEXT_FIELDS:
                val = getattr(msg, name, None)
                if isinstance(val, str):
                    setattr(msg, name, session.restore(val))
            for tc in getattr(msg, "tool_calls", None) or []:
                fn = getattr(tc, "function", None)
                if fn is not None and isinstance(getattr(fn, "arguments", None), str):
                    fn.arguments = restore_args(fn.arguments)
        return response

    async def async_post_call_failure_hook(  # type: ignore[override]
        self, request_data, original_exception, user_api_key_dict, traceback_str=None
    ):
        self._drop(request_data)

    # -- response, streaming -------------------------------------------------

    async def async_post_call_streaming_iterator_hook(  # type: ignore[override]
        self, user_api_key_dict, response: AsyncIterable[Any], request_data: dict
    ) -> AsyncGenerator[Any, None]:
        session = self._session_for(request_data)
        if session is None:
            async for item in response:
                yield item
            return

        state = _StreamState(session)
        last: ModelResponseStream | None = None
        try:
            async for chunk in response:
                if not isinstance(chunk, ModelResponseStream):
                    yield chunk  # raw bytes / non-chat events: pass through unchanged
                    continue
                last = chunk
                held: list[tuple[tuple, str]] = []
                for choice in chunk.choices:
                    delta = choice.delta
                    if delta is None:
                        continue
                    for name in _TEXT_FIELDS:
                        val = getattr(delta, name, None)
                        if isinstance(val, str):
                            setattr(delta, name, state.feed((choice.index, name), val))
                    for tc in getattr(delta, "tool_calls", None) or []:
                        fn = getattr(tc, "function", None)
                        args = getattr(fn, "arguments", None)
                        if isinstance(args, str):
                            fn.arguments = state.feed((choice.index, "tool", tc.index), args)
                    if choice.finish_reason:
                        held += state.flush(choice.index)
                if held:
                    # Text held back for finishing choices goes out before the
                    # chunk that carries finish_reason.
                    yield _tail_chunk(chunk, held)
                yield chunk
            held = state.flush()
            if held and last is not None:
                yield _tail_chunk(last, held)
        finally:
            self._drop(request_data)
