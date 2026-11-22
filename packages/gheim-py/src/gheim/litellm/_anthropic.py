"""The native Anthropic Messages route (``/v1/messages``) for :class:`GheimGuardrail`.

LiteLLM runs the guardrail's hooks there too (``call_type="anthropic_messages"``),
but every payload has the Anthropic shape:

- request: a top-level ``system`` (string or text blocks) and content blocks --
  ``text``, ``tool_result`` (string or text blocks), ``tool_use`` (a JSON
  ``input`` object);
- non-streamed response: a ``{"type": "message", "content": [...]}`` dict;
- streamed response: raw SSE bytes, cut wherever the upstream read ended, so
  an event (and a sentinel) can span several items.

Signed ``thinking`` blocks are left alone both ways: a client replays them with
their signature, and a block that differs from what the model produced is
rejected. Thinking therefore reads ``<PERSON_1>``.
"""
from __future__ import annotations

import json
from typing import Any

from ..core.session import Session
from ..core.stream import StreamDeanonymizer
from ..plain import anonymize_text

CALL_TYPE = "anthropic_messages"

# delta type -> the field carrying its text. thinking_delta is absent on purpose.
_DELTA_FIELDS = {"text_delta": "text", "input_json_delta": "partial_json"}


def _json_escape(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)[1:-1]


# -- request ----------------------------------------------------------------


def _json_leaves(value: Any, fn) -> Any:
    if isinstance(value, str):
        return fn(value)
    if isinstance(value, list):
        return [_json_leaves(v, fn) for v in value]
    if isinstance(value, dict):
        return {k: _json_leaves(v, fn) for k, v in value.items()}
    return value


def _text_or_blocks(value: Any, fn) -> Any:
    """A string, or a list whose ``text`` blocks get ``fn``; anything else as-is."""
    if isinstance(value, str):
        return fn(value)
    if not isinstance(value, list):
        return value
    return [
        {**b, "text": fn(b["text"])}
        if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)
        else b
        for b in value
    ]


def _block(block: Any, fn) -> Any:
    if not isinstance(block, dict):
        return block
    kind = block.get("type")
    if kind == "text" and isinstance(block.get("text"), str):
        return {**block, "text": fn(block["text"])}
    if kind == "tool_result":
        return {**block, "content": _text_or_blocks(block.get("content"), fn)}
    if kind == "tool_use":
        return {**block, "input": _json_leaves(block.get("input"), fn)}
    return block  # thinking, images, documents...


def anonymize_request(data: dict, session: Session) -> dict:
    """``system`` and every message's text, tool results and tool inputs, in place."""
    session.reserve([data.get("system"), data.get("messages")])

    def fn(text: str) -> str:
        return anonymize_text(text, session)

    if "system" in data:
        data["system"] = _text_or_blocks(data["system"], fn)
    messages = []
    for msg in data.get("messages") or []:
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list):
            msg = {**msg, "content": [_block(b, fn) for b in content]}
        elif isinstance(content, str):
            msg = {**msg, "content": fn(content)}
        messages.append(msg)
    data["messages"] = messages
    return data


# -- response, non-streaming --------------------------------------------------


def is_message(response: Any) -> bool:
    return isinstance(response, dict) and response.get("type") == "message"


def restore_message(response: dict, session: Session) -> dict:
    """Text blocks and tool_use inputs, in place; thinking stays as produced."""
    for block in response.get("content") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            block["text"] = session.restore(block["text"])
        elif block.get("type") == "tool_use":
            block["input"] = _json_leaves(block.get("input"), session.restore)
    return response


# -- response, streaming --------------------------------------------------------


def _split_event(buffer: bytes) -> tuple[bytes | None, bytes]:
    """(one whole event incl. its blank-line terminator, rest), or (None, buffer)."""
    ends = [(at, sep) for sep in (b"\r\n\r\n", b"\n\n") if (at := buffer.find(sep)) >= 0]
    if not ends:
        return None, buffer
    at, sep = min(ends)
    return buffer[: at + len(sep)], buffer[at + len(sep):]


def _parse(raw: bytes) -> tuple[str, str | None, dict] | None:
    """(newline, event name, data) for a JSON data event, else None."""
    name, data = None, None
    for line in raw.decode("utf-8").splitlines():
        if line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            try:
                data = json.loads(line[5:])
            except ValueError:
                return None
    if not isinstance(data, dict):
        return None
    return ("\r\n" if b"\r\n" in raw else "\n"), name, data


def _render(newline: str, name: str | None, data: dict) -> bytes:
    head = f"event: {name}{newline}" if name else ""
    return f"{head}data: {json.dumps(data, ensure_ascii=False)}{newline}{newline}".encode()


class SSERestorer:
    """Restores sentinels in an Anthropic Messages SSE stream, event by event.

    Items are re-framed into whole events; ``text_delta`` and
    ``input_json_delta`` run through one :class:`StreamDeanonymizer` per content
    block (tool input with JSON-escaped surfaces), so a sentinel split across
    deltas or reads is restored. Text held back as a possible sentinel prefix is
    released in its own delta just before the block stops. Every other event
    goes out byte-identical. Output items keep the type (str or bytes) of the
    input items.
    """

    def __init__(self, session: Session) -> None:
        self._maps = {
            "text_delta": dict(session.mapping),
            "input_json_delta": {k: _json_escape(v) for k, v in session.mapping.items()},
        }
        self._bufs: dict[tuple[Any, str], StreamDeanonymizer] = {}
        self._pending = b""
        self._as_str = False

    def feed(self, item: str | bytes) -> list[str | bytes]:
        self._as_str = isinstance(item, str)
        self._pending += item.encode() if isinstance(item, str) else item
        out: list[bytes] = []
        while True:
            raw, self._pending = _split_event(self._pending)
            if raw is None:
                return self._typed(out)
            out += self._event(raw)

    def finish(self) -> list[str | bytes]:
        out = self._release("\n", None)
        if self._pending:
            out.append(self._pending)
            self._pending = b""
        return self._typed(out)

    def _typed(self, out: list[bytes]) -> list[str | bytes]:
        return [b.decode() for b in out] if self._as_str else list(out)

    def _event(self, raw: bytes) -> list[bytes]:
        parsed = _parse(raw)
        if parsed is None:
            return [raw]
        newline, name, data = parsed
        kind = data.get("type")
        if kind == "content_block_delta":
            return self._delta(raw, newline, name, data)
        if kind == "content_block_stop":
            return self._release(newline, data.get("index")) + [raw]
        if kind == "message_stop":
            return self._release(newline, None) + [raw]
        return [raw]

    def _delta(self, raw: bytes, newline: str, name: str | None, data: dict) -> list[bytes]:
        delta = data.get("delta") or {}
        kind = delta.get("type")
        field = _DELTA_FIELDS.get(kind)
        if field is None or not isinstance(delta.get(field), str):
            return [raw]
        key = (data.get("index"), kind)
        if key not in self._bufs:
            self._bufs[key] = StreamDeanonymizer(self._maps[kind])
        text = self._bufs[key].feed(delta[field])
        if not text:
            return []
        return [_render(newline, name, {**data, "delta": {**delta, field: text}})]

    def _release(self, newline: str, index: Any) -> list[bytes]:
        out = []
        for (block, kind), buf in self._bufs.items():
            held = buf.flush() if index is None or block == index else ""
            if held:
                data = {"type": "content_block_delta", "index": block,
                        "delta": {"type": kind, _DELTA_FIELDS[kind]: held}}
                out.append(_render(newline, "content_block_delta", data))
        return out
