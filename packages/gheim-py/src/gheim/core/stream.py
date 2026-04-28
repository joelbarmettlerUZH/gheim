"""Streaming de-anonymizer.

Problem: the LLM produces tokens one at a time. A sentinel like ``<PERSON_1>`` may
split across chunks (``<PERS``, ``ON_1>``), or even across single-character tokens.
We cannot emit ``<`` to the end user until we know whether what follows it is a
sentinel (to be substituted) or incidental text like ``<html>`` or ``x < 3``.

Strategy: character-level state machine with a hold-back buffer.

- While scanning, flush everything up to but not including the first ``<``.
- Once we see ``<``, enter HOLDING. Keep appending.
- While HOLDING, after every append, ask three questions:
    1. Does the buffer end with a complete sentinel? → substitute, emit, drop buffer.
    2. Is the buffer still a possible sentinel prefix? → keep holding, wait for more.
    3. Otherwise: the ``<`` turned out to be incidental. Emit the ``<`` literally,
       rewind past it, and restart scanning at the next character.

The buffer only grows until we decide, so worst-case hold-back is bounded by the
longest plausible sentinel (a few dozen chars).
"""
from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Annotated

from .sentinels import SENTINEL_RE, is_possible_sentinel_prefix

# Hard upper bound. A sentinel has shape ``<TAG_N>`` with a short TAG and small N.
# If a hold-back ever exceeds this, it's definitely not a sentinel — emit verbatim.
MAX_SENTINEL_LEN = 64


@dataclass
class StreamDeanonymizer:
    """Incremental chunk-based deanonymizer.

    ``mapping`` is a dict of rendered sentinel (``"<PERSON_1>"``) → original surface.
    Unknown sentinels are passed through untouched.
    """

    mapping: Annotated[
        dict[str, str], "Sentinel string ('<PERSON_1>') → original surface form."
    ]
    _hold: str = field(default="", init=False, repr=False)

    def feed(
        self,
        chunk: Annotated[str, "Next text chunk from the upstream stream."],
    ) -> str:
        """Consume a chunk, return what can be safely emitted right now."""
        if not chunk:
            return ""

        buffer = self._hold + chunk
        out: list[str] = []
        i = 0
        n = len(buffer)

        while i < n:
            if buffer[i] != "<":
                lt = buffer.find("<", i)
                if lt == -1:
                    out.append(buffer[i:])
                    i = n
                    break
                out.append(buffer[i:lt])
                i = lt

            m = SENTINEL_RE.match(buffer, i)
            if m is not None:
                sentinel = m.group(0)
                out.append(self.mapping.get(sentinel, sentinel))
                i = m.end()
                continue

            tail = buffer[i:]
            if len(tail) < MAX_SENTINEL_LEN and is_possible_sentinel_prefix(tail):
                self._hold = tail
                return "".join(out)

            out.append("<")
            i += 1

        self._hold = ""
        return "".join(out)

    def flush(self) -> str:
        """Emit anything still held back. Call after the upstream stream ends."""
        held = self._hold
        self._hold = ""
        return held

    def transform(
        self,
        chunks: Annotated[Iterable[str], "Synchronous chunk iterable."],
    ) -> Iterator[str]:
        """Apply ``feed`` to each chunk and emit non-empty outputs; flush at end."""
        for chunk in chunks:
            out = self.feed(chunk)
            if out:
                yield out
        tail = self.flush()
        if tail:
            yield tail

    async def transform_async(
        self,
        chunks: Annotated[AsyncIterable[str], "Async chunk iterable (e.g. async SSE)."],
    ) -> AsyncIterator[str]:
        """Async counterpart of ``transform``."""
        async for chunk in chunks:
            out = self.feed(chunk)
            if out:
                yield out
        tail = self.flush()
        if tail:
            yield tail
