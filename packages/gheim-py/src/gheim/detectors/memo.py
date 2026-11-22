"""MemoDetector — remember detection results, so repeated text is detected once.

A chat client sends its whole history on every turn, and with a restoring
proxy that history is plain text again, so every turn would re-detect every
earlier message: the cost of a turn grows with the conversation. Wrapping the
detector in a memo makes each turn pay only for the text that is new.

The memo is keyed by a digest of the text (and the model id), never the text
itself, and stores span offsets, labels and scores; the surface is cut from the
caller's text again on a hit. So it holds no PII beyond what offsets reveal.
A detection that raises is not remembered, so the next call asks again.
"""
from __future__ import annotations

import hashlib
import inspect
import threading
from collections import OrderedDict
from typing import Annotated

from .base import Detector, Span

_Entry = tuple[tuple[str, int, int, float], ...]


def _takes_model(detector: Detector) -> bool:
    try:
        return "model" in inspect.signature(detector.detect).parameters
    except (TypeError, ValueError):
        return True


class MemoDetector:
    """Bounded, thread-safe LRU memo around another detector."""

    def __init__(
        self,
        detector: Annotated[Detector, "The detector whose results are remembered."],
        max_entries: Annotated[int, "Distinct texts remembered; least recently used go first."] = 10_000,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self.detector = detector
        self.max_entries = max_entries
        self._takes_model = _takes_model(detector)
        self._entries: OrderedDict[bytes, _Entry] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _key(text: str, model: str | None) -> bytes:
        digest = hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=20)
        digest.update(b"\0" + (model or "").encode())
        return digest.digest()

    def detect(self, text: str, *, model: str | None = None) -> list[Span]:
        key = self._key(text, model)
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
        if entry is None:
            spans = (self.detector.detect(text, model=model) if self._takes_model
                     else self.detector.detect(text))
            entry = tuple((s.label, s.start, s.end, s.score) for s in spans)
            with self._lock:
                self._entries[key] = entry
                self._entries.move_to_end(key)
                while len(self._entries) > self.max_entries:
                    self._entries.popitem(last=False)
        return [Span(label, start, end, text[start:end], score) for label, start, end, score in entry]
