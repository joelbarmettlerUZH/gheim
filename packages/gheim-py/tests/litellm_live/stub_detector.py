"""Deterministic detector for the live LiteLLM test (no model download needed)."""
import re

from gheim.detectors import Span

_PATTERNS = [
    ("private_person", re.compile(r'Joel "JB" Barmettler|Joel Barmettler|Anna Meier|Joel')),
    ("private_email", re.compile(r"[\w.]+@[\w.]+\.\w+")),
]


class StubDetector:
    def detect(self, text: str, *, model: str | None = None) -> list[Span]:
        spans = []
        for label, rx in _PATTERNS:
            for m in rx.finditer(text):
                spans.append(Span(label=label, start=m.start(), end=m.end(), text=m.group(0)))
        spans.sort(key=lambda s: (s.start, -(s.end - s.start)))
        out: list[Span] = []
        for s in spans:  # drop overlaps ("Joel" inside "Joel Barmettler")
            if not out or s.start >= out[-1].end:
                out.append(s)
        return out


def make() -> StubDetector:
    return StubDetector()
