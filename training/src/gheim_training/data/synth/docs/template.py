"""Template engine: ``{{slot:category}}`` placeholders → text + char-aligned spans.

A template is a string with placeholders like ``{{name:private_person}}`` or
``{{iban:account_number}}``. ``render(template, fillers)`` replaces each
placeholder by calling ``fillers[name]()`` and returns the rendered text plus a
list of (start, end, category) tuples for each filled slot — the foundation
for the BIOES alignment downstream.
"""
from __future__ import annotations

import re
from collections.abc import Callable

from ...label_space import CATEGORIES
from ...schema import Span

_PLACEHOLDER_RE = re.compile(r"\{\{(?P<slot>[a-z_][a-z0-9_]*):(?P<cat>[a-z_]+)\}\}")


def render(
    template: str,
    fillers: dict[str, Callable[[], str]],
) -> tuple[str, list[Span]]:
    """Fill placeholders, returning (rendered_text, spans).

    Each placeholder ``{{slot:cat}}`` calls ``fillers[slot]()`` to produce its
    surface form. The slot can repeat in the template; each occurrence is
    filled independently (so two ``{{name:private_person}}`` placeholders
    produce two different names).
    """
    out: list[str] = []
    spans: list[Span] = []
    cursor = 0
    char_pos = 0
    for m in _PLACEHOLDER_RE.finditer(template):
        # Emit the literal chunk before this placeholder.
        literal = template[cursor:m.start()]
        out.append(literal)
        char_pos += len(literal)

        slot = m.group("slot")
        cat = m.group("cat")
        if cat not in CATEGORIES:
            raise ValueError(f"Unknown category {cat!r} in template placeholder {m.group(0)!r}")
        if slot not in fillers:
            raise KeyError(f"No filler registered for slot {slot!r} (placeholder {m.group(0)!r})")
        surface = fillers[slot]()
        if not surface or not surface.strip():
            raise ValueError(f"Filler for slot {slot!r} returned empty/whitespace value")
        out.append(surface)
        spans.append(Span(start=char_pos, end=char_pos + len(surface), label=cat))
        char_pos += len(surface)
        cursor = m.end()
    # trailing literal
    out.append(template[cursor:])
    return "".join(out), spans
