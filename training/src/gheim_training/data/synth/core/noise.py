"""Surface-form noise: realistic corruption applied to a fraction of chunks.

Production text has OCR artifacts, broken capitalisation, markdown
wrapping, non-breaking spaces, HTML entities, etc. Previous-iteration synthetic
chunks are *too clean* — every email is perfectly formatted. The model
overfits to that and falls apart on real messy input.

This module applies one of several noise kinds to a chunk's text,
**adjusting span offsets** so spans still point to the right characters
after corruption.

Noise kinds:
  - ``nbsp``        : replace some spaces with non-breaking spaces (U+00A0)
  - ``broken_caps`` : random char up-/down-casing on non-span tokens
  - ``no_space``    : drop a space between two words (concatenation)
  - ``markdown``    : wrap a span in **bold** or *italic*
  - ``html_entity`` : replace umlauts with &auml; etc.
  - ``stray_punct`` : insert a stray comma or dot
  - ``ocr_swap``    : single-char swap (e→c, o→0, l→1) on a non-span char

For training, we want to apply noise PROBABILISTICALLY (not all
chunks get noise — production input is mostly clean). Default
probability is 15% per chunk.

All noise kinds preserve span correctness — either the corruption is
applied OUTSIDE spans (so spans don't move), or it's applied INSIDE a
span with explicit offset adjustment.
"""
from __future__ import annotations

import random
from typing import Literal

from .schema import Chunk, Span

NoiseKind = Literal[
    "nbsp", "broken_caps", "no_space", "markdown",
    "html_entity", "stray_punct", "ocr_swap",
]

ALL_NOISE_KINDS: tuple[NoiseKind, ...] = (
    "nbsp", "broken_caps", "no_space", "markdown",
    "html_entity", "stray_punct", "ocr_swap",
)


def _in_any_span(idx: int, spans: list[Span]) -> bool:
    return any(sp.start <= idx < sp.end for sp in spans)


def _shift_spans_after(spans: list[Span], from_idx: int, delta: int
                       ) -> list[Span]:
    """Shift any spans starting after ``from_idx`` by ``delta``."""
    out: list[Span] = []
    for sp in spans:
        if sp.start >= from_idx:
            out.append(Span(start=sp.start + delta, end=sp.end + delta,
                            label=sp.label))
        elif sp.end > from_idx:
            # Span overlaps the modified position; extend the end.
            out.append(Span(start=sp.start, end=sp.end + delta, label=sp.label))
        else:
            out.append(sp)
    return out


def _apply_nbsp(chunk: Chunk, rng: random.Random) -> Chunk:
    """Replace ~half the spaces (only those OUTSIDE PII spans) with NBSP.
    Spans don't shift because we replace 1 char with 1 char."""
    out = list(chunk.text)
    for i, ch in enumerate(out):
        if ch == " " and not _in_any_span(i, chunk.spans) and rng.random() < 0.4:
            out[i] = " "
    return Chunk(id=chunk.id, text="".join(out), spans=chunk.spans,
                 language=chunk.language, source=chunk.source,
                 template_id=chunk.template_id,
                 meta={**chunk.meta, "noise": "nbsp"})


def _apply_broken_caps(chunk: Chunk, rng: random.Random) -> Chunk:
    """Randomly upper/lower a few non-span letters."""
    out = list(chunk.text)
    candidates = [i for i, ch in enumerate(out)
                  if ch.isalpha() and not _in_any_span(i, chunk.spans)]
    n = min(len(candidates), max(1, len(candidates) // 20))
    for i in rng.sample(candidates, n) if candidates else []:
        out[i] = out[i].upper() if out[i].islower() else out[i].lower()
    return Chunk(id=chunk.id, text="".join(out), spans=chunk.spans,
                 language=chunk.language, source=chunk.source,
                 template_id=chunk.template_id,
                 meta={**chunk.meta, "noise": "broken_caps"})


def _apply_no_space(chunk: Chunk, rng: random.Random) -> Chunk:
    """Drop one space outside spans (concatenates two words)."""
    candidates = [i for i, ch in enumerate(chunk.text)
                  if ch == " " and not _in_any_span(i, chunk.spans)
                  and not _in_any_span(i - 1, chunk.spans)
                  and not _in_any_span(i + 1, chunk.spans)]
    if not candidates:
        return chunk
    drop_idx = rng.choice(candidates)
    new_text = chunk.text[:drop_idx] + chunk.text[drop_idx + 1:]
    new_spans = _shift_spans_after(chunk.spans, drop_idx, -1)
    return Chunk(id=chunk.id, text=new_text, spans=new_spans,
                 language=chunk.language, source=chunk.source,
                 template_id=chunk.template_id,
                 meta={**chunk.meta, "noise": "no_space"})


def _apply_markdown(chunk: Chunk, rng: random.Random) -> Chunk:
    """Wrap one span in ** ** or * *. Span offsets shift accordingly."""
    if not chunk.spans:
        return chunk
    target_idx = rng.randrange(len(chunk.spans))
    target = chunk.spans[target_idx]
    wrapper = rng.choice(("**", "*", "_"))
    new_text = (chunk.text[:target.start]
                + wrapper + chunk.text[target.start:target.end] + wrapper
                + chunk.text[target.end:])
    # Span boundaries: opening wrapper inserted BEFORE the span (shifts span
    # by +len(wrapper)). Closing wrapper AFTER span end (no further shift
    # for THIS span, but shifts subsequent spans by +2*len(wrapper)).
    wrap_len = len(wrapper)
    new_spans: list[Span] = []
    for i, sp in enumerate(chunk.spans):
        if i < target_idx:
            new_spans.append(sp)
        elif i == target_idx:
            new_spans.append(Span(start=sp.start + wrap_len,
                                  end=sp.end + wrap_len,
                                  label=sp.label))
        else:  # i > target_idx
            new_spans.append(Span(start=sp.start + 2 * wrap_len,
                                  end=sp.end + 2 * wrap_len,
                                  label=sp.label))
    return Chunk(id=chunk.id, text=new_text, spans=new_spans,
                 language=chunk.language, source=chunk.source,
                 template_id=chunk.template_id,
                 meta={**chunk.meta, "noise": "markdown"})


_HTML_ENTITY_MAP = {
    "ä": "&auml;", "ö": "&ouml;", "ü": "&uuml;",
    "Ä": "&Auml;", "Ö": "&Ouml;", "Ü": "&Uuml;",
    "ß": "&szlig;", "é": "&eacute;", "è": "&egrave;",
    "à": "&agrave;", "ç": "&ccedil;",
}


def _apply_html_entity(chunk: Chunk, rng: random.Random) -> Chunk:
    """Replace one umlaut/accent (outside spans) with its HTML entity."""
    candidates = [i for i, ch in enumerate(chunk.text)
                  if ch in _HTML_ENTITY_MAP and not _in_any_span(i, chunk.spans)]
    if not candidates:
        return chunk
    target = rng.choice(candidates)
    replacement = _HTML_ENTITY_MAP[chunk.text[target]]
    delta = len(replacement) - 1
    new_text = chunk.text[:target] + replacement + chunk.text[target + 1:]
    new_spans = _shift_spans_after(chunk.spans, target + 1, delta)
    return Chunk(id=chunk.id, text=new_text, spans=new_spans,
                 language=chunk.language, source=chunk.source,
                 template_id=chunk.template_id,
                 meta={**chunk.meta, "noise": "html_entity"})


def _apply_stray_punct(chunk: Chunk, rng: random.Random) -> Chunk:
    """Insert a stray comma or dot outside any span."""
    candidates = [i for i, ch in enumerate(chunk.text)
                  if ch == " " and not _in_any_span(i, chunk.spans)
                  and not _in_any_span(i - 1, chunk.spans)]
    if not candidates:
        return chunk
    pos = rng.choice(candidates)
    insert = rng.choice((",", ";", "."))
    new_text = chunk.text[:pos] + insert + chunk.text[pos:]
    new_spans = _shift_spans_after(chunk.spans, pos, 1)
    return Chunk(id=chunk.id, text=new_text, spans=new_spans,
                 language=chunk.language, source=chunk.source,
                 template_id=chunk.template_id,
                 meta={**chunk.meta, "noise": "stray_punct"})


_OCR_SWAPS = {"e": "c", "o": "0", "l": "1", "I": "l", "B": "8",
              "S": "5", "Z": "2", "G": "6", "a": "@"}


def _apply_ocr_swap(chunk: Chunk, rng: random.Random) -> Chunk:
    """OCR-style char swap on a single non-span letter."""
    candidates = [i for i, ch in enumerate(chunk.text)
                  if ch in _OCR_SWAPS and not _in_any_span(i, chunk.spans)]
    if not candidates:
        return chunk
    target = rng.choice(candidates)
    out = list(chunk.text)
    out[target] = _OCR_SWAPS[out[target]]
    return Chunk(id=chunk.id, text="".join(out), spans=chunk.spans,
                 language=chunk.language, source=chunk.source,
                 template_id=chunk.template_id,
                 meta={**chunk.meta, "noise": "ocr_swap"})


_DISPATCH = {
    "nbsp": _apply_nbsp,
    "broken_caps": _apply_broken_caps,
    "no_space": _apply_no_space,
    "markdown": _apply_markdown,
    "html_entity": _apply_html_entity,
    "stray_punct": _apply_stray_punct,
    "ocr_swap": _apply_ocr_swap,
}


def apply_noise(chunk: Chunk, *, kind: NoiseKind | None = None,
                rng: random.Random | None = None) -> Chunk:
    """Apply one noise transformation. If ``kind`` is None, picks at
    random. Validates the output chunk before returning."""
    rng = rng or random
    if kind is None:
        kind = rng.choice(ALL_NOISE_KINDS)
    noisy = _DISPATCH[kind](chunk, rng)
    noisy.validate()  # catch bugs in offset adjustment immediately
    return noisy


def maybe_apply_noise(chunk: Chunk, *, probability: float = 0.15,
                      rng: random.Random | None = None) -> Chunk:
    """With probability ``probability``, apply random noise; else return
    the chunk unchanged."""
    rng = rng or random
    if rng.random() < probability:
        return apply_noise(chunk, rng=rng)
    return chunk
