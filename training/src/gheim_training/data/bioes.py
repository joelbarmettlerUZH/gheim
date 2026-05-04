"""Char-span → token BIOES alignment using a fast tokenizer's offset mapping.

The Trainer expects per-token integer label IDs. We tokenize with
``return_offsets_mapping=True`` and assign each token the BIOES label of the
span it falls inside. Tokens that overlap a span boundary inherit the span's
category; tokens fully outside any span get ``O``. Special tokens (offset
``(0, 0)`` from a fast tokenizer) get ``-100`` so HF cross-entropy ignores them.

This is the alignment used by every data layer (synthetic, AI4Privacy,
Apertus, English anchor) so behaviour is consistent.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any

from .label_space import LABEL2ID, OUTSIDE
from .schema import Example, Span

IGNORE_INDEX = -100


def _tokens_in_span(
    offsets: Sequence[tuple[int, int]],
    span: Span,
) -> list[int]:
    """Indices of tokens whose char range overlaps ``span``.

    A token (a, b) belongs to the span if it overlaps [span.start, span.end).
    Special tokens are filtered out (offset (0, 0) when produced by a fast
    tokenizer for [CLS]/[SEP]/etc., assuming the real text has non-zero length).
    """
    out: list[int] = []
    for i, (a, b) in enumerate(offsets):
        if a == 0 and b == 0:  # special token
            continue
        if a >= span.end:
            break
        if b <= span.start:
            continue
        out.append(i)
    return out


def assign_bioes(
    offsets: Sequence[tuple[int, int]],
    spans: Sequence[Span],
) -> list[str]:
    """Return one BIOES label per offset entry. Special tokens get ``O`` here;
    the caller is responsible for masking them to ``-100`` via ``encode_example``."""
    n = len(offsets)
    labels = [OUTSIDE] * n
    sorted_spans = sorted(spans, key=lambda s: s.start)
    for sp in sorted_spans:
        tok_idxs = _tokens_in_span(offsets, sp)
        if not tok_idxs:
            continue  # span fell entirely inside whitespace stripped by tokenizer
        if len(tok_idxs) == 1:
            labels[tok_idxs[0]] = f"S-{sp.label}"
        else:
            labels[tok_idxs[0]] = f"B-{sp.label}"
            for j in tok_idxs[1:-1]:
                labels[j] = f"I-{sp.label}"
            labels[tok_idxs[-1]] = f"E-{sp.label}"
    return labels


def encode_example(
    example: Example,
    tokenizer: Any,
    *,
    max_length: int = 1024,
) -> dict[str, list[int]]:
    """Tokenize one Example into model inputs + integer label IDs.

    Returns the standard HF dict (``input_ids``, ``attention_mask``, ``labels``).
    Special tokens are masked to ``-100``.
    """
    enc = tokenizer(
        example.text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
    )
    offsets: list[tuple[int, int]] = enc.pop("offset_mapping")
    str_labels = assign_bioes(offsets, example.spans)
    label_ids: list[int] = []
    for (a, b), lab in zip(offsets, str_labels, strict=True):
        if a == 0 and b == 0:  # special token
            label_ids.append(IGNORE_INDEX)
        else:
            label_ids.append(LABEL2ID[lab])
    enc["labels"] = label_ids
    return enc


def decode_bioes_to_spans(
    text: str,
    offsets: Sequence[tuple[int, int]],
    label_ids: Sequence[int],
    id2label: dict[int, str],
) -> list[Span]:
    """Inverse of assign_bioes: greedy span reconstruction from per-token labels.

    Used by the eval harness to turn model predictions back into char spans for
    span-level F1.
    """
    spans: list[Span] = []
    cur_cat: str | None = None
    cur_start: int | None = None
    cur_end: int | None = None

    def flush() -> None:
        nonlocal cur_cat, cur_start, cur_end
        if cur_cat is not None and cur_start is not None and cur_end is not None:
            spans.append(Span(start=cur_start, end=cur_end, label=cur_cat))
        cur_cat = None
        cur_start = None
        cur_end = None

    for (a, b), lid in zip(offsets, label_ids, strict=True):
        if a == 0 and b == 0:
            continue
        lab = id2label.get(int(lid), OUTSIDE)
        if lab == OUTSIDE:
            flush()
            continue
        prefix, cat = lab.split("-", 1)
        if prefix == "S":
            flush()
            spans.append(Span(start=a, end=b, label=cat))
        elif prefix == "B":
            flush()
            cur_cat = cat
            cur_start = a
            cur_end = b
        elif prefix == "I":
            if cur_cat == cat:
                cur_end = b
            else:  # malformed: I without matching B; start a new span anyway
                flush()
                cur_cat = cat
                cur_start = a
                cur_end = b
        elif prefix == "E":
            if cur_cat == cat and cur_start is not None:
                cur_end = b
                spans.append(Span(start=cur_start, end=cur_end, label=cat))
                cur_cat = None
                cur_start = None
                cur_end = None
            else:  # malformed: standalone E, treat as S
                flush()
                spans.append(Span(start=a, end=b, label=cat))
    flush()
    # Trim whitespace at edges (matches gheim's LocalDetector behaviour)
    out: list[Span] = []
    for sp in spans:
        s, e = sp.start, sp.end
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            out.append(Span(start=s, end=e, label=sp.label))
    return out
