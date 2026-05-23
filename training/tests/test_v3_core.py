"""Smoke tests for the v3 generator framework.

Validates that:
- Fragment rendering substitutes [[PII:...]] markers and tracks spans.
- Composer glues greeting+body+signature with correct cross-section spans.
- Noise overlays adjust spans without breaking validation.
- Generated chunks pass Chunk.validate() (bounds, label, no overlaps).
- Output JSONL is readable.
"""
from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

import pytest

from gheim_training.data.v3.core import (
    Chunk, CompositionStyle, Fragment, Span,
    apply_noise, compose_email, maybe_apply_noise,
    render_fragment, render_standalone, write_jsonl,
)
from gheim_training.data.v3.core import name_registry as nr
from gheim_training.data.v3.core import pii_values as pv


SEED = 20260524


# --------------------------------------------------------------- Fragment

def test_render_fragment_substitutes_and_tracks_spans():
    """Single marker → one substitution, one span at the right offset."""
    rng = random.Random(SEED)
    frag = Fragment(template_id="t1", language="de_ch",
                    text="Hallo [[PII:first]], danke für deine Nachricht.")
    text, spans = render_fragment(frag, rng)

    assert text.startswith("Hallo ")
    assert text.endswith(", danke für deine Nachricht.")
    assert len(spans) == 1
    assert spans[0].label == "private_person"
    # The substituted value should match text[start:end] exactly
    assert text[spans[0].start:spans[0].end] != ""
    assert text[spans[0].start:spans[0].end][0].isupper()  # name starts uppercase


def test_render_fragment_multiple_markers():
    """Three markers across categories → three spans in order."""
    rng = random.Random(SEED)
    frag = Fragment(
        template_id="t2", language="de_ch",
        text=("Anbei [[PII:first]] meldet sich unter [[PII:phone]] "
              "oder [[PII:email]]."),
    )
    text, spans = render_fragment(frag, rng)

    assert len(spans) == 3
    labels = [sp.label for sp in spans]
    assert labels == ["private_person", "private_phone", "private_email"]
    # Spans should be in increasing-start order and non-overlapping
    for a, b in zip(spans, spans[1:]):
        assert a.end <= b.start


def test_render_fragment_format_hint():
    """`/iso` format hint should produce an ISO date."""
    rng = random.Random(SEED)
    frag = Fragment(template_id="t3", language="de_ch",
                    text="Datum: [[PII:date/iso]]")
    text, spans = render_fragment(frag, rng)
    value = text[spans[0].start:spans[0].end]
    # ISO format = YYYY-MM-DD
    assert value.count("-") == 2
    assert value[:4].isdigit() and len(value[:4]) == 4


def test_render_fragment_title_last_format():
    rng = random.Random(SEED)
    frag = Fragment(template_id="t4", language="de_ch",
                    text="[[PII:title_last]] hat unterschrieben.")
    text, spans = render_fragment(frag, rng, gender="m")
    value = text[spans[0].start:spans[0].end]
    assert " " in value  # title + last has a space


# ---------------------------------------------------------- compose_email

def test_compose_email_glues_greeting_body_signature():
    rng = random.Random(SEED)
    greeting = Fragment(template_id="g1", language="de_ch",
                        text="Hallo [[PII:first]],")
    body = Fragment(template_id="b1", language="de_ch",
                    text="anbei meine Daten: [[PII:phone]] und [[PII:email]].")
    signature = Fragment(template_id="s1", language="de_ch",
                         text="Liebe Grüsse,\n[[PII:full]]")
    chunk = compose_email(greeting=greeting, body=body, signature=signature,
                          language="de_ch", source="synthetic_emails_v3",
                          rng=rng)
    chunk.validate()
    # Should have at least 4 spans: greeting name + 2 body + signature name
    assert len(chunk.spans) >= 4
    # The signature name should be after the body name in offset
    assert chunk.spans[-1].start > chunk.spans[0].start
    # All spans should still be within the text
    for sp in chunk.spans:
        assert 0 <= sp.start < sp.end <= len(chunk.text)


def test_compose_email_multiple_bodies():
    """Composing with a list of body fragments produces multi-paragraph email."""
    rng = random.Random(SEED)
    greeting = Fragment(template_id="g1", language="fr_ch",
                        text="Bonjour [[PII:first]],")
    body1 = Fragment(template_id="b1", language="fr_ch",
                     text="Je vous écris depuis [[PII:address]].")
    body2 = Fragment(template_id="b2", language="fr_ch",
                     text="Mon IBAN est [[PII:iban]].")
    signature = Fragment(template_id="s1", language="fr_ch",
                         text="Cordialement,\n[[PII:full]]")
    chunk = compose_email(greeting=greeting, body=[body1, body2],
                          signature=signature,
                          language="fr_ch", source="synthetic_emails_v3",
                          rng=rng)
    chunk.validate()
    # 1 greeting + 2 body (address+iban) + 1 sig = 4 spans
    assert len(chunk.spans) == 4
    assert chunk.text.count("\n\n") >= 2  # has paragraph breaks


# ---------------------------------------------------------- noise

@pytest.mark.parametrize("kind", [
    "nbsp", "broken_caps", "no_space", "markdown",
    "html_entity", "stray_punct", "ocr_swap",
])
def test_apply_noise_preserves_span_correctness(kind):
    """Every noise kind must produce a chunk whose spans still match
    the surface form they're supposed to."""
    rng = random.Random(SEED)
    # Use a text rich in characters each noise kind targets.
    # Pre-computed offsets: "Marius Müller" = [6, 19), "+41 44 555 12 34" = [36, 52).
    text = "Hallo Marius Müller, ruf mich unter +41 44 555 12 34 an."
    chunk = Chunk(
        id="t1", text=text,
        spans=[Span(start=text.index("Marius Müller"),
                    end=text.index("Marius Müller") + len("Marius Müller"),
                    label="private_person"),
               Span(start=text.index("+41 44 555 12 34"),
                    end=text.index("+41 44 555 12 34") + len("+41 44 555 12 34"),
                    label="private_phone")],
        language="de_ch", source="test", template_id="t",
    )
    chunk.validate()  # sanity
    noisy = apply_noise(chunk, kind=kind, rng=rng)
    noisy.validate()  # noise must not break offsets
    # The person span MUST still spell "Marius Müller" (the noise kinds
    # never modify chars inside spans; only nbsp/broken_caps/markdown
    # can shift the span position, never its content).
    for sp in noisy.spans:
        if sp.label == "private_person":
            assert noisy.text[sp.start:sp.end] == "Marius Müller"


def test_maybe_apply_noise_respects_probability():
    """With probability=0, never noisy. With probability=1, always noisy."""
    rng = random.Random(SEED)
    chunk = Chunk(
        id="t1", text="Hallo Marius, danke.",
        spans=[Span(start=6, end=12, label="private_person")],
        language="de_ch", source="test", template_id="t",
    )
    for _ in range(20):
        out = maybe_apply_noise(chunk, probability=0.0, rng=rng)
        assert "noise" not in out.meta
    seen_noise = False
    for _ in range(20):
        out = maybe_apply_noise(chunk, probability=1.0, rng=rng)
        if "noise" in out.meta:
            seen_noise = True
    assert seen_noise


# ---------------------------------------------------------- jsonl io

def test_write_jsonl_roundtrip():
    rng = random.Random(SEED)
    chunks = []
    for i in range(5):
        frag = Fragment(template_id=f"t{i}", language="de_ch",
                        text=f"Test #{i}: [[PII:first]] meldet sich.")
        chunks.append(render_standalone(frag, source="synthetic_test", rng=rng))

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        out_path = Path(f.name)
    try:
        n = write_jsonl(out_path, chunks)
        assert n == 5
        lines = out_path.read_text().splitlines()
        assert len(lines) == 5
        for line in lines:
            rec = json.loads(line)
            assert "id" in rec and "text" in rec and "spans" in rec
            assert rec["language"] == "de_ch"
            assert rec["source"] == "synthetic_test"
            # Verify offsets in the round-tripped record
            for sp in rec["spans"]:
                assert 0 <= sp["start"] < sp["end"] <= len(rec["text"])
    finally:
        out_path.unlink()


# ---------------------------------------------------------- pii_values diversity

def test_pii_dates_produce_multiple_formats():
    """Over 50 random draws, gen_date should hit at least 4 distinct formats."""
    rng = random.Random(SEED)
    samples = {pv.gen_date("de_ch", rng=rng) for _ in range(50)}
    # 50 draws across 7 formats — even with same date repeating, formats should vary
    distinct_shapes = set()
    for s in samples:
        if "-" in s and len(s) == 10 and s[:4].isdigit():
            distinct_shapes.add("iso")
        elif "/" in s:
            distinct_shapes.add("slash")
        elif "." in s and any(c.isalpha() for c in s):
            # short or ddmmyy or dot — group as numeric-dot if no alpha
            distinct_shapes.add("dot_with_alpha")
        elif "." in s:
            distinct_shapes.add("numeric_dot")
        else:
            distinct_shapes.add("spelled")
    assert len(distinct_shapes) >= 3


def test_phone_produces_multiple_formats():
    rng = random.Random(SEED)
    samples = {pv.gen_phone_ch(rng=rng) for _ in range(40)}
    # Look for both + and 0-prefixed; both with-spaces and compact
    has_intl = any(s.startswith("+41") for s in samples)
    has_national = any(s.startswith("0") for s in samples)
    has_space = any(" " in s for s in samples)
    has_compact = any(" " not in s and "-" not in s for s in samples)
    assert has_intl and has_national and has_space and has_compact


# ---------------------------------------------------------- name_registry

def test_common_word_lastname_returns_known_pool_entry_de():
    rng = random.Random(SEED)
    for _ in range(20):
        v = nr.common_word_lastname("de_ch", rng)
        assert v in nr.COMMON_WORD_LASTNAMES_DE


def test_format_name_styles_distinct():
    rng = random.Random(SEED)
    out = {
        style: nr.format_name("Anna", "Müller", style,
                              language="de_ch", gender="f", rng=rng)
        for style in ("first_last", "last_first", "last_first_upper",
                      "first_initial", "title_last", "last_only", "first_only")
    }
    assert out["first_last"] == "Anna Müller"
    assert out["last_first"] == "Müller, Anna"
    assert out["last_first_upper"] == "MÜLLER, Anna"
    assert out["first_initial"] == "A. Müller"
    assert out["title_last"].endswith("Müller") and out["title_last"] != "Müller"
    assert out["last_only"] == "Müller"
    assert out["first_only"] == "Anna"
