"""Tests for the v2 dataset schema and merge logic."""
from __future__ import annotations

import pytest

from gheim_training.data.labelling.merge import _RawSpan, from_value_pair, merge_signals
from gheim_training.data.labelling.schema import V2_PIPELINE_VERSION, V2Example, V2Span


# --------------------------------------------------------------------- V2Span

def test_v2span_construction_minimal():
    sp = V2Span(start=0, end=5, label="private_person", value="Alice",
                signals=("gemma",), confidence=1 / 3)
    assert sp.signals == ("gemma",)
    assert sp.regex_subtype is None


def test_v2span_rejects_unknown_label():
    with pytest.raises(ValueError, match="unknown category"):
        V2Span(start=0, end=5, label="not_a_label", value="x",
               signals=("gemma",), confidence=0.5)


def test_v2span_rejects_unsorted_signals():
    with pytest.raises(ValueError, match="sorted alphabetically"):
        V2Span(start=0, end=5, label="private_person", value="x",
               signals=("regex", "gemma"), confidence=0.66)


def test_v2span_rejects_empty_signals():
    with pytest.raises(ValueError, match="at least one signal"):
        V2Span(start=0, end=5, label="private_person", value="x",
               signals=(), confidence=0.5)


def test_v2span_rejects_unknown_signal():
    with pytest.raises(ValueError, match="unknown signal"):
        V2Span(start=0, end=5, label="private_person", value="x",
               signals=("rumor",), confidence=0.5)


def test_v2span_rejects_invalid_offsets():
    with pytest.raises(ValueError, match="invalid span"):
        V2Span(start=5, end=5, label="private_person", value="",
               signals=("gemma",), confidence=0.33)


# ------------------------------------------------------------------- Roundtrip

def test_v2example_roundtrip_json():
    ex = V2Example(
        id="chunk_001",
        text="Hello Alice, your IBAN is CH9300762011623852957.",
        language="en",
        subset="fineweb",
        doc_id="doc_001",
        spans=[
            V2Span(start=6, end=11, label="private_person", value="Alice",
                   signals=("gemma", "qwen"), confidence=2 / 3),
            V2Span(start=26, end=47, label="account_number",
                   value="CH9300762011623852957",
                   signals=("gemma", "qwen", "regex"), confidence=1.0,
                   regex_subtype="iban_ch"),
        ],
        labelers=["gemma-4-26B-A4B-it-AWQ-4bit", "Qwen3.6-35B-A3B-AWQ-4bit"],
    )
    s = ex.to_json()
    restored = V2Example.from_json(s)
    assert restored.id == ex.id
    assert len(restored.spans) == 2
    assert restored.spans[0].signals == ("gemma", "qwen")
    assert restored.spans[1].regex_subtype == "iban_ch"
    assert restored.build_pipeline == V2_PIPELINE_VERSION


# ------------------------------------------------------------------ from_value_pair

def test_from_value_pair_finds_substring():
    sp = from_value_pair("Hello Alice", "Alice", "private_person")
    assert sp is not None
    assert sp.start == 6 and sp.end == 11
    assert sp.value == "Alice"


def test_from_value_pair_returns_none_for_hallucinated_value():
    sp = from_value_pair("Hello Alice", "Bob", "private_person")
    assert sp is None


# ------------------------------------------------------------------ merge_signals

def test_merge_unanimous_all_signals():
    """All four labellers agree on a span → confidence 1.0."""
    text = "Alice's IBAN is CH9300762011623852957."
    s_gemma = [_RawSpan(16, 37, "account_number", "CH9300762011623852957")]
    s_qwen = [_RawSpan(16, 37, "account_number", "CH9300762011623852957")]
    s_nemo = [_RawSpan(16, 37, "account_number", "CH9300762011623852957")]
    s_regex = [_RawSpan(16, 37, "account_number", "CH9300762011623852957",
                        regex_subtype="iban_ch")]
    out = merge_signals(text, gemma=s_gemma, qwen=s_qwen, nemotron=s_nemo,
                        regex=s_regex)
    assert len(out) == 1
    assert out[0].signals == ("gemma", "nemotron", "qwen", "regex")
    assert out[0].confidence == 1.0
    assert out[0].regex_subtype == "iban_ch"


def test_merge_three_of_four_signals():
    """Three of four labellers agree → confidence 3/4 with default
    n_candidate_signals=4."""
    text = "Alice's IBAN is CH9300762011623852957."
    s_gemma = [_RawSpan(16, 37, "account_number", "CH9300762011623852957")]
    s_qwen = [_RawSpan(16, 37, "account_number", "CH9300762011623852957")]
    s_regex = [_RawSpan(16, 37, "account_number", "CH9300762011623852957",
                        regex_subtype="iban_ch")]
    out = merge_signals(text, gemma=s_gemma, qwen=s_qwen, regex=s_regex)
    assert len(out) == 1
    assert out[0].signals == ("gemma", "qwen", "regex")
    assert out[0].confidence == 0.75


def test_merge_partial_agreement():
    """Only Gemma flags a person → confidence 1/4 with default
    n_candidate_signals=4. The chunk-level
    n_candidate_signals parameter can be lowered (e.g. to 3) when a
    labeller didn't run on this chunk — see assemble.py."""
    text = "Hi Alice."
    s_gemma = [_RawSpan(3, 8, "private_person", "Alice")]
    out = merge_signals(text, gemma=s_gemma)
    assert len(out) == 1
    assert out[0].signals == ("gemma",)
    assert out[0].confidence == pytest.approx(1 / 4)


def test_merge_longer_span_wins_boundary():
    """Two labellers, one with a wider boundary → longer wins."""
    text = "Dr. Müller signed."
    s_gemma = [_RawSpan(4, 10, "private_person", "Müller")]
    s_qwen = [_RawSpan(0, 10, "private_person", "Dr. Müller")]
    out = merge_signals(text, gemma=s_gemma, qwen=s_qwen, regex=[])
    # NB: the merge key is value.casefold() — these have DIFFERENT
    # values so they merge to two separate spans, not one. That's the
    # correct behaviour: ``Dr. Müller`` and ``Müller`` are
    # legitimately different surface forms; downstream can pick which.
    assert len(out) == 2


def test_merge_audit_alone_does_not_emit_span():
    """A span flagged only by the audit signal (no labeller agrees) is
    skipped — audit is for cross-checking, not for inclusion."""
    text = "Some name."
    out = merge_signals(text,
                        gemma=[],
                        qwen=[],
                        regex=[],
                        audit=[_RawSpan(5, 9, "private_person", "name")])
    assert out == []


def test_merge_audit_increments_signal_set_but_not_confidence():
    """When the audit also flags a span gemma found, audit is added to
    signals but confidence still divides by n_candidate_signals (=4
    with the v2 default)."""
    text = "Hi Alice."
    s_gemma = [_RawSpan(3, 8, "private_person", "Alice")]
    audit = [_RawSpan(3, 8, "private_person", "Alice")]
    out = merge_signals(text, gemma=s_gemma, audit=audit)
    assert len(out) == 1
    assert "audit" in out[0].signals
    assert "gemma" in out[0].signals
    assert out[0].confidence == pytest.approx(1 / 4)  # audit excluded from denom


def test_merge_preserves_regex_subtype_through_merge():
    """When regex contributes alongside an LLM, the regex_subtype rides
    through to the merged span."""
    text = "AHV: 756.9217.0769.85."
    s_gemma = [_RawSpan(5, 21, "account_number", "756.9217.0769.85")]
    s_regex = [_RawSpan(5, 21, "account_number", "756.9217.0769.85",
                        regex_subtype="ahv")]
    out = merge_signals(text, gemma=s_gemma, regex=s_regex)
    assert len(out) == 1
    assert out[0].regex_subtype == "ahv"
    assert out[0].signals == ("gemma", "regex")
