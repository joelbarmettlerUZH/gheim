"""Smoke tests that don't require torch/transformers/datasets.

Validates the core data primitives: label space integrity, BIOES round-trip
through a stub offset-tokenizer, synthetic template rendering with span
alignment, and slot-bag verification.
"""
from __future__ import annotations

from gheim_training.data import bioes
# gheim_training.data.apertus removed; tests below stripped accordingly
from gheim_training.data.label_space import (
    CATEGORIES,
    ID2LABEL,
    LABEL2ID,
    NUM_LABELS,
    category_of,
    prefix_of,
)
from gheim_training.data.schema import Example, Span
from gheim_training.data.synth import faker_ch as F
from gheim_training.data.synth.docs import generate as syn_generate
from gheim_training.data.synth.docs.template import render


def test_label_space_shape() -> None:
    assert NUM_LABELS == 33
    assert len(CATEGORIES) == 8
    assert ID2LABEL[0] == "O"
    for cat in CATEGORIES:
        for p in ("B", "I", "E", "S"):
            assert f"{p}-{cat}" in LABEL2ID
    assert category_of("O") is None
    assert category_of("B-private_person") == "private_person"
    assert prefix_of("S-account_number") == "S"


# --- BIOES alignment ---

class _StubTok:
    """Whitespace tokenizer that emits ``offset_mapping`` (mimics a fast tokenizer).

    Produces ``[CLS]`` and ``[SEP]`` with offset (0, 0) so the encoding path
    exercises the special-token mask.
    """

    def __call__(self, text, return_offsets_mapping=True, truncation=True,
                 max_length=1024, return_attention_mask=True):
        offsets: list[tuple[int, int]] = [(0, 0)]  # CLS
        ids: list[int] = [0]
        i = 0
        while i < len(text):
            if text[i].isspace():
                i += 1
                continue
            j = i
            while j < len(text) and not text[j].isspace():
                j += 1
            offsets.append((i, j))
            ids.append(len(ids))
            i = j
        offsets.append((0, 0))  # SEP
        ids.append(len(ids))
        if max_length and len(ids) > max_length:
            ids = ids[:max_length]
            offsets = offsets[:max_length]
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "offset_mapping": offsets,
        }


def test_bioes_round_trip_single_token() -> None:
    text = "Hello Joel here"
    spans = [Span(start=6, end=10, label="private_person")]
    tok = _StubTok()
    enc = tok(text)
    labels = bioes.assign_bioes(enc["offset_mapping"], spans)
    # Whitespace tokens: [CLS] Hello Joel here [SEP]
    assert labels[0] == "O"  # CLS
    assert labels[1] == "O"  # Hello
    assert labels[2] == "S-private_person"
    assert labels[3] == "O"  # here
    assert labels[4] == "O"  # SEP


def test_bioes_round_trip_multi_token() -> None:
    text = "Joel Barmettler lives in Zurich"
    spans = [
        Span(start=0, end=15, label="private_person"),
        Span(start=25, end=31, label="private_address"),
    ]
    tok = _StubTok()
    enc = tok(text)
    labels = bioes.assign_bioes(enc["offset_mapping"], spans)
    # [CLS] Joel Barmettler lives in Zurich [SEP]
    assert labels[1] == "B-private_person"
    assert labels[2] == "E-private_person"
    assert labels[3] == "O"  # lives
    assert labels[4] == "O"  # in
    assert labels[5] == "S-private_address"


def test_bioes_decode_inverse() -> None:
    text = "Joel Barmettler lives in Zurich"
    spans_in = [
        Span(start=0, end=15, label="private_person"),
        Span(start=25, end=31, label="private_address"),
    ]
    tok = _StubTok()
    enc = tok(text)
    str_labels = bioes.assign_bioes(enc["offset_mapping"], spans_in)
    label_ids = [LABEL2ID[lab] for lab in str_labels]
    spans_out = bioes.decode_bioes_to_spans(text, enc["offset_mapping"], label_ids, ID2LABEL)
    assert sorted(spans_out, key=lambda s: s.start) == sorted(spans_in, key=lambda s: s.start)


def test_encode_example_masks_specials() -> None:
    ex = Example(
        text="Joel Barmettler lives in Zurich",
        spans=[Span(start=0, end=15, label="private_person")],
        language="en",
        source="synthetic",
    )
    enc = bioes.encode_example(ex, _StubTok())
    assert enc["labels"][0] == bioes.IGNORE_INDEX
    assert enc["labels"][-1] == bioes.IGNORE_INDEX
    assert enc["labels"][1] == LABEL2ID["B-private_person"]
    assert enc["labels"][2] == LABEL2ID["E-private_person"]


# --- Synthetic generator ---

def test_synthetic_generation_validates() -> None:
    F.seed_all(17)
    examples = syn_generate.generate(50)
    assert len(examples) == 50
    for ex in examples:
        ex.validate_offsets()
        for sp in ex.spans:
            surface = ex.text[sp.start:sp.end]
            assert surface.strip() == surface
            assert sp.label in CATEGORIES


def test_template_render_offsets() -> None:
    text, spans = render(
        "Hello {{name:private_person}}, your IBAN is {{iban:account_number}}.",
        {"name": lambda: "Joel", "iban": lambda: "CH9300762011623852957"},
    )
    assert text == "Hello Joel, your IBAN is CH9300762011623852957."
    assert spans[0] == Span(start=6, end=10, label="private_person")
    assert spans[1] == Span(start=25, end=46, label="account_number")


def test_template_unknown_category_rejected() -> None:
    import pytest
    with pytest.raises(ValueError):
        render("{{x:not_a_category}}", {"x": lambda: "v"})


# Apertus slot-verification tests removed when the apertus labeller
# was retired in favour of Gemma+Qwen+Nemotron. See git history for the
# original tests if needed.


# --- Faker checksum sanity ---

def test_iban_ch_checksum_valid() -> None:
    for _ in range(20):
        iban = F.iban_ch(spaced=False)
        assert iban.startswith("CH") and len(iban) == 21
        # Recompute mod-97 to verify check digits.
        rearranged = iban[4:] + iban[0:4]
        numeric = "".join(str(ord(c) - 55) if c.isalpha() else c for c in rearranged)
        assert int(numeric) % 97 == 1


def test_ahv_checksum_valid() -> None:
    for _ in range(20):
        a = F.ahv(dotted=False)
        assert a.startswith("756") and len(a) == 13
        digits = [int(d) for d in a]
        s = sum(d * (3 if i % 2 else 1) for i, d in enumerate(digits[:12]))
        check = (10 - (s % 10)) % 10
        assert digits[12] == check


def test_credit_card_luhn_valid() -> None:
    for _ in range(20):
        cc = F.credit_card(spaced=False)
        digits = [int(d) for d in cc]
        s = 0
        for i, d in enumerate(reversed(digits)):
            if i % 2 == 1:
                d *= 2
                if d > 9:
                    d -= 9
            s += d
        assert s % 10 == 0
