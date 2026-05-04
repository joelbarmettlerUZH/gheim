"""Unit tests for the Layer 5 stub (apertus_label).

Covers the deterministic parts: chunking, prefilter, JSON parsing, verify +
combine-with-regex. The actual Apertus call is mocked.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from gheim_training.data.apertus_label import prefilter
from gheim_training.data.apertus_label.label import _parse_claims
from gheim_training.data.apertus_label.stream import _chunk_text, _classify
from gheim_training.data.apertus_label.verify import ApertusClaim, verify_and_combine

# --- chunking ---

def test_chunk_text_respects_target_window() -> None:
    text = ". ".join(f"Sentence number {i} with some filler words here" for i in range(60)) + "."
    chunks = list(_chunk_text(text, target_min=200, target_max=500, hard_min=50))
    assert chunks
    for c in chunks[:-1]:
        assert 200 <= len(c) <= 600  # last chunk may be smaller


def test_chunk_text_drops_tiny_trailing() -> None:
    chunks = list(_chunk_text("short text.", target_min=400, target_max=800, hard_min=200))
    assert chunks == []


# --- subset classifier ---

@pytest.mark.parametrize("rec_id,expected", [
    ("/entscheidsuche_html/filtered/documents_0046.jsonl.gz/0", "entscheidsuche"),
    ("/curia_vista/2024-spring/0042", "curia_vista"),
    ("/parlament/sessions/2023/123", "curia_vista"),
    ("/fineweb-2-swiss/dump-2024-10/000", "fineweb"),
    ("/romansh_corpus/articles/0001", "romansh"),
    ("/some/random/path", None),
])
def test_classify_dispatches_by_id(rec_id: str, expected: str | None) -> None:
    rec = {"id": rec_id, "metadata": {"file_path": rec_id}, "text": "x"}
    assert _classify(rec) == expected


# --- regex prefilter ---

def test_prefilter_passes_chunk_with_iban() -> None:
    text = "Die Beschwerdeführerin hat die IBAN CH93 0076 2011 6238 5295 7 angegeben."
    assert prefilter.passes(text)


def test_prefilter_rejects_pure_procedural_text() -> None:
    text = "Das Bundesgericht hat die Beschwerde abgewiesen und die Sache zur Neubeurteilung an die Vorinstanz zurückgewiesen."
    assert not prefilter.passes(text)


def test_find_structured_returns_non_overlapping() -> None:
    text = (
        "Versicherter: AHV 756.1234.5678.97, Tel. 044 668 18 00, "
        "E-Mail abc@example.ch, IBAN CH93 0076 2011 6238 5295 7, geboren 12.03.1985."
    )
    hits = prefilter.find_structured(text)
    labels = sorted({h.label for h in hits})
    assert "account_number" in labels
    assert "private_phone" in labels
    assert "private_email" in labels
    assert "private_date" in labels
    # Non-overlap invariant
    last = -1
    for h in sorted(hits, key=lambda h: h.start):
        assert h.start >= last
        last = h.end


# --- JSON parsing tolerance ---

def test_parse_claims_plain_json() -> None:
    out = _parse_claims('[{"value":"Anna","label":"private_person"},{"value":"Bern","label":"private_address"}]')
    assert len(out) == 2
    assert out[0].value == "Anna"
    assert out[1].label == "private_address"


def test_parse_claims_with_markdown_fence() -> None:
    out = _parse_claims('```json\n[{"value":"X","label":"private_person"}]\n```')
    assert out and out[0].value == "X"


def test_parse_claims_with_trailing_commentary() -> None:
    out = _parse_claims(
        '[{"value":"Maria","label":"private_person"}] note: this is a single match'
    )
    assert out and out[0].value == "Maria"


def test_parse_claims_handles_garbage() -> None:
    assert _parse_claims("nope, no JSON here at all") == []
    assert _parse_claims("") == []
    assert _parse_claims("```json\nthis is not valid JSON\n```") == []


# --- verify_and_combine ---

def test_verify_drops_hallucinated_value() -> None:
    chunk = "Die Beschwerdeführerin Anna Müller wohnt in Zürich."
    claims = [
        ApertusClaim("Anna Müller", "private_person"),
        ApertusClaim("Beat Brunner", "private_person"),  # not in text
        ApertusClaim("Zürich", "private_address"),
    ]
    ex = verify_and_combine(chunk, claims, language="de_ch", doc_id="doc1", subset="entscheidsuche")
    assert ex is not None
    surfaces = {ex.text[s.start:s.end] for s in ex.spans}
    assert "Anna Müller" in surfaces
    assert "Zürich" in surfaces
    assert "Beat Brunner" not in surfaces


def test_verify_regex_overrides_apertus_overlap() -> None:
    chunk = "Tel. 044 668 18 00 erreichbar."
    # Apertus mistakenly tags the phone as a person; regex should win.
    claims = [ApertusClaim("044 668 18 00", "private_person")]
    ex = verify_and_combine(chunk, claims, language="de_ch", doc_id="doc2", subset="entscheidsuche")
    assert ex is not None
    spans_by_label = [s.label for s in ex.spans]
    # Regex should claim it as private_phone, Apertus's person claim is dropped.
    assert "private_phone" in spans_by_label
    assert "private_person" not in spans_by_label


def test_verify_drops_blocklisted_person() -> None:
    chunk = "Der Beschwerdeführer hat einen Antrag gestellt."
    claims = [ApertusClaim("Der", "private_person")]
    ex = verify_and_combine(chunk, claims, language="de_ch", doc_id="d", subset="entscheidsuche")
    # No regex hits, no valid Apertus claims → None
    assert ex is None


def test_verify_combines_apertus_and_regex() -> None:
    chunk = "Anna Müller, geboren 12.03.1985, AHV 756.1234.5678.97."
    claims = [
        ApertusClaim("Anna Müller", "private_person"),
        ApertusClaim("12.03.1985", "private_date"),
    ]
    ex = verify_and_combine(chunk, claims, language="de_ch", doc_id="d", subset="entscheidsuche")
    assert ex is not None
    labels = {s.label for s in ex.spans}
    assert {"private_person", "private_date", "account_number"} <= labels


def test_verify_returns_none_on_empty_input() -> None:
    chunk = "Procedural boilerplate with no PII at all."
    ex = verify_and_combine(chunk, [], language="de_ch", doc_id="d", subset="entscheidsuche")
    assert ex is None


# --- audit gate ---

def test_audit_gate_passes_on_synthetic_perfect_match(tmp_path: Path) -> None:
    """If Apertus output exactly matches the gold, all gate criteria pass."""
    import json

    from gheim_training.data.apertus_label.audit import audit, check_gate

    gold = [
        {
            "text": "Anna Müller wohnt in Zürich.",
            "spans": [
                {"start": 0, "end": 11, "label": "private_person"},
                {"start": 21, "end": 27, "label": "private_address"},
            ],
            "language": "de_ch", "source": "real", "template_id": "t",
        },
        {
            "text": "Geboren am 12.03.1985.",
            "spans": [{"start": 11, "end": 21, "label": "private_date"}],
            "language": "de_ch", "source": "real", "template_id": "t",
        },
    ]
    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text("\n".join(json.dumps(g) for g in gold), encoding="utf-8")
    apertus_path = tmp_path / "apertus.jsonl"
    apertus_path.write_text("\n".join(json.dumps(g) for g in gold), encoding="utf-8")

    report = audit(apertus_path, gold_path)
    passed, failures = check_gate(report)
    assert passed, f"perfect match should pass gate, got failures: {failures}"
    for label in ("private_person", "private_address", "private_date"):
        assert report[label]["f1"] == 1.0


# --- M2 ALL-CAPS surname regex ---

def test_m2_tags_academic_citation_surnames() -> None:
    text = "Vgl. ROLAND PFÄFFLI, Neuerungen im Immobiliarsachenrecht."
    hits = prefilter.find_structured(text)
    persons = [h for h in hits if h.label == "private_person"]
    assert any(h.text == "ROLAND PFÄFFLI" for h in persons), hits


def test_m2_splits_slash_separated_coauthors() -> None:
    text = "STAEHELIN/GROLIMUND, Zivilprozessrecht."
    hits = prefilter.find_structured(text)
    persons = [h.text for h in hits if h.label == "private_person"]
    assert "STAEHELIN" in persons
    assert "GROLIMUND" in persons


def test_m2_blocks_institution_acronyms() -> None:
    text = "Die UBS AG hat MWST-Nr. CHE-123.456.789."
    hits = prefilter.find_structured(text)
    persons = [h for h in hits if h.label == "private_person"]
    assert persons == [], f"unexpected person hits on institution acronyms: {persons}"


def test_m2_handles_van_den_berg() -> None:
    text = "Cf. VAN DEN BERG, op. cit."
    hits = prefilter.find_structured(text)
    persons = [h for h in hits if h.label == "private_person"]
    assert any("VAN DEN BERG" in h.text for h in persons), hits


def test_m2_handles_pierre_a_karrer() -> None:
    text = "PIERRE A. KARRER, International Arbitration."
    hits = prefilter.find_structured(text)
    persons = [h for h in hits if h.label == "private_person"]
    assert any("KARRER" in h.text for h in persons), hits


# --- M3 date filter ---

def test_m3_accepts_real_dates() -> None:
    from gheim_training.data.apertus_label.verify import _date_ok
    assert _date_ok("12.03.1985")
    assert _date_ok("5. Juni 2012")
    assert _date_ok("29 ottobre 2001")
    assert _date_ok("Januar 2024")
    assert _date_ok("2024-03-15")


def test_m3_rejects_journal_refs_and_non_dates() -> None:
    from gheim_training.data.apertus_label.verify import _date_ok
    assert not _date_ok("11/1993")
    assert not _date_ok("74/1978")
    assert not _date_ok("1985")
    assert not _date_ok("im Jahr 2024")
    assert not _date_ok("Art. 12")
    assert not _date_ok("08:32 Uhr")


def test_m3_drops_apertus_journal_ref_date_claims() -> None:
    """Integration: Apertus tags '11/1993' as a date → verify drops it."""
    text = "Vgl. ROLAND PFÄFFLI, AJP 11/1993 S. 174."
    claims = [
        ApertusClaim("ROLAND PFÄFFLI", "private_person"),
        ApertusClaim("11/1993", "private_date"),  # Apertus mistake
    ]
    ex = verify_and_combine(text, claims, language="de_ch", doc_id="d", subset="entscheidsuche")
    assert ex is not None
    date_spans = [s for s in ex.spans if s.label == "private_date"]
    assert date_spans == [], f"M3 should drop journal ref date: {date_spans}"


def test_audit_gate_fails_when_recall_too_low(tmp_path: Path) -> None:
    """Apertus missing all person spans → gate fails on private_person recall."""
    import json

    from gheim_training.data.apertus_label.audit import audit, check_gate

    gold = [{"text": f"Person Anna Müller {i}.", "spans": [
        {"start": 7, "end": 18, "label": "private_person"}],
        "language": "de_ch", "source": "real", "template_id": "t"}
        for i in range(10)]
    apertus = [{"text": g["text"], "spans": [], "language": "de_ch",
                "source": "real", "template_id": "t"} for g in gold]
    gold_path = tmp_path / "gold.jsonl"
    apertus_path = tmp_path / "ap.jsonl"
    gold_path.write_text("\n".join(json.dumps(g) for g in gold), encoding="utf-8")
    apertus_path.write_text("\n".join(json.dumps(a) for a in apertus), encoding="utf-8")
    report = audit(apertus_path, gold_path)
    passed, failures = check_gate(report)
    assert not passed
    assert any("private_person" in f and "R=" in f for f in failures)
