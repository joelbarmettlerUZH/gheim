"""Tests for the coverage gate."""
from __future__ import annotations

from pathlib import Path

import pytest
from gheim_training.data import coverage_gate as g
from gheim_training.data.schema import Example, Span, write_jsonl


def _ex(text: str, spans: list[tuple[int, int, str]], lang: str) -> Example:
    return Example(
        text=text,
        spans=[Span(s, e, lab) for s, e, lab in spans],
        language=lang,  # type: ignore[arg-type]
        source="ai4privacy",
    )


def test_count_examples_one_per_category_per_example(tmp_path: Path) -> None:
    """An example with two private_person spans and one private_date counts
    as ONE in (lang, person) and ONE in (lang, date)."""
    f = tmp_path / "layer.jsonl"
    write_jsonl(f, [
        _ex("Alice and Bob met on 2025-01-15.",
            [(0, 5, "private_person"),
             (10, 13, "private_person"),
             (21, 31, "private_date")],
            "de_ch"),
        _ex("Carol's email is c@example.com.",
            [(0, 5, "private_person"),
             (17, 30, "private_email")],
            "de_ch"),
    ])
    counts = g.count_examples([f])
    assert counts[("de_ch", "private_person")] == 2
    assert counts[("de_ch", "private_date")] == 1
    assert counts[("de_ch", "private_email")] == 1


def test_count_examples_aggregates_across_layers(tmp_path: Path) -> None:
    f1 = tmp_path / "layer1.jsonl"
    f2 = tmp_path / "layer2.jsonl"
    write_jsonl(f1, [_ex("Anna.", [(0, 4, "private_person")], "de_ch")])
    write_jsonl(f2, [_ex("Hans.", [(0, 4, "private_person")], "de_ch")])
    counts = g.count_examples([f1, f2])
    assert counts[("de_ch", "private_person")] == 2


def test_count_examples_skips_missing(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    counts = g.count_examples([tmp_path / "does_not_exist.jsonl"])
    assert counts == {}
    err = capsys.readouterr().err
    assert "does not exist" in err


def test_report_passes_when_all_cells_meet_threshold(capsys: pytest.CaptureFixture) -> None:
    counts: dict[tuple[str, str], int] = {}
    for lang in ("de_ch", "fr_ch", "it_ch", "rm", "gsw", "en"):
        for cat in ("account_number", "private_address", "private_date",
                    "private_email", "private_person", "private_phone",
                    "private_url", "secret"):
            counts[(lang, cat)] = 5000
    passed, failures = g.report(counts, min_per_cell=1000, waivers=set())
    assert passed is True
    assert failures == []


def test_report_fails_on_thin_cell() -> None:
    counts = {("de_ch", "secret"): 50}  # everything else is 0
    passed, failures = g.report(counts, min_per_cell=1000, waivers=set())
    assert passed is False
    assert any("de_ch/secret" in f for f in failures)
    # Other (lang, cat) cells are also 0 → also fail
    assert any("rm/private_person" in f for f in failures)


def test_waiver_excludes_cell_from_failure_list() -> None:
    counts = {("gsw", "secret"): 0}
    waivers = {("gsw", "secret")}
    passed, failures = g.report(counts, min_per_cell=1000, waivers=waivers)
    assert not any("gsw/secret" in f for f in failures)
