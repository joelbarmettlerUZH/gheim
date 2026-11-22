"""MemoDetector: a conversation re-sends its whole history every turn; only new
text should reach the wrapped detector."""
from concurrent.futures import ThreadPoolExecutor

import pytest
from gheim.detectors import MemoDetector, Span

TEXT = "Anna Meier, mail anna@example.com"


class _Recorder:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def detect(self, text, *, model=None):
        self.calls.append((text, model))
        if self.fail:
            raise RuntimeError("detector down")
        out = []
        for label, needle in (("private_person", "Anna Meier"), ("private_email", "anna@example.com")):
            i = text.find(needle)
            if i >= 0:
                out.append(Span(label, i, i + len(needle), needle, 0.9))
        return out


class _NoModelKwarg:
    """Like CompositeDetector: ``detect(text)`` only."""

    def detect(self, text):
        return [Span("private_person", 0, 4, text[:4])]


def test_the_wrapped_detector_sees_each_distinct_text_once():
    inner = _Recorder()
    memo = MemoDetector(inner, max_entries=8)
    first = memo.detect(TEXT)
    assert memo.detect(TEXT) == first == _Recorder().detect(TEXT)
    memo.detect("nobody here")
    assert [c[0] for c in inner.calls] == [TEXT, "nobody here"]


def test_the_model_is_part_of_the_key_and_is_forwarded():
    inner = _Recorder()
    memo = MemoDetector(inner, max_entries=8)
    memo.detect(TEXT)
    memo.detect(TEXT, model="m2")
    memo.detect(TEXT, model="m2")
    assert inner.calls == [(TEXT, None), (TEXT, "m2")]


def test_a_detector_without_a_model_parameter_still_works():
    assert MemoDetector(_NoModelKwarg(), max_entries=2).detect("Joel x")[0].text == "Joel"


def test_least_recently_used_entry_is_evicted():
    inner = _Recorder()
    memo = MemoDetector(inner, max_entries=2)
    for text in ("a", "b", "a", "c", "a", "b"):
        memo.detect(text)
    assert [c[0] for c in inner.calls] == ["a", "b", "c", "b"]


def test_a_failure_is_not_remembered():
    inner = _Recorder(fail=True)
    memo = MemoDetector(inner, max_entries=8)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            memo.detect(TEXT)
    assert len(inner.calls) == 2


def test_the_memo_holds_no_surface_text():
    memo = MemoDetector(_Recorder(), max_entries=8)
    memo.detect(TEXT)
    held = repr(memo._entries)
    assert "Anna" not in held and "example.com" not in held


def test_returned_spans_are_fresh_lists():
    memo = MemoDetector(_Recorder(), max_entries=8)
    memo.detect(TEXT).clear()
    assert len(memo.detect(TEXT)) == 2


def test_max_entries_must_be_positive():
    with pytest.raises(ValueError):
        MemoDetector(_Recorder(), max_entries=0)


def test_concurrent_callers_share_one_memo():
    inner = _Recorder()
    memo = MemoDetector(inner, max_entries=64)
    texts = [f"{TEXT} {i % 8}" for i in range(400)]
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(memo.detect, texts))
    assert {c[0] for c in inner.calls} == set(texts[:8])
    assert len(memo._entries) == 8
