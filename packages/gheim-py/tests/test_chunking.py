"""Long-input chunking + batching tests for ``LocalDetector``.

These would have caught the silent-truncation bug where the underlying
HF pipeline truncates anything past ~512 tokens without warning, so
PII near the end of a long document went un-redacted.

Uses a fake pipeline + fake tokenizer with 1-token-per-character so the
chunking math is exactly predictable. The live-model variant is in
``test_live_local.py``.
"""
from __future__ import annotations

from typing import Any

import pytest
from gheim.detectors.local import LocalDetector


class _FakeTokenizer:
    """1-token-per-character tokenizer so token offsets == char offsets.
    With this, ``chunk_tokens=N`` means each chunk covers exactly N chars."""

    model_max_length = 512

    def __call__(self, text: str, **_: Any) -> dict[str, Any]:
        return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}


class _FakePipeline:
    """Records every call and returns spans that match a fixed needle."""

    def __init__(self, needle: str = "Joel", label: str = "private_person"):
        self.needle = needle
        self.label = label
        self.tokenizer = _FakeTokenizer()
        self.calls: list[dict[str, Any]] = []

    def __call__(self, texts: str | list[str], **kwargs: Any) -> Any:
        if isinstance(texts, str):
            self.calls.append({"n": 1, "batch_size": kwargs.get("batch_size")})
            return self._scan(texts)
        self.calls.append({"n": len(texts), "batch_size": kwargs.get("batch_size")})
        return [self._scan(t) for t in texts]

    def _scan(self, text: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        i = 0
        while True:
            idx = text.find(self.needle, i)
            if idx == -1:
                break
            out.append({
                "start": idx,
                "end": idx + len(self.needle),
                "entity_group": self.label,
                "score": 0.99,
            })
            i = idx + len(self.needle)
        return out


def _detector(pipeline: _FakePipeline, **kw: Any) -> LocalDetector:
    """Build a LocalDetector with a pre-injected fake pipeline (skips _load)."""
    det = LocalDetector(**kw)
    det._pipeline = pipeline
    return det


# ---------------------------------------------------------------------------
# Single-input chunking
# ---------------------------------------------------------------------------

def test_short_input_uses_fast_path_no_chunking():
    """Inputs under chunk_tokens go through the pipeline once with no
    reprojection overhead."""
    pipe = _FakePipeline()
    det = _detector(pipe, chunk_tokens=100, chunk_overlap_tokens=10)
    spans = det.detect("Hello Joel.")
    assert len(spans) == 1
    assert spans[0].start == 6
    assert spans[0].end == 10
    assert spans[0].text == "Joel"
    # Single batched call (the fast path still goes through detect_batch).
    assert len(pipe.calls) == 1
    assert pipe.calls[0]["n"] == 1


def test_long_input_finds_pii_at_the_end():
    """The bug we're fixing: PII near the end of a long document got
    silently truncated. With 1k chars and chunk_tokens=200, the input
    needs ~5 chunks; the Joel at char 950 was unreachable before."""
    text = "x" * 950 + "Joel was here." + "y" * 50
    pipe = _FakePipeline()
    det = _detector(pipe, chunk_tokens=200, chunk_overlap_tokens=20)
    spans = det.detect(text)
    assert len(spans) == 1
    assert text[spans[0].start:spans[0].end] == "Joel"
    assert spans[0].start == 950
    # Multiple chunks were emitted (1k chars / (200 - 20) stride = ~6 chunks).
    assert pipe.calls[0]["n"] >= 5


def test_long_input_finds_pii_at_start_middle_and_end():
    """Sanity: PII anywhere in the document gets surfaced exactly once."""
    pieces = [
        ("a" * 50) + "Joel" + ("b" * 50),       # block 0: 104 chars, Joel at 50
        ("c" * 400),                            # block 1: 400 chars
        ("d" * 50) + "Joel" + ("e" * 50),       # block 2: 104 chars, Joel at 554
        ("f" * 400),                            # block 3: 400 chars
        ("g" * 50) + "Joel" + ("h" * 50),       # block 4: 104 chars, Joel at 1058
    ]
    text = "".join(pieces)
    pipe = _FakePipeline()
    det = _detector(pipe, chunk_tokens=200, chunk_overlap_tokens=32)
    spans = det.detect(text)
    starts = sorted(s.start for s in spans)
    assert starts == [50, 554, 1058], starts
    for s in spans:
        assert text[s.start:s.end] == "Joel"


def test_pii_in_overlap_region_is_deduplicated():
    """A PII span that falls into the overlap between two chunks must
    appear in both chunk's raw output but get deduplicated to one span."""
    # Place Joel exactly inside the overlap window of two chunks.
    # chunk_tokens=100, overlap=20 → stride=80. Chunk 1 covers [0, 100),
    # chunk 2 covers [80, 180). A Joel at [85, 89) lives in both.
    text = ("x" * 85) + "Joel" + ("y" * 100)
    pipe = _FakePipeline()
    det = _detector(pipe, chunk_tokens=100, chunk_overlap_tokens=20)
    spans = det.detect(text)
    assert len(spans) == 1
    assert spans[0].start == 85
    assert spans[0].end == 89


def test_chunk_overlap_must_be_smaller_than_chunk_tokens():
    pipe = _FakePipeline()
    det = LocalDetector(chunk_tokens=10, chunk_overlap_tokens=10)
    det._pipeline = pipe
    with pytest.raises(ValueError, match="chunk_overlap_tokens"):
        det.detect("x" * 100)


def test_empty_input_returns_empty_no_pipeline_call():
    pipe = _FakePipeline()
    det = _detector(pipe, chunk_tokens=100)
    assert det.detect("") == []
    assert pipe.calls == []  # empty inputs short-circuit


# ---------------------------------------------------------------------------
# Cross-input batching (detect_batch)
# ---------------------------------------------------------------------------

def test_detect_batch_runs_one_pipeline_call_for_many_short_inputs():
    """Three short inputs → one batched pipeline call (not three sequential)."""
    pipe = _FakePipeline()
    det = _detector(pipe, chunk_tokens=100, batch_size=4)
    results = det.detect_batch(["I am Joel.", "Joel and Joel.", "Nobody."])
    assert len(results) == 3
    assert len(results[0]) == 1
    assert len(results[1]) == 2
    assert len(results[2]) == 0
    # One pipeline invocation total — three chunks bundled.
    assert len(pipe.calls) == 1
    assert pipe.calls[0]["n"] == 3
    assert pipe.calls[0]["batch_size"] == 4


def test_detect_batch_chunks_long_inputs_and_batches_chunks():
    """Long inputs become multiple chunks; the chunks across all inputs
    flow through a single batched call."""
    short = "Joel here."
    long_ = ("x" * 200) + "Joel" + ("y" * 200)  # needs ~3 chunks at chunk_tokens=180
    pipe = _FakePipeline()
    det = _detector(pipe, chunk_tokens=180, chunk_overlap_tokens=20, batch_size=8)
    results = det.detect_batch([short, long_])
    # Both inputs find their Joels; long_ has its Joel at char 200.
    assert len(results[0]) == 1
    assert len(results[1]) == 1
    assert results[1][0].start == 200
    # Single batched call covering all chunks across both inputs.
    assert len(pipe.calls) == 1
    assert pipe.calls[0]["n"] >= 3  # 1 chunk for short + ≥2 for long


def test_detect_batch_preserves_ordering_with_empty_inputs():
    """Empty texts must produce ``[]`` at the matching index without
    breaking the alignment of the other inputs' results."""
    pipe = _FakePipeline()
    det = _detector(pipe, chunk_tokens=100)
    results = det.detect_batch(["I am Joel.", "", "Hi Joel!", ""])
    assert len(results) == 4
    assert len(results[0]) == 1
    assert results[1] == []
    assert len(results[2]) == 1
    assert results[3] == []


def test_detect_batch_empty_list_returns_empty():
    pipe = _FakePipeline()
    det = _detector(pipe, chunk_tokens=100)
    assert det.detect_batch([]) == []
    assert pipe.calls == []


def test_detect_batch_all_empty_inputs():
    pipe = _FakePipeline()
    det = _detector(pipe, chunk_tokens=100)
    assert det.detect_batch(["", "", ""]) == [[], [], []]
    # No pipeline call needed when there's nothing to scan.
    assert pipe.calls == []
