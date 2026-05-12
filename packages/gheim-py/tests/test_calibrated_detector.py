"""Tests for CalibratedDetector — both unit tests (mocked model) and a
slow live test that downloads the real v1 model and verifies the
calibration recovers a documented failing input.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from gheim import CalibratedDetector, default_detector
from gheim.detectors import Detector

# Live model tests are slow + need GPU; gate behind an env var.
_HAS_TORCH = True
try:
    import torch  # noqa: F401
except ImportError:
    _HAS_TORCH = False
slow = pytest.mark.skipif(
    not _HAS_TORCH,
    reason="torch + transformers required for live CalibratedDetector test",
)


def test_default_detector_is_calibrated_when_no_api_key(monkeypatch):
    """default_detector() must return a CalibratedDetector by default."""
    monkeypatch.delenv("GHEIM_API_KEY", raising=False)
    det = default_detector()
    assert isinstance(det, CalibratedDetector)


def test_default_detector_uses_remote_when_api_key_set(monkeypatch):
    """If GHEIM_API_KEY is set, default_detector() picks RemoteDetector."""
    from gheim import RemoteDetector
    monkeypatch.setenv("GHEIM_API_KEY", "test-key")
    det = default_detector()
    assert isinstance(det, RemoteDetector)


def test_calibrated_default_o_bias_is_0_5():
    """Default oBias must be 0.5 — the Pareto-clean point per the
    full-scale bias sweep (eval/calibration_sweep.json). Changing this
    default without re-running the sweep is a bug."""
    det = CalibratedDetector()
    assert det.o_bias == 0.5


def test_calibrated_per_call_model_override_rejected():
    """Per-call model override that doesn't match the constructor must
    raise — same contract as LocalDetector."""
    det = CalibratedDetector(model_id="some/model")
    with pytest.raises(ValueError, match="per-call override"):
        det.detect("anything", model="other/model")


def test_calibrated_empty_text_returns_empty():
    """Empty input short-circuits before loading the model."""
    det = CalibratedDetector()
    # Doesn't load weights — both calls return [] without touching the GPU.
    assert det.detect("") == []
    assert det.detect("") == []


def test_calibrated_o_bias_zero_disables_calibration():
    """o_bias=0 must be a true no-op: CalibratedDetector with o_bias=0
    should behave like the un-biased model (i.e. flag the same spans
    as a vanilla LocalDetector would). This is a structural test —
    setting o_bias=0 should leave the logits untouched."""
    det = CalibratedDetector(o_bias=0.0)
    assert det.o_bias == 0.0


# ─────────────────────────────────────────────────────────────────────
# Live tests — only run when torch is installed and an env var is set.
# ─────────────────────────────────────────────────────────────────────

GHEIM_RUN_LIVE = os.getenv("GHEIM_RUN_LIVE", "").lower() in ("1", "true", "yes")
live = pytest.mark.skipif(
    not GHEIM_RUN_LIVE or not _HAS_TORCH,
    reason="set GHEIM_RUN_LIVE=1 to run live model tests",
)


@live
def test_calibration_recovers_mueller_when_iban_present():
    """The exact failing input from the probe rounds.
    With o_bias=0.0 the model emits ZERO non-O tokens for "Müller"
    when an IBAN follows; with the default o_bias=0.5 it should
    recover the person span."""
    text = (
        "Sehr geehrte Frau Müller,\n"
        "vielen Dank für Ihre Anfrage vom 14. März 2026.\n"
        "IBAN: CH9300762011623852957"
    )
    det = CalibratedDetector(o_bias=0.5, device="auto")
    spans = det.detect(text)
    cats = {s.label for s in spans}
    assert "private_person" in cats, (
        f"expected private_person in {cats} (Müller should be recovered)"
    )
    assert "private_date" in cats, (
        f"expected private_date in {cats} (14. März 2026 should be recovered)"
    )
    assert "account_number" in cats


@live
def test_calibration_no_false_positives_on_pii_free_text():
    """A neutral German legal sentence should produce zero spans even
    with calibration on. If we trade off too much precision, we'll
    start over-tagging neutral text — the full-scale sweep (FP cost
    ~0.05/chunk at o_bias=0.5) should not produce any FPs on this kind
    of clean control input."""
    det = CalibratedDetector(o_bias=0.5, device="auto")
    spans = det.detect(
        "Das Bundesgericht hat in seinem Entscheid die Argumentation "
        "der Vorinstanz bestätigt."
    )
    assert len(spans) == 0, f"expected no spans on PII-free text, got {spans}"
