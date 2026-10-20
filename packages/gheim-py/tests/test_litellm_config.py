"""YAML-knob → detector mapping for the LiteLLM guardrail (no model is loaded)."""
import pytest

pytest.importorskip("litellm")

from gheim.detectors import (  # noqa: E402
    CalibratedDetector,
    CompositeDetector,
    LocalDetector,
    RemoteDetector,
)
from gheim.litellm import GheimGuardrail  # noqa: E402
from gheim.litellm._config import build_detector, parse_custom_patterns  # noqa: E402


def test_default_is_composite_over_calibrated():
    d = build_detector()
    assert isinstance(d, CompositeDetector) and isinstance(d.model, CalibratedDetector)
    assert d.propagate_names is True and d.propagate_surnames is False


def test_all_composite_knobs_reach_their_targets():
    d = build_detector(
        "composite",
        detector_model="org/model", o_bias=1.0, device="cpu", chunk_tokens=300,
        chunk_overlap_tokens=16, batch_size=2, trim_whitespace=False,
        propagate_names=False, propagate_surnames=True,
        custom_patterns=[{"label": "case_id", "regex": r"CASE-(\d{4})", "group": 1}],
    )
    m = d.model
    assert (m.model_id, m.o_bias, m.device, m.chunk_tokens, m.chunk_overlap_tokens,
            m.batch_size, m.trim_whitespace) == ("org/model", 1.0, "cpu", 300, 16, 2, False)
    assert d.propagate_names is False and d.propagate_surnames is True
    assert [(s.label, s.text) for s in d.patterns[0].find("see case-1234 now")] == [("case_id", "1234")]


def test_kinds():
    assert isinstance(build_detector("local", batch_size=4), LocalDetector)
    assert isinstance(build_detector("calibrated", o_bias=0.0), CalibratedDetector)
    r = build_detector("remote", api_key="k", api_base="http://x/", timeout=5, detector_model="m")
    assert isinstance(r, RemoteDetector)
    assert (r.api_key, r.base_url, r.timeout, r.model) == ("k", "http://x", 5, "m")
    assert isinstance(build_detector(api_key="k"), RemoteDetector)  # api_key implies remote


def test_pattern_flags_validator_and_errors():
    (p,) = parse_custom_patterns([
        {"label": "x", "regex": "abc", "flags": "MULTILINE", "validator": "builtins:bool"}
    ])
    assert p.regex.flags & 8 and not p.regex.flags & 2  # MULTILINE set, IGNORECASE not
    assert p.validator is bool and len(p.find("abc")) == 1  # imported "module:name" validator
    with pytest.raises(ValueError, match="unknown keys"):
        parse_custom_patterns([{"label": "x", "regex": "a", "grup": 1}])
    with pytest.raises(ValueError, match="unknown regex flag"):
        parse_custom_patterns([{"label": "x", "regex": "a", "flags": "NOPE"}])
    with pytest.raises(ValueError, match="label"):
        parse_custom_patterns([{"label": "1bad", "regex": "a"}])


@pytest.mark.parametrize(
    "kind, knob",
    [("local", {"o_bias": 0.5}), ("calibrated", {"propagate_names": True}),
     ("remote", {"device": "cpu"}), ("remote", {"custom_patterns": [{"label": "a", "regex": "a"}]}),
     ("pkg.mod:make", {"o_bias": 1.0})],
)
def test_ignored_knobs_fail_loudly(kind, knob):
    with pytest.raises(ValueError):
        build_detector(kind, **knob)


def test_unknown_detector():
    with pytest.raises(ValueError, match="unknown detector"):
        build_detector("nope")


def test_guardrail_takes_yaml_shaped_params_and_normalizers():
    g = GheimGuardrail(
        "t", detector="calibrated", o_bias=0.75, normalizers={"private_phone": "e164"},
        default_on=True,
    )
    assert g.detector.o_bias == 0.75 and g.normalizers == {"private_phone": "e164"}


def test_composite_over_remote_model_keeps_regex_front_end():
    d = build_detector(
        "composite", model_detector="remote", api_base="http://gheim:8080", timeout=7,
        custom_patterns=[{"label": "case_id", "regex": "ZH-\\d+"}],
    )
    assert isinstance(d, CompositeDetector) and isinstance(d.model, RemoteDetector)
    assert (d.model.base_url, d.model.timeout) == ("http://gheim:8080", 7)
    assert len(d.patterns) == 1


@pytest.mark.parametrize(
    "knobs",
    [{"model_detector": "remote", "device": "cpu"},   # model knob, remote model
     {"model_detector": "calibrated", "api_base": "http://x"},  # remote knob, local model
     {"model_detector": "nope"}],
)
def test_composite_inner_model_knob_mismatch_fails(knobs):
    with pytest.raises(ValueError):
        build_detector("composite", **knobs)
