"""Build a gheim detector from plain ``config.yaml`` values.

Every knob of the Python API that makes sense in YAML is accepted by name::

    detector: composite          # composite (default) | calibrated | local | remote | pkg.mod:factory
    detector_model: joelbarmettler/gheim-ch-560m
    o_bias: 0.5                  # calibrated / composite
    device: auto                 # auto | cpu | cuda | cuda:N
    dtype: bfloat16              # name of a torch dtype
    chunk_tokens: 510
    chunk_overlap_tokens: 32
    batch_size: 8
    trim_whitespace: true
    propagate_names: true        # composite
    propagate_surnames: false    # composite
    custom_patterns:             # composite
      - label: insurance_card
        regex: '\\b756\\.\\d{4}\\.\\d{4}\\.\\d{2}\\b'
        group: 0
        flags: IGNORECASE        # name, "A|B" or list of names
        validator: mypkg.checks:is_valid_card   # optional, str -> bool
    api_key: os.environ/GHEIM_API_KEY            # remote
    api_base: https://api.gheim.ch               # remote
    timeout: 30                                   # remote
"""
from __future__ import annotations

import importlib
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ..detectors import (
    CalibratedDetector,
    CompositeDetector,
    CustomPattern,
    Detector,
    LocalDetector,
    RemoteDetector,
)

_KINDS = ("composite", "calibrated", "local", "remote")

# Knobs and the detector kinds that understand them. A knob set for a kind
# that ignores it is an error: silently dropping a custom pattern or a
# calibration bias is a PII leak waiting to happen.
_MODEL_KNOBS = {"device", "dtype", "chunk_tokens", "chunk_overlap_tokens", "batch_size"}
_ALLOWED: dict[str, set[str]] = {
    "composite": _MODEL_KNOBS | {"o_bias", "trim_whitespace", "detector_model",
                                 "custom_patterns", "propagate_names", "propagate_surnames"},
    "calibrated": _MODEL_KNOBS | {"o_bias", "trim_whitespace", "detector_model"},
    "local": _MODEL_KNOBS | {"trim_whitespace", "detector_model"},
    "remote": {"api_key", "api_base", "timeout", "trim_whitespace", "detector_model"},
}


def _import_ref(ref: str) -> Any:
    mod, sep, attr = ref.partition(":")
    if not sep or not mod or not attr:
        raise ValueError(f"expected 'package.module:name', got {ref!r}")
    return getattr(importlib.import_module(mod), attr)


def _parse_flags(value: Any) -> int:
    if value is None:
        return re.IGNORECASE
    if isinstance(value, int):
        return value
    names = value.replace("|", " ").split() if isinstance(value, str) else list(value)
    flags = 0
    for name in names:
        try:
            flags |= re.RegexFlag[str(name).upper()]
        except KeyError:
            raise ValueError(f"unknown regex flag {name!r}") from None
    return flags


def parse_custom_patterns(raw: Sequence[Mapping[str, Any]] | None) -> list[CustomPattern]:
    patterns = []
    for i, item in enumerate(raw or ()):
        unknown = set(item) - {"label", "regex", "group", "flags", "validator", "name"}
        if unknown:
            raise ValueError(f"custom_patterns[{i}]: unknown keys {sorted(unknown)}")
        if "label" not in item or "regex" not in item:
            raise ValueError(f"custom_patterns[{i}]: 'label' and 'regex' are required")
        validator: Callable[[str], bool] | None = None
        if item.get("validator"):
            validator = _import_ref(item["validator"])
        patterns.append(
            CustomPattern(
                label=item["label"],
                regex=item["regex"],
                group=int(item.get("group", 0)),
                flags=_parse_flags(item.get("flags")),
                validator=validator,
                name=item.get("name"),
            )
        )
    return patterns


def _torch_dtype(name: Any) -> Any:
    if name is None or not isinstance(name, str):
        return name
    import torch

    dtype = getattr(torch, name.removeprefix("torch."), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"dtype {name!r} is not a torch dtype (e.g. bfloat16, float16)")
    return dtype


def build_detector(detector: Detector | str | None = None, **knobs: Any) -> Detector:
    """Return a detector. ``knobs`` that are ``None`` count as unset."""
    if detector is not None and not isinstance(detector, str):
        if any(v is not None for v in knobs.values()):
            raise ValueError("detector options cannot be combined with a detector instance")
        return detector

    given = {k: v for k, v in knobs.items() if v is not None}
    kind = (detector or "").strip()
    if ":" in kind:
        if given:
            raise ValueError(f"detector options {sorted(given)} cannot be combined with {kind!r}")
        return _import_ref(kind)()
    if not kind:
        # Same choice as gheim.default_detector(), but with the knobs applied.
        import os

        kind = "remote" if given.get("api_key") or os.getenv("GHEIM_API_KEY") else "composite"
    if kind not in _KINDS:
        raise ValueError(
            f"unknown detector {detector!r}; expected one of {', '.join(_KINDS)} or 'module:factory'"
        )
    stray = set(given) - _ALLOWED[kind]
    if stray:
        raise ValueError(f"{sorted(stray)} have no effect with detector '{kind}'")

    if kind == "remote":
        opts = {k: given[k] for k in ("api_key", "timeout", "trim_whitespace") if k in given}
        if "api_base" in given:
            opts["base_url"] = given["api_base"]
        if "detector_model" in given:
            opts["model"] = given["detector_model"]
        return RemoteDetector(**opts)

    opts = {k: given[k] for k in
            (_MODEL_KNOBS | {"o_bias", "trim_whitespace"}) & given.keys()}
    if "dtype" in opts:
        opts["dtype"] = _torch_dtype(opts["dtype"])
    if "detector_model" in given:
        opts["model_id"] = given["detector_model"]
    if kind == "local":
        return LocalDetector(**opts)
    model = CalibratedDetector(**opts)
    if kind == "calibrated":
        return model
    return CompositeDetector(
        model=model,
        patterns=parse_custom_patterns(given.get("custom_patterns")),
        **{k: given[k] for k in ("propagate_names", "propagate_surnames") if k in given},
    )
