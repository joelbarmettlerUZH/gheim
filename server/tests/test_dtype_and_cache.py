"""GHEIM_DEVICE / GHEIM_DTYPE reach LocalDetector; on MPS, inference is serialised
and every detect hands the allocator's cached GPU memory back.

Measured on an Apple-silicon Mac with gheim-ch-560m: fp16 on MPS returns the same
spans as fp32 (96/96 test sentences, 4179/4181 on a 100 KB document) at 40 vs
35 KB/s, and with the cache released after each request the idle GPU footprint
drops from ~3.1 GB to ~1.1 GB. GPU memory is wired on Apple silicon, so that
difference is RAM the rest of the machine gets back.
"""
from __future__ import annotations

import importlib
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient


def _reload_registry(monkeypatch: pytest.MonkeyPatch, **env: str):
    monkeypatch.setenv("GHEIM_DEVICE", "cpu")
    monkeypatch.delenv("GHEIM_DTYPE", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    import gheim_server.registry as reg_mod
    return importlib.reload(reg_mod)


def _built_with(reg_mod, *, mps: bool = True, cuda: bool = False) -> dict:
    """The keyword arguments the registry hands gheim.LocalDetector."""
    with (
        patch("torch.backends.mps.is_available", return_value=mps),
        patch("torch.cuda.is_available", return_value=cuda),
        patch("gheim.LocalDetector") as detector,
    ):
        reg_mod.ModelRegistry._load("some/model")
    return detector.call_args.kwargs


def _loaded_dtype(reg_mod) -> torch.dtype:
    return _built_with(reg_mod)["dtype"]


def test_unset_dtype_keeps_the_models_own(monkeypatch, caplog):
    assert _loaded_dtype(_reload_registry(monkeypatch)) is None
    assert "GHEIM_DTYPE" not in caplog.text
    assert "dtype=model default" in caplog.text


def test_the_load_log_names_the_dtype(monkeypatch, caplog):
    _loaded_dtype(_reload_registry(monkeypatch, GHEIM_DTYPE="fp16"))
    assert "device=cpu dtype=float16" in caplog.text


@pytest.mark.parametrize(("name", "dtype"), [("float16", torch.float16), ("bfloat16", torch.bfloat16),
                                             ("FP16", torch.float16), ("float32", torch.float32)])
def test_dtype_is_read_from_env(monkeypatch, name, dtype):
    assert _loaded_dtype(_reload_registry(monkeypatch, GHEIM_DTYPE=name)) is dtype


def test_unknown_dtype_keeps_the_models_own_with_a_warning(monkeypatch, caplog):
    reg_mod = _reload_registry(monkeypatch, GHEIM_DTYPE="int3")
    assert _loaded_dtype(reg_mod) is None
    assert "GHEIM_DTYPE='int3'" in caplog.text


def test_release_cache_empties_the_mps_allocator_only_on_mps(monkeypatch):
    reg_mod = _reload_registry(monkeypatch, GHEIM_DEVICE="mps")
    with patch("torch.mps.empty_cache") as empty:
        reg_mod.ModelRegistry().release_cache()
    empty.assert_called_once_with()

    reg_mod = _reload_registry(monkeypatch, GHEIM_DEVICE="cpu")
    with patch("torch.mps.empty_cache") as empty:
        reg_mod.ModelRegistry().release_cache()
    empty.assert_not_called()


def test_detect_releases_the_cache_after_inference(monkeypatch):
    monkeypatch.setenv("GHEIM_SKIP_WARMUP", "1")
    monkeypatch.setenv("GHEIM_ALLOWED_MODELS", "openai/privacy-filter")
    import gheim_server.registry as reg_mod
    importlib.reload(reg_mod)
    import gheim_server.main as main_mod
    importlib.reload(main_mod)

    order: list[str] = []
    fake = MagicMock()
    fake.detect.side_effect = lambda text: order.append("infer") or []
    with (
        patch.object(main_mod.registry, "get", return_value=fake),
        patch.object(main_mod.registry, "release_cache", side_effect=lambda: order.append("release")),
        TestClient(main_mod.app) as client,
    ):
        assert client.post("/v1/detect", json={"text": "x"}).status_code == 200
        assert client.post("/v1/detect", json={"text": ""}).status_code == 200
    assert order == ["infer", "release"]


@pytest.mark.parametrize(
    ("env", "mps", "cuda", "device"),
    [
        ("mps", True, False, "mps"),
        ("mps", False, False, "cpu"),
        ("cuda", False, False, "cpu"),
        ("cuda", False, True, "cuda"),
        ("cpu", True, True, "cpu"),
        ("", False, False, "cpu"),
        ("", False, True, "cuda"),
    ],
)
def test_device_choice_and_fallback(monkeypatch, env, mps, cuda, device):
    reg_mod = _reload_registry(monkeypatch, GHEIM_DEVICE=env)
    assert _built_with(reg_mod, mps=mps, cuda=cuda) == {
        "model_id": "some/model", "device": device, "dtype": None}


def test_inference_is_serialised_on_mps_only(monkeypatch):
    lock_type = type(threading.Lock())
    assert isinstance(_reload_registry(monkeypatch, GHEIM_DEVICE="mps").ModelRegistry().inference_guard, lock_type)
    assert not isinstance(_reload_registry(monkeypatch, GHEIM_DEVICE="cpu").ModelRegistry().inference_guard, lock_type)


def test_elapsed_ms_measures_the_inference_in_milliseconds(monkeypatch):
    monkeypatch.setenv("GHEIM_SKIP_WARMUP", "1")
    monkeypatch.setenv("GHEIM_ALLOWED_MODELS", "openai/privacy-filter")
    import gheim_server.registry as reg_mod
    importlib.reload(reg_mod)
    import gheim_server.main as main_mod
    importlib.reload(main_mod)

    slow = MagicMock()
    slow.detect.side_effect = lambda text: time.sleep(0.05) or []
    with patch.object(main_mod.registry, "get", return_value=slow), TestClient(main_mod.app) as client:
        elapsed = client.post("/v1/detect", json={"text": "x"}).json()["elapsed_ms"]
    assert 45 <= elapsed < 5_000
