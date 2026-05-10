"""Server input-validation + concurrency tests using a fake pipeline."""
from __future__ import annotations

import importlib
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

SERVER_SRC = Path(__file__).resolve().parents[1] / "src"
if str(SERVER_SRC) not in sys.path:
    sys.path.insert(0, str(SERVER_SRC))


class _FakePipeline:
    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.call_count = 0
        self._lock = threading.Lock()

    def __call__(self, text: str) -> list[dict[str, Any]]:
        with self._lock:
            self.call_count += 1
        if self.delay:
            time.sleep(self.delay)
        idx = text.find("Joel")
        if idx == -1:
            return []
        return [
            {"entity_group": "private_person", "start": idx, "end": idx + 4, "score": 0.99}
        ]


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GHEIM_SKIP_WARMUP", "1")
    monkeypatch.setenv("GHEIM_ALLOWED_MODELS", "openai/privacy-filter")
    import gheim_server.registry as reg_mod
    importlib.reload(reg_mod)
    import gheim_server.main as main_mod
    importlib.reload(main_mod)
    fake = _FakePipeline()
    with (
        patch.object(main_mod.registry, "get", return_value=fake),
        TestClient(main_mod.app) as c,
    ):
        yield c, fake


# ---------- C.2 pydantic validation ----------

def test_missing_text_field_returns_422(client):
    c, _ = client
    r = c.post("/v1/detect", json={"model": "openai/privacy-filter"})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert any("text" in e["loc"] for e in detail)


def test_text_with_wrong_type_returns_422(client):
    c, _ = client
    r = c.post("/v1/detect", json={"text": 12345})
    assert r.status_code == 422


def test_invalid_json_returns_422(client):
    c, _ = client
    r = c.post(
        "/v1/detect",
        content=b"{ not valid json",
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 422


def test_extra_fields_in_request_are_ignored(client):
    c, _ = client
    r = c.post(
        "/v1/detect",
        json={"text": "Joel", "model": "openai/privacy-filter", "future_flag": True},
    )
    assert r.status_code == 200


def test_empty_text_returns_empty_spans_without_calling_pipeline(client):
    c, fake = client
    r = c.post("/v1/detect", json={"text": ""})
    assert r.status_code == 200
    assert r.json()["spans"] == []
    assert fake.call_count == 0


def test_default_model_used_when_body_omits_it(client):
    c, _ = client
    r = c.post("/v1/detect", json={"text": "Joel"})
    assert r.status_code == 200
    assert r.json()["model"] == "openai/privacy-filter"


# ---------- C.3 large body ----------

def test_large_text_body_handled(client):
    c, _ = client
    big = ("Joel " * 50_000).strip()  # ~250 KB
    r = c.post("/v1/detect", json={"text": big})
    assert r.status_code == 200
    # FakePipeline only finds the first occurrence; just make sure the request didn't choke.
    body = r.json()
    assert body["model"] == "openai/privacy-filter"
    assert isinstance(body["spans"], list)


# ---------- C.4 concurrent requests through shared registry ----------

def test_concurrent_requests_share_pipeline_safely(client):
    """Hammer the endpoint with parallel requests to surface any threading issues
    in registry / pipeline access."""
    c, fake = client

    def one(_: int) -> int:
        r = c.post("/v1/detect", json={"text": "Hello Joel and Joel."})
        return r.status_code

    with ThreadPoolExecutor(max_workers=16) as pool:
        statuses = list(pool.map(one, range(64)))

    assert all(s == 200 for s in statuses)
    assert fake.call_count == 64


def test_registry_get_is_thread_safe_for_concurrent_first_calls(monkeypatch: pytest.MonkeyPatch):
    """Two threads racing to load the same model id must end up with the same
    pipeline instance — the lock in ModelRegistry must serialize the load."""
    monkeypatch.setenv("GHEIM_SKIP_WARMUP", "1")
    monkeypatch.setenv("GHEIM_ALLOWED_MODELS", "openai/privacy-filter")
    import gheim_server.registry as reg_mod
    importlib.reload(reg_mod)

    load_count = {"n": 0}
    sentinel_pipe = object()

    def slow_load(model_id: str) -> Any:
        load_count["n"] += 1
        time.sleep(0.05)  # widen the race window
        return sentinel_pipe

    monkeypatch.setattr(reg_mod.ModelRegistry, "_load", staticmethod(slow_load))

    registry = reg_mod.ModelRegistry()
    results: list[Any] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for fut in [pool.submit(registry.get, "openai/privacy-filter") for _ in range(8)]:
            results.append(fut.result())

    assert all(r is sentinel_pipe for r in results)
    assert load_count["n"] == 1  # loaded exactly once despite 8 racing callers
