"""HTTP-level tests using a fake registry — no model load required."""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Ensure the server src is importable.
SERVER_SRC = Path(__file__).resolve().parents[1] / "src"
if str(SERVER_SRC) not in sys.path:
    sys.path.insert(0, str(SERVER_SRC))


class _FakePipeline:
    """Returns a fixed span for any text containing 'Joel'."""

    def __call__(self, text: str):
        out = []
        idx = text.find("Joel")
        if idx >= 0:
            out.append(
                {"entity_group": "private_person", "start": idx, "end": idx + 4, "score": 0.95}
            )
        return out


@pytest.fixture
def client(monkeypatch):
    # Skip lifespan warmup so we don't hit the network.
    monkeypatch.setenv("GHEIM_SKIP_WARMUP", "1")
    monkeypatch.setenv("GHEIM_ALLOWED_MODELS", "openai/privacy-filter,custom/test")
    # Reload registry module so the env vars take effect.
    import importlib

    import gheim_server.registry as reg_mod
    importlib.reload(reg_mod)
    import gheim_server.main as main_mod
    importlib.reload(main_mod)

    fake = _FakePipeline()
    with (
        patch.object(main_mod.registry, "get", return_value=fake),
        TestClient(main_mod.app) as c,
    ):
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_detect_returns_spans(client):
    r = client.post("/v1/detect", json={"text": "Hi, my name is Joel"})
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "openai/privacy-filter"
    assert body["spans"] == [
        {"label": "private_person", "start": 15, "end": 19, "score": 0.95}
    ]
    assert body["elapsed_ms"] >= 0


def test_detect_empty_text(client):
    r = client.post("/v1/detect", json={"text": ""})
    assert r.status_code == 200
    assert r.json()["spans"] == []


def test_detect_unknown_model_rejected(client, monkeypatch):
    import gheim_server.main as main_mod
    monkeypatch.setattr(
        main_mod.registry,
        "get",
        lambda mid: (_ for _ in ()).throw(ValueError(f"model {mid!r} is not in GHEIM_ALLOWED_MODELS")),
    )
    r = client.post("/v1/detect", json={"text": "x", "model": "evil/model"})
    assert r.status_code == 400
    assert "not in GHEIM_ALLOWED_MODELS" in r.json()["detail"]


def test_auth_disabled_when_no_keys_configured(client, monkeypatch):
    monkeypatch.delenv("GHEIM_API_KEYS", raising=False)
    r = client.post("/v1/detect", json={"text": "Joel"})
    assert r.status_code == 200


def test_auth_required_when_keys_configured(monkeypatch):
    monkeypatch.setenv("GHEIM_API_KEYS", "secret-key-1")
    monkeypatch.setenv("GHEIM_SKIP_WARMUP", "1")
    monkeypatch.setenv("GHEIM_ALLOWED_MODELS", "openai/privacy-filter")
    import importlib

    import gheim_server.main as main_mod
    importlib.reload(main_mod)
    fake = _FakePipeline()
    with (
        patch.object(main_mod.registry, "get", return_value=fake),
        TestClient(main_mod.app) as c,
    ):
        # No auth header
        r = c.post("/v1/detect", json={"text": "Joel"})
        assert r.status_code == 401
        # Wrong key
        r = c.post(
            "/v1/detect",
            json={"text": "Joel"},
            headers={"Authorization": "Bearer wrong"},
        )
        assert r.status_code == 401
        # Right key
        r = c.post(
            "/v1/detect",
            json={"text": "Joel"},
            headers={"Authorization": "Bearer secret-key-1"},
        )
        assert r.status_code == 200
