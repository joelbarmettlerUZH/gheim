"""RemoteDetector unit tests using httpx.MockTransport — no live server needed."""
from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from gheim import RemoteDetector


def _client(handler: Any) -> httpx.Client:
    return httpx.Client(
        base_url="http://mock", transport=httpx.MockTransport(handler), timeout=5.0
    )


def test_basic_detect_request_shape_and_response_parsing():
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content.decode())
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={
                "model": "openai/privacy-filter",
                "spans": [
                    {"label": "private_person", "start": 11, "end": 16, "score": 0.99},
                ],
                "elapsed_ms": 12.3,
            },
        )

    det = RemoteDetector(api_key="secret-abc", client=_client(handler))
    spans = det.detect("My name is Alice.")

    assert captured["method"] == "POST"
    assert captured["url"].endswith("/v1/detect")
    assert captured["body"] == {"model": "openai/privacy-filter", "text": "My name is Alice."}
    assert captured["auth"] == "Bearer secret-abc"
    assert len(spans) == 1
    assert spans[0].label == "private_person"
    assert spans[0].text == "Alice"
    assert spans[0].score == pytest.approx(0.99)


def test_no_auth_header_when_api_key_is_none():
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"model": "x", "spans": []})

    det = RemoteDetector(api_key=None, client=_client(handler))
    det.detect("anything")
    assert captured["auth"] is None


def test_per_call_model_override_goes_into_request_body():
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={"model": "x", "spans": []})

    det = RemoteDetector(client=_client(handler))
    det.detect("hi", model="custom/model")
    assert captured["body"]["model"] == "custom/model"


def test_empty_text_short_circuits_no_request():
    called = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        called["n"] += 1
        return httpx.Response(200, json={"model": "x", "spans": []})

    det = RemoteDetector(client=_client(handler))
    out = det.detect("")
    assert out == []
    assert called["n"] == 0


def test_500_response_raises_for_status():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "boom"})

    det = RemoteDetector(client=_client(handler))
    with pytest.raises(httpx.HTTPStatusError):
        det.detect("anything")


def test_401_response_raises_for_status():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "missing bearer token"})

    det = RemoteDetector(client=_client(handler))
    with pytest.raises(httpx.HTTPStatusError) as exc:
        det.detect("anything")
    assert exc.value.response.status_code == 401


def test_trim_whitespace_strips_leading_space_from_remote_spans():
    """Remote server may return spans with leading whitespace (BPE artifact);
    RemoteDetector trims them by default."""
    text = "Hi, my name is Alice."

    def handler(_: httpx.Request) -> httpx.Response:
        # Simulate pipeline emitting a span that starts with the leading space.
        return httpx.Response(
            200,
            json={
                "model": "x",
                "spans": [
                    {"label": "private_person", "start": 14, "end": 20, "score": 0.99}
                ],
            },
        )

    det = RemoteDetector(client=_client(handler))
    spans = det.detect(text)
    assert len(spans) == 1
    assert spans[0].text == "Alice"  # leading space trimmed
    assert spans[0].start == 15


def test_trim_whitespace_can_be_disabled():
    text = "Hi, my name is Alice."

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "x",
                "spans": [
                    {"label": "private_person", "start": 14, "end": 20, "score": 0.99}
                ],
            },
        )

    det = RemoteDetector(trim_whitespace=False, client=_client(handler))
    spans = det.detect(text)
    assert spans[0].text == " Alice"  # leading space preserved


def test_whitespace_only_span_is_dropped_when_trimming():
    text = "abcdef"

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"model": "x", "spans": [{"label": "x", "start": 0, "end": 0, "score": 0.5}]},
        )

    det = RemoteDetector(client=_client(handler))
    spans = det.detect(text)
    assert spans == []


def test_close_releases_owned_client():
    """Closing the detector closes a client it owns. A pre-built client passed in
    by the user is NOT closed (caller's responsibility)."""
    det = RemoteDetector()  # owns its client
    # Force lazy creation of the underlying client without making a request.
    _ = det._http()
    det.close()
    assert det.client is None

    user_client = httpx.Client()
    det = RemoteDetector(client=user_client)
    det.close()
    # User's client must still be open.
    assert not user_client.is_closed
    user_client.close()
