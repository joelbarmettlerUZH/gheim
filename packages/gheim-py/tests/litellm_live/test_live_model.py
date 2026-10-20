"""Live tests with the REAL gheim model behind the proxy (config_model.yaml).

The upstream echoes its input, so an exact round trip means: PII was replaced
by sentinels on the way out and restored on the way back, in both modes.
"""
import json
import os

import httpx
import pytest
from openai import OpenAI

PROXY = os.getenv("GHEIM_LITELLM_URL")
UPSTREAM = os.getenv("GHEIM_UPSTREAM_URL", "http://127.0.0.1:9100")
pytestmark = pytest.mark.skipif(not PROXY, reason="GHEIM_LITELLM_URL not set")

TEXT = (
    "Guten Tag, mein Name ist Lukas Brunner, wohnhaft an der Bahnhofstrasse 12, 8001 Zürich. "
    "Erreichbar unter lukas.brunner@bluewin.ch oder +41 79 123 45 67. "
    "Bitte überweisen Sie auf CH93 0076 2011 6238 5295 7. Freundliche Grüsse, Lukas Brunner"
)
MUST_NOT_LEAK = ["Lukas", "Brunner", "lukas.brunner@bluewin.ch", "CH93 0076 2011 6238 5295 7"]


@pytest.fixture(scope="module")
def client():
    return OpenAI(base_url=f"{PROXY}/v1", api_key="sk-test", timeout=120)


def _sent() -> str:
    return json.dumps(httpx.get(f"{UPSTREAM}/_received").json(), ensure_ascii=False)


@pytest.mark.parametrize("stream", [False, True])
def test_real_model_roundtrip(client, stream):
    httpx.post(f"{UPSTREAM}/_reset")
    r = client.chat.completions.create(
        model="fake", messages=[{"role": "user", "content": TEXT}], stream=stream
    )
    if stream:
        out = "".join(c.choices[0].delta.content or "" for c in r if c.choices)
    else:
        out = r.choices[0].message.content
    sent = _sent()
    print("\nUPSTREAM SAW:", sent[:600])
    for secret in MUST_NOT_LEAK:
        assert secret not in sent, f"{secret!r} leaked upstream"
    assert out == "Echo: " + TEXT
