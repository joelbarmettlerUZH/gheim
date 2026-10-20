"""Live tests against a running LiteLLM proxy (docker) + fake upstream.

Run via ``run_live.sh``; skipped unless ``GHEIM_LITELLM_URL`` is set.
"""
import json
import os
import time

import httpx
import pytest
from openai import OpenAI

PROXY = os.getenv("GHEIM_LITELLM_URL")
UPSTREAM = os.getenv("GHEIM_UPSTREAM_URL", "http://127.0.0.1:9100")
pytestmark = pytest.mark.skipif(not PROXY, reason="GHEIM_LITELLM_URL not set")

PII = "Joel Barmettler"
EMAIL = "joel@example.com"


@pytest.fixture(scope="module")
def client():
    return OpenAI(base_url=f"{PROXY}/v1", api_key="sk-test")


@pytest.fixture(autouse=True)
def _reset_upstream():
    httpx.post(f"{UPSTREAM}/_reset")


def _upstream_bodies() -> str:
    return json.dumps(httpx.get(f"{UPSTREAM}/_received").json())


def _chat(client, content, **kw):
    return client.chat.completions.create(
        model="fake", messages=[{"role": "user", "content": content}], **kw
    )


def test_non_stream_roundtrip(client):
    r = _chat(client, f"Hi, I am {PII}, mail {EMAIL}")
    assert r.choices[0].message.content == f"Echo: Hi, I am {PII}, mail {EMAIL}"
    sent = _upstream_bodies()
    assert PII not in sent and EMAIL not in sent
    assert "<PERSON_1>" in sent


def test_stream_roundtrip_and_sentinel_split_across_chunks(client):
    stream = _chat(client, f"Hi, I am {PII}, mail {EMAIL}", stream=True)
    pieces = [c.choices[0].delta.content for c in stream if c.choices and c.choices[0].delta.content]
    text = "".join(pieces)
    assert text == f"Echo: Hi, I am {PII}, mail {EMAIL}"
    assert "<PERSON" not in text and "<EMAIL" not in text
    assert PII not in _upstream_bodies()
    # Not buffered into one chunk (the Presidio behaviour).
    assert len(pieces) > 5


def test_stream_is_incremental(client):
    t0 = time.monotonic()
    first = None
    n = 0
    for c in _chat(client, f"[slow] hello {PII} and again {PII}", stream=True):
        if c.choices and c.choices[0].delta.content:
            n += 1
            first = first or time.monotonic() - t0
    total = time.monotonic() - t0
    assert n > 5
    assert first < total * 0.4, f"first content at {first:.2f}s of {total:.2f}s — stream was buffered"


def test_repeated_name_same_sentinel_and_two_people(client):
    r = _chat(client, f"{PII} met Anna Meier. {PII} left.")
    assert r.choices[0].message.content == f"Echo: {PII} met Anna Meier. {PII} left."
    sent = _upstream_bodies()
    assert sent.count("<PERSON_1>") == 2 and "<PERSON_2>" in sent


def test_incidental_angle_brackets_and_truncated_sentinel_prefix(client):
    text = "".join(
        c.choices[0].delta.content or ""
        for c in _chat(client, "[trunc] x", stream=True)
        if c.choices
    )
    assert text == "Echo: a < b and then <PERS"


def test_usage_chunk_survives(client):
    chunks = list(_chat(client, f"hi {PII}", stream=True, stream_options={"include_usage": True}))
    assert any(getattr(c, "usage", None) for c in chunks)
    assert chunks[-1].choices == [] or chunks[-1].usage is not None
    assert sum(1 for c in chunks if c.choices and c.choices[0].finish_reason) == 1


TOOLS = [{"type": "function", "function": {
    "name": "lookup", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}]


def test_tool_call_arguments_restored_non_stream_and_stream(client):
    r = _chat(client, f"find {PII}", tools=TOOLS)
    assert json.loads(r.choices[0].message.tool_calls[0].function.arguments) == {"q": f"find {PII}"}

    args = ""
    for c in _chat(client, f"find {PII}", tools=TOOLS, stream=True):
        for tc in (c.choices[0].delta.tool_calls or []) if c.choices else []:
            args += tc.function.arguments or ""
    assert json.loads(args) == {"q": f"find {PII}"}


def test_tool_call_arguments_json_escaping(client):
    # The surface contains quotes: restoring it raw into the JSON arguments
    # string would produce invalid JSON.
    tricky = 'Joel "JB" Barmettler'
    r = _chat(client, tricky, tools=TOOLS)
    assert json.loads(r.choices[0].message.tool_calls[0].function.arguments) == {"q": tricky}
    args = ""
    for c in _chat(client, tricky, tools=TOOLS, stream=True):
        for tc in (c.choices[0].delta.tool_calls or []) if c.choices else []:
            args += tc.function.arguments or ""
    assert json.loads(args) == {"q": tricky}


def test_multimodal_text_parts_anonymized(client):
    r = client.chat.completions.create(
        model="fake",
        messages=[{"role": "user", "content": [{"type": "text", "text": f"I am {PII}"}]}],
    )
    assert PII in r.choices[0].message.content
    assert PII not in _upstream_bodies()


def test_concurrent_streams_do_not_cross_sessions(client):
    import concurrent.futures as cf

    def run(i):
        name = ["Joel Barmettler", "Anna Meier"][i % 2]
        text = "".join(
            c.choices[0].delta.content or ""
            for c in _chat(client, f"[slow] hello {name}", stream=True)
            if c.choices
        )
        return name, text

    with cf.ThreadPoolExecutor(8) as ex:
        for name, text in ex.map(run, range(16)):
            assert text == f"Echo: [slow] hello {name}"
