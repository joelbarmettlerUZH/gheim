"""The LiteLLM guardrail on the native Anthropic Messages route (``/v1/messages``).

LiteLLM calls the same hooks there, with ``call_type="anthropic_messages"``:
the request carries ``system`` and content blocks, a non-streamed response is
an Anthropic ``message`` dict, and a streamed one reaches the iterator hook as
raw SSE bytes, cut wherever the upstream read ended. Live: tests/litellm_live/.
"""
import json

import pytest

pytest.importorskip("litellm")

from gheim.detectors import Span  # noqa: E402
from gheim.litellm import GheimGuardrail  # noqa: E402

JOEL = "Joel Barmettler"
ANNA = "Anna Meier"
TRICKY = 'Joel "JB" Barmettler'


class _Names:
    def detect(self, text):
        out = []
        for name in (TRICKY, JOEL, ANNA):
            i = text.find(name)
            if i >= 0 and not any(s.start <= i < s.end for s in out):
                out.append(Span("private_person", i, i + len(name), name))
        return sorted(out, key=lambda s: s.start)


def _guard():
    return GheimGuardrail("t", detector=_Names(), default_on=True)


def _request(**extra):
    return {
        "litellm_call_id": "call-1",
        "system": f"You help {ANNA}",
        "messages": [
            {"role": "user", "content": f"Hi, I am {JOEL}"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": f"{JOEL} again", "signature": "sig"},
                {"type": "tool_use", "id": "t1", "name": "save",
                 "input": {"note": f"for {JOEL}", "tags": [ANNA], "n": 3}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": f"saved {ANNA}"},
                {"type": "tool_result", "tool_use_id": "t2",
                 "content": [{"type": "text", "text": f"also {JOEL}"}]},
                {"type": "text", "text": f"thanks, {JOEL}"},
            ]},
        ],
        **extra,
    }


async def _masked(g, data):
    return await g.async_pre_call_hook(None, None, data, "anthropic_messages")


async def test_every_anthropic_text_field_is_anonymized():
    out = await _masked(_guard(), _request(system=[{"type": "text", "text": f"You help {ANNA}"}]))
    sent = json.dumps(out)
    assert ANNA not in sent.replace(f"{JOEL} again", "")
    assert out["system"] == [{"type": "text", "text": "You help <PERSON_1>"}]
    blocks = out["messages"][2]["content"]
    assert blocks[0]["content"] == "saved <PERSON_1>"
    assert blocks[1]["content"] == [{"type": "text", "text": "also <PERSON_2>"}]
    assert blocks[2]["text"] == "thanks, <PERSON_2>"
    assert out["messages"][1]["content"][1]["input"] == {
        "note": "for <PERSON_2>", "tags": ["<PERSON_1>"], "n": 3}


async def test_string_system_is_anonymized_and_signed_thinking_left_verbatim():
    out = await _masked(_guard(), _request())
    assert out["system"] == "You help <PERSON_1>"
    # A signed thinking block must come back byte-identical or the provider
    # rejects the turn.
    assert out["messages"][1]["content"][0] == {
        "type": "thinking", "thinking": f"{JOEL} again", "signature": "sig"}


async def test_the_call_key_never_enters_the_forwarded_request():
    # /v1/messages forwards `metadata` to the provider, whose schema rejects
    # unknown fields; LiteLLM's own call id is the key.
    data = _request()
    out = await _masked(_guard(), data)
    assert "metadata" not in out


async def test_non_streamed_message_is_restored_but_thinking_stays_masked():
    g = _guard()
    data = await _masked(g, _request())
    response = {"type": "message", "role": "assistant", "content": [
        {"type": "thinking", "thinking": "about <PERSON_1>", "signature": "sig"},
        {"type": "text", "text": "Hello <PERSON_2> and <PERSON_1>"},
        {"type": "tool_use", "id": "t9", "name": "save", "input": {"to": ["<PERSON_1>"], "n": 1}},
    ]}
    out = await g.async_post_call_success_hook(data, None, response)
    assert out["content"][1]["text"] == f"Hello {JOEL} and {ANNA}"
    assert out["content"][2]["input"] == {"to": [ANNA], "n": 1}
    assert out["content"][0]["thinking"] == "about <PERSON_1>"
    assert g._session_for(data) is None


def _sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _delta(index, kind, field, text):
    return _sse("content_block_delta", {"type": "content_block_delta", "index": index,
                                        "delta": {"type": kind, field: text}})


def _stream(text, args):
    events = [_sse("message_start", {"type": "message_start", "message": {"content": []}})]
    events += [_delta(0, "text_delta", "text", text[i:i + 3]) for i in range(0, len(text), 3)]
    events.append(_sse("content_block_stop", {"type": "content_block_stop", "index": 0}))
    events += [_delta(1, "input_json_delta", "partial_json", args[i:i + 4])
               for i in range(0, len(args), 4)]
    events.append(_sse("content_block_stop", {"type": "content_block_stop", "index": 1}))
    events.append(_sse("message_stop", {"type": "message_stop"}))
    return "".join(events)


def _cut(raw, size):
    """Raw bytes cut at arbitrary points, the way an upstream read returns them."""
    data = raw.encode()
    return [data[i:i + size] for i in range(0, len(data), size)]


def _deltas(out):
    text, args, kinds = "", "", []
    raw = b"".join(x if isinstance(x, bytes) else x.encode() for x in out).decode()
    for block in raw.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                event = json.loads(line[6:])
                kinds.append(event["type"])
                delta = event.get("delta") or {}
                text += delta.get("text", "")
                args += delta.get("partial_json", "")
    return text, args, kinds


async def _collect(g, data, items):
    async def agen():
        for item in items:
            yield item
    return [x async for x in g.async_post_call_streaming_iterator_hook(None, agen(), data)]


@pytest.mark.parametrize("size", [1, 7, 50, 10_000])
async def test_sse_stream_is_restored_whatever_the_byte_cuts(size):
    g = _guard()
    data = await _masked(g, {**_request(), "system": TRICKY})
    args = json.dumps({"to": "<PERSON_1>", "cc": "<PERSON_3>"})
    out = await _collect(g, data, _cut(_stream("Hi <PERSON_2>, bye <PERSON_3>", args), size))
    text, restored_args, kinds = _deltas(out)
    assert text == f"Hi {JOEL}, bye {ANNA}"
    assert json.loads(restored_args) == {"to": TRICKY, "cc": ANNA}
    assert kinds[0] == "message_start" and kinds[-1] == "message_stop"
    assert all(isinstance(x, bytes) for x in out)
    assert g._session_for(data) is None


async def test_text_held_as_a_possible_sentinel_goes_out_before_its_block_stops():
    g = _guard()
    data = await _masked(g, _request())
    out = await _collect(g, data, [_stream("a < b <PERS", "{}")])
    text, _, kinds = _deltas(out)
    assert text == "a < b <PERS"
    first_stop = kinds.index("content_block_stop")
    assert kinds[first_stop - 1] == "content_block_delta"


async def test_str_items_and_unknown_events_pass_through_and_keep_their_type():
    g = _guard()
    data = await _masked(g, _request())
    ping = _sse("ping", {"type": "ping"})
    out = await _collect(g, data, [ping, _delta(0, "text_delta", "text", "<PERSON_2>"), ": comment\n\n"])
    assert all(isinstance(x, str) for x in out)
    assert out[0] == ping and out[-1] == ": comment\n\n"
    assert _deltas(out)[0] == JOEL


async def test_a_trailing_partial_event_is_released_at_stream_end():
    g = _guard()
    data = await _masked(g, _request())
    out = await _collect(g, data, [b"event: ping\ndata: {\"type\""])
    assert b"".join(out) == b"event: ping\ndata: {\"type\""


async def test_a_second_request_pass_of_the_same_call_extends_its_session():
    # LiteLLM can run the request hook again for one call (same litellm_call_id,
    # a different dict) on the already-anonymized request. A span found only in
    # that pass must stay restorable, and the first pass's sentinels with it.
    g = _guard()
    first = await g.async_pre_call_hook(
        None, None, {"litellm_call_id": "c", "messages": [{"role": "user", "content": f"hi {JOEL}"}]},
        "completion")
    second = {"litellm_call_id": "c", "messages": [
        {"role": "user", "content": first["messages"][0]["content"] + f" and {ANNA}"}]}
    out = await g.async_pre_call_hook(None, None, second, "completion")
    assert out["messages"][0]["content"] == "hi <PERSON_1> and <PERSON_2>"
    response = {"type": "message", "content": [{"type": "text", "text": "<PERSON_1>, <PERSON_2>"}]}
    restored = await g.async_post_call_success_hook(first, None, response)
    assert restored["content"][0]["text"] == f"{JOEL}, {ANNA}"
