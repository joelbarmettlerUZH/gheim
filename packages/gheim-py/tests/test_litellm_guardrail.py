"""In-process tests for the LiteLLM guardrail (no proxy). Live tests: tests/litellm_live/."""
import pytest

pytest.importorskip("litellm")

from gheim.detectors import Span  # noqa: E402
from gheim.litellm import GheimGuardrail  # noqa: E402
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices  # noqa: E402


class _Names:
    def detect(self, text):  # like CompositeDetector: no `model` kwarg
        return [
            Span("private_person", i, i + len(n), n)
            for n in ("Joel Barmettler", "Anna Meier")
            if (i := text.find(n)) >= 0
        ]


def _chunk(*choices):
    return ModelResponseStream(
        id="x", model="m",
        choices=[StreamingChoices(index=i, delta=Delta(content=c), finish_reason=f)
                 for i, c, f in choices],
    )


async def _agen(chunks):
    for c in chunks:
        yield c


async def _setup(text="Hi Joel Barmettler and Anna Meier", g=None):
    g = g or GheimGuardrail("t", detector=_Names(), default_on=True)
    data = {"messages": [{"role": "user", "content": text}], "metadata": {}}
    out = await g.async_pre_call_hook(None, None, data, "completion")
    assert "Joel" not in out["messages"][0]["content"]
    return g, out


async def _collect(g, data, chunks):
    return [c async for c in g.async_post_call_streaming_iterator_hook(None, _agen(chunks), data)]


def _texts(chunks, idx=0):
    return "".join(
        ch.delta.content or "" for c in chunks for ch in c.choices if ch.index == idx
    )


def test_forces_post_call_mode():
    g = GheimGuardrail("t", detector=_Names(), event_hook="pre_call")
    assert {h.value for h in g.event_hook} == {"pre_call", "post_call"}


async def test_choices_do_not_share_hold_buffers():
    g, data = await _setup()
    # choice 0 is mid-sentinel while choice 1 emits text: must not interleave.
    out = await _collect(g, data, [
        _chunk((0, "<PERS", None), (1, "<PERSON_2", None)),
        _chunk((0, "ON_1> ok", "stop"), (1, "> hi", "stop")),
    ])
    assert _texts(out, 0) == "Joel Barmettler ok"
    assert _texts(out, 1) == "Anna Meier hi"


async def test_stream_ending_without_finish_reason_flushes_held_text():
    g, data = await _setup()
    out = await _collect(g, data, [_chunk((0, "a <PERSON_1> and <PER", None))])
    assert _texts(out) == "a Joel Barmettler and <PER"


async def test_no_session_is_passthrough():
    g = GheimGuardrail("t", detector=_Names())
    out = await _collect(g, {"metadata": {}}, [_chunk((0, "<PERSON_1>", "stop"))])
    assert _texts(out) == "<PERSON_1>"


async def test_session_dropped_after_stream_and_ttl_sweep():
    g, data = await _setup()
    await _collect(g, data, [_chunk((0, "x", "stop"))])
    assert not g._sessions

    g.session_ttl_seconds = -1  # everything already registered counts as expired
    await _setup(g=g)
    await _setup(g=g)  # sweep runs before the new session is registered
    assert len(g._sessions) == 1


async def test_preexisting_sentinel_kept_verbatim_both_directions():
    # Issue #4: a <PERSON_1> from an earlier manual anonymisation stays as-is,
    # and a newly detected person gets a different placeholder.
    g, data = await _setup("<PERSON_1> asked Anna Meier to call back")
    assert data["messages"][0]["content"] == "<PERSON_1> asked <PERSON_2> to call back"
    out = await _collect(g, data, [_chunk((0, "<PERSON_1> and <PERSON_2>", "stop"))])
    assert _texts(out) == "<PERSON_1> and Anna Meier"


class _Counting(_Names):
    def __init__(self):
        self.texts = []

    def detect(self, text):
        self.texts.append(text)
        return super().detect(text)


async def test_a_later_turn_only_detects_its_new_text():
    # The client replays the (restored) history: only the new message is detected.
    det = _Counting()
    g = GheimGuardrail("t", detector=det, default_on=True)
    first = {"role": "user", "content": "Hi Joel Barmettler"}
    await g.async_pre_call_hook(None, None, {"messages": [first], "metadata": {}}, "completion")
    turn2 = {"messages": [first, {"role": "assistant", "content": "Hello Joel Barmettler"},
                          {"role": "user", "content": "and Anna Meier"}], "metadata": {}}
    out = await g.async_pre_call_hook(None, None, turn2, "completion")
    assert [m["content"] for m in out["messages"]] == [
        "Hi <PERSON_1>", "Hello <PERSON_1>", "and <PERSON_2>"]
    assert det.texts == ["Hi Joel Barmettler", "Hello Joel Barmettler", "and Anna Meier"]


async def test_memo_entries_zero_turns_the_memo_off():
    det = _Counting()
    g = GheimGuardrail("t", detector=det, default_on=True, memo_entries=0)
    for _ in range(2):
        await g.async_pre_call_hook(
            None, None, {"messages": [{"role": "user", "content": "Hi Joel Barmettler"}],
                         "metadata": {}}, "completion")
    assert det.texts == ["Hi Joel Barmettler"] * 2
    assert g.detector is det and g._detect is det
