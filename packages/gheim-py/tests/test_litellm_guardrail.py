"""In-process tests for the LiteLLM guardrail (no proxy). Live tests: tests/litellm_live/."""
import pytest

pytest.importorskip("litellm")

from litellm.types.utils import ModelResponseStream, StreamingChoices, Delta  # noqa: E402

from gheim.detectors import Span  # noqa: E402
from gheim.litellm import GheimGuardrail  # noqa: E402


class _Names:
    def detect(self, text, *, model=None):
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
