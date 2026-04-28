"""Edge-case tests for gheim.openai sync + async wrappers (Batch A)."""
from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest

import gheim.openai as gheim_openai_mod
from gheim import Session, Span
from gheim.detectors.base import Detector


# ---------- shared fakes ----------

class FakeDetector:
    def __init__(self, surfaces: dict[str, str]):
        self._surfaces = surfaces

    def detect(self, text: str, *, model: str | None = None) -> list[Span]:
        out: list[Span] = []
        for surface, label in self._surfaces.items():
            i = 0
            while True:
                idx = text.find(surface, i)
                if idx == -1:
                    break
                out.append(Span(label=label, start=idx, end=idx + len(surface), text=surface))
                i = idx + len(surface)
        return out


@dataclass
class FakeDelta:
    content: str | None = None
    role: str | None = None
    tool_calls: list[Any] | None = None


@dataclass
class FakeChoiceChunk:
    delta: FakeDelta
    index: int = 0
    finish_reason: str | None = None


@dataclass
class FakeChunk:
    choices: list[FakeChoiceChunk]
    id: str = "chunk-x"
    model: str = "gpt-4o"

    def model_copy(self, *, deep: bool = False) -> "FakeChunk":
        return FakeChunk(
            id=self.id,
            model=self.model,
            choices=[
                FakeChoiceChunk(
                    delta=FakeDelta(
                        content=c.delta.content,
                        role=c.delta.role,
                        tool_calls=c.delta.tool_calls,
                    ),
                    index=c.index,
                    finish_reason=c.finish_reason,
                )
                for c in self.choices
            ],
        )


@dataclass
class FakeFunction:
    name: str
    arguments: str


@dataclass
class FakeToolCall:
    id: str
    function: FakeFunction
    type: str = "function"


@dataclass
class FakeMessage:
    role: str
    content: str | None
    tool_calls: list[FakeToolCall] | None = None


@dataclass
class FakeChoice:
    message: FakeMessage
    index: int = 0
    finish_reason: str | None = "stop"


@dataclass
class FakeCompletion:
    choices: list[FakeChoice]
    id: str = "cmpl-x"


# ---------- sync fake client ----------

@dataclass
class _SyncCompletions:
    create_calls: list[dict[str, Any]] = field(default_factory=list)
    response: Any = None

    def create(self, *, messages: list[Any], stream: bool = False, **kw: Any) -> Any:
        self.create_calls.append({"messages": messages, "stream": stream, **kw})
        if stream:
            # Allow `response` to be either a list or an iterable.
            return iter(self.response)
        return self.response


@dataclass
class _SyncChat:
    completions: _SyncCompletions = field(default_factory=_SyncCompletions)


class FakeSyncOpenAI:
    def __init__(self, *_: Any, **__: Any):
        self._chat = _SyncChat()

    @property
    def chat(self) -> _SyncChat:
        return self._chat


# ---------- async fake client ----------

class FakeAsyncStream:
    """Minimal async iterable mirroring openai's AsyncStream surface.

    Supports: ``async for``, ``aclose()``, raises ``StopAsyncIteration`` cleanly,
    and also lets the test inject an exception mid-iteration.
    """

    def __init__(self, chunks: list[FakeChunk] | list[Any]):
        self._chunks = list(chunks)

    def __aiter__(self) -> AsyncIterator[Any]:
        async def gen() -> AsyncIterator[Any]:
            for c in self._chunks:
                if isinstance(c, BaseException):
                    raise c
                yield c
        return gen()

    async def close(self) -> None:
        return None


@dataclass
class _AsyncCompletions:
    create_calls: list[dict[str, Any]] = field(default_factory=list)
    response: Any = None

    async def create(self, *, messages: list[Any], stream: bool = False, **kw: Any) -> Any:
        self.create_calls.append({"messages": messages, "stream": stream, **kw})
        if stream:
            return FakeAsyncStream(self.response)
        return self.response


@dataclass
class _AsyncChat:
    completions: _AsyncCompletions = field(default_factory=_AsyncCompletions)


class FakeAsyncOpenAI:
    def __init__(self, *_: Any, **__: Any):
        self._chat = _AsyncChat()

    @property
    def chat(self) -> _AsyncChat:
        return self._chat


# ---------- fixtures ----------

@pytest.fixture
def sync_client(monkeypatch: pytest.MonkeyPatch):
    fake_module = type("_M", (), {"OpenAI": FakeSyncOpenAI, "AsyncOpenAI": object})
    monkeypatch.setattr(gheim_openai_mod, "_import_openai", lambda: fake_module)
    cls = gheim_openai_mod.__getattr__("OpenAI")
    return cls(gheim_detector=FakeDetector({"Joel": "private_person"}))


@pytest.fixture
def async_client(monkeypatch: pytest.MonkeyPatch):
    fake_module = type("_M", (), {"OpenAI": object, "AsyncOpenAI": FakeAsyncOpenAI})
    monkeypatch.setattr(gheim_openai_mod, "_import_openai", lambda: fake_module)
    cls = gheim_openai_mod.__getattr__("AsyncOpenAI")
    return cls(gheim_detector=FakeDetector({"Joel": "private_person"}))


# ---------- A.1 tool-call argument restoration ----------

def test_sync_tool_call_arguments_are_restored(sync_client):
    inner = sync_client.chat.completions.inner
    inner.response = FakeCompletion(
        choices=[
            FakeChoice(
                message=FakeMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        FakeToolCall(
                            id="call_1",
                            function=FakeFunction(
                                name="lookup_user",
                                arguments='{"name": "<PERSON_1>"}',
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ]
    )
    sess = Session(detector=FakeDetector({"Joel": "private_person"}))
    sess.allocate("private_person", "Joel")
    out = sync_client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "Joel"}], gheim_session=sess
    )
    args = out.choices[0].message.tool_calls[0].function.arguments
    assert args == '{"name": "Joel"}'


async def test_async_tool_call_arguments_are_restored(async_client):
    inner = async_client.chat.completions.inner
    inner.response = FakeCompletion(
        choices=[
            FakeChoice(
                message=FakeMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        FakeToolCall(
                            id="call_1",
                            function=FakeFunction(
                                name="lookup_user",
                                arguments='{"name": "<PERSON_1>"}',
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ]
    )
    sess = Session(detector=FakeDetector({"Joel": "private_person"}))
    sess.allocate("private_person", "Joel")
    out = await async_client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "Joel"}], gheim_session=sess
    )
    assert out.choices[0].message.tool_calls[0].function.arguments == '{"name": "Joel"}'


# ---------- A.2 gheim_session reuse across calls ----------

def test_sync_session_reuse_across_calls_keeps_sentinel_stable(sync_client):
    inner = sync_client.chat.completions.inner
    inner.response = FakeCompletion(
        choices=[FakeChoice(message=FakeMessage(role="assistant", content="<PERSON_1> noted"))]
    )
    sess = Session()  # explicit reuse

    sync_client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "I am Joel"}], gheim_session=sess
    )
    first_msgs = inner.create_calls[-1]["messages"]
    assert first_msgs[0]["content"] == "I am <PERSON_1>"

    sync_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Joel asked again"}],
        gheim_session=sess,
    )
    second_msgs = inner.create_calls[-1]["messages"]
    # Same Joel must reuse <PERSON_1> — multi-turn coherence.
    assert second_msgs[0]["content"] == "<PERSON_1> asked again"
    assert sess.mapping == {"<PERSON_1>": "Joel"}


async def test_async_session_reuse_across_calls_keeps_sentinel_stable(async_client):
    inner = async_client.chat.completions.inner
    inner.response = FakeCompletion(
        choices=[FakeChoice(message=FakeMessage(role="assistant", content="<PERSON_1> noted"))]
    )
    sess = Session()
    await async_client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "I am Joel"}], gheim_session=sess
    )
    await async_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Joel asked again"}],
        gheim_session=sess,
    )
    assert inner.create_calls[-1]["messages"][0]["content"] == "<PERSON_1> asked again"


# ---------- A.3 chunks with delta.content == None pass through ----------

def _role_only_chunk() -> FakeChunk:
    return FakeChunk(choices=[FakeChoiceChunk(delta=FakeDelta(role="assistant", content=None))])


def _finish_only_chunk() -> FakeChunk:
    return FakeChunk(
        choices=[FakeChoiceChunk(delta=FakeDelta(content=None), finish_reason="stop")]
    )


def test_sync_streaming_passes_role_only_and_finish_only_chunks(sync_client):
    inner = sync_client.chat.completions.inner
    inner.response = [
        _role_only_chunk(),
        FakeChunk(choices=[FakeChoiceChunk(delta=FakeDelta(content="Hi "))]),
        FakeChunk(choices=[FakeChoiceChunk(delta=FakeDelta(content="<PERSON_1>"))]),
        _finish_only_chunk(),
    ]
    sess = Session(detector=FakeDetector({"Joel": "private_person"}))
    sess.allocate("private_person", "Joel")
    chunks = list(sync_client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "Joel"}], stream=True, gheim_session=sess
    ))
    # All four upstream chunks must pass through; first/last have no content to mutate.
    assert len(chunks) == 4
    assert chunks[0].choices[0].delta.role == "assistant"
    assert chunks[0].choices[0].delta.content is None
    assert chunks[1].choices[0].delta.content == "Hi "
    assert chunks[2].choices[0].delta.content == "Joel"
    assert chunks[3].choices[0].finish_reason == "stop"


async def test_async_streaming_passes_role_only_and_finish_only_chunks(async_client):
    inner = async_client.chat.completions.inner
    inner.response = [
        _role_only_chunk(),
        FakeChunk(choices=[FakeChoiceChunk(delta=FakeDelta(content="Hi "))]),
        FakeChunk(choices=[FakeChoiceChunk(delta=FakeDelta(content="<PERSON_1>"))]),
        _finish_only_chunk(),
    ]
    sess = Session(detector=FakeDetector({"Joel": "private_person"}))
    sess.allocate("private_person", "Joel")
    stream = await async_client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "Joel"}], stream=True, gheim_session=sess
    )
    collected = []
    async for c in stream:
        collected.append(c)
    assert len(collected) == 4
    assert collected[2].choices[0].delta.content == "Joel"
    assert collected[3].choices[0].finish_reason == "stop"


# ---------- A.4 early break in async iteration ----------

async def test_async_streaming_early_break_does_not_raise(async_client):
    inner = async_client.chat.completions.inner
    inner.response = [
        FakeChunk(choices=[FakeChoiceChunk(delta=FakeDelta(content="Hi "))]),
        FakeChunk(choices=[FakeChoiceChunk(delta=FakeDelta(content="<PERSON_1>"))]),
        FakeChunk(choices=[FakeChoiceChunk(delta=FakeDelta(content=", how can I help?"))]),
    ]
    sess = Session(detector=FakeDetector({"Joel": "private_person"}))
    sess.allocate("private_person", "Joel")
    stream = await async_client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "Joel"}], stream=True, gheim_session=sess
    )
    seen = []
    async for c in stream:
        seen.append(c.choices[0].delta.content)
        if len(seen) == 1:
            break
    assert seen == ["Hi "]


# ---------- A.5 exception inside async iteration ----------

async def test_async_streaming_propagates_upstream_exception(async_client):
    inner = async_client.chat.completions.inner
    inner.response = [
        FakeChunk(choices=[FakeChoiceChunk(delta=FakeDelta(content="Hi <"))]),
        RuntimeError("upstream blew up"),
    ]
    sess = Session(detector=FakeDetector({"Joel": "private_person"}))
    sess.allocate("private_person", "Joel")
    stream = await async_client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "Joel"}], stream=True, gheim_session=sess
    )
    collected: list[str | None] = []
    with pytest.raises(RuntimeError, match="upstream blew up"):
        async for c in stream:
            collected.append(c.choices[0].delta.content)
    # We received the first chunk before the exception; held-back '<' is buried in the buffer.
    assert collected == ["Hi "]


# ---------- A.6 detector override per-call ----------

def test_sync_per_call_detector_override(sync_client):
    inner = sync_client.chat.completions.inner
    inner.response = FakeCompletion(
        choices=[FakeChoice(message=FakeMessage(role="assistant", content="ok"))]
    )
    # Wrapper default detector recognizes "Joel"; override only recognizes "Alice".
    alice_only = FakeDetector({"Alice": "private_person"})
    sync_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Joel and Alice"}],
        gheim_detector=alice_only,
    )
    sent = inner.create_calls[-1]["messages"][0]["content"]
    # "Joel" should NOT be redacted under the override; "Alice" should be.
    assert sent == "Joel and <PERSON_1>"
