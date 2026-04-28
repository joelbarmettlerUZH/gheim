"""Tests for gheim.openai — both sync (OpenAI) and async (AsyncOpenAI) wrappers.

We don't hit the live OpenAI API. Instead we build fake completion / chunk
objects and a fake openai.OpenAI / openai.AsyncOpenAI base class that the
wrapper subclasses, then drive everything through the proxy.
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any

import pytest

import gheim.openai as gheim_openai_mod
from gheim import Session, Span
from gheim.detectors.base import Detector


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


# Match the openai SDK's pydantic-ish chunk shape closely enough for the wrapper.
@dataclass
class FakeDelta:
    content: str | None = None
    role: str | None = None


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
    # Stand-in for pydantic's model_copy so the wrapper's tail-cloning path works.

    def model_copy(self, *, deep: bool = False) -> "FakeChunk":
        return FakeChunk(
            id=self.id,
            model=self.model,
            choices=[FakeChoiceChunk(delta=FakeDelta(content=c.delta.content, role=c.delta.role),
                                      index=c.index, finish_reason=c.finish_reason)
                     for c in self.choices],
        )


@dataclass
class FakeMessage:
    role: str
    content: str | None
    tool_calls: list[Any] | None = None


@dataclass
class FakeChoice:
    message: FakeMessage
    index: int = 0
    finish_reason: str | None = "stop"


@dataclass
class FakeCompletion:
    choices: list[FakeChoice]
    id: str = "cmpl-x"


def _make_chunks(deltas: list[str]) -> list[FakeChunk]:
    return [FakeChunk(choices=[FakeChoiceChunk(delta=FakeDelta(content=d))]) for d in deltas]


# ---------- Sync fake openai client ----------

@dataclass
class _SyncCompletions:
    last_messages: list[Any] = field(default_factory=list)
    response: Any = None

    def create(self, *, messages: list[Any], stream: bool = False, **_: Any) -> Any:
        # Capture what was actually sent to "OpenAI".
        self.last_messages = messages
        if stream:
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


@pytest.fixture
def sync_client(monkeypatch: pytest.MonkeyPatch):
    """Replace openai.OpenAI with FakeSyncOpenAI, then build a gheim wrapper."""
    fake_module = type("_M", (), {"OpenAI": FakeSyncOpenAI, "AsyncOpenAI": object})
    monkeypatch.setattr(gheim_openai_mod, "_import_openai", lambda: fake_module)
    cls = gheim_openai_mod.__getattr__("OpenAI")
    detector: Detector = FakeDetector({"Joel": "private_person"})
    client = cls(gheim_detector=detector)
    return client


def test_sync_non_streaming_round_trip(sync_client):
    inner = sync_client._gheim_chat.completions.inner
    inner.response = FakeCompletion(
        choices=[FakeChoice(message=FakeMessage(role="assistant", content="Hello <PERSON_1>!"))]
    )
    sess = Session(detector=FakeDetector({"Joel": "private_person"}))
    out = sync_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "I am Joel"}],
        gheim_session=sess,
    )
    # OpenAI saw the redacted form.
    assert inner.last_messages[0]["content"] == "<PERSON_1> "[: -1] or inner.last_messages[0]["content"] == "I am <PERSON_1>"
    assert inner.last_messages[0]["content"] == "I am <PERSON_1>"
    # End user sees the restored form.
    assert out.choices[0].message.content == "Hello Joel!"


def test_sync_streaming_round_trip(sync_client):
    inner = sync_client._gheim_chat.completions.inner
    inner.response = _make_chunks(["Hi ", "<PER", "SON_1>", ", how can I help?"])
    sess = Session(detector=FakeDetector({"Joel": "private_person"}))
    stream = sync_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "I am Joel"}],
        stream=True,
        gheim_session=sess,
    )
    collected = "".join(chunk.choices[0].delta.content or "" for chunk in stream)
    assert collected == "Hi Joel, how can I help?"


# ---------- Async fake openai client ----------

class FakeAsyncStream:
    def __init__(self, chunks: list[FakeChunk]):
        self._chunks = chunks

    def __aiter__(self) -> AsyncIterator[FakeChunk]:
        async def gen() -> AsyncIterator[FakeChunk]:
            for c in self._chunks:
                yield c
        return gen()

    async def close(self) -> None:
        pass


@dataclass
class _AsyncCompletions:
    last_messages: list[Any] = field(default_factory=list)
    response: Any = None

    async def create(self, *, messages: list[Any], stream: bool = False, **_: Any) -> Any:
        self.last_messages = messages
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


@pytest.fixture
def async_client(monkeypatch: pytest.MonkeyPatch):
    fake_module = type("_M", (), {"OpenAI": object, "AsyncOpenAI": FakeAsyncOpenAI})
    monkeypatch.setattr(gheim_openai_mod, "_import_openai", lambda: fake_module)
    cls = gheim_openai_mod.__getattr__("AsyncOpenAI")
    detector: Detector = FakeDetector({"Joel": "private_person"})
    client = cls(gheim_detector=detector)
    return client


async def test_async_non_streaming_round_trip(async_client):
    inner = async_client._gheim_chat.completions.inner
    inner.response = FakeCompletion(
        choices=[FakeChoice(message=FakeMessage(role="assistant", content="Hello <PERSON_1>!"))]
    )
    sess = Session(detector=FakeDetector({"Joel": "private_person"}))
    out = await async_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "I am Joel"}],
        gheim_session=sess,
    )
    assert inner.last_messages[0]["content"] == "I am <PERSON_1>"
    assert out.choices[0].message.content == "Hello Joel!"


async def test_async_streaming_round_trip(async_client):
    inner = async_client._gheim_chat.completions.inner
    inner.response = _make_chunks(["Hi ", "<PER", "SON_1>", ", how can I help?"])
    sess = Session(detector=FakeDetector({"Joel": "private_person"}))
    stream = await async_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "I am Joel"}],
        stream=True,
        gheim_session=sess,
    )
    collected = ""
    async for chunk in stream:
        collected += chunk.choices[0].delta.content or ""
    assert collected == "Hi Joel, how can I help?"


async def test_async_streaming_handles_held_back_tail(async_client):
    """Tail buffer must surface as a synthetic final chunk."""
    inner = async_client._gheim_chat.completions.inner
    # Last delta ends with an incomplete sentinel — flush() must emit it after stream end.
    inner.response = _make_chunks(["done <PERSON_"])
    sess = Session(detector=FakeDetector({"Joel": "private_person"}))
    sess.allocate("private_person", "Joel")  # so <PERSON_1> would map; the partial <PERSON_ doesn't
    stream = await async_client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "x"}], stream=True, gheim_session=sess
    )
    out = ""
    async for chunk in stream:
        out += chunk.choices[0].delta.content or ""
    assert out == "done <PERSON_"
