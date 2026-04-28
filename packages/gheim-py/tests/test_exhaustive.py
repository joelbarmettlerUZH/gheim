"""Exhaustive coverage for cross-cutting concerns the per-endpoint suites miss.

Topics:
  - Cross-endpoint session reuse (same Session passed to chat → embeddings → responses)
  - Multi-tool-call streaming + multiple choices per chunk
  - Nested chat content parts (multimodal text + image)
  - Empty / null content, no messages, no tools
  - Concurrent requests through one client (threading + asyncio)
  - Strict + raw + loose interactions
  - Session JSON shape unchanged after refactor (backward-compat)
  - Sentinel survival across every wrapped endpoint in one Session
"""
from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import pytest

import gheim.openai as gheim_openai_mod
from gheim import Session, Span
from gheim.detectors.base import Detector


# ============================================================
# Shared fakes (mirror test_openai_endpoints.py shapes)
# ============================================================

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
class _Capture:
    create_calls: list[dict[str, Any]] = field(default_factory=list)
    response: Any = None

    def create(self, *args: Any, **kwargs: Any) -> Any:
        self.create_calls.append(kwargs)
        if kwargs.get("stream"):
            return iter(self.response or [])
        return self.response


@dataclass
class _AsyncCapture:
    create_calls: list[dict[str, Any]] = field(default_factory=list)
    response: Any = None

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        self.create_calls.append(kwargs)
        if kwargs.get("stream"):
            return _FakeAsyncIter(self.response or [])
        return self.response


class _FakeAsyncIter:
    def __init__(self, items: list[Any]):
        self._items = items

    def __aiter__(self) -> AsyncIterator[Any]:
        async def gen() -> AsyncIterator[Any]:
            for it in self._items:
                yield it
        return gen()


@dataclass
class _ChatInner:
    completions: Any = field(default_factory=_Capture)


@dataclass
class _AsyncChatInner:
    completions: Any = field(default_factory=_AsyncCapture)


@dataclass
class _AudioInner:
    speech: Any = field(default_factory=_Capture)
    transcriptions: Any = field(default_factory=_Capture)
    translations: Any = field(default_factory=_Capture)


@dataclass
class _AsyncAudioInner:
    speech: Any = field(default_factory=_AsyncCapture)
    transcriptions: Any = field(default_factory=_AsyncCapture)
    translations: Any = field(default_factory=_AsyncCapture)


@dataclass
class _ImagesCapture:
    generate_calls: list[dict[str, Any]] = field(default_factory=list)
    edit_calls: list[dict[str, Any]] = field(default_factory=list)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        self.generate_calls.append(kwargs)
        return {"data": [{"url": "x"}]}

    def edit(self, *args: Any, **kwargs: Any) -> Any:
        self.edit_calls.append(kwargs)
        return {"data": [{"url": "x"}]}


class _FakeFullSync:
    def __init__(self, *_: Any, **__: Any):
        d = self.__dict__
        d["chat"] = _ChatInner()
        d["completions"] = _Capture()
        d["responses"] = _Capture()
        d["embeddings"] = _Capture()
        d["moderations"] = _Capture()
        d["audio"] = _AudioInner()
        d["images"] = _ImagesCapture()
        d["beta"] = object()
        d["batches"] = object()
        d["files"] = object()


class _FakeFullAsync:
    def __init__(self, *_: Any, **__: Any):
        d = self.__dict__
        d["chat"] = _AsyncChatInner()
        d["completions"] = _AsyncCapture()
        d["responses"] = _AsyncCapture()
        d["embeddings"] = _AsyncCapture()
        d["moderations"] = _AsyncCapture()
        d["audio"] = _AsyncAudioInner()
        d["images"] = _ImagesCapture()
        d["beta"] = object()


@pytest.fixture
def sync_client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        gheim_openai_mod,
        "_import_openai",
        lambda: type("_M", (), {"OpenAI": _FakeFullSync, "AsyncOpenAI": object}),
    )
    cls = gheim_openai_mod.__getattr__("OpenAI")
    return cls(gheim_detector=FakeDetector({"Joel": "private_person", "Alice": "private_person"}))


@pytest.fixture
def async_client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        gheim_openai_mod,
        "_import_openai",
        lambda: type("_M", (), {"OpenAI": object, "AsyncOpenAI": _FakeFullAsync}),
    )
    cls = gheim_openai_mod.__getattr__("AsyncOpenAI")
    return cls(gheim_detector=FakeDetector({"Joel": "private_person", "Alice": "private_person"}))


# ============================================================
# Cross-endpoint session reuse
# ============================================================

@dataclass
class _Msg:
    role: str
    content: str | None
    tool_calls: list[Any] | None = None


@dataclass
class _Choice:
    message: _Msg
    index: int = 0
    finish_reason: str | None = "stop"


@dataclass
class _Completion:
    choices: list[_Choice]


def test_session_reused_across_chat_then_embeddings_keeps_sentinel(sync_client):
    """Same Joel passed through chat first, then embeddings, must use <PERSON_1> in both."""
    sync_client.chat.completions.inner.response = _Completion(
        choices=[_Choice(message=_Msg(role="assistant", content="ok"))]
    )
    sync_client.embeddings.inner.response = {"data": []}

    sess = Session()
    sync_client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "I am Joel"}], gheim_session=sess
    )
    sync_client.embeddings.create(
        model="text-embedding-3-small", input="Joel works at Acme", gheim_session=sess
    )
    assert (
        sync_client.chat.completions.inner.create_calls[0]["messages"][0]["content"]
        == "I am <PERSON_1>"
    )
    assert sync_client.embeddings.inner.create_calls[0]["input"] == "<PERSON_1> works at Acme"
    assert sess.mapping == {"<PERSON_1>": "Joel"}


def test_session_reused_across_chat_responses_audio_images(sync_client):
    """One Session threaded through every wrapped endpoint — sentinels stay coherent."""
    sync_client.chat.completions.inner.response = _Completion(
        choices=[_Choice(message=_Msg(role="assistant", content="ok"))]
    )
    sync_client.responses.inner.response = type("R", (), {"output": []})()
    sync_client.embeddings.inner.response = {"data": []}
    sync_client.moderations.inner.response = {"results": []}
    sync_client.audio.speech.inner.response = b""
    sync_client.audio.transcriptions.inner.response = type("T", (), {"text": "ok"})()
    sync_client.completions.inner.response = type(
        "C", (), {"choices": [type("CC", (), {"text": "ok"})()]}
    )()

    sess = Session()
    sync_client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "Joel here"}], gheim_session=sess
    )
    sync_client.responses.create(model="gpt-5", input="Hello Joel", gheim_session=sess)
    sync_client.embeddings.create(model="x", input="Joel works", gheim_session=sess)
    sync_client.moderations.create(model="x", input="Joel said x", gheim_session=sess)
    sync_client.audio.speech.create(model="tts-1", voice="alloy", input="Hi Joel", gheim_session=sess)
    sync_client.audio.transcriptions.create(file=b"", model="whisper-1", prompt="Joel speaks", gheim_session=sess)
    sync_client.completions.create(model="gpt-3.5-turbo-instruct", prompt="Joel said", gheim_session=sess)
    sync_client.images.generate(model="dall-e-3", prompt="Portrait of Joel", gheim_session=sess)
    sync_client.images.edit(image=b"", prompt="Make Joel smile", gheim_session=sess)

    # All calls must have routed Joel through <PERSON_1> — no second sentinel.
    assert sess.mapping == {"<PERSON_1>": "Joel"}

    # Spot-check the actual outbound payloads.
    assert sync_client.chat.completions.inner.create_calls[0]["messages"][0]["content"] == "<PERSON_1> here"
    assert sync_client.responses.inner.create_calls[0]["input"] == "Hello <PERSON_1>"
    assert sync_client.embeddings.inner.create_calls[0]["input"] == "<PERSON_1> works"
    assert sync_client.moderations.inner.create_calls[0]["input"] == "<PERSON_1> said x"
    assert sync_client.audio.speech.inner.create_calls[0]["input"] == "Hi <PERSON_1>"
    assert sync_client.audio.transcriptions.inner.create_calls[0]["prompt"] == "<PERSON_1> speaks"
    assert sync_client.completions.inner.create_calls[0]["prompt"] == "<PERSON_1> said"
    assert sync_client.images.generate_calls[0]["prompt"] == "Portrait of <PERSON_1>"
    assert sync_client.images.edit_calls[0]["prompt"] == "Make <PERSON_1> smile"


# ============================================================
# Multi-tool-call streaming + multi-choice
# ============================================================

@dataclass
class _Delta:
    content: str | None = None
    role: str | None = None
    tool_calls: list[Any] | None = None


@dataclass
class _ChoiceChunk:
    delta: _Delta
    index: int = 0
    finish_reason: str | None = None


@dataclass
class _Chunk:
    choices: list[_ChoiceChunk]
    id: str = "x"
    model: str = "gpt-4o"

    def model_copy(self, *, deep: bool = False) -> "_Chunk":
        return _Chunk(
            id=self.id,
            model=self.model,
            choices=[
                _ChoiceChunk(
                    delta=_Delta(content=c.delta.content, role=c.delta.role, tool_calls=c.delta.tool_calls),
                    index=c.index,
                    finish_reason=c.finish_reason,
                )
                for c in self.choices
            ],
        )


def test_streaming_chunk_with_multiple_choices_each_has_content(sync_client):
    """Some models stream multiple choices in parallel (n>1). Each delta must restore."""
    sess = Session()
    sess.allocate("private_person", "Joel")
    sync_client.chat.completions.inner.response = [
        _Chunk(
            choices=[
                _ChoiceChunk(delta=_Delta(content="A: hi <PERSON_1>"), index=0),
                _ChoiceChunk(delta=_Delta(content="B: hello <PERSON_1>"), index=1),
            ]
        )
    ]
    chunks = list(sync_client.chat.completions.create(
        model="gpt-4o", n=2, messages=[{"role": "user", "content": "Joel"}], stream=True, gheim_session=sess
    ))
    assert chunks[0].choices[0].delta.content == "A: hi Joel"
    assert chunks[0].choices[1].delta.content == "B: hello Joel"


def test_non_streaming_response_with_multiple_choices(sync_client):
    sess = Session()
    sess.allocate("private_person", "Joel")
    sync_client.chat.completions.inner.response = _Completion(
        choices=[
            _Choice(message=_Msg(role="assistant", content="A: <PERSON_1>"), index=0),
            _Choice(message=_Msg(role="assistant", content="B: <PERSON_1>"), index=1),
        ]
    )
    out = sync_client.chat.completions.create(
        model="gpt-4o", n=2, messages=[{"role": "user", "content": "Joel"}], gheim_session=sess
    )
    assert out.choices[0].message.content == "A: Joel"
    assert out.choices[1].message.content == "B: Joel"


def test_response_with_multiple_tool_calls_all_restored(sync_client):
    @dataclass
    class _Fn:
        name: str
        arguments: str

    @dataclass
    class _TC:
        id: str
        function: _Fn
        type: str = "function"

    sess = Session()
    sess.allocate("private_person", "Joel")
    sync_client.chat.completions.inner.response = _Completion(
        choices=[
            _Choice(
                message=_Msg(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        _TC(id="1", function=_Fn(name="lookup", arguments='{"name": "<PERSON_1>"}')),
                        _TC(id="2", function=_Fn(name="audit", arguments='{"who": "<PERSON_1>"}')),
                    ],
                ),
                finish_reason="tool_calls",
            )
        ]
    )
    out = sync_client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "lookup Joel"}], gheim_session=sess
    )
    assert out.choices[0].message.tool_calls[0].function.arguments == '{"name": "Joel"}'
    assert out.choices[0].message.tool_calls[1].function.arguments == '{"who": "Joel"}'


# ============================================================
# Tool definition redaction (input side, new in this commit)
# ============================================================

def test_tool_function_description_is_anonymized(sync_client):
    sync_client.chat.completions.inner.response = _Completion(
        choices=[_Choice(message=_Msg(role="assistant", content="ok"))]
    )
    sync_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up Joel's records",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )
    sent = sync_client.chat.completions.inner.create_calls[0]
    assert sent["tools"][0]["function"]["description"] == "Look up <PERSON_1>'s records"
    # Parameters must NOT be touched (per design decision).
    assert sent["tools"][0]["function"]["parameters"] == {"type": "object", "properties": {}}


def test_tool_without_description_passes_through(sync_client):
    sync_client.chat.completions.inner.response = _Completion(
        choices=[_Choice(message=_Msg(role="assistant", content="ok"))]
    )
    sync_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "x"}}],
    )
    sent = sync_client.chat.completions.inner.create_calls[0]
    assert sent["tools"] == [{"type": "function", "function": {"name": "x"}}]


def test_message_name_field_is_anonymized(sync_client):
    sync_client.chat.completions.inner.response = _Completion(
        choices=[_Choice(message=_Msg(role="assistant", content="ok"))]
    )
    sync_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hi", "name": "Joel"}],
    )
    sent_msgs = sync_client.chat.completions.inner.create_calls[0]["messages"]
    assert sent_msgs[0]["name"] == "<PERSON_1>"


# ============================================================
# Multimodal content parts
# ============================================================

def test_multimodal_message_text_parts_redacted_image_passes(sync_client):
    sync_client.chat.completions.inner.response = _Completion(
        choices=[_Choice(message=_Msg(role="assistant", content="ok"))]
    )
    sync_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "This is Joel"},
                    {"type": "image_url", "image_url": {"url": "https://x.png"}},
                    {"type": "text", "text": "with another mention of Joel"},
                ],
            }
        ],
    )
    parts = sync_client.chat.completions.inner.create_calls[0]["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "This is <PERSON_1>"}
    assert parts[1] == {"type": "image_url", "image_url": {"url": "https://x.png"}}
    assert parts[2] == {"type": "text", "text": "with another mention of <PERSON_1>"}


def test_multimodal_with_no_text_parts(sync_client):
    sync_client.chat.completions.inner.response = _Completion(
        choices=[_Choice(message=_Msg(role="assistant", content="ok"))]
    )
    sync_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "https://x"}}],
            }
        ],
    )
    parts = sync_client.chat.completions.inner.create_calls[0]["messages"][0]["content"]
    assert parts == [{"type": "image_url", "image_url": {"url": "https://x"}}]


# ============================================================
# Empty / null edge cases per endpoint
# ============================================================

def test_chat_with_no_tools_field_does_not_crash(sync_client):
    sync_client.chat.completions.inner.response = _Completion(
        choices=[_Choice(message=_Msg(role="assistant", content="ok"))]
    )
    sync_client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "Joel"}]
    )
    # No assertion needed beyond "didn't raise".


def test_chat_with_empty_messages_list(sync_client):
    sync_client.chat.completions.inner.response = _Completion(
        choices=[_Choice(message=_Msg(role="assistant", content="ok"))]
    )
    sync_client.chat.completions.create(model="gpt-4o", messages=[])
    assert sync_client.chat.completions.inner.create_calls[0]["messages"] == []


def test_chat_with_null_content_in_message(sync_client):
    sync_client.chat.completions.inner.response = _Completion(
        choices=[_Choice(message=_Msg(role="assistant", content="ok"))]
    )
    sync_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "assistant", "content": None, "tool_calls": []},
            {"role": "user", "content": "Joel"},
        ],
    )
    sent = sync_client.chat.completions.inner.create_calls[0]["messages"]
    assert sent[0]["content"] is None
    assert sent[1]["content"] == "<PERSON_1>"


def test_embeddings_with_empty_string_input_produces_no_sentinels(sync_client):
    sync_client.embeddings.inner.response = {"data": []}
    sess = Session()
    sync_client.embeddings.create(model="x", input="", gheim_session=sess)
    assert sess.mapping == {}


def test_responses_without_instructions_works(sync_client):
    sync_client.responses.inner.response = type("R", (), {"output": []})()
    sync_client.responses.create(model="gpt-5", input="Joel here")
    sent = sync_client.responses.inner.create_calls[0]
    assert sent["input"] == "<PERSON_1> here"
    assert "instructions" not in sent


# ============================================================
# Concurrency
# ============================================================

def test_concurrent_threaded_chat_calls_share_no_session_state(sync_client):
    """No shared Session → independent counters per call. Each call gets <PERSON_1>."""
    sync_client.chat.completions.inner.response = _Completion(
        choices=[_Choice(message=_Msg(role="assistant", content="ok"))]
    )
    barrier = threading.Barrier(8)

    def call(_: int) -> str:
        barrier.wait()
        sync_client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "Joel"}]
        )
        return sync_client.chat.completions.inner.create_calls[-1]["messages"][0]["content"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(call, range(8)))
    # All callers should have produced the same redacted form.
    assert all(r == "<PERSON_1>" for r in results)
    assert len(sync_client.chat.completions.inner.create_calls) == 8


async def test_concurrent_async_chat_calls(async_client):
    """asyncio.gather concurrent async create calls."""
    async_client.chat.completions.inner.response = _Completion(
        choices=[_Choice(message=_Msg(role="assistant", content="ok"))]
    )

    async def one():
        await async_client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "Joel"}]
        )

    await asyncio.gather(*(one() for _ in range(20)))
    assert len(async_client.chat.completions.inner.create_calls) == 20


# ============================================================
# Strict + raw + loose interaction
# ============================================================

def test_strict_raise_through_chained_attribute_access(sync_client):
    """client.beta.assistants.create chained access still raises (proxy descends)."""
    with pytest.raises(RuntimeError):
        sync_client.beta.assistants.create(name="x")
    with pytest.raises(RuntimeError):
        sync_client.beta.assistants.update(assistant_id="x")


def test_raw_works_after_strict_raise(sync_client):
    """Even after a strict raise, client.raw is still functional."""
    try:
        sync_client.beta.assistants.create(name="x")
    except RuntimeError:
        pass
    raw = sync_client.raw
    assert raw.chat is not sync_client.chat  # raw is the underlying object


def test_loose_mode_does_not_warn_on_wrapped_endpoints(monkeypatch: pytest.MonkeyPatch):
    """Embeddings is wrapped — no warning even in loose mode."""
    monkeypatch.setattr(
        gheim_openai_mod,
        "_import_openai",
        lambda: type("_M", (), {"OpenAI": _FakeFullSync, "AsyncOpenAI": object}),
    )
    cls = gheim_openai_mod.__getattr__("OpenAI")
    client = cls(
        gheim_detector=FakeDetector({"Joel": "private_person"}),
        gheim_strict=False,
    )
    client.embeddings.inner.response = {"data": []}
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # turn warnings into exceptions
        client.embeddings.create(model="x", input="Joel")
    # If we got here, no warning fired.


# ============================================================
# Session JSON shape stability (cross-language + backward compat)
# ============================================================

def test_session_to_json_shape_unchanged_after_endpoint_refactor():
    """to_json should still emit just {'mapping': ...}. Never add fields without
    bumping a version key — JS Session.fromJSON is paired to this shape."""
    sess = Session()
    sess.allocate("private_person", "Joel")
    sess.allocate("private_email", "joel@x")
    raw = sess.to_json()
    parsed = json.loads(raw)
    assert set(parsed.keys()) == {"mapping"}
    assert parsed["mapping"] == {
        "<PERSON_1>": "Joel",
        "<EMAIL_1>": "joel@x",
    }


def test_session_from_json_with_unknown_fields_works():
    """Forward-compat: extra top-level fields don't break restore."""
    sess = Session.from_json(json.dumps({
        "mapping": {"<PERSON_1>": "Joel"},
        "schema_version": "0.2",
        "future": True,
    }))
    assert sess.mapping == {"<PERSON_1>": "Joel"}


# ============================================================
# Sentinel survival — every wrapped endpoint, one session
# ============================================================

def test_every_wrapped_endpoint_uses_same_sentinel_for_same_surface(sync_client):
    """Build a single Session, hit every endpoint, and assert exactly ONE sentinel
    was allocated for "Joel". This is the single most important property of the
    cross-endpoint round-trip story."""
    sess = Session()
    # Stub responses that don't matter for input verification.
    sync_client.chat.completions.inner.response = _Completion(
        choices=[_Choice(message=_Msg(role="assistant", content="ok"))]
    )
    sync_client.responses.inner.response = type("R", (), {"output": []})()
    sync_client.embeddings.inner.response = {"data": []}
    sync_client.moderations.inner.response = {"results": []}
    sync_client.audio.speech.inner.response = b""
    sync_client.audio.transcriptions.inner.response = type("T", (), {"text": "ok"})()
    sync_client.audio.translations.inner.response = type("T", (), {"text": "ok"})()
    sync_client.completions.inner.response = type(
        "C", (), {"choices": [type("CC", (), {"text": "ok"})()]}
    )()

    # Hit every wrapped endpoint with the SAME PII surface.
    text_with_joel = "Joel here"

    sync_client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": text_with_joel}], gheim_session=sess
    )
    sync_client.responses.create(model="gpt-5", input=text_with_joel, gheim_session=sess)
    sync_client.completions.create(model="x", prompt=text_with_joel, gheim_session=sess)
    sync_client.embeddings.create(model="x", input=text_with_joel, gheim_session=sess)
    sync_client.moderations.create(model="x", input=text_with_joel, gheim_session=sess)
    sync_client.audio.speech.create(model="tts", voice="alloy", input=text_with_joel, gheim_session=sess)
    sync_client.audio.transcriptions.create(file=b"", model="whisper-1", prompt=text_with_joel, gheim_session=sess)
    sync_client.audio.translations.create(file=b"", model="whisper-1", prompt=text_with_joel, gheim_session=sess)
    sync_client.images.generate(model="dall-e-3", prompt=text_with_joel, gheim_session=sess)
    sync_client.images.edit(image=b"", prompt=text_with_joel, gheim_session=sess)

    # ONE sentinel total — coalescing held across all endpoints.
    assert sess.mapping == {"<PERSON_1>": "Joel"}


# ============================================================
# Default session not provided — each call gets a fresh one
# ============================================================

def test_no_explicit_session_means_fresh_session_per_call(sync_client):
    """Without `gheim_session=`, two independent calls each get their own Session,
    so each call's "Joel" gets re-numbered as <PERSON_1>."""
    sync_client.embeddings.inner.response = {"data": []}
    sync_client.embeddings.create(model="x", input="Joel here")
    sync_client.embeddings.create(model="x", input="Joel again")
    a = sync_client.embeddings.inner.create_calls[0]["input"]
    b = sync_client.embeddings.inner.create_calls[1]["input"]
    assert a == "<PERSON_1> here"
    assert b == "<PERSON_1> again"  # restarts at 1 (independent Session)


# ============================================================
# gheim_detector per-call override on every endpoint
# ============================================================

def test_per_call_detector_override_on_embeddings(sync_client):
    """Default detector knows Joel; override only knows Alice. Embedding input must
    use the override."""
    sync_client.embeddings.inner.response = {"data": []}
    alice_only = FakeDetector({"Alice": "private_person"})
    sync_client.embeddings.create(
        model="x", input="Joel and Alice", gheim_detector=alice_only
    )
    sent = sync_client.embeddings.inner.create_calls[0]["input"]
    assert sent == "Joel and <PERSON_1>"  # Joel passes through; Alice gets sentinel


def test_per_call_detector_override_on_responses(sync_client):
    sync_client.responses.inner.response = type("R", (), {"output": []})()
    alice_only = FakeDetector({"Alice": "private_person"})
    sync_client.responses.create(
        model="gpt-5", input="Joel and Alice", gheim_detector=alice_only
    )
    sent = sync_client.responses.inner.create_calls[0]["input"]
    assert sent == "Joel and <PERSON_1>"
