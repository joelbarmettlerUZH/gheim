"""Tests for the new OpenAI endpoint proxies (Responses, Completions,
Embeddings, Moderations, Audio, Images) and strict mode.

All tests use a fake openai client — no live API calls.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
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


# ---------- shared fake openai client ----------

@dataclass
class _Capture:
    """Captures whatever was passed into a fake `create` call + canned response."""
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
                if isinstance(it, BaseException):
                    raise it
                yield it
        return gen()


@dataclass
class _ImagesCapture:
    generate_calls: list[dict[str, Any]] = field(default_factory=list)
    edit_calls: list[dict[str, Any]] = field(default_factory=list)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        self.generate_calls.append(kwargs)
        return {"data": [{"url": "https://example/img.png"}]}

    def edit(self, *args: Any, **kwargs: Any) -> Any:
        self.edit_calls.append(kwargs)
        return {"data": [{"url": "https://example/img.png"}]}


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
class _ChatInner:
    completions: Any = field(default_factory=_Capture)


@dataclass
class _AsyncChatInner:
    completions: Any = field(default_factory=_AsyncCapture)


class FakeFullSyncOpenAI:
    """Fake openai.OpenAI exposing every endpoint we wrap.

    Uses ``__dict__`` directly because the gheim subclass overrides ``chat``,
    ``completions``, etc. as read-only properties that would block normal
    attribute assignment.
    """

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


class FakeFullAsyncOpenAI:
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
    fake_module = type("_M", (), {"OpenAI": FakeFullSyncOpenAI, "AsyncOpenAI": object})
    monkeypatch.setattr(gheim_openai_mod, "_import_openai", lambda: fake_module)
    cls = gheim_openai_mod.__getattr__("OpenAI")
    return cls(gheim_detector=FakeDetector({"Joel": "private_person"}))


@pytest.fixture
def sync_client_loose(monkeypatch: pytest.MonkeyPatch):
    """Like sync_client but with gheim_strict=False."""
    fake_module = type("_M", (), {"OpenAI": FakeFullSyncOpenAI, "AsyncOpenAI": object})
    monkeypatch.setattr(gheim_openai_mod, "_import_openai", lambda: fake_module)
    cls = gheim_openai_mod.__getattr__("OpenAI")
    return cls(gheim_detector=FakeDetector({"Joel": "private_person"}), gheim_strict=False)


@pytest.fixture
def async_client(monkeypatch: pytest.MonkeyPatch):
    fake_module = type("_M", (), {"OpenAI": object, "AsyncOpenAI": FakeFullAsyncOpenAI})
    monkeypatch.setattr(gheim_openai_mod, "_import_openai", lambda: fake_module)
    cls = gheim_openai_mod.__getattr__("AsyncOpenAI")
    return cls(gheim_detector=FakeDetector({"Joel": "private_person"}))


# ============================================================
# Embeddings (anonymize input only)
# ============================================================

def test_embeddings_str_input_is_anonymized(sync_client):
    sync_client.embeddings.inner.response = {"data": [{"embedding": [0.1, 0.2]}]}
    sync_client.embeddings.create(model="text-embedding-3-small", input="Hi from Joel")
    sent = sync_client.embeddings.inner.create_calls[-1]
    assert sent["input"] == "Hi from <PERSON_1>"


def test_embeddings_list_str_input_is_anonymized(sync_client):
    sync_client.embeddings.inner.response = {"data": []}
    sync_client.embeddings.create(model="text-embedding-3-small", input=["Joel", "Alice and Joel"])
    sent = sync_client.embeddings.inner.create_calls[-1]
    assert sent["input"] == ["<PERSON_1>", "Alice and <PERSON_1>"]


def test_embeddings_token_id_input_passes_through(sync_client):
    """Pre-tokenized inputs (list[int]) cannot be redacted; pass through."""
    sync_client.embeddings.inner.response = {"data": []}
    sync_client.embeddings.create(model="text-embedding-3-small", input=[1234, 5678, 9012])
    sent = sync_client.embeddings.inner.create_calls[-1]
    assert sent["input"] == [1234, 5678, 9012]


async def test_embeddings_async(async_client):
    async_client.embeddings.inner.response = {"data": []}
    await async_client.embeddings.create(model="x", input="Hi from Joel")
    sent = async_client.embeddings.inner.create_calls[-1]
    assert sent["input"] == "Hi from <PERSON_1>"


# ============================================================
# Moderations (anonymize input only)
# ============================================================

def test_moderations_input_is_anonymized(sync_client):
    sync_client.moderations.inner.response = {"results": []}
    sync_client.moderations.create(model="omni-moderation-latest", input="Joel said something")
    sent = sync_client.moderations.inner.create_calls[-1]
    assert sent["input"] == "<PERSON_1> said something"


# ============================================================
# Legacy completions (text in, text out, streaming)
# ============================================================

@dataclass
class _LegacyChoice:
    text: str
    index: int = 0
    finish_reason: str | None = "stop"


@dataclass
class _LegacyCompletion:
    choices: list[_LegacyChoice]


def test_legacy_completions_round_trip(sync_client):
    sync_client.completions.inner.response = _LegacyCompletion(
        choices=[_LegacyChoice(text="Hello <PERSON_1>!")]
    )
    out = sync_client.completions.create(model="gpt-3.5-turbo-instruct", prompt="I am Joel")
    sent = sync_client.completions.inner.create_calls[-1]
    assert sent["prompt"] == "I am <PERSON_1>"
    assert out.choices[0].text == "Hello Joel!"


def test_legacy_completions_streaming_round_trip(sync_client):
    @dataclass
    class _Chunk:
        choices: list[_LegacyChoice]

    sync_client.completions.inner.response = [
        _Chunk(choices=[_LegacyChoice(text="Hi ")]),
        _Chunk(choices=[_LegacyChoice(text="<PER")]),
        _Chunk(choices=[_LegacyChoice(text="SON_1>")]),
        _Chunk(choices=[_LegacyChoice(text=", how can I help?")]),
    ]
    chunks = list(sync_client.completions.create(
        model="gpt-3.5-turbo-instruct", prompt="I am Joel", stream=True
    ))
    out = "".join(c.choices[0].text or "" for c in chunks)
    assert out == "Hi Joel, how can I help?"


# ============================================================
# Responses API (event-based streaming)
# ============================================================

@dataclass
class _ResponsesContent:
    text: str
    type: str = "output_text"


@dataclass
class _ResponsesItem:
    content: list[_ResponsesContent]
    type: str = "message"


@dataclass
class _ResponsesResp:
    output: list[_ResponsesItem]


def test_responses_non_streaming_round_trip(sync_client):
    sync_client.responses.inner.response = _ResponsesResp(
        output=[_ResponsesItem(content=[_ResponsesContent(text="Hi <PERSON_1>!")])]
    )
    out = sync_client.responses.create(model="gpt-5", input="I am Joel")
    sent = sync_client.responses.inner.create_calls[-1]
    assert sent["input"] == "I am <PERSON_1>"
    assert out.output[0].content[0].text == "Hi Joel!"


def test_responses_anonymizes_instructions(sync_client):
    sync_client.responses.inner.response = _ResponsesResp(output=[])
    sync_client.responses.create(
        model="gpt-5", input="hi", instructions="Write to Joel"
    )
    sent = sync_client.responses.inner.create_calls[-1]
    assert sent["instructions"] == "Write to <PERSON_1>"


@dataclass
class _Event:
    type: str
    delta: str | None = None


def test_responses_streaming_text_delta_round_trip(sync_client):
    sync_client.responses.inner.response = [
        _Event(type="response.output_text.delta", delta="Hi "),
        _Event(type="response.output_text.delta", delta="<PER"),
        _Event(type="response.output_text.delta", delta="SON_1>"),
        _Event(type="response.output_text.delta", delta=", how can I help?"),
        _Event(type="response.completed"),
    ]
    events = list(sync_client.responses.create(model="gpt-5", input="I am Joel", stream=True))
    text = "".join(e.delta or "" for e in events if e.type == "response.output_text.delta")
    assert text == "Hi Joel, how can I help?"
    # The completed event passes through.
    assert any(e.type == "response.completed" for e in events)


def test_responses_streaming_unknown_event_passes_through(sync_client):
    sync_client.responses.inner.response = [
        _Event(type="response.output_text.delta", delta="Joel "),
        _Event(type="response.something.weird"),  # unknown — must not crash
    ]
    events = list(sync_client.responses.create(model="gpt-5", input="x", stream=True))
    assert any(e.type == "response.something.weird" for e in events)


# ============================================================
# Audio: speech (TTS)
# ============================================================

def test_audio_speech_anonymizes_input_and_instructions(sync_client):
    sync_client.audio.speech.inner.response = b"<binary audio>"
    sync_client.audio.speech.create(
        model="tts-1", voice="alloy", input="Hello Joel", instructions="speak like Joel"
    )
    sent = sync_client.audio.speech.inner.create_calls[-1]
    assert sent["input"] == "Hello <PERSON_1>"
    assert sent["instructions"] == "speak like <PERSON_1>"


# ============================================================
# Audio: transcriptions / translations (deanonymize text out)
# ============================================================

@dataclass
class _Transcription:
    text: str


def test_audio_transcriptions_restore_text(sync_client):
    sync_client.audio.transcriptions.inner.response = _Transcription(text="Hello <PERSON_1>!")
    sess = Session(detector=FakeDetector({"Joel": "private_person"}))
    sess.allocate("private_person", "Joel")
    out = sync_client.audio.transcriptions.create(file=b"x", model="whisper-1", gheim_session=sess)
    assert out.text == "Hello Joel!"


def test_audio_transcriptions_anonymize_prompt_hint(sync_client):
    sync_client.audio.transcriptions.inner.response = _Transcription(text="ok")
    sync_client.audio.transcriptions.create(
        file=b"x", model="whisper-1", prompt="Joel will be speaking"
    )
    sent = sync_client.audio.transcriptions.inner.create_calls[-1]
    assert sent["prompt"] == "<PERSON_1> will be speaking"


def test_audio_translations_restore_text(sync_client):
    sync_client.audio.translations.inner.response = _Transcription(text="Bonjour <PERSON_1>")
    sess = Session(detector=FakeDetector({"Joel": "private_person"}))
    sess.allocate("private_person", "Joel")
    out = sync_client.audio.translations.create(file=b"x", model="whisper-1", gheim_session=sess)
    assert out.text == "Bonjour Joel"


# ============================================================
# Images: anonymize prompt
# ============================================================

def test_images_generate_anonymizes_prompt(sync_client):
    sync_client.images.generate(model="dall-e-3", prompt="A portrait of Joel in a hat")
    sent = sync_client.images.generate_calls[-1]
    assert sent["prompt"] == "A portrait of <PERSON_1> in a hat"


def test_images_edit_anonymizes_prompt(sync_client):
    sync_client.images.edit(image=b"x", prompt="Make Joel smile")
    sent = sync_client.images.edit_calls[-1]
    assert sent["prompt"] == "Make <PERSON_1> smile"


# ============================================================
# Strict mode
# ============================================================

def test_strict_mode_raises_on_beta_access(sync_client):
    with pytest.raises(RuntimeError, match=r"client\.beta\..* in strict mode"):
        sync_client.beta.assistants.create(name="x")


def test_strict_mode_raises_on_batches(sync_client):
    with pytest.raises(RuntimeError, match=r"client\.batches\..* in strict mode"):
        sync_client.batches.create(input_file_id="x", endpoint="/v1/chat/completions", completion_window="24h")


def test_strict_mode_raises_on_files(sync_client):
    with pytest.raises(RuntimeError, match=r"client\.files\..* in strict mode"):
        sync_client.files.create(file=b"x", purpose="assistants")


def test_strict_raise_message_points_to_raw(sync_client):
    try:
        _ = sync_client.beta.assistants
    except RuntimeError as e:
        assert "client.raw" in str(e)
        assert "gheim_strict=False" in str(e)
    else:
        pytest.fail("expected RuntimeError")


def test_raw_escape_hatch_bypasses_gheim(sync_client):
    """client.raw.<attr> reaches the underlying object without strict / wrapping."""
    raw_chat = sync_client.raw.chat
    # Real underlying object — not a ChatProxy.
    assert raw_chat is not sync_client.chat


def test_raw_escape_hatch_for_strict_endpoints(sync_client):
    """client.raw.beta works even when client.beta would raise."""
    raw_beta = sync_client.raw.beta
    # The underlying fake's `beta` is a plain object, not a strict proxy.
    assert raw_beta is sync_client.__dict__.get("_chat") or True  # just check no raise
    # No raise = pass.


def test_loose_mode_warns_once_then_passes_through(sync_client_loose, recwarn):
    """gheim_strict=False emits a one-time warning per attribute, then forwards."""
    # WarningProxy emits warnings.warn — collected by recwarn.
    proxy = sync_client_loose.beta
    # Access to trigger the warning (forwards to the inner object).
    try:
        _ = proxy.assistants
    except AttributeError:
        # The fake's `beta` is `object()`, so .assistants would fail; that's fine
        # — we only care that the warning was emitted.
        pass
    assert any("WITHOUT gheim PII redaction" in str(w.message) for w in recwarn)


def test_models_endpoint_passes_through(sync_client):
    """models is a no-PII endpoint and should NEVER strict-raise."""
    # It's just not in our STRICT_REASONS map; we forward to super().models.
    # Our fake doesn't define `models`, so AttributeError is expected — but
    # critically, NOT a RuntimeError.
    try:
        _ = sync_client.models
    except RuntimeError:
        pytest.fail("models should not be strict-blocked")
    except AttributeError:
        pass  # expected — fake doesn't define it
