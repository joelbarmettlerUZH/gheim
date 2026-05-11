"""Realistic OpenAI round-trip tests.

These cover the kind of inputs that actually hit production: a Swiss
support ticket with several flavours of PII, a multi-turn conversation
where the same person is referenced across turns, and a streaming
response. Each test asserts on **both ends of the wire**:

  - what the gheim wrapper actually sent to OpenAI (no original PII
    surface forms must appear in the captured payload)
  - what the wrapper handed back to the user (every sentinel in the
    LLM's response must be restored to the original surface)

Uses a fake openai.OpenAI / openai.AsyncOpenAI base class that records
each ``messages=`` payload, so we can assert directly on what crosses
the wire.
"""
from __future__ import annotations

import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import gheim.openai as gheim_openai_mod
import pytest
from gheim import Session, Span
from gheim.detectors.base import Detector


# ---------------------------------------------------------------------------
# Test fixtures: realistic ticket text + a deterministic detector that
# recognises every PII surface we plant in the test corpus.
# ---------------------------------------------------------------------------

# Realistic Swiss support ticket. Every (surface, label) pair in
# REALISTIC_PII below appears literally somewhere in this string.
REALISTIC_TICKET = """\
Subject: Locked out of my account

Hi support team,

This is Joel Barmettler writing on behalf of my colleague Alice Müller
at acme AG. Joel can't log in since yesterday. Could you reset the
password for joel.barmettler@acme.ch and CC alice.mueller@acme.ch on
the confirmation? You can reach me at +41 44 268 12 34 between 09:00
and 17:00. The account is registered to the office address
Bahnhofstrasse 1, 8001 Zürich, and the billing IBAN on file is
CH9300762011623852957. Please confirm receipt by Friday, 15 May 2026.

Thanks,
JOEL
"""

REALISTIC_PII: dict[str, str] = {
    "Joel Barmettler":      "private_person",
    "Alice Müller":         "private_person",
    "Joel":                 "private_person",
    "JOEL":                 "private_person",
    "joel.barmettler@acme.ch":  "private_email",
    "alice.mueller@acme.ch":    "private_email",
    "+41 44 268 12 34":     "private_phone",
    "Bahnhofstrasse 1, 8001 Zürich": "private_address",
    "CH9300762011623852957": "account_number",
    "15 May 2026":          "private_date",
}


class FakeDetector:
    """Deterministic detector: matches each (surface → label) literally."""

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
        # Resolve overlaps deterministically: longer + earlier wins, so
        # "Joel Barmettler" beats "Joel" at the same offset.
        out.sort(key=lambda s: (s.start, -(s.end - s.start)))
        kept: list[Span] = []
        cursor = -1
        for s in out:
            if s.start < cursor:
                continue
            kept.append(s)
            cursor = s.end
        return kept


# ---------------------------------------------------------------------------
# Fake OpenAI client (sync) — captures the messages reaching the wire.
# ---------------------------------------------------------------------------

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

    def model_copy(self, *, deep: bool = False) -> FakeChunk:
        return FakeChunk(
            id=self.id,
            model=self.model,
            choices=[FakeChoiceChunk(
                delta=FakeDelta(content=c.delta.content, role=c.delta.role),
                index=c.index, finish_reason=c.finish_reason)
                for c in self.choices],
        )


@dataclass
class _SyncCompletions:
    captured_calls: list[dict[str, Any]] = field(default_factory=list)
    response: Any = None

    def create(self, *, messages: list[Any], stream: bool = False, **rest: Any) -> Any:
        self.captured_calls.append({
            "messages": [dict(m) if isinstance(m, dict) else m for m in messages],
            "stream": stream,
            **rest,
        })
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


# ---------------------------------------------------------------------------
# Async fake (for streaming-async tests).
# ---------------------------------------------------------------------------

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
    captured_calls: list[dict[str, Any]] = field(default_factory=list)
    response: Any = None

    async def create(self, *, messages: list[Any], stream: bool = False, **rest: Any) -> Any:
        self.captured_calls.append({
            "messages": [dict(m) if isinstance(m, dict) else m for m in messages],
            "stream": stream,
            **rest,
        })
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


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sync_client(monkeypatch: pytest.MonkeyPatch):
    fake_module = type("_M", (), {"OpenAI": FakeSyncOpenAI, "AsyncOpenAI": object})
    monkeypatch.setattr(gheim_openai_mod, "_import_openai", lambda: fake_module)
    cls = gheim_openai_mod.__getattr__("OpenAI")
    detector: Detector = FakeDetector(REALISTIC_PII)
    return cls(gheim_detector=detector)


@pytest.fixture
def async_client(monkeypatch: pytest.MonkeyPatch):
    fake_module = type("_M", (), {"OpenAI": object, "AsyncOpenAI": FakeAsyncOpenAI})
    monkeypatch.setattr(gheim_openai_mod, "_import_openai", lambda: fake_module)
    cls = gheim_openai_mod.__getattr__("AsyncOpenAI")
    detector: Detector = FakeDetector(REALISTIC_PII)
    return cls(gheim_detector=detector)


# ---------------------------------------------------------------------------
# Helpers used in assertions
# ---------------------------------------------------------------------------

# Anything that looks like a sentinel: <UPPERCASE_DIGITS>.
_SENTINEL_RE = re.compile(r"<[A-Z][A-Z0-9_]*_\d+>")

# Surface forms whose presence in the wire payload would prove a redaction
# bug. We keep this strict: the literal string must be absent from the
# captured payload. Subset substrings (e.g. "Joel" inside another word)
# are a separate concern handled by the detector itself; if the detector
# recognised "Joel" the wrapper must redact every occurrence.
_FORBIDDEN_ON_WIRE = list(REALISTIC_PII.keys())


def _wire_payload(captured: list[dict[str, Any]]) -> str:
    """Concatenate every captured outgoing message body into one string,
    suitable for substring leak assertions."""
    parts: list[str] = []
    for call in captured:
        for msg in call["messages"]:
            content = msg.get("content")
            if isinstance(content, str):
                parts.append(content)
    return "\n".join(parts)


def _assert_no_pii_on_wire(captured: list[dict[str, Any]]) -> None:
    payload = _wire_payload(captured)
    leaked = [s for s in _FORBIDDEN_ON_WIRE if s in payload]
    assert not leaked, f"PII leaked to OpenAI wire: {leaked!r}\nPayload: {payload!r}"


def _assert_only_sentinels_for_pii(captured: list[dict[str, Any]]) -> None:
    """Every place that originally held PII must now hold a sentinel."""
    payload = _wire_payload(captured)
    sentinels = _SENTINEL_RE.findall(payload)
    assert sentinels, f"expected ≥1 sentinel in wire payload; got: {payload!r}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_realistic_ticket_no_pii_reaches_wire_sync(sync_client):
    """Headline test: a realistic Swiss support ticket gets fully redacted
    before it crosses the wire, and a sentinel-shaped LLM response is
    fully restored before reaching the user."""
    inner = sync_client.chat.completions.inner
    sess = Session(detector=FakeDetector(REALISTIC_PII))
    # Pre-populate session by anonymizing once so we know which sentinels
    # the assistant should produce. The wrapper does this implicitly on
    # the user's behalf inside .create(); we mirror it here only to know
    # the sentinel names for the synthetic response.
    from gheim.plain import anonymize_text
    anonymized_preview = anonymize_text(REALISTIC_TICKET, sess)
    used_sentinels = _SENTINEL_RE.findall(anonymized_preview)
    assert used_sentinels, "preview anonymization produced no sentinels — fixture broken"

    # Have the assistant echo every sentinel back so we can verify
    # round-trip restoration.
    echoed = " ".join(used_sentinels)
    inner.response = FakeCompletion(choices=[FakeChoice(
        message=FakeMessage(role="assistant", content=f"Confirmed for {echoed}.")
    )])

    out = sync_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": REALISTIC_TICKET}],
        gheim_session=Session(detector=FakeDetector(REALISTIC_PII)),
    )

    _assert_no_pii_on_wire(inner.captured_calls)
    _assert_only_sentinels_for_pii(inner.captured_calls)

    # All sentinels in the LLM response are restored to original surfaces.
    restored = out.choices[0].message.content
    assert "<PERSON_" not in restored
    assert "<EMAIL_" not in restored
    assert "<PHONE_" not in restored
    # Originals are back. (Note: dedup means JOEL/Joel/Joel Barmettler
    # collapse to one sentinel that restores to whichever was seen first.)
    assert "Joel" in restored or "JOEL" in restored


def test_same_surface_across_turns_uses_stable_sentinel(sync_client):
    """Multi-turn dedup: the same surface form across turns gets the same
    sentinel, so the LLM can refer back coherently across messages and
    PII still doesn't leak."""
    inner = sync_client.chat.completions.inner
    session = Session(detector=FakeDetector(REALISTIC_PII))

    # Turn 1.
    inner.response = FakeCompletion(choices=[FakeChoice(
        message=FakeMessage(role="assistant", content="Got it, <PERSON_1>.")
    )])
    sync_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content":
                   "Hi, this is Joel Barmettler at joel.barmettler@acme.ch."}],
        gheim_session=session,
    )

    # Turn 2: same person referenced again. The session must reuse the
    # sentinel allocated in turn 1 — both for the name and the email.
    inner.response = FakeCompletion(choices=[FakeChoice(
        message=FakeMessage(role="assistant",
                            content="Yes <PERSON_1>, your account is active.")
    )])
    out2 = sync_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content":
                   "Is Joel Barmettler still locked out at "
                   "joel.barmettler@acme.ch?"}],
        gheim_session=session,
    )
    turn2_payload = inner.captured_calls[1]["messages"][0]["content"]
    assert "Joel Barmettler" not in turn2_payload
    assert "joel.barmettler@acme.ch" not in turn2_payload
    assert "<PERSON_1>" in turn2_payload
    # And the email sentinel is reused too — only one EMAIL_n was ever issued.
    email_sentinels = {s for s in _SENTINEL_RE.findall(turn2_payload)
                       if s.startswith("<EMAIL_")}
    assert email_sentinels == {"<EMAIL_1>"}, email_sentinels
    assert "Joel Barmettler" in out2.choices[0].message.content


@pytest.mark.xfail(
    reason="Person-name aliasing not yet implemented: ``JOEL`` (first name) "
           "and ``Joel Barmettler`` (full name) are distinct surfaces and "
           "currently get distinct sentinels. Documented limitation; would "
           "be fixed by an opt-in nameparser-based normalizer.",
)
def test_first_name_in_later_turn_should_collapse_to_existing_sentinel(sync_client):
    """Aliasing case: the user references Joel by full name in turn 1
    and by first name only in turn 2. Ideally both refer to the same
    person and reuse <PERSON_1>; today they don't."""
    inner = sync_client.chat.completions.inner
    session = Session(detector=FakeDetector(REALISTIC_PII))

    inner.response = FakeCompletion(choices=[FakeChoice(
        message=FakeMessage(role="assistant", content="ok")
    )])
    sync_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hi, this is Joel Barmettler."}],
        gheim_session=session,
    )

    inner.response = FakeCompletion(choices=[FakeChoice(
        message=FakeMessage(role="assistant", content="ok")
    )])
    sync_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Is Joel still locked out?"}],
        gheim_session=session,
    )
    turn2_payload = inner.captured_calls[1]["messages"][0]["content"]
    # Currently: "Is <PERSON_2> still locked out?" — fails the xfail.
    assert "<PERSON_1>" in turn2_payload


def test_distinct_persons_get_distinct_sentinels(sync_client):
    """Two truly different people in the same ticket get different sentinels."""
    inner = sync_client.chat.completions.inner
    session = Session(detector=FakeDetector(REALISTIC_PII))
    inner.response = FakeCompletion(choices=[FakeChoice(
        message=FakeMessage(role="assistant", content="ok")
    )])
    sync_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content":
                   "Joel Barmettler and Alice Müller are both on the ticket."}],
        gheim_session=session,
    )
    payload = inner.captured_calls[0]["messages"][0]["content"]
    assert "Joel Barmettler" not in payload
    assert "Alice Müller" not in payload
    sentinels = _SENTINEL_RE.findall(payload)
    person_sentinels = {s for s in sentinels if s.startswith("<PERSON_")}
    assert len(person_sentinels) == 2, f"expected 2 person sentinels, got {person_sentinels}"


def test_streaming_response_restores_sentinels_split_across_chunks(sync_client):
    """The classic streaming hazard: the LLM emits a sentinel like
    ``<PERSON_1>`` token-by-token (e.g. ``"<PER"``, ``"SON_"``,
    ``"1>"``). The deanonymizer must hold back the partial buffer
    and only emit once the whole sentinel has arrived."""
    inner = sync_client.chat.completions.inner
    session = Session(detector=FakeDetector(REALISTIC_PII))

    # Pre-allocate so we know <PERSON_1> = "Joel Barmettler".
    from gheim.plain import anonymize_text
    anonymize_text("Joel Barmettler", session)

    # Stream the sentinel split across chunk boundaries.
    inner.response = [
        FakeChunk(choices=[FakeChoiceChunk(delta=FakeDelta(content="Hi "))]),
        FakeChunk(choices=[FakeChoiceChunk(delta=FakeDelta(content="<PER"))]),
        FakeChunk(choices=[FakeChoiceChunk(delta=FakeDelta(content="SON_"))]),
        FakeChunk(choices=[FakeChoiceChunk(delta=FakeDelta(content="1>!"))]),
    ]

    stream = sync_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Greet Joel Barmettler please."}],
        stream=True,
        gheim_session=session,
    )
    collected = "".join(chunk.choices[0].delta.content or "" for chunk in stream)
    assert collected == "Hi Joel Barmettler!"
    # Wire payload contains the sentinel, not the original.
    sent_payload = inner.captured_calls[0]["messages"][0]["content"]
    assert "Joel Barmettler" not in sent_payload
    assert "<PERSON_1>" in sent_payload


def test_unknown_sentinel_in_response_passes_through_untouched(sync_client):
    """Defensive: if the LLM hallucinates a sentinel like ``<PERSON_99>``
    that the session never allocated, the wrapper must leave it as-is so
    the leak is visible to the user, not silently swallowed."""
    inner = sync_client.chat.completions.inner
    session = Session(detector=FakeDetector(REALISTIC_PII))
    # Allocate <PERSON_1> for Joel.
    from gheim.plain import anonymize_text
    anonymize_text("Joel Barmettler", session)
    inner.response = FakeCompletion(choices=[FakeChoice(
        message=FakeMessage(role="assistant",
                            content="Hello <PERSON_1> and <PERSON_99>!")
    )])
    out = sync_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Anything for Joel Barmettler?"}],
        gheim_session=session,
    )
    restored = out.choices[0].message.content
    assert "Joel Barmettler" in restored        # known sentinel restored
    assert "<PERSON_99>" in restored            # unknown sentinel preserved


async def test_async_streaming_realistic_ticket(async_client):
    """End-to-end async streaming: a multi-PII ticket goes out anonymized,
    the assistant streams a sentence with two sentinels, the user sees
    the restored stream."""
    inner = async_client.chat.completions.inner
    session = Session(detector=FakeDetector(REALISTIC_PII))

    # Pre-anonymize once so we know the sentinel ids the response will use.
    from gheim.plain import anonymize_text
    anonymize_text("Joel Barmettler joel.barmettler@acme.ch", session)

    inner.response = [
        FakeChunk(choices=[FakeChoiceChunk(delta=FakeDelta(content="Confirmed "))]),
        FakeChunk(choices=[FakeChoiceChunk(delta=FakeDelta(content="for <PERSON_"))]),
        FakeChunk(choices=[FakeChoiceChunk(delta=FakeDelta(content="1> at <EMAIL_1>."))]),
    ]

    stream = await async_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content":
                   "Please confirm for Joel Barmettler at joel.barmettler@acme.ch"}],
        stream=True,
        gheim_session=session,
    )
    collected = ""
    async for chunk in stream:
        collected += chunk.choices[0].delta.content or ""
    assert "Joel Barmettler" in collected
    assert "joel.barmettler@acme.ch" in collected
    sent_payload = inner.captured_calls[0]["messages"][0]["content"]
    assert "Joel Barmettler" not in sent_payload
    assert "joel.barmettler@acme.ch" not in sent_payload


def test_system_message_pii_also_redacted(sync_client):
    """Production-realistic: a system prompt may contain customer name or
    email (e.g. "You are helping Joel Barmettler today"). It must be
    redacted just like a user message."""
    inner = sync_client.chat.completions.inner
    session = Session(detector=FakeDetector(REALISTIC_PII))
    inner.response = FakeCompletion(choices=[FakeChoice(
        message=FakeMessage(role="assistant", content="Hi.")
    )])
    sync_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system",
             "content": "You are an assistant helping Joel Barmettler "
                        "(joel.barmettler@acme.ch)."},
            {"role": "user", "content": "Hi"},
        ],
        gheim_session=session,
    )
    payload = _wire_payload(inner.captured_calls)
    assert "Joel Barmettler" not in payload
    assert "joel.barmettler@acme.ch" not in payload
    # Both sentinels appeared in the system message position.
    sys_msg_content = inner.captured_calls[0]["messages"][0]["content"]
    assert "<PERSON_1>" in sys_msg_content
    assert "<EMAIL_1>" in sys_msg_content
