"""Framework-agnostic surface — works with any LLM client.

Two flows:

1. **One-shot text**:
   ```
   session = Session(detector=LocalDetector())
   clean = anonymize_text("Hi, my name is Joel", session)
   # ... call any LLM with clean ...
   final = deanonymize_text(response_text, session)
   ```

2. **Streaming**:
   ```
   for restored in deanonymize_stream(chunks, session):
       print(restored, end="", flush=True)
   ```

The chat-message helpers (`anonymize_messages`, `deanonymize_messages`) accept the
common ``[{"role": "...", "content": "..."}]`` shape and pass through unrelated fields.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Annotated, Any

from .core.session import Session
from .core.stream import StreamDeanonymizer
from .detectors import Detector, default_detector

ChatMessage = dict[str, Any]


def _resolve(
    session: Session | None,
    detector: Detector | None,
) -> tuple[Session, Detector]:
    sess = session or Session()
    det = detector or sess.detector or default_detector()
    if sess.detector is None:
        sess.detector = det
    return sess, det


def anonymize_text(
    text: Annotated[str, "Source text containing PII to redact."],
    session: Annotated[
        Session | None,
        "Session that tracks sentinels. A fresh session is created if None.",
    ] = None,
    *,
    detector: Annotated[
        Detector | None,
        "Detector backend. Falls back to session.detector or default_detector().",
    ] = None,
    model: Annotated[
        str | None,
        "Optional per-call model id override (passed to the detector).",
    ] = None,
) -> str:
    """Detect PII in ``text`` and replace each span with a stable sentinel."""
    sess, det = _resolve(session, detector)
    spans = det.detect(text, model=model)
    return sess.apply_spans(text, spans)


def deanonymize_text(
    text: Annotated[str, "Text containing sentinels (typically an LLM response)."],
    session: Annotated[Session, "Session whose mapping is used for restoration."],
) -> str:
    """Replace every known sentinel in ``text`` with its original surface form."""
    return session.restore(text)


def anonymize_messages(
    messages: Annotated[
        Iterable[ChatMessage],
        "Chat messages to anonymize, e.g. ``[{'role': 'user', 'content': '...'}, ...]``.",
    ],
    session: Annotated[Session | None, "Session for sentinel allocation."] = None,
    *,
    detector: Annotated[Detector | None, "Detector backend override."] = None,
    model: Annotated[str | None, "Optional per-call model id override."] = None,
) -> list[ChatMessage]:
    """Anonymize the ``content`` field of each chat message in-order.

    Sentinels are coalesced across messages — the same name in two messages will
    map to the same sentinel.

    String content is processed directly. List-of-parts content (multimodal) has
    its ``"text"`` parts redacted; non-text parts (images, audio) are passed through.
    Other fields (``role``, ``name``, ``tool_call_id``, ...) are preserved.
    """
    sess, _ = _resolve(session, detector)
    out: list[ChatMessage] = []
    for msg in messages:
        out.append(_anonymize_message(msg, sess, detector=detector, model=model))
    return out


def _anonymize_message(
    msg: ChatMessage,
    session: Session,
    *,
    detector: Detector | None,
    model: str | None,
) -> ChatMessage:
    content = msg.get("content")
    if content is None:
        return dict(msg)
    if isinstance(content, str):
        new_content: Any = anonymize_text(content, session, detector=detector, model=model)
    elif isinstance(content, list):
        new_parts: list[Any] = []
        for part in content:
            if (
                isinstance(part, dict)
                and part.get("type") == "text"
                and isinstance(part.get("text"), str)
            ):
                redacted = anonymize_text(part["text"], session, detector=detector, model=model)
                new_parts.append({**part, "text": redacted})
            else:
                new_parts.append(part)
        new_content = new_parts
    else:
        new_content = content
    return {**msg, "content": new_content}


def deanonymize_stream(
    chunks: Annotated[Iterable[str], "Iterable of text chunks (e.g. SSE delta payloads)."],
    session: Annotated[Session, "Session whose mapping is used for restoration."],
) -> Iterator[str]:
    """Wrap an iterable of text chunks with a streaming deanonymizer.

    Each yielded chunk has any complete sentinels replaced with their originals.
    Partial sentinels are held back across chunk boundaries until they complete or
    are proven not to be sentinels.
    """
    deanonymizer = StreamDeanonymizer(session.mapping)
    yield from deanonymizer.transform(chunks)


# Convenience aliases — same operation, shorter name.
anonymize = anonymize_messages
deanonymize = deanonymize_text
