"""gheim — PII round-trip for LLM APIs.

Public API:

- ``Session`` — per-request state (sentinel mapping)
- ``LocalDetector`` / ``RemoteDetector`` — pluggable PII detection backends
- ``anonymize`` / ``deanonymize`` — message-level helpers
- ``anonymize_text`` / ``deanonymize_text`` — string-level helpers
- ``deanonymize_stream`` — wrap a chunk iterator
- ``gheim.openai.OpenAI`` — drop-in OpenAI client (subpackage import)
"""
from .core.session import Session
from .core.stream import StreamDeanonymizer
from .detectors import Detector, LocalDetector, RemoteDetector, Span, default_detector
from .plain import (
    anonymize,
    anonymize_messages,
    anonymize_text,
    deanonymize,
    deanonymize_stream,
    deanonymize_text,
)

__version__ = "0.1.0"

__all__ = [
    "Detector",
    "LocalDetector",
    "RemoteDetector",
    "Session",
    "Span",
    "StreamDeanonymizer",
    "__version__",
    "anonymize",
    "anonymize_messages",
    "anonymize_text",
    "deanonymize",
    "deanonymize_stream",
    "deanonymize_text",
    "default_detector",
]
