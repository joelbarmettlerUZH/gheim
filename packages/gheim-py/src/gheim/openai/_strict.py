"""Strict-mode proxy that raises on access to unwrapped endpoints.

The point: customers who installed gheim believing PII redaction is automatic
should never silently leak data through endpoints we don't yet wrap. Strict mode
makes the failure mode loud and actionable instead of silent.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Annotated, Any

# Wrapped endpoint groups that should NOT raise (they're already protected).
# Anything else on the openai client gets the strict treatment.
WRAPPED_ATTRS = frozenset({
    "chat",
    "completions",
    "responses",
    "embeddings",
    "moderations",
    "audio",
    "images",
    "videos",
    # Read-only / no-PII attrs that always pass through.
    "models",
})

# Reasons we deliberately don't wrap each known unwrapped attribute.
STRICT_REASONS: dict[str, str] = {
    "beta": (
        "the Assistants API (beta.assistants.*, beta.threads.*) is stateful and "
        "stores instructions + messages on OpenAI's servers across calls — "
        "wrapping it correctly requires cross-call session persistence which is "
        "not yet implemented. OpenAI is steering everyone to the new Responses "
        "API; consider client.responses.create(...) instead."
    ),
    "batches": (
        "the Batch API takes a JSONL file containing nested chat/completions/embeddings "
        "request bodies; wrapping requires JSONL rewrite + per-line restoration which "
        "is not yet implemented. Anonymize each line yourself with gheim.anonymize_text "
        "before uploading the batch input file."
    ),
    "files": (
        "files.create uploads binary blobs (PDFs, CSVs, audio); document-level "
        "redaction is outside gheim's scope. Use a dedicated PDF/document redactor."
    ),
    "uploads": (
        "uploads.* is the multi-step file upload protocol; same reason as files.*."
    ),
    "fine_tuning": (
        "fine-tuning sends a training data file by ID — gheim doesn't redact files. "
        "Pre-anonymize your training data file before uploading."
    ),
    "vector_stores": (
        "vector_stores ingest pre-uploaded files; redact those files first."
    ),
    "containers": (
        "containers manage server-side execution environments; not in scope."
    ),
}


@dataclass
class StrictProxy:
    """Replaces an unwrapped attribute on the gheim client.

    Any attribute access (or call) raises ``RuntimeError`` with a clear remediation
    pointer. ``client.raw.<attr>`` is the documented escape hatch.
    """

    attr_name: Annotated[str, "Name of the unwrapped client attribute (e.g. 'beta')."]
    raw_path: Annotated[str, "Suggested escape hatch path, e.g. 'client.raw.beta'."]
    reason: Annotated[str, "Human-readable explanation of why this is blocked."]

    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(self._message(name))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(self._message("__call__"))

    def _message(self, sub: str) -> str:
        full_path = f"client.{self.attr_name}.{sub}" if sub != "__call__" else f"client.{self.attr_name}(...)"
        return (
            f"gheim.openai blocks {full_path} in strict mode: {self.reason}\n"
            f"To call it anyway with raw PII flowing to OpenAI, use "
            f"{self.raw_path}.{sub}(...). Or construct the client with "
            f"gheim_strict=False to silence this check (one-time warning)."
        )


@dataclass
class WarningProxy:
    """Non-strict counterpart: warn once, then forward to the underlying attribute."""

    attr_name: Annotated[str, "Name of the unwrapped client attribute."]
    inner: Annotated[Any, "The real openai attribute being wrapped."]
    reason: Annotated[str, "Same explanation used by StrictProxy."]
    _warned: bool = False

    def _maybe_warn(self) -> None:
        if not self._warned:
            warnings.warn(
                f"gheim.openai: client.{self.attr_name}.* sends data to OpenAI "
                f"WITHOUT gheim PII redaction. {self.reason}",
                stacklevel=3,
            )
            self._warned = True

    def __getattr__(self, name: str) -> Any:
        self._maybe_warn()
        return getattr(self.inner, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._maybe_warn()
        return self.inner(*args, **kwargs)


def make_unwrapped(
    name: str,
    inner_attr: Any | None,
    *,
    strict: bool,
) -> StrictProxy | WarningProxy | Any:
    """Build the right wrapper for an unwrapped attribute.

    If we have no documented reason for blocking it, pass through unchanged
    (forward-compatible: new openai endpoints we don't know about don't break).
    """
    reason = STRICT_REASONS.get(name)
    if reason is None:
        # Unknown attribute — neither strict nor warn; just forward.
        return inner_attr
    if strict:
        return StrictProxy(attr_name=name, raw_path="client.raw", reason=reason)
    if inner_attr is None:
        return None
    return WarningProxy(attr_name=name, inner=inner_attr, reason=reason)
