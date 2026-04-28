"""Remote detector backend — calls a gheim-server (or compatible) over HTTP."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Annotated, Any

import httpx

from .base import Span, trim_span

DEFAULT_BASE_URL = "https://api.gheim.ch"
DEFAULT_MODEL = "openai/privacy-filter"


@dataclass
class RemoteDetector:
    """POSTs ``text`` to ``{base_url}/v1/detect`` and returns spans.

    Wire protocol:
        request:  { "model": "...", "text": "..." }
        response: { "spans": [ { "label", "start", "end", "score" }, ... ] }

    The server only ever sees the input text and returns spans — never the
    LLM-bound traffic, never the eventual response.
    """

    base_url: Annotated[str | None, "Base URL of the detection server. Falls back to GHEIM_BASE_URL."] = None
    api_key: Annotated[str | None, "Bearer token. Falls back to GHEIM_API_KEY."] = None
    model: Annotated[str, "Default model id passed in the request body."] = DEFAULT_MODEL
    timeout: Annotated[float, "Per-request timeout in seconds."] = 30.0
    trim_whitespace: Annotated[
        bool, "Strip leading/trailing whitespace from each span (BPE artifact fix)."
    ] = True
    client: Annotated[httpx.Client | None, "Pre-built httpx client; created on demand if None."] = None
    _owns_client: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.base_url = (
            self.base_url or os.getenv("GHEIM_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        if self.api_key is None:
            self.api_key = os.getenv("GHEIM_API_KEY")
        self._owns_client = self.client is None

    def _http(self) -> httpx.Client:
        if self.client is None:
            self.client = httpx.Client(
                base_url=self.base_url or DEFAULT_BASE_URL,
                timeout=self.timeout,
            )
        return self.client

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def detect(
        self,
        text: Annotated[str, "Source text to scan for PII."],
        *,
        model: Annotated[str | None, "Per-call model id override."] = None,
    ) -> list[Span]:
        if not text:
            return []
        payload = {"model": model or self.model, "text": text}
        resp = self._http().post("/v1/detect", json=payload, headers=self._headers())
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        spans: list[Span] = []
        for item in data.get("spans", []):
            start = int(item["start"])
            end = int(item["end"])
            if self.trim_whitespace:
                trimmed = trim_span(text, start, end)
                if trimmed is None:
                    continue
                start, end = trimmed
            spans.append(
                Span(
                    label=str(item["label"]),
                    start=start,
                    end=end,
                    text=text[start:end],
                    score=float(item.get("score", 1.0)),
                )
            )
        return spans

    def close(self) -> None:
        if self.client is not None and self._owns_client:
            self.client.close()
            self.client = None

    def __enter__(self) -> RemoteDetector:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
