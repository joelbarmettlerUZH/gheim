"""Client class factories — build the gheim OpenAI subclasses with all proxies wired."""
from __future__ import annotations

from typing import Any

from ..detectors import Detector, default_detector
from ._anonymize_only import (
    AsyncEmbeddingsProxy,
    AsyncImagesProxy,
    AsyncModerationsProxy,
    EmbeddingsProxy,
    ImagesProxy,
    ModerationsProxy,
)
from ._audio import AsyncAudioProxy, AudioProxy
from ._chat import AsyncChatProxy, ChatProxy
from ._completions import AsyncCompletionsProxy, CompletionsProxy
from ._responses import AsyncResponsesProxy, ResponsesProxy
from ._strict import STRICT_REASONS, make_unwrapped


def _import_openai() -> Any:
    try:
        import openai
    except ImportError as e:  # pragma: no cover - import guard
        raise ImportError(
            "gheim.openai requires the `openai` package. Install with: uv add 'gheim[openai]'"
        ) from e
    return openai


# Names of the wrapped endpoint attributes — used by __getattribute__ to know
# which attrs to skip the strict guard for.
WRAPPED_PROXY_ATTRS = frozenset({
    "chat",
    "completions",
    "responses",
    "embeddings",
    "moderations",
    "audio",
    "images",
    "raw",
    "models",
})


def _get_inner_attr(client: Any, attr_name: str) -> Any:
    """Look up an attribute on the client BYPASSING our subclass's overrides.

    Walks the MRO past the gheim subclass and returns the first base class
    that defines ``attr_name`` (as a property, descriptor, or class var). Falls
    back to the instance ``__dict__`` (for attrs the base sets in ``__init__``).
    """
    cls = type(client)
    for base in cls.__mro__[1:]:
        if attr_name in vars(base):
            descr = vars(base)[attr_name]
            if hasattr(descr, "__get__"):
                return descr.__get__(client, base)
            return descr
    if attr_name in client.__dict__:
        return client.__dict__[attr_name]
    raise AttributeError(
        f"underlying openai client has no attribute {attr_name!r}"
    )


def _lazy_proxy(client: Any, attr_name: str, cache_attr: str, factory: Any) -> Any:
    """Build the proxy on first access; cache it on the client. Lazy so that
    test fakes which only mock a subset of endpoints don't blow up at construction."""
    cached = client.__dict__.get(cache_attr)
    if cached is not None:
        return cached
    inner = _get_inner_attr(client, attr_name)
    detector = client.__dict__["_gheim_detector"]
    proxy = factory(inner=inner, detector=detector)
    client.__dict__[cache_attr] = proxy
    return proxy


def make_sync_client_class(base_cls: Any) -> Any:
    """Build a subclass of openai.OpenAI that wraps every text-carrying endpoint."""

    class _GheimOpenAI(base_cls):  # type: ignore[misc, valid-type]
        """Drop-in replacement for ``openai.OpenAI`` with PII round-trip on every text endpoint."""

        def __init__(
            self,
            *args: Any,
            gheim_detector: Detector | None = None,
            gheim_strict: bool = True,
            **kwargs: Any,
        ):
            super().__init__(*args, **kwargs)
            self.__dict__["_gheim_detector"] = gheim_detector or default_detector()
            self.__dict__["_gheim_strict"] = gheim_strict

        @property
        def raw(self) -> Any:
            return _RawClientView(self, base_cls)

        @property
        def chat(self):  # type: ignore[override]
            return _lazy_proxy(self, "chat", "_gheim_chat", ChatProxy)

        @property
        def completions(self):  # type: ignore[override]
            return _lazy_proxy(self, "completions", "_gheim_completions", CompletionsProxy)

        @property
        def responses(self):  # type: ignore[override]
            return _lazy_proxy(self, "responses", "_gheim_responses", ResponsesProxy)

        @property
        def embeddings(self):  # type: ignore[override]
            return _lazy_proxy(self, "embeddings", "_gheim_embeddings", EmbeddingsProxy)

        @property
        def moderations(self):  # type: ignore[override]
            return _lazy_proxy(self, "moderations", "_gheim_moderations", ModerationsProxy)

        @property
        def audio(self):  # type: ignore[override]
            return _lazy_proxy(self, "audio", "_gheim_audio", AudioProxy)

        @property
        def images(self):  # type: ignore[override]
            return _lazy_proxy(self, "images", "_gheim_images", ImagesProxy)

        def __getattribute__(self, name: str) -> Any:
            if name.startswith("_") or name in WRAPPED_PROXY_ATTRS:
                return object.__getattribute__(self, name)
            if name in STRICT_REASONS:
                strict = object.__getattribute__(self, "_gheim_strict")
                inner_attr = None if strict else super().__getattribute__(name)
                return make_unwrapped(name, inner_attr, strict=strict)
            return super().__getattribute__(name)

    _GheimOpenAI.__name__ = "OpenAI"
    _GheimOpenAI.__qualname__ = "OpenAI"
    return _GheimOpenAI


def make_async_client_class(base_cls: Any) -> Any:
    """Build a subclass of openai.AsyncOpenAI with the same wiring."""

    class _GheimAsyncOpenAI(base_cls):  # type: ignore[misc, valid-type]
        """Drop-in replacement for ``openai.AsyncOpenAI``."""

        def __init__(
            self,
            *args: Any,
            gheim_detector: Detector | None = None,
            gheim_strict: bool = True,
            **kwargs: Any,
        ):
            super().__init__(*args, **kwargs)
            self.__dict__["_gheim_detector"] = gheim_detector or default_detector()
            self.__dict__["_gheim_strict"] = gheim_strict

        @property
        def raw(self) -> Any:
            return _RawClientView(self, base_cls)

        @property
        def chat(self):  # type: ignore[override]
            return _lazy_proxy(self, "chat", "_gheim_chat", AsyncChatProxy)

        @property
        def completions(self):  # type: ignore[override]
            return _lazy_proxy(self, "completions", "_gheim_completions", AsyncCompletionsProxy)

        @property
        def responses(self):  # type: ignore[override]
            return _lazy_proxy(self, "responses", "_gheim_responses", AsyncResponsesProxy)

        @property
        def embeddings(self):  # type: ignore[override]
            return _lazy_proxy(self, "embeddings", "_gheim_embeddings", AsyncEmbeddingsProxy)

        @property
        def moderations(self):  # type: ignore[override]
            return _lazy_proxy(self, "moderations", "_gheim_moderations", AsyncModerationsProxy)

        @property
        def audio(self):  # type: ignore[override]
            return _lazy_proxy(self, "audio", "_gheim_audio", AsyncAudioProxy)

        @property
        def images(self):  # type: ignore[override]
            return _lazy_proxy(self, "images", "_gheim_images", AsyncImagesProxy)

        def __getattribute__(self, name: str) -> Any:
            if name.startswith("_") or name in WRAPPED_PROXY_ATTRS:
                return object.__getattribute__(self, name)
            if name in STRICT_REASONS:
                strict = object.__getattribute__(self, "_gheim_strict")
                inner_attr = None if strict else super().__getattribute__(name)
                return make_unwrapped(name, inner_attr, strict=strict)
            return super().__getattribute__(name)

    _GheimAsyncOpenAI.__name__ = "AsyncOpenAI"
    _GheimAsyncOpenAI.__qualname__ = "AsyncOpenAI"
    return _GheimAsyncOpenAI


class _RawClientView:
    """Escape-hatch view: ``client.raw.<attr>`` reaches the underlying openai
    attribute bypassing every gheim wrapper."""

    __slots__ = ("_client", "_base_cls")

    def __init__(self, client: Any, base_cls: Any):
        object.__setattr__(self, "_client", client)
        object.__setattr__(self, "_base_cls", base_cls)

    def __getattr__(self, name: str) -> Any:
        client = object.__getattribute__(self, "_client")
        return _get_inner_attr(client, name)
