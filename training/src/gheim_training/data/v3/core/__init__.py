"""V3 generator framework — core modules.

Public API for Phase 2 template authors:

    from gheim_training.data.v3.core import (
        Chunk, Span, Fragment, write_jsonl,
        render_fragment, render_standalone,
        compose_email, CompositionStyle,
        apply_noise, maybe_apply_noise,
    )
    from gheim_training.data.v3.core import name_registry, pii_values

Subagents writing Phase 2 templates should NOT import from anywhere
else — these primitives are sufficient. Each template file should
contain a hand-written list of :class:`Fragment` objects (one entry
per template, no `for` loops generating templates).
"""
from .schema import Chunk, Span, write_jsonl, locate, spans_from_values
from .composers import (
    Fragment, render_fragment, render_standalone,
    compose_email, CompositionStyle, repeat,
)
from .noise import apply_noise, maybe_apply_noise, ALL_NOISE_KINDS, NoiseKind

__all__ = [
    "Chunk", "Span", "Fragment",
    "write_jsonl", "locate", "spans_from_values",
    "render_fragment", "render_standalone",
    "compose_email", "CompositionStyle", "repeat",
    "apply_noise", "maybe_apply_noise", "ALL_NOISE_KINDS", "NoiseKind",
]
