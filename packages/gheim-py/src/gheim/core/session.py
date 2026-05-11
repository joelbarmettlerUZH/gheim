"""Session state: tracks which sentinels map to which originals for one round-trip."""
from __future__ import annotations

import json
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any

from .sentinels import SENTINEL_RE, Sentinel, label_tag

if TYPE_CHECKING:
    from ..detectors.base import Detector, Span
    from ..normalizers import Normalizer


# Maximum number of whitespace characters allowed between two spans for them to
# count as "adjacent" and be merged. Covers the common BPE artifact where the
# tokenizer splits an email into two adjacent same-label spans.
_MERGE_GAP = 1


def _normalize_surface(s: str) -> str:
    """Normalize a surface form for sentinel-coalescing key purposes.

    NFKC unicode normalization (so "ﬁ" ligature == "fi"), Unicode-aware
    casefolding (so "Joel" == "JOEL" == "joel" and "Müller" == "MÜLLER";
    casefold also expands German "ß" → "ss"), then collapse all
    whitespace runs to a single space and strip. The mapping still
    stores the original surface so restoration preserves the first-seen
    casing; only the lookup key is normalized.
    """
    return " ".join(unicodedata.normalize("NFKC", s).casefold().split())


def _merge_adjacent(text: str, spans: list[Span]) -> list[tuple[int, int, str]]:
    """Merge runs of same-label spans that touch (or are separated by ≤ ``_MERGE_GAP``
    whitespace characters)."""
    out: list[tuple[int, int, str]] = []
    for span in spans:
        if span.start < 0 or span.end > len(text) or span.start >= span.end:
            continue
        if not out:
            out.append((span.start, span.end, span.label))
            continue
        prev_start, prev_end, prev_label = out[-1]
        if span.start < prev_end:
            # Overlap — drop the later span (greedy first-wins).
            continue
        gap = text[prev_end : span.start]
        if (
            span.label == prev_label
            and len(gap) <= _MERGE_GAP
            and gap.strip() == ""
        ):
            out[-1] = (prev_start, span.end, prev_label)
        else:
            out.append((span.start, span.end, span.label))
    return out


@dataclass
class Session:
    """Mutable per-request state for anonymizing outbound text and restoring inbound text.

    - ``mapping`` goes from rendered sentinel (``"<PERSON_1>"``) to the original surface form.
    - Same (tag, surface) reuses the same sentinel so the LLM can refer back coherently
      ("Joel said Joel's email is ...").

    Sessions are JSON-serializable so callers can persist them across requests if they want
    long-lived conversations (same `Joel` keeps mapping to `<PERSON_1>` across turns).
    """

    mapping: Annotated[
        dict[str, str], "Rendered sentinel ('<PERSON_1>') → original surface form."
    ] = field(default_factory=dict)
    _coalesce: dict[tuple[str, str], str] = field(default_factory=dict)
    _counters: dict[str, int] = field(default_factory=dict)
    detector: Annotated[
        Detector | None, "Optional default detector reused across calls in this session."
    ] = None
    normalizers: Annotated[
        Mapping[str, "str | Normalizer"] | None,
        "Optional per-label canonical-form normalizers, e.g. "
        "``{'private_phone': 'e164', 'private_date': 'iso_date'}`` or "
        "``{'private_phone': e164(region='CH')}``. A label not listed "
        "here uses the default NFKC+casefold key. See "
        "``gheim.normalizers``.",
    ] = None
    _resolved_normalizers: dict[str, "Normalizer"] = field(
        default_factory=dict, init=False, repr=False,
    )

    def __post_init__(self) -> None:
        if self.normalizers:
            from ..normalizers import resolve_normalizer
            self._resolved_normalizers = {
                label: resolve_normalizer(spec)
                for label, spec in self.normalizers.items()
            }

    def allocate(
        self,
        label: Annotated[str, "Detector label (e.g. 'private_person')."],
        surface: Annotated[str, "Original surface form to be hidden behind a sentinel."],
    ) -> str:
        """Return (or allocate) the sentinel for this (label, surface) pair.

        The coalesce key uses a normalized form of ``surface`` (NFKC +
        casefold + collapsed whitespace) so trivial variants of the same
        identity collapse to one sentinel: ``Joel`` / ``JOEL`` / ``joel``,
        ``Müller`` / ``MÜLLER``, ``"alice@example.com"`` / ``"ALICE@EXAMPLE.COM"``.
        When a per-label normalizer is configured (via ``normalizers=``)
        it overrides the default key for that label — used to collapse
        e.g. phone-format variants under E.164. If the normalizer
        returns ``None`` (un-parseable), the default NFKC+casefold key
        is used as a safe fallback so the surface still gets a sentinel.

        The mapping stores the original (first-seen) surface so
        restoration preserves casing.
        """
        tag = label_tag(label)
        norm_fn = self._resolved_normalizers.get(label)
        normalized: str | None = None
        if norm_fn is not None:
            normalized = norm_fn(surface)
        if normalized is None:
            normalized = _normalize_surface(surface)
        key = (tag, normalized)
        existing = self._coalesce.get(key)
        if existing is not None:
            return existing
        self._counters[tag] = self._counters.get(tag, 0) + 1
        sentinel = Sentinel(tag=tag, index=self._counters[tag]).render()
        self._coalesce[key] = sentinel
        self.mapping[sentinel] = surface
        return sentinel

    def apply_spans(
        self,
        text: Annotated[str, "Original text the spans were detected in."],
        spans: Annotated[list[Span], "Detected PII spans, possibly unsorted."],
    ) -> str:
        """Replace each span in ``text`` with an allocated sentinel. Left-to-right, non-overlapping.

        Overlapping spans are resolved by keeping the earliest-starting, longest span.
        Adjacent same-label spans (``alice@example`` + ``.com``) are merged before
        allocation so a split email doesn't end up as two separate sentinels.
        """
        if not spans:
            return text
        sorted_spans = sorted(spans, key=lambda s: (s.start, -(s.end - s.start)))
        merged = _merge_adjacent(text, sorted_spans)
        out: list[str] = []
        cursor = 0
        for span_start, span_end, label in merged:
            if span_start < cursor:
                continue
            if span_start < 0 or span_end > len(text) or span_start >= span_end:
                continue
            surface = text[span_start:span_end]
            out.append(text[cursor:span_start])
            out.append(self.allocate(label, surface))
            cursor = span_end
        out.append(text[cursor:])
        return "".join(out)

    def restore(
        self,
        text: Annotated[str, "Text containing sentinels (typically an LLM response)."],
    ) -> str:
        """Replace every known sentinel in ``text`` with its original surface form.

        Unknown sentinels (ones not in ``mapping``) are left as-is so callers can see leaks.
        """
        if not self.mapping:
            return text

        def _sub(m: Any) -> str:
            s = m.group(0)
            return self.mapping.get(s, s)

        return SENTINEL_RE.sub(_sub, text)

    def to_json(self) -> str:
        return json.dumps({"mapping": self.mapping})

    @classmethod
    def from_json(
        cls,
        raw: Annotated[str, "JSON string previously produced by Session.to_json()."],
        *,
        normalizers: Annotated[
            Mapping[str, "str | Normalizer"] | None,
            "Re-apply the same normalizers the session was created with. "
            "Required if the session originally used custom normalizers; "
            "otherwise dedup of new variants will fall back to NFKC+casefold "
            "and miss e.g. phone-format collapses.",
        ] = None,
    ) -> Session:
        data = json.loads(raw)
        mapping: dict[str, str] = data.get("mapping", {})
        s = cls(mapping=dict(mapping), normalizers=normalizers)
        # The persisted form is just the {sentinel: surface} mapping. To
        # rebuild the coalesce map we re-normalize each surface, using
        # any per-label normalizer that was passed back in.
        for sentinel, surface in mapping.items():
            parsed = Sentinel.parse(sentinel)
            if parsed is None:
                continue
            # Reverse-lookup the label name from the tag — we don't store
            # the original label, so try every label that maps to this tag.
            normalized: str | None = None
            for label, fn in s._resolved_normalizers.items():
                if label_tag(label) == parsed.tag:
                    normalized = fn(surface)
                    if normalized is not None:
                        break
            if normalized is None:
                normalized = _normalize_surface(surface)
            s._coalesce[(parsed.tag, normalized)] = sentinel
            s._counters[parsed.tag] = max(s._counters.get(parsed.tag, 0), parsed.index)
        return s
