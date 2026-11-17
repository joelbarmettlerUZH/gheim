"""Session state: tracks which sentinels map to which originals for one round-trip."""
from __future__ import annotations

import json
import re
import unicodedata
from bisect import bisect_left
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any

from .sentinels import SENTINEL_RE, Sentinel, find_sentinels, label_tag

if TYPE_CHECKING:
    from ..detectors.base import Detector, Span
    from ..normalizers import Normalizer


# Maximum number of whitespace characters allowed between two spans for them to
# count as "adjacent" and be merged. Covers the common BPE artifact where the
# tokenizer splits an email into two adjacent same-label spans.
_MERGE_GAP = 1


@dataclass(frozen=True, slots=True)
class MergedSpan:
    """A span after merge: same shape as :class:`Span` but without
    ``text``/``score`` carried. Returned by :meth:`Session.apply_spans_detailed`."""

    start: int
    end: int
    label: str


@dataclass(frozen=True, slots=True)
class ApplySpansResult:
    """Return type of :meth:`Session.apply_spans_detailed`."""

    redacted: Annotated[str, "Text with each detected span replaced by its sentinel."]
    merged: Annotated[list[MergedSpan], "The post-merge span list (offsets into the original text)."]


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

    def reserve(
        self,
        value: Annotated[
            Any,
            "Text that may already carry sentinels from an earlier anonymisation, "
            "or a list/tuple/dict of them (searched recursively, e.g. a whole "
            "chat-message list or request kwargs).",
        ],
    ) -> None:
        """Make sure no sentinel already present in ``value`` is ever allocated again.

        Raises each tag's counter to at least every sentinel index found, so a
        caller's ``<PERSON_1>`` keeps its meaning and a newly detected entity
        gets a fresh index. ``apply_spans`` does this for its own text; when one
        session anonymises several texts, reserve all of them first (as
        ``anonymize_messages`` and the OpenAI proxies do), otherwise allocating
        for the first text can hand out an index a later one already carries.
        """
        if isinstance(value, str):
            for _, _, sentinel in find_sentinels(value):
                self._bump_counter(sentinel)
        elif isinstance(value, Mapping):
            for v in value.values():
                self.reserve(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                self.reserve(v)

    def _bump_counter(self, sentinel: Sentinel) -> None:
        self._counters[sentinel.tag] = max(self._counters.get(sentinel.tag, 0), sentinel.index)

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
        return self.apply_spans_detailed(text, spans).redacted

    def apply_spans_detailed(
        self,
        text: Annotated[str, "Original text the spans were detected in."],
        spans: Annotated[list[Span], "Detected PII spans, possibly unsorted."],
    ) -> "ApplySpansResult":
        """Like :meth:`apply_spans` but also returns the post-merge span list.

        UI consumers (segment renderers, bar overlays, ablation scripts)
        typically need both the redacted text *and* the merged spans so
        they can highlight the redacted runs at their original character
        positions. Without this method they'd have to call the internal
        merge again on the same input. Pure convenience — no extra cost
        over ``apply_spans``.
        """
        taken: list[tuple[int, int]] = []
        for start, end, sentinel in find_sentinels(text):
            self._bump_counter(sentinel)
            taken.append((start, end))
        if taken:
            spans = _cut_out_sentinels(text, spans, taken)
        if not spans:
            return ApplySpansResult(redacted=text, merged=[])
        sorted_spans = sorted(spans, key=lambda s: (s.start, -(s.end - s.start)))
        merged_tuples = _merge_adjacent(text, sorted_spans)
        out: list[str] = []
        cursor = 0
        for span_start, span_end, label in merged_tuples:
            if span_start < cursor:
                continue
            if span_start < 0 or span_end > len(text) or span_start >= span_end:
                continue
            surface = text[span_start:span_end]
            out.append(text[cursor:span_start])
            out.append(self.allocate(label, surface))
            cursor = span_end
        out.append(text[cursor:])
        redacted = "".join(out)

        # Duplicate-surface recovery: the model often catches one occurrence
        # of a name (greeting) but misses a second occurrence in the
        # signature. Once we know "Lukas Brunner" is a person, every
        # remaining occurrence in the text is also a person — coalesces
        # to the same sentinel by design. Patches BOTH the redacted
        # string AND the merged span list so UI consumers (bar overlays,
        # highlighters) see the recovered occurrences too.
        redacted, merged_tuples = _recover_duplicate_occurrences(
            text, redacted, merged_tuples, self.mapping,
        )

        merged_spans = [
            MergedSpan(start=s, end=e, label=l) for s, e, l in merged_tuples
        ]
        return ApplySpansResult(redacted=redacted, merged=merged_spans)

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
            s._bump_counter(parsed)
        return s


def _cut_out_sentinels(
    text: str,
    spans: list[Span],
    taken: list[tuple[int, int]],
) -> list[Span]:
    """Remove pre-existing sentinel ranges from ``spans``.

    A sentinel already in the text is never PII, but a detector span may run
    into one (``"<PERSON_1> Meier"``). Dropping such a span whole would leak
    the real PII next to the sentinel, so each span is split around the
    sentinels it touches and the remaining pieces, trimmed of surrounding
    whitespace, are kept with the span's label. ``taken`` is sorted by start
    and non-overlapping (it comes from one left-to-right regex scan).
    """
    starts = [ts for ts, _ in taken]
    out: list[Span] = []
    for span in spans:
        # First sentinel that could overlap: the one starting before span.end,
        # walking back from there until sentinels end before span.start.
        i = bisect_left(starts, span.end)
        hits = []
        while i > 0 and taken[i - 1][1] > span.start:
            i -= 1
            hits.append(taken[i])
        if not hits:
            out.append(span)
            continue
        cursor = span.start
        for ts, te in sorted(hits):
            out.extend(_trimmed_piece(text, span, cursor, ts))
            cursor = max(cursor, te)
        out.extend(_trimmed_piece(text, span, cursor, span.end))
    return out


def _trimmed_piece(text: str, span: Span, start: int, end: int) -> list[Span]:
    from ..detectors.base import Span

    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return []
    return [Span(span.label, start, end, text[start:end], span.score)]


def _recover_duplicate_occurrences(
    text: str,
    redacted: str,
    merged_tuples: list[tuple[int, int, str]],
    mapping: dict[str, str],
) -> tuple[str, list[tuple[int, int, str]]]:
    """Substitute remaining occurrences of each known surface with its sentinel,
    AND emit merged-span entries for those occurrences in the original text.

    Catches the common case where a name appears in the greeting AND in
    the signature but the model only flagged one occurrence. Coalesces
    to the same sentinel by design.

    Patches BOTH outputs in lockstep:
      - ``redacted`` gets each remaining occurrence replaced with its sentinel
      - ``merged_tuples`` gets a new (start, end, label) per remaining
        occurrence (offsets into the ORIGINAL text), so bar overlays and
        other span consumers see them too.

    Word-boundary safe via Unicode-aware regex: ``Müller.`` matches but
    ``Müllergasse`` doesn't. Surfaces are tried longest-first so a
    "Joel Barmettler" sentinel doesn't get partially shadowed by a
    separate "Joel" sentinel.
    """
    # surface -> label: derived from the spans the detector already
    # emitted, since `mapping` only carries {sentinel -> surface}.
    surface_to_label: dict[str, str] = {
        text[s:e]: lbl for s, e, lbl in merged_tuples
    }
    covered: list[tuple[int, int]] = [(s, e) for s, e, _ in merged_tuples]

    def _overlaps(s: int, e: int) -> bool:
        return any(s < ce and e > cs for cs, ce in covered)

    items = sorted(
        ((sentinel, surface) for sentinel, surface in mapping.items() if surface),
        key=lambda x: -len(x[1]),
    )
    result = redacted
    additional: list[tuple[int, int, str]] = []
    for sentinel, surface in items:
        # ``re`` `\w` is unicode-aware on str patterns; surround the
        # surface with negative lookbehind/lookahead for "letter, digit,
        # or underscore" so we don't match inside a longer word.
        pattern = re.compile(rf"(?<!\w){re.escape(surface)}(?!\w)")
        # 1. patch the redacted string
        result = pattern.sub(sentinel, result)
        # 2. find ORIGINAL-text occurrences and stage merged-span entries
        label = surface_to_label.get(surface)
        if label is None:
            continue
        for m in pattern.finditer(text):
            start, end = m.start(), m.end()
            if _overlaps(start, end):
                continue
            additional.append((start, end, label))
            covered.append((start, end))

    if not additional:
        return result, merged_tuples
    augmented = sorted(merged_tuples + additional, key=lambda t: t[0])
    return result, augmented
