"""Language detection for Swiss-PII training data.

Uses lingua for DE/FR/IT/EN detection (high accuracy on short text) plus a
Romansh-marker heuristic since lingua doesn't ship a Romansh model. The
Romansh heuristic uses dialect-distinctive words that don't appear in
standard DE/FR/IT.

Output codes match our schema's ``language`` field:
  de_ch, fr_ch, it_ch, rm, gsw, en

Caveats:
  - GSW (Swiss German dialect) typically detects as DE_CH; we have no
    automatic way to distinguish them at scale.
  - Lingua needs at least ~50 chars of text for reliable detection; below
    that we fall back to the subset-implied default language.
"""
from __future__ import annotations

import re
from functools import lru_cache

from .schema import Language

# Romansh-distinctive tokens (case-insensitive). These appear frequently in
# Romansh text but not in standard DE/FR/IT. Picked for high specificity.
#
# CAREFUL: each language collision below was an empirical false positive on
# FineWeb-Swiss test_v1 surfacing. Removed:
#   - "che"   → most common IT word; drove ~94% of RM FPs
#   - "schi"  → DE colloquial ski-related
#   - "nun"   → EN/DE colloquial
#   - "ils"   → 3rd-person FR pronoun
#   - "ina","ino","tar","dapi","duai" → too short / too ambiguous, false-fire
# The remaining markers are 5+ chars or clearly RM-shaped.
#
# Also: this heuristic alone is not enough. ``detect()`` cross-checks with
# lingua confidence and overrides the marker count when lingua is highly
# confident about FR/IT/DE — see the implementation.
_RM_MARKERS = {
    # Verbs / particles distinct from DE/FR/IT
    "vegnan", "vegn", "vegnir", "stuess", "nunch", "dapertut", "uschia",
    "tschertgar", "exitser",
    # Identity / place names — high specificity
    "rumantsch", "romansh", "grischun", "vallader", "sursilvan",
    "surmiran", "sutsilvan", "puter", "rumantscha",
    # Common nouns characteristic
    "chantun", "chantuns", "cumun", "cumuns", "biarut", "veia", "plazza",
    "plaz", "stradun", "magister", "scolar", "scoula",
    # Multi-word fragments — high specificity, low collision risk
    "ed in", "ed ina", "ed il", "tar la", "tar il", "tar ils",
}

# Lingua confidence above which we trust its language verdict over the
# RM-marker heuristic. Empirically, lingua at >0.80 on a known language
# is rarely wrong; at lower confidence, the markers are useful tie-breakers.
_LINGUA_OVERRIDE_CONF = 0.80

_RM_MARKER_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in _RM_MARKERS) + r")\b",
    flags=re.IGNORECASE,
)

LingPair = tuple[str, float]  # (iso639_1_lower, confidence_0_to_1)


@lru_cache(maxsize=1)
def _detector():
    from lingua import Language, LanguageDetectorBuilder
    langs = [Language.GERMAN, Language.FRENCH, Language.ITALIAN, Language.ENGLISH]
    return LanguageDetectorBuilder.from_languages(*langs).build()


def _lingua_top(text: str) -> LingPair | None:
    if len(text.strip()) < 30:
        return None
    det = _detector()
    confs = det.compute_language_confidence_values(text)
    if not confs:
        return None
    top = max(confs, key=lambda c: c.value)
    return (top.language.iso_code_639_1.name.lower(), top.value)


def _count_rm_markers(text: str) -> int:
    return len(_RM_MARKER_RE.findall(text))


def detect(text: str, subset: str | None = None) -> Language:
    """Return the detected language as one of the 6 gheim Language codes.

    Decision tree:
      1. Use lingua's top guess if it's highly confident (>0.80) — it has
         proper trained models for DE/FR/IT/EN, more reliable than a
         keyword heuristic.
      2. Else if text has ≥3 RM markers → ``rm`` (lingua doesn't ship a
         Romansh model, so the heuristic is the only RM signal we have).
      3. Else if subset == ``romansh`` and lingua isn't very confident → ``rm``
      4. Else use lingua's top guess (de→de_ch, fr→fr_ch, it→it_ch, en→en)
      5. Fallback to subset-implied default

    Never raises on short or empty text — falls through to ``_subset_default``.
    """
    top = _lingua_top(text)
    # Lingua-first when it's very confident: prevents RM-marker false positives
    # on French/Italian text that incidentally contains RM-shaped words.
    if top is not None and top[1] >= _LINGUA_OVERRIDE_CONF:
        code, _ = top
        if code == "de":
            return "de_ch"
        if code == "fr":
            return "fr_ch"
        if code == "it":
            return "it_ch"
        if code == "en":
            return "en"

    n_rm = _count_rm_markers(text)
    if n_rm >= 3:
        return "rm"
    # Strong RM signal: subset says romansh, no strong non-RM detection
    if subset == "romansh":
        if top is None or top[1] < 0.85:
            return "rm"
        # If text is very clearly DE/FR/IT (e.g. a translated paragraph in
        # the multilingual GR corpus), trust lingua despite subset tag
        if top[0] == "de":
            return "de_ch"
        if top[0] == "fr":
            return "fr_ch"
        if top[0] == "it":
            return "it_ch"
        return "rm"

    if top is None:
        # Tiny text, no signal. Fall back to subset-implied default.
        return _subset_default(subset)

    code, _ = top
    if code == "de":
        return "de_ch"
    if code == "fr":
        return "fr_ch"
    if code == "it":
        return "it_ch"
    if code == "en":
        return "en"
    return _subset_default(subset)


def _subset_default(subset: str | None) -> Language:
    if subset == "romansh":
        return "rm"
    # Default for entscheidsuche / curia_vista / unknown is de_ch (majority
    # CH legal language).
    return "de_ch"
