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

# Romansh-distinctive tokens (case-insensitive). These appear frequently in
# Romansh text but not in standard DE/FR/IT. Picked for high specificity.
_RM_MARKERS = {
    # Articles, pronouns, prepositions distinct from DE/FR/IT
    "ils", "ina", "ino", "tar", "dapi", "duai", "vegnan", "vegn",
    "che", "schi", "stuess", "nun", "nunch", "dapertut", "uschia",
    # Verbs / particles
    "vegnir", "tschertgar", "exitser",
    # Idiom-distinctive
    "rumantsch", "romansh", "grischun", "vallader", "sursilvan",
    "surmiran", "sutsilvan", "puter", "rumantscha",
    # Place / common nouns characteristic
    "chantun", "chantuns", "cumun", "cumuns", "biarut", "veia", "plazza",
    "plaz", "stradun", "magister", "scolar", "scoula",
    # Dialect-distinctive grammar fragments
    "ed in", "ed ina", "ed il", "tar la", "tar il", "tar ils",
}

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


def detect(text: str, subset: str | None = None) -> str:
    """Return one of: de_ch, fr_ch, it_ch, rm, gsw, en.

    Decision tree:
      1. If text has ≥3 RM markers → rm
      2. Else if subset == 'romansh' and lingua isn't very confident → rm
      3. Else use lingua's top guess (de→de_ch, fr→fr_ch, it→it_ch, en→en)
      4. Fallback to subset-implied default
    """
    n_rm = _count_rm_markers(text)
    if n_rm >= 3:
        return "rm"

    top = _lingua_top(text)
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

    code, conf = top
    if code == "de":
        return "de_ch"
    if code == "fr":
        return "fr_ch"
    if code == "it":
        return "it_ch"
    if code == "en":
        return "en"
    return _subset_default(subset)


def _subset_default(subset: str | None) -> str:
    if subset == "romansh":
        return "rm"
    if subset in ("entscheidsuche", "curia_vista"):
        return "de_ch"  # majority CH legal lang
    return "de_ch"  # safe default
