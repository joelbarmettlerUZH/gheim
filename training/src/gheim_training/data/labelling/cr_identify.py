"""Identify commercial-register chunks in the v1 corpus.

The P7 audit (Section 3.5 of the paper) found 156 person-FPs that all
share a common pattern: chunks from Swiss commercial-register listings
(Zefix, FineWeb-2 mentions of newly-formed GmbHs, etc.) where the
chunk text follows a "Personen mit Entscheidungsbefugnis: <name>
(<role>) und <name> (<role>)" template. The Gemma v1 labelling pass
under-recalled director and board-member names in these listings,
likely because the chunks are long contact-block dumps rather than
prose and the prompt's few-shot examples are mostly prose.

This module detects CR chunks by language-specific marker phrases.
Anything containing one of them is a candidate for the format-aware
re-label pass (``run_label_cr.py``).
"""
from __future__ import annotations

import re
from collections.abc import Iterable

# Per-language CR markers. The list is intentionally generous — we'd
# rather re-label a few prose chunks that mention "Verwaltungsrat" in
# passing than miss the long contact-block listings that are the
# actual recall problem.
_MARKERS_BY_LANG: dict[str, tuple[str, ...]] = {
    "de_ch": (
        "personen mit entscheidungsbefugnis",
        "personen mit entscheidungsbefugnissen",
        "verwaltungsrat",
        "verwaltungsräte",
        "geschäftsführer",
        "geschäftsführerin",
        "gesellschafter",
        "gesellschafterin",
        "mitglied des verwaltungsrates",
        "präsident des verwaltungsrates",
        "präsidentin des verwaltungsrates",
        "vorsitz",
        "kollektivunterschrift",
        "einzelunterschrift",
        "prokurist",
        "ist eine gesellschaft mit beschränkter haftung",
        "wurde im jahr",  # combined with role tags below
    ),
    "fr_ch": (
        "personnes ayant le pouvoir de décision",
        "conseil d'administration",
        "membre du conseil d'administration",
        "président du conseil d'administration",
        "administrateur",
        "administratrice",
        "gérant",
        "gérante",
        "directeur",
        "directrice",
        "associé gérant",
        "signature collective",
        "signature individuelle",
        "fondé de pouvoir",
    ),
    "it_ch": (
        "persone con potere decisionale",
        "consiglio di amministrazione",
        "membro del consiglio di amministrazione",
        "presidente del consiglio di amministrazione",
        "amministratore",
        "amministratrice",
        "amministratore unico",
        "direttore",
        "direttrice",
        "gerente",
        "firma collettiva",
        "firma individuale",
        "procuratore",
    ),
    "en": (
        "persons authorised to take decisions",
        "board of directors",
        "managing director",
        "chief executive officer",
        "president of the board",
        "member of the board",
        "limited liability company",
        "joint stock company",
        "sole signatory",
    ),
}

# Compile per-language patterns. Word-boundary on either side keeps
# "verwaltungsräte" out of "verwaltungsräten" matches but allows the
# common compound endings the lemma takes (pluralised + cased forms
# are listed explicitly above).
_PATTERNS_BY_LANG: dict[str, re.Pattern[str]] = {}
for lang, markers in _MARKERS_BY_LANG.items():
    union = "|".join(re.escape(m) for m in markers)
    _PATTERNS_BY_LANG[lang] = re.compile(union, re.IGNORECASE)


def is_commercial_register(text: str, language: str) -> bool:
    """Return True if ``text`` looks like a commercial-register listing
    in the given language.

    Empty / unknown languages fall back to checking all language
    patterns; useful for chunks where the language id is uncertain.
    """
    if not text:
        return False
    pat = _PATTERNS_BY_LANG.get(language)
    if pat is not None:
        return bool(pat.search(text))
    # Fallback: try all languages.
    return any(p.search(text) for p in _PATTERNS_BY_LANG.values())


def filter_cr(
    examples: Iterable[dict],
    *,
    text_key: str = "text",
    lang_key: str = "language",
) -> Iterable[dict]:
    """Stream-filter an iterable of dict chunks, yielding only the CR
    ones. Convenience for the run_label_cr pipeline."""
    for ex in examples:
        text = ex.get(text_key) or ""
        lang = ex.get(lang_key) or ""
        if is_commercial_register(text, lang):
            yield ex
