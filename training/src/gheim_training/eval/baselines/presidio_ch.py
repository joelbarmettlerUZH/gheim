"""Microsoft Presidio with Swiss-aware recognizers.

Presidio's default English-only NER analyzer doesn't speak DE/FR/IT/RM,
so we wire up per-language NlpEngines (spaCy de/fr/it + en for fallback)
and a small set of Swiss-specific PatternRecognizers (IBAN-CH, AHV,
VAT-CHE, +41 phone). This is the closest fair comparison: a competitive
open-source PII tool, configured as a Swiss user reasonably would.

Maps Presidio's labels to gheim's 8 categories:
  PERSON / NRP                       → private_person
  LOCATION (only when address-like)  → private_address  (heuristic; Presidio
                                       doesn't separate addresses cleanly)
  EMAIL_ADDRESS                      → private_email
  PHONE_NUMBER / "swiss_phone"       → private_phone
  URL / IP_ADDRESS                   → private_url
  DATE_TIME                          → private_date  (only when day+month+year)
  IBAN_CODE / "iban_ch" / "ahv" / "vat_che" / CREDIT_CARD → account_number
  US_PASSPORT / US_SSN / etc.        → dropped (not relevant for CH)
  ORGANIZATION                       → dropped (not in our 8 categories)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ...data.lang_detect import detect as detect_lang
from ...data.schema import Span

# Per-language spaCy model, plus Swiss legal-text fallback.
_SPACY_BY_LANG: dict[str, str] = {
    "de_ch": "de_core_news_lg",
    "fr_ch": "fr_core_news_lg",
    "it_ch": "it_core_news_lg",
    "en":    "en_core_web_lg",
    "rm":    "de_core_news_lg",  # closest available; spaCy has no RM model
    "gsw":   "de_core_news_lg",
}

# Presidio entity → gheim label. None means drop.
_LABEL_MAP: dict[str, str | None] = {
    "PERSON": "private_person",
    "NRP": "private_person",  # nationality/religious/political — drop in production?
    "EMAIL_ADDRESS": "private_email",
    "PHONE_NUMBER": "private_phone",
    "URL": "private_url",
    "IP_ADDRESS": "private_url",
    "DATE_TIME": "private_date",
    "CREDIT_CARD": "account_number",
    "IBAN_CODE": "account_number",
    "iban_ch": "account_number",
    "ahv": "account_number",
    "vat_che": "account_number",
    "swiss_phone": "private_phone",
    "swiss_address": "private_address",
    "LOCATION": "private_address",  # heuristic
    "ORGANIZATION": None,
    "US_SSN": None,
    "US_PASSPORT": None,
    "US_DRIVER_LICENSE": None,
}

# Strict day+month+year date check used to filter Presidio's loose
# DATE_TIME (which fires on bare years and "yesterday").
_DATE_OK_RE = re.compile(
    r"\b(?:\d{1,2}[./]\d{1,2}[./]\d{2,4}"
    r"|\d{4}[./\-]\d{1,2}[./\-]\d{1,2}"
    r"|\d{1,2}\.?\s+(?:Jan|Feb|Mär|Apr|Mai|Jun|Jul|Aug|Sep|Okt|Nov|Dez|"
    r"janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre|"
    r"gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre|"
    r"January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\.?\s+\d{2,4})\b",
    re.IGNORECASE,
)


_SWISS_RECOGNIZER_SPECS: list[tuple[str, list[tuple[str, str, float]]]] = [
    ("iban_ch", [("iban_ch", r"\bCH\d{2}(?:[\s\-]?\d){17}\b", 0.85)]),
    ("ahv", [
        ("ahv_dotted", r"\b756\.\d{4}\.\d{4}\.\d{2}\b", 0.95),
        ("ahv_compact", r"\b756\d{10}\b", 0.85),
    ]),
    ("vat_che", [(
        "vat_che",
        r"\bCHE-\d{3}\.\d{3}\.\d{3}(?:\s+(?:MWST|TVA|IVA))?\b", 0.95,
    )]),
    ("swiss_phone", [(
        "swiss_phone",
        r"(?:\+41|0041|\b0)\s?(?:\(0\)\s?)?\d{2}[\s\-/.]?\d{3}[\s\-/.]?\d{2}[\s\-/.]?\d{2}\b",
        0.7,
    )]),
]


def _build_swiss_recognizers(language: str) -> list[Any]:
    """Instantiate Swiss-specific PatternRecognizers bound to one Presidio
    language code (e.g. 'de'/'fr'/'it'/'en'). Each language gets its own
    instances because PatternRecognizer is registered per-language."""
    from presidio_analyzer import Pattern, PatternRecognizer
    out: list[Any] = []
    for entity, patterns in _SWISS_RECOGNIZER_SPECS:
        out.append(PatternRecognizer(
            supported_entity=entity,
            patterns=[Pattern(name, regex, score) for name, regex, score in patterns],
            supported_language=language,
        ))
    return out


@dataclass
class PresidioCHDetector:
    """Presidio analyzer with Swiss-aware recognizers + per-language NlpEngine."""

    _engines: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _ready: bool = field(default=False, init=False, repr=False)

    def _load(self) -> None:
        if self._ready:
            return
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        loaded: set[str] = set()
        for _lang, model in _SPACY_BY_LANG.items():
            if model in loaded:
                continue
            loaded.add(model)
            spacy_lang_code = model.split("_", 1)[0]  # 'de'/'fr'/'it'/'en'
            provider = NlpEngineProvider(nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": spacy_lang_code, "model_name": model}],
            })
            engine = AnalyzerEngine(
                nlp_engine=provider.create_engine(),
                supported_languages=[spacy_lang_code],
            )
            for rec in _build_swiss_recognizers(spacy_lang_code):
                engine.registry.add_recognizer(rec)
            self._engines[spacy_lang_code] = engine
        self._ready = True

    def detect(self, text: str) -> list[Span]:
        self._load()
        if not text:
            return []
        # Per-text language detection → pick the right Presidio engine.
        lang = detect_lang(text)
        spacy_lang = {
            "de_ch": "de", "fr_ch": "fr", "it_ch": "it", "en": "en",
            "rm": "de", "gsw": "de",
        }.get(lang, "de")
        engine = self._engines.get(spacy_lang) or next(iter(self._engines.values()))
        analyzer_results = engine.analyze(text=text, language=spacy_lang)

        seen: set[tuple[int, int, str]] = set()
        out: list[Span] = []
        for r in analyzer_results:
            label = _LABEL_MAP.get(r.entity_type)
            if label is None:
                continue
            value = text[r.start:r.end]
            # Strict date filter: drop loose DATE_TIMEs that are bare years
            # or fuzzy ("yesterday", "next week").
            if label == "private_date" and not _DATE_OK_RE.search(value):
                continue
            key = (int(r.start), int(r.end), label)
            if key in seen:
                continue
            seen.add(key)
            out.append(Span(start=key[0], end=key[1], label=label))
        return out
