"""SwissBERT-NER + regex hybrid. The bar to beat per the brief.

SwissBERT was pre-trained on Swiss text in DE/FR/IT/RM via the XMOD
architecture: one shared body + four language-specific adapter sets.
The community ``ZurichNLP/swissbert-ner`` head outputs PERSON/LOC/ORG, so
we map:
  - PERSON → private_person
  - LOC    → private_address  (LOC includes addresses + place names)
  - ORG    → dropped (no gheim category for orgs)

XMOD requires a default language before each forward — we use lingua to
detect per chunk (DE/FR/IT primarily, RM via gheim's heuristic) and fall
back to ``de_CH`` for unknowns. Without the per-call language set, the
model raises "Input language unknown" at inference.

For categories SwissBERT-NER doesn't cover (email, phone, IBAN, AHV, VAT, URL,
date, secret) we layer on regex recognizers. This is the "real" comparison:
a non-trivial multilingual NER plus deterministic structured-PII detection.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ...data.lang_detect import detect as detect_lang
from ...data.schema import Span

# Map gheim's Language values → SwissBERT XMOD adapter codes. XMOD only
# ships adapters for the 4 official Swiss languages; GSW transfers from
# de_CH (closest morphology), EN falls back to de_CH (no English adapter).
_GHEIM_TO_XMOD: dict[str, str] = {
    "de_ch": "de_CH",
    "fr_ch": "fr_CH",
    "it_ch": "it_CH",
    "rm": "rm_CH",
    "gsw": "de_CH",
    "en": "de_CH",
}

# Regex-extractable PII (categories not covered by SwissBERT-NER).
_REGEXES: list[tuple[str, re.Pattern]] = [
    ("private_email", re.compile(r"\b[\w._%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")),
    ("private_url", re.compile(r"https?://[^\s)>]+")),
    ("private_phone", re.compile(
        r"(?:\+41|0041|0)\s?(?:\(0\)\s?)?\d{2}\s?\d{3}\s?\d{2}\s?\d{2}"
    )),
    ("account_number", re.compile(r"\bCH\d{2}(?:\s?\d){17}\b")),  # IBAN-CH
    ("account_number", re.compile(r"\b756\.\d{4}\.\d{4}\.\d{2}\b")),  # AHV dotted
    ("account_number", re.compile(r"\b756\d{10}\b")),  # AHV compact
    ("account_number", re.compile(r"\bCHE-\d{3}\.\d{3}\.\d{3}(?:\s+(?:MWST|TVA|IVA))?\b")),
    ("account_number", re.compile(r"\b(?:\d[\s-]?){15,16}\d\b")),  # CC
    ("private_date", re.compile(r"\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b")),
    ("secret", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}\b")),
    ("secret", re.compile(r"\bghp_[A-Za-z0-9]{36}\b")),
    ("secret", re.compile(r"\bAKIA[A-Z0-9]{16}\b")),
]


@dataclass
class SwissBertNERDetector:
    model_id: str = "ZurichNLP/swissbert-ner"
    _pipe: Any = field(default=None, init=False, repr=False)

    def _load(self) -> None:
        if self._pipe is not None:
            return
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
        tok = AutoTokenizer.from_pretrained(self.model_id)
        mdl = AutoModelForTokenClassification.from_pretrained(self.model_id)
        device = 0 if torch.cuda.is_available() else -1
        self._pipe = pipeline(
            "token-classification",
            model=mdl,
            tokenizer=tok,
            device=device,
            aggregation_strategy="simple",
        )

    def detect(self, text: str) -> list[Span]:
        self._load()
        # XMOD requires a language adapter selected before each forward.
        # Detect via lingua and fall back to de_CH (the largest training
        # language) for unknowns / very short text.
        try:
            lang = detect_lang(text)
        except Exception:
            lang = "de_ch"
        adapter = _GHEIM_TO_XMOD.get(lang or "de_ch", "de_CH")
        self._pipe.model.set_default_language(adapter)
        spans: list[Span] = []
        for r in self._pipe(text):
            ent = (r.get("entity_group") or r.get("entity") or "").upper()
            if "PER" in ent:
                cat = "private_person"
            elif "LOC" in ent:
                cat = "private_address"
            else:
                continue
            s, e = int(r["start"]), int(r["end"])
            while s < e and text[s].isspace():
                s += 1
            while e > s and text[e - 1].isspace():
                e -= 1
            if e > s:
                spans.append(Span(start=s, end=e, label=cat))
        # Layer regexes for non-NER categories.
        for cat, pat in _REGEXES:
            for m in pat.finditer(text):
                spans.append(Span(start=m.start(), end=m.end(), label=cat))
        # Resolve overlaps (NER + regex can overlap; prefer earlier-start, longer span).
        spans.sort(key=lambda s: (s.start, -(s.end - s.start)))
        out: list[Span] = []
        last_end = -1
        for sp in spans:
            if sp.start < last_end:
                continue
            out.append(sp)
            last_end = sp.end
        return out
