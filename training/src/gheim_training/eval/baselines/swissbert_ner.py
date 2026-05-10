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
from typing import Annotated, Any, Literal

from ...data.lang_detect import detect as detect_lang
from ...data.schema import Language, Span

# XMOD adapter codes shipped by SwissBERT. The set is closed; type tightens
# the codomain of `_GHEIM_TO_XMOD` so downstream lookups can't accidentally
# return something that isn't a valid adapter.
XmodAdapter = Literal["de_CH", "fr_CH", "it_CH", "rm_CH"]

# Map gheim's Language values → SwissBERT XMOD adapter codes. XMOD only
# ships adapters for the 4 official Swiss languages; GSW transfers from
# de_CH (closest morphology), EN falls back to de_CH (no English adapter).
# The dict is total over Language so detect() never returns a key miss.
_GHEIM_TO_XMOD: dict[Language, XmodAdapter] = {
    "de_ch": "de_CH",
    "fr_ch": "fr_CH",
    "it_ch": "it_CH",
    "rm": "rm_CH",
    "gsw": "de_CH",
    "en": "de_CH",
}

# Regex-extractable PII (categories not covered by SwissBERT-NER).
_REGEXES: list[tuple[str, re.Pattern[str]]] = [
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
    model_id: Annotated[
        str,
        "HuggingFace id of the SwissBERT-NER checkpoint. Default is the "
        "community NER head; subclasses may swap in a different head.",
    ] = "ZurichNLP/swissbert-ner"
    # transformers.Pipeline doesn't have a clean public type annotation we
    # can reuse here. _load() guarantees non-None before .detect() runs.
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
        # _pipe is non-None after _load(); HF pipelines wrap a model with
        # set_default_language for XMOD. Asserts narrow ty's view (which
        # treats _pipe as Any) and document the precondition.
        assert self._pipe is not None
        # Lingua wrapper never raises (returns _subset_default for short text);
        # the dict is total over Language so the lookup never KeyErrors.
        lang = detect_lang(text)
        adapter = _GHEIM_TO_XMOD[lang]
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
