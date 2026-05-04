"""Layer 2: AI4Privacy pii-masking-300k → gheim's 8 categories.

The pii-masking-300k dataset uses a much finer label taxonomy (LASTNAME,
FIRSTNAME, STREET, BUILDINGNUMBER, CREDITCARDNUMBER, IBAN, SOCIALNUM, ...).
We collapse those down to the 8 OpenAI categories. Anything we can't confidently
map (e.g. JOBTITLE, COMPANYNAME) is dropped as a span — left as ``O`` — rather
than mis-labeled.

We filter to Swiss locales: ``de`` (proxy for de-CH; the dataset only has 'de'),
``fr`` and ``it``. Caveat documented in the audit log: AI4Privacy ``de`` is
not Swiss-specific German; we accept this contamination because it improves
coverage of multi-tenant German PII patterns.

Run:
    uv run python -m gheim_training.data.ai4privacy --out data/layer2.jsonl
"""
from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from .schema import Example, Span, write_jsonl

# Maps AI4Privacy label → gheim category, or None to drop.
# Source: ai4privacy/pii-masking-300k label list (v1.0). When in doubt, drop.
REMAP: dict[str, str | None] = {
    # Persons
    "FIRSTNAME": "private_person",
    "LASTNAME": "private_person",
    "MIDDLENAME": "private_person",
    "GIVENNAME": "private_person",
    "USERNAME": "private_person",  # often masquerades as a name
    "PREFIX": None,
    "SUFFIX": None,
    "TITLE": None,
    # Email / URL
    "EMAIL": "private_email",
    "URL": "private_url",
    "IPV4": "private_url",  # arguable; treat IPs like URL
    "IPV6": "private_url",
    "MAC": None,
    # Phone
    "PHONE_NUMBER": "private_phone",
    "PHONEIMEI": None,  # device-id, not strictly PII for our use
    # Address — collapsed (the model gets char-spans, not substructure)
    "STREET": "private_address",
    "BUILDINGNUMBER": "private_address",
    "CITY": "private_address",
    "STATE": "private_address",
    "COUNTY": "private_address",
    "ZIPCODE": "private_address",
    "SECONDARYADDRESS": "private_address",
    "STREETADDRESS": "private_address",
    # Date
    "DATE": "private_date",
    "TIME": None,
    "DOB": "private_date",
    # Account-like
    "IBAN": "account_number",
    "BIC": "account_number",
    "ACCOUNTNUMBER": "account_number",
    "ACCOUNTNAME": None,  # name on account; covered by FIRSTNAME/LASTNAME
    "CREDITCARDNUMBER": "account_number",
    "CREDITCARDISSUER": None,
    "CREDITCARDCVV": "secret",
    "PIN": "secret",
    "SSN": "account_number",  # US analog; treat as account_number for transfer
    "SOCIALNUM": "account_number",
    "VEHICLEVIN": None,
    "VEHICLEVRM": None,
    # Secrets
    "PASSWORD": "secret",
    # Geographic catch-alls
    "ORDINALDIRECTION": None,
    "NEARBYGPSCOORDINATE": None,
    "CURRENCY": None,
    "CURRENCYCODE": None,
    "CURRENCYNAME": None,
    "CURRENCYSYMBOL": None,
    # Org / job — drop (not in our 8 categories)
    "COMPANYNAME": None,
    "JOBTITLE": None,
    "JOBTYPE": None,
    "JOBAREA": None,
    # Misc
    "AGE": None,
    "GENDER": None,
    "ETHNICCATEGORY": None,
    "MASKEDNUMBER": None,
}

# Locales we keep. AI4Privacy uses ISO-639 codes; we map to our Language enum.
LOCALE_TO_LANG: dict[str, str] = {
    "German": "de_ch",
    "French": "fr_ch",
    "Italian": "it_ch",
}


def _spans_from_record(record: dict, *, language: str) -> Example | None:
    """Convert one AI4Privacy record to an Example, or None if too lossy.

    AI4Privacy records typically expose ``source_text`` plus ``privacy_mask``
    (a list of {value, label, start, end}) or a ``mbert_text_tokens`` +
    ``mbert_bio_labels`` view. We use the privacy_mask form because it gives
    us char offsets directly.
    """
    text = record.get("source_text") or record.get("unmasked_text") or record.get("text")
    mask = record.get("privacy_mask") or []
    if not text or not mask or not isinstance(mask, list):
        return None

    spans: list[Span] = []
    for m in mask:
        # Normalize field names across dataset versions.
        lab = m.get("label") or m.get("entity_type") or m.get("entity")
        s = m.get("start") if "start" in m else m.get("start_position")
        e = m.get("end") if "end" in m else m.get("end_position")
        if lab is None or s is None or e is None:
            continue
        # AI4Privacy uses numbered variants (LASTNAME1, LASTNAME2, ...) when
        # multiple instances appear in one record. Strip trailing digits.
        norm = lab.upper().rstrip("0123456789")
        cat = REMAP.get(norm)
        if cat is None:
            continue
        if e <= s or e > len(text):
            continue
        if not text[s:e].strip():
            continue
        spans.append(Span(start=int(s), end=int(e), label=cat))

    if not spans:
        return None

    # Resolve overlaps: prefer the first by start position, drop later overlaps.
    spans.sort(key=lambda sp: (sp.start, -sp.end))
    deduped: list[Span] = []
    last_end = -1
    for sp in spans:
        if sp.start < last_end:
            continue
        deduped.append(sp)
        last_end = sp.end

    if not deduped:
        return None

    ex = Example(
        text=text,
        spans=deduped,
        language=language,  # type: ignore[arg-type]
        source="ai4privacy",
    )
    try:
        ex.validate_offsets()
    except ValueError:
        return None
    return ex


def load(
    dataset_name: str = "ai4privacy/pii-masking-300k",
    languages: Iterable[str] = ("German", "French", "Italian"),
    split: str = "train",
    max_per_language: int | None = None,
) -> list[Example]:
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split=split)
    out: list[Example] = []
    counts: dict[str, int] = {l: 0 for l in languages}
    for rec in ds:
        loc = rec.get("language") or rec.get("locale")
        if loc not in languages:
            continue
        if max_per_language and counts[loc] >= max_per_language:
            continue
        ex = _spans_from_record(rec, language=LOCALE_TO_LANG[loc])
        if ex is None:
            continue
        out.append(ex)
        counts[loc] += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dataset", default="ai4privacy/pii-masking-300k")
    ap.add_argument("--max-per-language", type=int, default=20_000)
    args = ap.parse_args()

    examples = load(args.dataset, max_per_language=args.max_per_language)
    n = write_jsonl(args.out, examples)
    print(f"wrote {n} examples → {args.out}")


if __name__ == "__main__":
    main()
