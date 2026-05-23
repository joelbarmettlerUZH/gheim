"""Hand-crafted edge-case probe for gheim-ch-v2.

NOT a metric — a forensic test. Each case is a short multi-language
example designed to stress a known weakness pattern that v1 failed on:

1. Greeting + bare first name (v1 missed; tracked in user-memory as the
   ``gheim-ch-560m v1 misses bare first names in greeting positions``
   gap).
2. Title + bare last name (Herr/Frau/Dr./Sgnr. + Müller).
3. Bare first name in mid-sentence narrative.
4. Bare last name in narrative.
5. Hyphenated / compound names.
6. Initials.
7. Signature line first name only.
8. RM names with RM titles.

For each case the script prints the chunk, the gold spans, the model's
predicted spans, and a hit/miss diff.

Run
---
    uv run python -m gheim_training.eval.probe_edge_cases \\
        --model checkpoints/gheim-ch-v2
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

# ---- Probe cases ---------------------------------------------------------


@dataclass
class Probe:
    case_id: str
    description: str
    language: str
    text: str
    expected: list[tuple[int, int, str]]  # (start, end, label) per gold span


def _at(text: str, value: str) -> tuple[int, int]:
    """Locate ``value`` in ``text`` and return its (start, end). Raises if
    the value isn't present — keeps the gold spans in sync with the text."""
    i = text.find(value)
    if i < 0:
        raise ValueError(f"{value!r} not in {text!r}")
    return (i, i + len(value))


def _gold(text: str, *expected: tuple[str, str]) -> list[tuple[int, int, str]]:
    """Build [(start, end, label), ...] from (value, label) pairs by
    locating each value in text. Multiple occurrences of the same value
    only mark the FIRST — caller adds more pairs to cover others."""
    out: list[tuple[int, int, str]] = []
    cursor = 0
    for value, label in expected:
        i = text.find(value, cursor)
        if i < 0:
            raise ValueError(f"{value!r} not found in {text!r} after pos {cursor}")
        out.append((i, i + len(value), label))
        cursor = i + len(value)
    out.sort(key=lambda x: x[0])
    return out


# The original site-demo email — multi-PII baseline.
DEMO_EMAIL = """Hallo Team,

mein Name ist Lukas Brunner. Bitte erreichen Sie mich unter
+41 44 555 12 34 oder lukas.brunner@example.ch. Meine Adresse ist
Bahnhofstrasse 10, 8001 Zürich.

Am 30. Juni 2026 läuft mein Vertrag aus. Bitte überweisen Sie
CHF 230.00 auf IBAN CH9300762011623852957.

Anbei der Staging-Schlüssel zum Testen:
sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789AbCdEfGhIjKl

Freundliche Grüsse,
Lukas Brunner"""


PROBES: list[Probe] = [
    # ---- 0. Baseline: demo email (multi-PII, model should nail this) ----
    Probe(
        case_id="00_demo_email",
        description="Site demo email (multi-PII baseline)",
        language="de_ch",
        text=DEMO_EMAIL,
        expected=_gold(
            DEMO_EMAIL,
            ("Lukas Brunner", "private_person"),
            ("+41 44 555 12 34", "private_phone"),
            ("lukas.brunner@example.ch", "private_email"),
            ("Bahnhofstrasse 10, 8001 Zürich", "private_address"),
            ("30. Juni 2026", "private_date"),
            ("CH9300762011623852957", "account_number"),
            ("sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789AbCdEfGhIjKl",
             "secret"),
            ("Lukas Brunner", "private_person"),  # 2nd occurrence
        ),
    ),

    # ---- 1. Greeting + bare first name (the documented v1 gap) ----
    Probe(
        case_id="01_greet_firstname_de",
        description="Hallo + bare first name (DE)",
        language="de_ch",
        text="Hallo Marius, danke für deine Nachricht.",
        expected=_gold("Hallo Marius, danke für deine Nachricht.",
                       ("Marius", "private_person")),
    ),
    Probe(
        case_id="02_greet_firstname_de_liebe",
        description="Liebe + bare first name (DE)",
        language="de_ch",
        text="Liebe Anna, ich habe deine Frage erhalten.",
        expected=_gold("Liebe Anna, ich habe deine Frage erhalten.",
                       ("Anna", "private_person")),
    ),
    Probe(
        case_id="03_greet_firstname_fr",
        description="Bonjour + bare first name (FR)",
        language="fr_ch",
        text="Bonjour Jean, merci de votre message.",
        expected=_gold("Bonjour Jean, merci de votre message.",
                       ("Jean", "private_person")),
    ),
    Probe(
        case_id="04_greet_firstname_it",
        description="Ciao + bare first name (IT)",
        language="it_ch",
        text="Ciao Marco, grazie del tuo messaggio.",
        expected=_gold("Ciao Marco, grazie del tuo messaggio.",
                       ("Marco", "private_person")),
    ),

    # ---- 2. Title + last name (Herr/Frau/Dr.) ----
    Probe(
        case_id="05_herr_lastname",
        description="Sehr geehrter Herr + last name",
        language="de_ch",
        text="Sehr geehrter Herr Müller, gerne komme ich auf Ihre Anfrage zurück.",
        expected=_gold(
            "Sehr geehrter Herr Müller, gerne komme ich auf Ihre Anfrage zurück.",
            # Both "Herr Müller" and just "Müller" are arguably correct.
            # Conservative: include the title in the span (more useful to
            # redact).
            ("Herr Müller", "private_person"),
        ),
    ),
    Probe(
        case_id="06_dr_lastname",
        description="Dr. + last name",
        language="de_ch",
        text="Bitte kontaktieren Sie Dr. Schmidt für weitere Details.",
        expected=_gold(
            "Bitte kontaktieren Sie Dr. Schmidt für weitere Details.",
            ("Dr. Schmidt", "private_person"),
        ),
    ),
    Probe(
        case_id="07_frau_lastname",
        description="Frau + last name (FR-style surname in DE context)",
        language="de_ch",
        text="Frau Dubois hat heute angerufen.",
        expected=_gold("Frau Dubois hat heute angerufen.",
                       ("Frau Dubois", "private_person")),
    ),
    Probe(
        case_id="08_monsieur_lastname",
        description="Monsieur + last name (FR)",
        language="fr_ch",
        text="Monsieur Dupont, vous trouverez ci-joint le document signé.",
        expected=_gold(
            "Monsieur Dupont, vous trouverez ci-joint le document signé.",
            ("Monsieur Dupont", "private_person"),
        ),
    ),
    Probe(
        case_id="09_madame_lastname",
        description="Madame + last name (FR)",
        language="fr_ch",
        text="Madame Müller a accepté la proposition.",
        expected=_gold("Madame Müller a accepté la proposition.",
                       ("Madame Müller", "private_person")),
    ),
    Probe(
        case_id="10_signor_lastname",
        description="Signor + last name (IT)",
        language="it_ch",
        text="Signor Rossi, la riunione è confermata per giovedì.",
        expected=_gold(
            "Signor Rossi, la riunione è confermata per giovedì.",
            ("Signor Rossi", "private_person"),
        ),
    ),

    # ---- 3. Bare first name mid-sentence (no greeting cue) ----
    Probe(
        case_id="11_bare_firstname_de",
        description="Bare first name in mid-sentence (DE narrative)",
        language="de_ch",
        text="Anna meinte gestern, dass die Lieferung morgen kommt.",
        expected=_gold("Anna meinte gestern, dass die Lieferung morgen kommt.",
                       ("Anna", "private_person")),
    ),
    Probe(
        case_id="12_bare_firstname_fr",
        description="Bare first name in mid-sentence (FR)",
        language="fr_ch",
        text="Jean a confirmé hier que le contrat sera signé demain.",
        expected=_gold("Jean a confirmé hier que le contrat sera signé demain.",
                       ("Jean", "private_person")),
    ),

    # ---- 4. Bare last name mid-sentence ----
    Probe(
        case_id="13_bare_lastname_de",
        description="Bare last name in mid-sentence (DE)",
        language="de_ch",
        text="Müller hat heute eine ausführliche Stellungnahme geschrieben.",
        expected=_gold(
            "Müller hat heute eine ausführliche Stellungnahme geschrieben.",
            ("Müller", "private_person"),
        ),
    ),
    Probe(
        case_id="14_bare_lastname_it",
        description="Bare last name in mid-sentence (IT)",
        language="it_ch",
        text="Rossi ha presentato il rapporto alla riunione di ieri.",
        expected=_gold(
            "Rossi ha presentato il rapporto alla riunione di ieri.",
            ("Rossi", "private_person"),
        ),
    ),

    # ---- 5. Hyphenated / compound names ----
    Probe(
        case_id="15_hyphenated_compound",
        description="Hyphenated first + double surname (DE)",
        language="de_ch",
        text="Hans-Peter Bürgi-Stocker hat das Protokoll unterzeichnet.",
        expected=_gold(
            "Hans-Peter Bürgi-Stocker hat das Protokoll unterzeichnet.",
            ("Hans-Peter Bürgi-Stocker", "private_person"),
        ),
    ),

    # ---- 6. Title abbrev + initial + last name ----
    Probe(
        case_id="16_abbrev_initial",
        description="Hr. + initial + last name",
        language="de_ch",
        text="Hr. M. Müller hat mich heute Morgen kontaktiert.",
        expected=_gold("Hr. M. Müller hat mich heute Morgen kontaktiert.",
                       ("Hr. M. Müller", "private_person")),
    ),

    # ---- 7. Signature: first name only ----
    Probe(
        case_id="17_signature_firstname",
        description="Email signature with first name only",
        language="de_ch",
        text=(
            "Vielen Dank für die schnelle Bearbeitung.\n\n"
            "Liebe Grüsse,\n"
            "Marius"
        ),
        expected=_gold(
            (
                "Vielen Dank für die schnelle Bearbeitung.\n\n"
                "Liebe Grüsse,\n"
                "Marius"
            ),
            ("Marius", "private_person"),
        ),
    ),

    # ---- 8. Two names in one sentence (multi-PII per chunk) ----
    Probe(
        case_id="18_two_names_de",
        description="Two titled names in one sentence",
        language="de_ch",
        text="Frau Müller und Herr Schmidt haben gemeinsam unterzeichnet.",
        expected=_gold(
            "Frau Müller und Herr Schmidt haben gemeinsam unterzeichnet.",
            ("Frau Müller", "private_person"),
            ("Herr Schmidt", "private_person"),
        ),
    ),

    # ---- 9. RM ----
    Probe(
        case_id="19_rm_title_lastname",
        description="RM: Sgnr. + last name (Romansh)",
        language="rm",
        text="Bun di, Sgnr. Caduff, perinclis arrivar a las 14:00 a Cuera.",
        expected=_gold(
            "Bun di, Sgnr. Caduff, perinclis arrivar a las 14:00 a Cuera.",
            ("Sgnr. Caduff", "private_person"),
        ),
    ),

    # ---- 10. The deliberately-tricky case: name that's also a common word ----
    Probe(
        case_id="20_common_word_name",
        description="Surname that's a common word (Bach)",
        language="de_ch",
        text="Herr Bach hat das Konzert dirigiert; wir trafen ihn am Bach.",
        expected=_gold(
            "Herr Bach hat das Konzert dirigiert; wir trafen ihn am Bach.",
            # Only first "Bach" (the person); second is a creek.
            ("Herr Bach", "private_person"),
        ),
    ),
]


# ---- Runner --------------------------------------------------------------

def _predict_spans(model: any, tokenizer: any, text: str, max_len: int
                   ) -> list[tuple[int, int, str]]:
    """Run the model on one text and decode BIOES → char spans."""
    import numpy as np
    import torch

    from ..data.bioes import decode_bioes_to_spans
    from ..data.label_space import ID2LABEL

    enc = tokenizer(
        text, return_offsets_mapping=True, truncation=True,
        max_length=max_len, return_tensors="pt",
    )
    offsets = enc.pop("offset_mapping")[0].tolist()
    ids = enc["input_ids"]
    am = enc["attention_mask"]
    if torch.cuda.is_available():
        ids = ids.cuda()
        am = am.cuda()
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=am).logits.float().cpu().numpy()
    preds = np.argmax(logits[0], axis=-1).tolist()
    spans = decode_bioes_to_spans(text, offsets, preds, ID2LABEL)
    return [(sp.start, sp.end, sp.label) for sp in spans]


def _format_spans(text: str,
                  spans: list[tuple[int, int, str]],
                  ) -> str:
    if not spans:
        return "  (none)"
    out = []
    for s, e, lab in sorted(spans, key=lambda x: x[0]):
        out.append(f"  [{s:>3}-{e:<3}] {lab:<18} \"{text[s:e]}\"")
    return "\n".join(out)


def _diff_spans(
    gold: list[tuple[int, int, str]],
    pred: list[tuple[int, int, str]],
    text: str,
) -> dict:
    """Compute exact (start, end, label) match, plus value-level match
    (catches boundary noise like 'Müller' vs 'Herr Müller')."""
    gold_set = set(gold)
    pred_set = set(pred)
    exact_tp = gold_set & pred_set
    exact_fp = pred_set - gold_set
    exact_fn = gold_set - pred_set

    # Soft match: predicted span overlaps gold span on the same label
    # (counts boundary disagreements as partial hits).
    def overlaps(a: tuple[int, int, str],
                 b: tuple[int, int, str]) -> bool:
        return a[2] == b[2] and not (a[1] <= b[0] or b[1] <= a[0])

    soft_tp = sum(
        1 for g in gold if any(overlaps(g, p) for p in pred)
    )
    soft_fn = len(gold) - soft_tp
    soft_fp = sum(
        1 for p in pred if not any(overlaps(g, p) for g in gold)
    )
    return {
        "exact_tp": len(exact_tp), "exact_fp": len(exact_fp),
        "exact_fn": len(exact_fn),
        "soft_tp": soft_tp, "soft_fp": soft_fp, "soft_fn": soft_fn,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path("checkpoints/gheim-ch-v2"))
    ap.add_argument("--max-seq-length", type=int, default=512)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    print(f"Loading model: {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(str(args.model), use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        str(args.model),
    ).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    print()

    total_exact_tp = total_exact_fp = total_exact_fn = 0
    total_soft_tp = total_soft_fp = total_soft_fn = 0
    n_perfect = 0

    for probe in PROBES:
        print(f"=== [{probe.case_id}] {probe.description} ({probe.language}) ===")
        print(f"  text: {probe.text!r}")
        pred = _predict_spans(model, tok, probe.text, args.max_seq_length)

        print(f"  gold ({len(probe.expected)}):")
        print(_format_spans(probe.text, probe.expected))
        print(f"  pred ({len(pred)}):")
        print(_format_spans(probe.text, pred))

        d = _diff_spans(probe.expected, pred, probe.text)
        total_exact_tp += d["exact_tp"]
        total_exact_fp += d["exact_fp"]
        total_exact_fn += d["exact_fn"]
        total_soft_tp += d["soft_tp"]
        total_soft_fp += d["soft_fp"]
        total_soft_fn += d["soft_fn"]

        if d["soft_fn"] == 0 and d["soft_fp"] == 0:
            verdict = "PASS (perfect)"
            n_perfect += 1
        elif d["soft_fn"] == 0:
            verdict = (
                f"PARTIAL (all gold hit but +{d['soft_fp']} unexpected pred)"
            )
        elif d["soft_tp"] == 0:
            verdict = "FAIL (zero hits)"
        else:
            verdict = (
                f"PARTIAL "
                f"(soft hits {d['soft_tp']}/{len(probe.expected)}, "
                f"missed {d['soft_fn']}, extra {d['soft_fp']})"
            )
        print(f"  → {verdict}")
        print()

    print("=" * 60)
    print(f"OVERALL: {n_perfect}/{len(PROBES)} cases pass perfectly")
    print(f"  Exact match: TP={total_exact_tp} FP={total_exact_fp} "
          f"FN={total_exact_fn}")
    print(f"  Soft match : TP={total_soft_tp} FP={total_soft_fp} "
          f"FN={total_soft_fn}")
    if total_soft_tp + total_soft_fp > 0 and total_soft_tp + total_soft_fn > 0:
        sp = total_soft_tp / (total_soft_tp + total_soft_fp)
        sr = total_soft_tp / (total_soft_tp + total_soft_fn)
        f1 = 2 * sp * sr / (sp + sr) if (sp + sr) else 0
        print(f"  Soft P={sp:.3f} R={sr:.3f} F1={f1:.3f}")


if __name__ == "__main__":
    main()
