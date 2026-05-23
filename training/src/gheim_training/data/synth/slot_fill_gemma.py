"""P9: Synthetic gap-fill data via Gemma slot-fill.

Pre-generates realistic PII payload values, prompts Gemma in each target
language to embed them VERBATIM into natural prose, verifies via
text.find(value), builds spans, writes to data/gheim_synthetic.jsonl.

Targets (chunks per template per lang):
  dev_chat        (secret + person + email):      3000 each × 5 langs
  customer_record (acct + person + addr + email + phone):  2800 each × {it_ch, rm, en}
  incident_report (secret + person + phone + email):       1500 × en

Total: 24,900 chunks. ~6h on Gemma 4 26B-A4B-it-AWQ-4bit, tp=2.

Resumable: counts existing output lines on startup and skips already-done
(template, lang) buckets that have hit their target.

Usage:
  uv run python -m gheim_training.data.synth.slot_fill_gemma --smoke    # 50 chunks total, sanity check
  uv run python -m gheim_training.data.synth.slot_fill_gemma            # full run (no flag)
"""
from __future__ import annotations

import argparse
import json
import random
import string
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import cast

from .synth import faker_ch
from .synth.swiss_address import Lang
from .synth.swiss_address import address as swiss_address

OUT_PATH = Path("data/gheim_synthetic.jsonl")

LANGS = ["de_ch", "fr_ch", "it_ch", "rm", "en"]
SEED = 42
BATCH_SIZE = 32          # vLLM batch
MAX_RETRIES = 3
TEMPERATURE = 0.85
MAX_NEW_TOKENS = 600

# Per (template, lang) targets
TARGETS: dict[tuple[str, str], int] = {
    # dev_chat: secret + person + email — all langs
    ("dev_chat", "de_ch"): 3000,
    ("dev_chat", "fr_ch"): 3000,
    ("dev_chat", "it_ch"): 3000,
    ("dev_chat", "rm"):    3000,
    ("dev_chat", "en"):    3000,
    # customer_record: account + person + address + email + phone — sparse-cell langs
    ("customer_record", "it_ch"): 2800,
    ("customer_record", "rm"):    2800,
    ("customer_record", "en"):    2800,
    # incident_report: secret + person + phone + email — en only
    ("incident_report", "en"): 1500,
}

# Slot → label mapping
SLOT_LABELS: dict[str, str] = {
    "SECRET":  "secret",
    "PERSON":  "private_person",
    "EMAIL":   "private_email",
    "PHONE":   "private_phone",
    "ADDRESS": "private_address",
    "ACCOUNT": "account_number",
    "URL":     "private_url",
}


# =================================================================
# Payload generators
# =================================================================

def _name(lang: str) -> str:
    return {
        "de_ch": faker_ch.name_de,
        "fr_ch": faker_ch.name_fr,
        "it_ch": faker_ch.name_it,
        "rm":    faker_ch.name_rm,
        "en":    faker_ch._FAKER_EN.name,
    }[lang]()


def _address(lang: str) -> str:
    if lang == "en":
        # Use a Swiss-format address even for en chunks (en context can mention CH addresses)
        return swiss_address("de")
    return swiss_address(cast(Lang, lang.removesuffix("_ch")))


def _email(lang: str, person_hint: str | None = None) -> str:
    return faker_ch.email_ch(name_hint=person_hint)


def _phone() -> str:
    return faker_ch.phone_ch()


def _account() -> str:
    """Pick uniformly from CH-relevant account types."""
    kind = random.choice(("iban", "ahv", "vat", "cc"))
    if kind == "iban":
        return faker_ch.iban_ch()
    if kind == "ahv":
        return faker_ch.ahv()
    if kind == "vat":
        return faker_ch.vat_che()
    return faker_ch.credit_card()


def _secret() -> str:
    """Diverse secret formats."""
    kind = random.choice((
        "openai_proj", "openai_classic", "github_pat", "github_classic",
        "slack_bot", "slack_user", "aws_akid", "google_api", "stripe_live",
        "stripe_test", "jwt", "hex64", "base64_token",
    ))
    if kind == "openai_proj":
        return "sk-proj-" + _rand(43, "alnum_dash")
    if kind == "openai_classic":
        return "sk-" + _rand(48, "alnum")
    if kind == "github_pat":
        return "github_pat_" + _rand(82, "alnum")
    if kind == "github_classic":
        return "ghp_" + _rand(36, "alnum")
    if kind == "slack_bot":
        return f"xoxb-{random.randint(10**11, 10**12 - 1)}-{random.randint(10**11, 10**12 - 1)}-{_rand(24, 'alnum')}"
    if kind == "slack_user":
        return f"xoxp-{random.randint(10**11, 10**12 - 1)}-{random.randint(10**11, 10**12 - 1)}-{_rand(24, 'alnum')}"
    if kind == "aws_akid":
        return "AKIA" + _rand(16, "upper_digit")
    if kind == "google_api":
        return "AIza" + _rand(35, "alnum_dash")
    if kind == "stripe_live":
        return "sk_live_" + _rand(48, "alnum")
    if kind == "stripe_test":
        return "sk_test_" + _rand(48, "alnum")
    if kind == "jwt":
        return f"{_rand(32, 'alnum_dash')}.{_rand(64, 'alnum_dash')}.{_rand(43, 'alnum_dash')}"
    if kind == "hex64":
        return "".join(random.choices("0123456789abcdef", k=64))
    return _rand(40, "base64")  # generic base64-ish


def _rand(n: int, alphabet: str) -> str:
    pools = {
        "alnum":      string.ascii_letters + string.digits,
        "alnum_dash": string.ascii_letters + string.digits + "_-",
        "upper_digit": string.ascii_uppercase + string.digits,
        "base64":     string.ascii_letters + string.digits + "+/=",
    }
    return "".join(random.choices(pools[alphabet], k=n))


# =================================================================
# Per-template payload bundles
# =================================================================

def build_payload(template: str, lang: str) -> dict[str, str]:
    if template == "dev_chat":
        person = _name(lang)
        return {
            "SECRET": _secret(),
            "PERSON": person,
            "EMAIL":  _email(lang, person_hint=person),
        }
    if template == "customer_record":
        person = _name(lang)
        return {
            "PERSON":  person,
            "ADDRESS": _address(lang),
            "EMAIL":   _email(lang, person_hint=person),
            "PHONE":   _phone(),
            "ACCOUNT": _account(),
        }
    if template == "incident_report":
        person = _name("en")
        return {
            "SECRET": _secret(),
            "PERSON": person,
            "PHONE":  _phone(),
            "EMAIL":  _email("en", person_hint=person),
        }
    raise ValueError(f"Unknown template: {template}")


# =================================================================
# Prompts (system + user) per (template, lang)
# =================================================================

# System prompts: per (template, lang). Sets role + format.
# User prompts: built dynamically with the slot values.

_SYSTEM_DEV_CHAT = {
    "de_ch": (
        "Du simulierst eine kurze Slack- oder Teams-Nachricht zwischen "
        "zwei Software-Entwicklern in einem Schweizer Tech-Unternehmen. "
        "Schreibe 2 bis 4 Sätze realistischer Entwickler-zu-Entwickler-"
        "Konversation. Themen: einen geleakten Schlüssel rotieren, "
        "Staging-Zugang teilen, Token widerrufen, Authentifizierungs-"
        "probleme debuggen. Du MUSST alle untenstehenden Werte WÖRTLICH "
        "und EXAKT in deine Antwort einbauen — verändere, paraphrasiere "
        "oder zitiere sie nicht. Gib NUR den Nachrichtentext aus, ohne "
        "Anführungszeichen, ohne Markdown-Backticks, ohne Vorrede."
    ),
    "fr_ch": (
        "Tu simules un bref message Slack ou Teams entre deux développeurs "
        "logiciels d'une entreprise tech suisse. Écris 2 à 4 phrases de "
        "conversation réaliste entre développeurs. Sujets: faire tourner "
        "une clé compromise, partager un accès staging, révoquer un "
        "token, déboguer un problème d'authentification. Tu DOIS inclure "
        "toutes les valeurs ci-dessous TEXTUELLEMENT et EXACTEMENT dans "
        "ta réponse — ne les modifie pas, ne les paraphrase pas, ne les "
        "mets pas entre guillemets. Affiche UNIQUEMENT le corps du "
        "message, sans guillemets, sans markdown, sans préambule."
    ),
    "it_ch": (
        "Stai simulando un breve messaggio Slack o Teams tra due "
        "sviluppatori software in un'azienda tech svizzera. Scrivi da 2 "
        "a 4 frasi di conversazione realistica tra sviluppatori. Temi: "
        "ruotare una chiave compromessa, condividere accesso allo staging, "
        "revocare un token, eseguire il debug di un problema di "
        "autenticazione. DEVI includere tutti i valori sottostanti "
        "LETTERALMENTE ed ESATTAMENTE nella tua risposta — non "
        "modificarli, non parafrasarli, non metterli tra virgolette. "
        "Output SOLO il corpo del messaggio, senza virgolette, senza "
        "markdown, senza preamboli."
    ),
    "rm": (
        "Ti simuleschas in curt messadi Slack u Teams tranter dus "
        "sviluppaders da software en in'interpresa tech svizra. Scriva "
        "da 2 fin 4 frasas dad ina conversaziun realistica tranter "
        "sviluppaders. Tematicas: rotar ina clav cumpromettida, partir "
        "in access al staging, revocar in token, debugar in problem "
        "d'autentificaziun. Ti DASTGAS includer tut las valuras sutvart "
        "LITTERALMAIN ed EXAGTAMAIN en tia resposta — na las midar "
        "betg, na las parafrasar betg, na las metter betg tranter "
        "virgulettas. Mussa MO il corp dal messadi, senza virgulettas, "
        "senza markdown, senza prefaziun."
    ),
    "en": (
        "You are simulating a brief Slack or Teams message between two "
        "software developers at a Swiss tech company. Write 2 to 4 "
        "sentences of realistic developer-to-developer conversation. "
        "Topics: rotating a leaked credential, sharing staging access, "
        "revoking a token, debugging an authentication problem. You MUST "
        "embed every value below VERBATIM and EXACTLY in your reply — do "
        "not modify, paraphrase, or quote them. Output ONLY the message "
        "body — no quotation marks, no markdown fences, no preamble."
    ),
}

_SYSTEM_CUSTOMER_RECORD = {
    "it_ch": (
        "Stai generando una singola voce dal sistema CRM/fatturazione "
        "interno di un'azienda svizzera. Formattala come 4-8 righe di "
        "prosa naturale o di campi etichettati (Nome:, Indirizzo:, ecc.). "
        "Varia la struttura: conferma di registrazione, voce di "
        "fatturazione, nota di un caso di supporto, aggiornamento di "
        "contatto. DEVI includere TUTTI i valori sottostanti LETTERALMENTE "
        "ed ESATTAMENTE — non modificarli, non parafrasarli. Output SOLO "
        "il testo della voce, senza markdown, senza preamboli."
    ),
    "rm": (
        "Ti generescha ina suletta endataziun dal sistem CRM/facturaziun "
        "intern dad ina interpresa svizra. Formatescha la sco 4-8 lingias "
        "da prosa naturala u champs marcads (Num:, Adressa:, etc.). "
        "Varia la structura: confirmaziun da registraziun, endataziun da "
        "facturaziun, nota dad in cas da support, actualisaziun dad in "
        "contact. Ti DASTGAS includer TUT las valuras sutvart "
        "LITTERALMAIN ed EXAGTAMAIN — na las midar betg, na las "
        "parafrasar betg. Mussa MO il text da l'endataziun, senza "
        "markdown, senza prefaziun."
    ),
    "en": (
        "You are generating a single entry from a Swiss company's "
        "internal CRM or billing system. Format it as 4-8 lines of "
        "natural prose or labelled fields (Name:, Address:, Email:, etc.). "
        "Vary the structure: registration confirmation, billing entry, "
        "support case note, contact update. You MUST include EVERY value "
        "below VERBATIM and EXACTLY — do not modify or paraphrase them. "
        "Output ONLY the entry text, no markdown, no preamble."
    ),
}

_SYSTEM_INCIDENT_REPORT = {
    "en": (
        "You are writing a brief internal security incident report (3 to "
        "6 sentences) about a leaked or exposed credential at a Swiss "
        "tech company. Mention the leaked credential, the responsible "
        "engineer, an oncall phone contact, and a security team email. "
        "You MUST embed every value below VERBATIM and EXACTLY — do not "
        "modify, paraphrase, or quote them. Output ONLY the report body, "
        "no headers, no markdown."
    ),
}

SYSTEM_PROMPTS: dict[tuple[str, str], str] = {}
for lang, sys_text in _SYSTEM_DEV_CHAT.items():
    SYSTEM_PROMPTS[("dev_chat", lang)] = sys_text
for lang, sys_text in _SYSTEM_CUSTOMER_RECORD.items():
    SYSTEM_PROMPTS[("customer_record", lang)] = sys_text
for lang, sys_text in _SYSTEM_INCIDENT_REPORT.items():
    SYSTEM_PROMPTS[("incident_report", lang)] = sys_text


def build_user_prompt(template: str, lang: str, payload: dict[str, str]) -> str:
    """Build a user prompt that lists the slot values."""
    if template == "dev_chat":
        return (
            f"Embed these three values verbatim:\n"
            f"  API key / token: {payload['SECRET']}\n"
            f"  Sender name: {payload['PERSON']}\n"
            f"  Email: {payload['EMAIL']}"
        )
    if template == "customer_record":
        return (
            f"Embed these five values verbatim:\n"
            f"  Name: {payload['PERSON']}\n"
            f"  Address: {payload['ADDRESS']}\n"
            f"  Email: {payload['EMAIL']}\n"
            f"  Phone: {payload['PHONE']}\n"
            f"  ID / IBAN / VAT / CC: {payload['ACCOUNT']}"
        )
    if template == "incident_report":
        return (
            f"Embed these four values verbatim:\n"
            f"  Leaked credential: {payload['SECRET']}\n"
            f"  Responsible engineer: {payload['PERSON']}\n"
            f"  Oncall phone: {payload['PHONE']}\n"
            f"  Security email: {payload['EMAIL']}"
        )
    raise ValueError(template)


# =================================================================
# Verifier
# =================================================================

def extract_spans(
    text: str, payload: dict[str, str]
) -> list[dict] | None:
    """Find each payload value in text. Return list of span dicts, or None
    if any value is missing (chunk should be rejected)."""
    spans: list[dict] = []
    for slot, value in payload.items():
        if not value:
            continue
        idx = text.find(value)
        if idx < 0:
            return None
        spans.append({
            "start": idx,
            "end": idx + len(value),
            "label": SLOT_LABELS[slot],
            "value": value,
            "source": "synthetic",
        })
    spans.sort(key=lambda sp: sp["start"])
    return spans


# =================================================================
# Driver
# =================================================================

def count_existing(out_path: Path) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    if not out_path.exists():
        return counts
    with out_path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            counts[(rec.get("synthetic_template", "?"), rec.get("language", "?"))] += 1
    return counts


def remaining_buckets(targets: dict, existing: Counter) -> list[tuple[str, str, int]]:
    """Return [(template, lang, n_remaining)] sorted by n_remaining desc."""
    out = []
    for (template, lang), target in targets.items():
        n = max(0, target - existing.get((template, lang), 0))
        if n > 0:
            out.append((template, lang, n))
    out.sort(key=lambda x: -x[2])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Generate 10 per (template, lang), exit. For pipeline check.")
    parser.add_argument("--max-attempts", type=int, default=None,
                        help="Stop after this many gemma attempts (debugging).")
    args = parser.parse_args()

    random.seed(SEED)
    faker_ch.seed_all(SEED)

    targets = TARGETS.copy()
    if args.smoke:
        targets = {k: 10 for k in targets}
        print("[SMOKE MODE] target = 10 per (template, lang)")

    existing = count_existing(OUT_PATH)
    if existing:
        print(f"Found existing {OUT_PATH}: {sum(existing.values()):,} chunks already done")
        for k, n in sorted(existing.items()):
            print(f"  {k[0]:<18} {k[1]:<6} {n:>6,}")

    buckets = remaining_buckets(targets, existing)
    if not buckets:
        print("All targets met. Nothing to do.")
        return
    print()
    print(f"Remaining: {sum(b[2] for b in buckets):,} chunks across {len(buckets)} buckets")
    for tmpl, lang, n in buckets:
        print(f"  {tmpl:<18} {lang:<6} {n:>6,}")

    print()
    print("Loading Gemma vLLM ...")
    from .gemma.client import GemmaClient
    gemma = GemmaClient()
    gemma._load()
    print("Gemma loaded.")

    out_f = OUT_PATH.open("a")
    n_attempts = 0
    n_accepted = 0
    n_rejected = 0
    rejects_per_bucket: Counter[tuple[str, str]] = Counter()
    accepts_per_bucket: Counter[tuple[str, str]] = Counter()
    t0 = time.time()
    last_report = t0

    try:
        while buckets:
            # Build a batch by drawing from buckets in priority order.
            # Each batch slot picks the (template, lang) with most remaining,
            # so we make balanced progress.
            batch: list[tuple[str, str, dict]] = []  # (template, lang, payload)
            for _ in range(BATCH_SIZE):
                if not buckets:
                    break
                buckets.sort(key=lambda x: -x[2])
                tmpl, lang, _n = buckets[0]
                payload = build_payload(tmpl, lang)
                batch.append((tmpl, lang, payload))
                # Decrement provisionally; will re-add if rejected
                buckets[0] = (tmpl, lang, buckets[0][2] - 1)
                if buckets[0][2] <= 0:
                    buckets.pop(0)

            if not batch:
                break

            # Build prompts
            messages_batch = [
                [
                    {"role": "system", "content": SYSTEM_PROMPTS[(t, la)]},
                    {"role": "user", "content": build_user_prompt(t, la, p)},
                ]
                for (t, la, p) in batch
            ]

            # Generate
            outputs = gemma.chat_messages(
                messages_batch,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=0.95,
            )
            n_attempts += len(batch)

            # Verify each
            for (tmpl, lang, payload), text in zip(batch, outputs, strict=True):
                spans = extract_spans(text, payload)
                if spans is None:
                    n_rejected += 1
                    rejects_per_bucket[(tmpl, lang)] += 1
                    # Re-add this bucket as needing one more
                    found = False
                    for i, (t, la, n) in enumerate(buckets):
                        if t == tmpl and la == lang:
                            buckets[i] = (t, la, n + 1)
                            found = True
                            break
                    if not found:
                        buckets.append((tmpl, lang, 1))
                    continue

                rec = {
                    "id": f"synthetic#{tmpl}#{lang}#{accepts_per_bucket[(tmpl, lang)] + 1}",
                    "text": text,
                    "language": lang,
                    "subset": "synthetic",
                    "source_dataset": "gheim-synthetic",
                    "doc_id": f"synthetic_{tmpl}",
                    "chunk_index_in_doc": accepts_per_bucket[(tmpl, lang)],
                    "spans": spans,
                    "labeler": "gemma-template-fill+verifier:gheim_training.data.synth.slot_fill_gemma",
                    "prompt_version": "p9-v1-2026-05-08",
                    "labeled_at": datetime.now().isoformat(timespec="seconds"),
                    "synthetic": True,
                    "synthetic_template": tmpl,
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                n_accepted += 1
                accepts_per_bucket[(tmpl, lang)] += 1

            # Periodic report
            if time.time() - last_report > 60:
                last_report = time.time()
                elapsed = time.time() - t0
                rate = n_attempts / elapsed if elapsed > 0 else 0
                remaining = sum(b[2] for b in buckets)
                eta = remaining / rate if rate > 0 else float('inf')
                reject_rate = n_rejected / n_attempts if n_attempts > 0 else 0
                print(f"[{int(elapsed)}s] attempts={n_attempts:,} accepted={n_accepted:,} "
                      f"rejected={n_rejected:,} ({100*reject_rate:.1f}%) "
                      f"rate={rate:.2f}/s remaining={remaining:,} "
                      f"eta={int(eta/60)}min", flush=True)
                # Per-bucket reject rates (to spot RM degenerating)
                if rejects_per_bucket:
                    print("  reject rates per bucket:")
                    for (t, la), n_rej in sorted(rejects_per_bucket.items()):
                        n_acc = accepts_per_bucket.get((t, la), 0)
                        rr = n_rej / (n_rej + n_acc) if (n_rej + n_acc) > 0 else 0
                        print(f"    {t:<18} {la:<6} rej={n_rej:>5,} acc={n_acc:>5,} ({100*rr:.0f}% rej)", flush=True)

            if args.max_attempts and n_attempts >= args.max_attempts:
                print(f"Reached --max-attempts={args.max_attempts}, stopping.")
                break
    finally:
        out_f.close()

    elapsed = time.time() - t0
    print()
    print("=" * 80)
    print(f"DONE in {int(elapsed)}s ({elapsed/3600:.2f}h)")
    print(f"  attempts:  {n_attempts:,}")
    print(f"  accepted:  {n_accepted:,}")
    print(f"  rejected:  {n_rejected:,}  ({100*n_rejected/max(n_attempts,1):.1f}%)")
    print()
    print("Per-bucket final counts:")
    for (t, la), n in sorted(accepts_per_bucket.items()):
        n_rej = rejects_per_bucket.get((t, la), 0)
        print(f"  {t:<18} {la:<6} accepted={n:>6,}  rejected={n_rej:>6,}")


if __name__ == "__main__":
    main()
