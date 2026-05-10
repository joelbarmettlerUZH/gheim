"""Prompt builders for Apertus slot-filled generation.

We pre-fill a "slot bag" of PII values from Faker_CH, hand them to Apertus,
and ask it to write a natural Swiss text that **uses every value verbatim**.
After generation we re-locate each value in the output to recover char spans.

v2: few-shot prompting with 3 in-context examples per language (DE/FR/IT,
shared for RM/GSW since base patterns transfer). Lifts slot-fill yield from
~46% (cold) to ~80% (per the Phase 2 plan in CLAUDE.md).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from . import slots

# Per-language scenario pool. Each scenario hints at register/genre.
SCENARIOS: dict[str, tuple[str, ...]] = {
    "de_ch": (
        "eine kurze Kunden-E-Mail an eine Schweizer Bank",
        "ein interner HR-Vermerk in einem Schweizer Unternehmen",
        "ein Arztbericht aus einer Hausarztpraxis",
        "ein Support-Ticket bei einem Schweizer Telekom-Anbieter",
        "ein Vertragsentwurf zwischen zwei KMU",
        "eine WhatsApp-Nachricht an einen Kollegen über ein Konto-Problem",
    ),
    "fr_ch": (
        "un courriel de client à une banque suisse",
        "une note RH interne d'une entreprise suisse",
        "un rapport médical de cabinet généraliste",
        "un ticket de support chez un opérateur télécom suisse",
        "un projet de contrat entre deux PME",
        "un message WhatsApp à un collègue concernant un problème de compte",
    ),
    "it_ch": (
        "un'e-mail di un cliente a una banca svizzera",
        "una nota interna delle risorse umane di un'azienda svizzera",
        "un referto medico di un medico di famiglia",
        "un ticket di supporto presso un operatore di telecomunicazioni svizzero",
        "una bozza di contratto tra due PMI",
        "un messaggio WhatsApp a un collega riguardo a un problema di conto",
    ),
    "rm": (
        "in e-mail dad in client tar ina banca svizra",
        "ina nota interna da las resursas umanas en in'interpresa svizra",
        "in tichet da support tar in furnider da telecommunicaziuns svizzer",
    ),
    "gsw": (
        "es kurzes WhatsApp-Gschpräch zwüsche zwöi Fründe über es Konto-Problem",
        "es informells E-Mail vomne Kunde a sini Bank uf Schwiizerdütsch",
        "e Notiz vomne Aazteller über sini Patiente",
    ),
}

LANGUAGE_NAMES: dict[str, str] = {
    "de_ch": "Standard German (Swiss)",
    "fr_ch": "Swiss French",
    "it_ch": "Swiss Italian",
    "rm": "Rumantsch Grischun",
    "gsw": "Swiss German dialect (Schweizerdeutsch, written informally)",
}


@dataclass(frozen=True)
class GenerationRequest:
    language: str
    scenario: str
    slot_bag: list[tuple[str, str]]  # (category, value) pairs

    def to_user_prompt(self) -> str:
        lang_label = LANGUAGE_NAMES[self.language]
        bullets = "\n".join(f"  - {value}" for _, value in self.slot_bag)
        return (
            f"Write {self.scenario}. The output MUST be in {lang_label}.\n\n"
            f"You MUST include every one of the following values verbatim, exactly "
            f"as written (same casing, same punctuation, same spaces):\n{bullets}\n\n"
            f"Rules:\n"
            f"  1. Use each value at least once, in a natural place in the text.\n"
            f"  2. Do NOT add brackets, labels, quotes, or any markup around the "
            f"     values — write them inline as ordinary parts of the sentence.\n"
            f"  3. Do not invent additional names, e-mails, phone numbers, IBANs, "
            f"     account numbers, dates, addresses, URLs, or secrets — only the "
            f"     values listed above may appear as PII.\n"
            f"  4. Keep the text between 60 and 220 words.\n"
            f"  5. Output ONLY the text itself — no preamble, no explanation, "
            f"     no markdown headers, no quotation marks around the result.\n"
        )


SYSTEM_PROMPT = (
    "You are a Swiss content generator producing realistic short documents in "
    "Swiss languages (German, French, Italian, Rumantsch, Swiss German dialect). "
    "Always follow user instructions exactly, including verbatim inclusion of "
    "supplied values and the requested output language. Do not add explanations, "
    "preambles, markdown, or quotes around the result. The values you receive "
    "MUST appear verbatim — same casing, same punctuation, same spaces — in "
    "your output."
)


# In-context demos. Three per "register" — banking, medical, contract — covering
# the most common slot-fill failure modes (Apertus paraphrasing IBANs by
# inserting spaces, dropping AHV dots, omitting trailing chars). Demos are
# designed for DE; FR/IT/RM/GSW reuse the same demos because the verbatim-
# inclusion behaviour transfers across languages.
_FEWSHOT_DEMOS: tuple[dict, ...] = (
    {
        "user": (
            "Write a customer email to a Swiss bank. The output MUST be in "
            "Standard German (Swiss).\n\n"
            "You MUST include every one of the following values verbatim, "
            "exactly as written (same casing, same punctuation, same spaces):\n"
            "  - Anna Müller\n"
            "  - CH9300762011623852957\n"
            "  - 12.03.2024\n"
            "  - anna.mueller@bluewin.ch\n\n"
            "Rules: use each value at least once, no markup, no extra invented "
            "PII, 60-220 words, output only the text."
        ),
        "assistant": (
            "Sehr geehrte Damen und Herren\n\n"
            "Ich, Anna Müller, möchte Sie auf einen Fehler in meinem letzten "
            "Kontoauszug hinweisen. Am 12.03.2024 wurde von meinem Konto mit "
            "der IBAN CH9300762011623852957 ein Betrag belastet, den ich nicht "
            "veranlasst habe. Bitte prüfen Sie diesen Vorgang und melden Sie "
            "sich bei mir per E-Mail unter anna.mueller@bluewin.ch oder "
            "telefonisch im Geschäft. Ich bitte um zeitnahe Klärung.\n\n"
            "Freundliche Grüsse\nAnna Müller"
        ),
    },
    {
        "user": (
            "Write a doctor's note. The output MUST be in Standard German (Swiss).\n\n"
            "You MUST include every one of the following values verbatim, "
            "exactly as written:\n"
            "  - Hans Brunner\n"
            "  - 5. November 1962\n"
            "  - 756.4488.2901.34\n"
            "  - +41 31 311 22 33\n\n"
            "Rules: use each value at least once, no markup, 60-220 words."
        ),
        "assistant": (
            "Arztbericht\n\nPatient: Hans Brunner, geboren am 5. November 1962. "
            "AHV-Nummer 756.4488.2901.34. Der Patient stellte sich heute in "
            "meiner Praxis mit anhaltenden Rückenschmerzen vor. Die klinische "
            "Untersuchung ergab keine neurologischen Ausfälle. Verordnet wurde "
            "Physiotherapie sowie eine Schmerztherapie für drei Wochen. Eine "
            "Kontrollkonsultation ist in einem Monat vorgesehen. Bei Verschlimmerung "
            "ist eine sofortige Meldung erforderlich; ich bin telefonisch unter "
            "+41 31 311 22 33 erreichbar."
        ),
    },
    {
        "user": (
            "Write a French banking customer message. The output MUST be in Swiss French.\n\n"
            "You MUST include every one of the following values verbatim:\n"
            "  - Pierre Dubois\n"
            "  - CH56 0483 5012 3456 7800 9\n"
            "  - 15/01/2024\n\n"
            "Rules: use each value at least once, no markup, 60-220 words."
        ),
        "assistant": (
            "Madame, Monsieur,\n\nJe vous écris pour signaler un problème "
            "concernant un virement effectué le 15/01/2024 sur le compte avec "
            "l'IBAN CH56 0483 5012 3456 7800 9. Le bénéficiaire indiqué était "
            "moi-même, Pierre Dubois, mais le montant n'apparaît pas sur mon "
            "relevé. Pouvez-vous vérifier ce qu'il s'est passé et me confirmer "
            "le statut de l'opération ? Je reste à votre disposition pour tout "
            "complément d'information.\n\nMeilleures salutations,\nPierre Dubois"
        ),
    },
)


def build_messages(language: str, *, slot_bag_size: int | None = None) -> tuple[list[dict], GenerationRequest]:
    """Build chat-template messages with few-shot demos + a fresh request.

    Returns the message list AND the GenerationRequest so the caller can
    later verify slot values appear in the output.
    """
    bag = slots.sample_slot_bag(language, n=slot_bag_size)
    scenario = random.choice(SCENARIOS[language])
    req = GenerationRequest(language=language, scenario=scenario, slot_bag=bag)
    msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for demo in _FEWSHOT_DEMOS:
        msgs.append({"role": "user", "content": demo["user"]})
        msgs.append({"role": "assistant", "content": demo["assistant"]})
    msgs.append({"role": "user", "content": req.to_user_prompt()})
    return msgs, req


def build_request(language: str, *, slot_bag_size: int | None = None) -> GenerationRequest:
    """Backwards-compat: returns the GenerationRequest only. Use build_messages
    when you also want the chat-template messages with few-shot demos."""
    bag = slots.sample_slot_bag(language, n=slot_bag_size)
    scenario = random.choice(SCENARIOS[language])
    return GenerationRequest(language=language, scenario=scenario, slot_bag=bag)
