"""Synthetic name-pattern chunks targeting v2 edge-case failures.

The edge-case probe (eval.probe_edge_cases) caught 14/21 failures on the
v2 model. Pattern analysis:

- bare first name in greeting (``Hallo Marius,``) — 4/4 langs failed
- bare first/last name in narrative (``Anna meinte gestern...``) — failed
- signature with bare first name (``Liebe Grüsse,\\nMarius``) — failed
- title + lastname abbreviations (``Hr. Müller``) — failed
- title + common-word surname (``Herr Bach``) — failed
- RM title + lastname (``Sgnr. Caduff``) — failed

Root cause: Apertus stream sources (encyclopedia, court, parliament) do
not produce email-correspondence patterns. The model never sees
"Hallo {first_name},". This module fills that gap with hand-written
templates rendered with Faker_CH first/last/full names.

Six template families, all four target languages (de/fr/it/rm), one
each for the explicit failure patterns above. Total ~5000-7000 chunks
written to ``data/layer_name_patterns.jsonl`` and consumed by the V2-9
balancer as the source ``synthetic_name_patterns``.

Run
---
    uv run python -m gheim_training.data.v2.synth_name_patterns
"""
from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..synthetic.faker_ch import (
    first_name_de, first_name_fr, first_name_it, first_name_rm,
    last_name_de, last_name_fr, last_name_it, last_name_rm,
    name_de, name_fr, name_it, name_rm,
    seed_all,
)

OUT_PATH = Path("data/layer_name_patterns.jsonl")
SEED = 20260523

# How many renders per (template_id × language). Each render uses fresh
# Faker values so the same template produces dozens of distinct chunks.
# ~6 families × ~6 templates each × 4 langs × 30 renders ≈ 4300 chunks.
RENDERS_PER_TEMPLATE = 30

# Per-language name generators.
FIRST_NAME = {
    "de_ch": first_name_de, "fr_ch": first_name_fr,
    "it_ch": first_name_it, "rm": first_name_rm,
}
LAST_NAME = {
    "de_ch": last_name_de, "fr_ch": last_name_fr,
    "it_ch": last_name_it, "rm": last_name_rm,
}
FULL_NAME = {
    "de_ch": name_de, "fr_ch": name_fr, "it_ch": name_it, "rm": name_rm,
}


@dataclass(frozen=True)
class Template:
    """One template, one language. ``body`` must contain exactly one
    ``{name}`` placeholder. ``name_kind`` selects which generator (first,
    last, full, title+last)."""
    template_id: str
    language: str
    body: str
    name_kind: str  # "first" | "last" | "full" | "title_last" | "title_first_last"


# ---------------------------------------------------------------------------
# Family 1 — Greeting + bare first name
# ---------------------------------------------------------------------------

_GREETING_FIRSTNAME: list[Template] = [
    # DE
    Template("greet_firstname_v1", "de_ch",
             "Hallo {name}, danke für deine Nachricht.", "first"),
    Template("greet_firstname_v2", "de_ch",
             "Liebe {name}, ich habe deine Frage erhalten.", "first"),
    Template("greet_firstname_v3", "de_ch",
             "Lieber {name}, vielen Dank für die schnelle Antwort.", "first"),
    Template("greet_firstname_v4", "de_ch",
             "Hi {name}, kurze Rückfrage zu deinem letzten Mail.", "first"),
    Template("greet_firstname_v5", "de_ch",
             "Guten Tag {name}, anbei die gewünschten Unterlagen.", "first"),
    Template("greet_firstname_v6", "de_ch",
             "Servus {name}, melde dich gerne, falls etwas unklar ist.", "first"),
    # FR
    Template("greet_firstname_v1", "fr_ch",
             "Bonjour {name}, merci de votre message.", "first"),
    Template("greet_firstname_v2", "fr_ch",
             "Salut {name}, j'espère que tout va bien.", "first"),
    Template("greet_firstname_v3", "fr_ch",
             "Cher {name}, veuillez trouver ci-joint le document.", "first"),
    Template("greet_firstname_v4", "fr_ch",
             "Chère {name}, je vous remercie pour votre retour rapide.", "first"),
    Template("greet_firstname_v5", "fr_ch",
             "Coucou {name}, dis-moi si tu as un moment cette semaine.", "first"),
    # IT
    Template("greet_firstname_v1", "it_ch",
             "Ciao {name}, grazie del tuo messaggio.", "first"),
    Template("greet_firstname_v2", "it_ch",
             "Caro {name}, ti allego il documento richiesto.", "first"),
    Template("greet_firstname_v3", "it_ch",
             "Cara {name}, grazie per la rapida risposta.", "first"),
    Template("greet_firstname_v4", "it_ch",
             "Buongiorno {name}, ho ricevuto la tua nota.", "first"),
    Template("greet_firstname_v5", "it_ch",
             "Salve {name}, le scrivo riguardo all'ordine.", "first"),
    # RM
    Template("greet_firstname_v1", "rm",
             "Hai {name}, grazia per tia communicaziun.", "first"),
    Template("greet_firstname_v2", "rm",
             "Char {name}, ti agiat de rispunder rapidamain.", "first"),
    Template("greet_firstname_v3", "rm",
             "Bun di {name}, en agiunta chattas ti il document.", "first"),
]

# ---------------------------------------------------------------------------
# Family 2 — Narrative + bare first name
# ---------------------------------------------------------------------------

_NARRATIVE_FIRSTNAME: list[Template] = [
    # DE
    Template("narr_first_v1", "de_ch",
             "{name} meinte gestern, dass die Lieferung morgen kommt.", "first"),
    Template("narr_first_v2", "de_ch",
             "Heute Morgen hat {name} im Büro angerufen.", "first"),
    Template("narr_first_v3", "de_ch",
             "Laut {name} ist der Termin auf nächsten Dienstag verschoben.", "first"),
    Template("narr_first_v4", "de_ch",
             "{name} und ich treffen uns am Mittwoch im Restaurant.", "first"),
    Template("narr_first_v5", "de_ch",
             "Ich habe {name} gebeten, das Dokument durchzusehen.", "first"),
    # FR
    Template("narr_first_v1", "fr_ch",
             "{name} a confirmé hier que le contrat sera signé demain.", "first"),
    Template("narr_first_v2", "fr_ch",
             "{name} m'a appelé ce matin pour clarifier les détails.", "first"),
    Template("narr_first_v3", "fr_ch",
             "D'après {name}, la réunion est reportée à mardi prochain.", "first"),
    Template("narr_first_v4", "fr_ch",
             "Je dois discuter de ce point avec {name} cette semaine.", "first"),
    # IT
    Template("narr_first_v1", "it_ch",
             "{name} ha confermato ieri che il contratto sarà firmato domani.", "first"),
    Template("narr_first_v2", "it_ch",
             "Stamattina {name} mi ha chiamato per chiarire i dettagli.", "first"),
    Template("narr_first_v3", "it_ch",
             "Secondo {name}, la riunione è rinviata a martedì prossimo.", "first"),
    Template("narr_first_v4", "it_ch",
             "Devo parlare con {name} di questo punto questa settimana.", "first"),
    # RM
    Template("narr_first_v1", "rm",
             "{name} ha confermau gliordi che la furniziun arriva damaun.", "first"),
    Template("narr_first_v2", "rm",
             "Quest mintga {name} m'ha clamau per discutar.", "first"),
    Template("narr_first_v3", "rm",
             "Tenor {name}, la sesida vegn spustada a mardi.", "first"),
]

# ---------------------------------------------------------------------------
# Family 3 — Narrative + bare last name
# ---------------------------------------------------------------------------

_NARRATIVE_LASTNAME: list[Template] = [
    # DE
    Template("narr_last_v1", "de_ch",
             "{name} hat das Protokoll heute Morgen unterzeichnet.", "last"),
    Template("narr_last_v2", "de_ch",
             "Wie {name} bereits erwähnte, beginnt die Sitzung um 14:00.", "last"),
    Template("narr_last_v3", "de_ch",
             "Die Beschwerde wurde von {name} eingereicht.", "last"),
    Template("narr_last_v4", "de_ch",
             "Laut Aussage von {name} liegt der Fehler beim Lieferanten.", "last"),
    Template("narr_last_v5", "de_ch",
             "{name} und {other_last} arbeiten am selben Dossier.", "last"),
    # FR
    Template("narr_last_v1", "fr_ch",
             "{name} a signé le protocole ce matin.", "last"),
    Template("narr_last_v2", "fr_ch",
             "Comme {name} l'a mentionné, la séance commence à 14h.", "last"),
    Template("narr_last_v3", "fr_ch",
             "La plainte a été déposée par {name}.", "last"),
    Template("narr_last_v4", "fr_ch",
             "Selon {name}, l'erreur vient du fournisseur.", "last"),
    # IT
    Template("narr_last_v1", "it_ch",
             "{name} ha firmato il verbale stamattina.", "last"),
    Template("narr_last_v2", "it_ch",
             "Come ha già detto {name}, la seduta inizia alle 14.", "last"),
    Template("narr_last_v3", "it_ch",
             "Il reclamo è stato presentato da {name}.", "last"),
    Template("narr_last_v4", "it_ch",
             "Secondo {name}, l'errore è del fornitore.", "last"),
    # RM
    Template("narr_last_v1", "rm",
             "{name} ha suttascrit il protocoll quest mintga.", "last"),
    Template("narr_last_v2", "rm",
             "Sco {name} ha gia menziunà, la sesida cumenza a las 14:00.", "last"),
    Template("narr_last_v3", "rm",
             "La reclamaziun ei vegnida inoltrada da {name}.", "last"),
]

# ---------------------------------------------------------------------------
# Family 4 — Signature line with bare first name
# ---------------------------------------------------------------------------

_SIGNATURE_FIRSTNAME: list[Template] = [
    Template("sig_first_v1", "de_ch",
             "Vielen Dank für die schnelle Bearbeitung.\n\n"
             "Liebe Grüsse,\n{name}", "first"),
    Template("sig_first_v2", "de_ch",
             "Bei Fragen melde dich jederzeit.\n\n"
             "Beste Grüsse\n{name}", "first"),
    Template("sig_first_v3", "de_ch",
             "Ich freue mich auf deine Rückmeldung.\n\n"
             "Gruss\n{name}", "first"),
    Template("sig_first_v1", "fr_ch",
             "Merci d'avance pour votre retour.\n\n"
             "Cordialement,\n{name}", "first"),
    Template("sig_first_v2", "fr_ch",
             "N'hésitez pas à me contacter pour toute question.\n\n"
             "Bien à vous,\n{name}", "first"),
    Template("sig_first_v1", "it_ch",
             "Grazie in anticipo per il riscontro.\n\n"
             "Cordiali saluti,\n{name}", "first"),
    Template("sig_first_v2", "it_ch",
             "Resto a disposizione per ulteriori chiarimenti.\n\n"
             "Saluti,\n{name}", "first"),
    Template("sig_first_v1", "rm",
             "Grazia per Vossa attenziun.\n\n"
             "Cordials salids,\n{name}", "first"),
]

# ---------------------------------------------------------------------------
# Family 5 — Title (abbrev or full) + last name
# ---------------------------------------------------------------------------

_TITLE_LASTNAME: list[Template] = [
    # DE abbreviations
    Template("title_abbrev_v1", "de_ch",
             "Hr. {name} hat mich heute Morgen kontaktiert.", "last"),
    Template("title_abbrev_v2", "de_ch",
             "Fr. {name} ist nicht im Büro diese Woche.", "last"),
    Template("title_abbrev_v3", "de_ch",
             "Bitte richten Sie meine Grüsse an Hr. {name} aus.", "last"),
    Template("title_abbrev_v4", "de_ch",
             "Dr. {name} hat den Befund bestätigt.", "last"),
    Template("title_abbrev_v5", "de_ch",
             "Prof. {name} leitet die Sitzung am Donnerstag.", "last"),
    Template("title_full_v1", "de_ch",
             "Sehr geehrter Herr {name}, anbei das gewünschte Dokument.", "last"),
    Template("title_full_v2", "de_ch",
             "Sehr geehrte Frau {name}, vielen Dank für Ihre Anfrage.", "last"),
    # FR
    Template("title_abbrev_v1", "fr_ch",
             "M. {name} a confirmé sa présence à la réunion.", "last"),
    Template("title_abbrev_v2", "fr_ch",
             "Mme {name} sera absente jusqu'à vendredi.", "last"),
    Template("title_abbrev_v3", "fr_ch",
             "Dr {name} a signé le rapport médical.", "last"),
    Template("title_full_v1", "fr_ch",
             "Cher Monsieur {name}, je vous remercie de votre courrier.", "last"),
    Template("title_full_v2", "fr_ch",
             "Chère Madame {name}, veuillez trouver ci-joint le devis.", "last"),
    # IT
    Template("title_abbrev_v1", "it_ch",
             "Sig. {name} ha confermato la sua presenza alla riunione.", "last"),
    Template("title_abbrev_v2", "it_ch",
             "Sig.ra {name} sarà assente fino a venerdì.", "last"),
    Template("title_abbrev_v3", "it_ch",
             "Dott. {name} ha firmato il rapporto medico.", "last"),
    Template("title_full_v1", "it_ch",
             "Egregio Signor {name}, La ringrazio per la sua lettera.", "last"),
    Template("title_full_v2", "it_ch",
             "Gentile Signora {name}, in allegato il preventivo richiesto.", "last"),
    # RM
    Template("title_abbrev_v1", "rm",
             "Sgnr. {name} ha confermau sia preschientscha.", "last"),
    Template("title_abbrev_v2", "rm",
             "Sgnra. {name} ei absenta entro venderdi.", "last"),
    Template("title_full_v1", "rm",
             "Stimà Sgnr. {name}, grazia per Vossa cuminonza.", "last"),
    Template("title_full_v2", "rm",
             "Stimada Sgnra. {name}, en agiunta chattas il document.", "last"),
]

# ---------------------------------------------------------------------------
# Family 6 — Common-word surnames (avoid common-noun false negatives)
# ---------------------------------------------------------------------------

# These surnames are also everyday German nouns. Without explicit
# training, the model treats them as common words and misses the person.
_COMMON_WORD_LAST_DE = (
    "Bach", "Berg", "Stein", "Stern", "Frank", "Sommer", "Winter",
    "Herzog", "König", "Schmied", "Vogel", "Fischer", "Bauer", "Müller",
    "Schneider", "Wagner", "Becker", "Hofer", "Wolf", "Hahn",
)

_COMMON_WORD_TEMPLATES: list[Template] = [
    Template("commonword_v1", "de_ch",
             "Herr {name} hat das Konzert dirigiert.", "last"),
    Template("commonword_v2", "de_ch",
             "Frau {name} hat die Schule geleitet.", "last"),
    Template("commonword_v3", "de_ch",
             "Dr. {name} hat den Bericht verfasst.", "last"),
    Template("commonword_v4", "de_ch",
             "{name} hat den Vertrag heute unterzeichnet.", "last"),
    Template("commonword_v5", "de_ch",
             "Wie {name} bereits sagte, ist die Sitzung verschoben.", "last"),
]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _gen_name(lang: str, kind: str) -> str:
    """Generate one name string for the given kind."""
    if kind == "first":
        return FIRST_NAME[lang]()
    if kind == "last":
        return LAST_NAME[lang]()
    if kind == "full":
        return FULL_NAME[lang]()
    raise ValueError(f"unknown name_kind {kind!r}")


def _render(template: Template, rng: random.Random,
            name_pool: Callable[[], str] | None = None,
            ) -> tuple[str, list[dict]]:
    """Render one template instance. Returns (text, [span dict])."""
    name = name_pool() if name_pool else _gen_name(template.language, template.name_kind)
    text = template.body.format(name=name, other_last=_gen_name(template.language, "last"))
    # Locate name span (the first occurrence; templates have exactly one).
    i = text.find(name)
    if i < 0:
        raise RuntimeError(f"name {name!r} not in rendered template: {text!r}")
    spans = [{"start": i, "end": i + len(name), "label": "private_person"}]
    return text, spans


def _emit(templates: list[Template], n_per: int, rng: random.Random,
          out_lines: list[str], counter: list[int],
          *, name_pool: Callable[[], str] | None = None) -> None:
    for tpl in templates:
        for _ in range(n_per):
            text, spans = _render(tpl, rng, name_pool=name_pool)
            rec = {
                "id": f"name_pattern_{counter[0]:06d}",
                "text": text,
                "spans": spans,
                "language": tpl.language,
                "source": "synthetic_name_patterns",
                "template_id": tpl.template_id,
                "meta": {"template_id": tpl.template_id,
                         "family": tpl.template_id.split("_")[0]},
            }
            out_lines.append(json.dumps(rec, ensure_ascii=False))
            counter[0] += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-template", type=int, default=RENDERS_PER_TEMPLATE)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    seed_all(args.seed)
    rng = random.Random(args.seed)

    print(f"Generating name-pattern synthetic chunks (seed={args.seed})")
    out_lines: list[str] = []
    counter = [0]

    for family_name, templates in (
        ("greeting_firstname", _GREETING_FIRSTNAME),
        ("narrative_firstname", _NARRATIVE_FIRSTNAME),
        ("narrative_lastname", _NARRATIVE_LASTNAME),
        ("signature_firstname", _SIGNATURE_FIRSTNAME),
        ("title_lastname", _TITLE_LASTNAME),
    ):
        start = counter[0]
        _emit(templates, args.n_per_template, rng, out_lines, counter)
        print(f"  {family_name:<22}  {len(templates)} templates × {args.n_per_template} = {counter[0] - start} chunks")

    # Family 6: common-word surnames. Restricted pool of surnames that
    # are also common DE nouns; sample from the pool instead of Faker so
    # the model specifically sees these surnames in person contexts.
    def _cw_pool() -> str:
        return rng.choice(_COMMON_WORD_LAST_DE)

    start = counter[0]
    _emit(_COMMON_WORD_TEMPLATES, args.n_per_template * 2, rng, out_lines,
          counter, name_pool=_cw_pool)
    print(f"  {'common_word_lastname':<22}  {len(_COMMON_WORD_TEMPLATES)} templates × "
          f"{args.n_per_template * 2} = {counter[0] - start} chunks")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for line in out_lines:
            f.write(line)
            f.write("\n")
    print(f"\nWrote {len(out_lines):,} chunks to {args.out}")


if __name__ == "__main__":
    main()
