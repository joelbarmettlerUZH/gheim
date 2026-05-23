"""V3 forensic edge-case probe — 200+ hand-crafted cases.

NOT a metric — a forensic test suite. Each case is a short text designed
to stress one specific failure pattern. Cases are tagged by
``pattern_tag`` so the report can group passes/fails by pattern, not
just count totals.

Pattern groups (the user-decision-driven v3 scope):

PERSON
- greet_first        — "Hallo Marius,"           (the v1 gap)
- greet_first_var    — variations: punct, line breaks, multi-greeting
- narr_first         — "Marius meinte gestern..."
- narr_last          — "Müller hat unterzeichnet..."
- sig_first          — "Liebe Grüsse,\\nMarius"
- sig_full           — "Liebe Grüsse,\\nMarius Müller"
- title_full         — "Sehr geehrter Herr Müller"
- title_abbrev       — "Hr. Müller / Dr. Müller / Sgnr. Caduff"
- title_firstlast    — "Dr. Anna Müller"
- multitle           — "Dr. med. Anna Müller, MBA"
- last_first_fmt     — "MÜLLER, Anna" / "Müller, Anna"
- initials           — "A. Müller", "Hr. M. Müller"
- compound_name      — "Hans-Peter Bürgi-Stocker"
- common_word_sur    — "Herr Bach hat dirigiert"
- adv_neg_person     — "Am Bach steht ein Baum" (no person)

DATE
- date_dot           — "30.06.2026"
- date_slash         — "30/06/2026"
- date_spelled       — "30. Juni 2026" / "le 30 juin 2026"
- date_iso           — "2026-06-30"
- date_modifier      — "Am 30.06., Bis 15. Mai"
- adv_neg_date       — "Zimmer 30.06 ist frei" (number, not date)

ADDRESS
- addr_full          — "Bahnhofstrasse 10, 8001 Zürich"
- addr_street_only   — "Bahnhofstrasse 10"
- addr_with_canton   — "Bahnhofstrasse 10, 8001 Zürich ZH"
- addr_city_only     — "Zürich" / "Cuera"
- adv_neg_addr       — "Maienfeld ist hübsch" (city as nominal)

EMAIL
- email_std          — "alice@example.ch"
- email_special      — "alice.müller+tag@example.ch"
- email_in_sig       — embedded in signature line
- adv_neg_email      — "info@example.ch" in API docs (template, not real)

PHONE
- phone_intl         — "+41 44 555 12 34"
- phone_national     — "044 555 12 34"
- phone_compact      — "+41445551234"
- adv_neg_phone      — long digit sequence that's not a phone

URL
- url_bare           — "example.ch"
- url_full           — "https://example.ch/path?q=1"
- adv_neg_url        — DOI citation, RFC reference

ACCOUNT
- iban_spaced        — "CH93 0076 2011 6238 5295 7"
- iban_unspaced      — "CH9300762011623852957"
- ahv                — "756.9217.0769.85"
- vat_che            — "CHE-123.456.789"
- credit_card        — Luhn-valid CC
- adv_neg_account    — number that LOOKS like IBAN but isn't

SECRET
- secret_openai      — "sk-proj-..."
- secret_github      — "ghp_..."
- secret_aws         — "AKIA..."
- secret_env_file    — secret in .env / config context

MULTI
- multi_email        — full email with 6+ PII spans
- multi_form         — KYC form with 8+ PII spans

The v3 probe report groups by ``pattern_tag`` so we can see which
classes of failure persist after v3 training.

Run
---
    uv run python -m gheim_training.eval.probe_v3 --model checkpoints/gheim-ch-v3
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_LABEL_ALIAS = {
    "person":   "private_person",
    "address":  "private_address",
    "date":     "private_date",
    "email":    "private_email",
    "phone":    "private_phone",
    "url":      "private_url",
    "secret":   "secret",
    "account":  "account_number",
}


@dataclass
class Probe:
    case_id: str
    language: str
    text: str
    expected: list[tuple[int, int, str]] = field(default_factory=list)
    pattern_tag: str = "uncategorized"


def P(case_id: str, lang: str, text: str,
      pairs: list[tuple[str, str]], tag: str) -> Probe:
    """Build a Probe from terse (value, short_label) pairs.

    ``pairs`` uses short labels ('person', 'date', ...) that we expand to
    canonical category names. Multiple occurrences of the same value
    advance a cursor so consecutive entries find consecutive positions.
    """
    expected: list[tuple[int, int, str]] = []
    cursor = 0
    for value, short_label in pairs:
        i = text.find(value, cursor)
        if i < 0:
            raise ValueError(
                f"{case_id}: value {value!r} not found in {text!r} "
                f"after pos {cursor}"
            )
        full_label = _LABEL_ALIAS.get(short_label, short_label)
        expected.append((i, i + len(value), full_label))
        cursor = i + len(value)
    return Probe(case_id=case_id, language=lang, text=text,
                 expected=expected, pattern_tag=tag)


# ---------------------------------------------------------------------------
# DE_CH cases
# ---------------------------------------------------------------------------

_DE: list[Probe] = [
    # ---- greet_first (the v1 gap)
    P("de_greet_first_1", "de_ch", "Hallo Marius, danke für deine Nachricht.",
      [("Marius", "person")], "greet_first"),
    P("de_greet_first_2", "de_ch", "Liebe Anna, ich melde mich nächste Woche.",
      [("Anna", "person")], "greet_first"),
    P("de_greet_first_3", "de_ch", "Hi Lukas, kurze Rückfrage:",
      [("Lukas", "person")], "greet_first"),
    P("de_greet_first_4", "de_ch", "Guten Tag Sophie, anbei der Vertrag.",
      [("Sophie", "person")], "greet_first"),
    P("de_greet_first_5", "de_ch", "Servus Tobias!",
      [("Tobias", "person")], "greet_first"),
    P("de_greet_first_6", "de_ch", "Lieber Benedikt, vielen Dank.",
      [("Benedikt", "person")], "greet_first"),

    # ---- greet_first_var (punctuation / line variants)
    P("de_greet_first_var_1", "de_ch", "Hallo, Marius — kannst du das prüfen?",
      [("Marius", "person")], "greet_first_var"),
    P("de_greet_first_var_2", "de_ch", "Hallo!\nAnna,\nkurz zur Lieferung:",
      [("Anna", "person")], "greet_first_var"),
    P("de_greet_first_var_3", "de_ch", "Hallöchen Lia, danke fürs Mitdenken!",
      [("Lia", "person")], "greet_first_var"),
    P("de_greet_first_var_4", "de_ch", "Grüezi Herr Bürki, hier die Unterlagen.",
      [("Herr Bürki", "person")], "greet_first_var"),

    # ---- narr_first
    P("de_narr_first_1", "de_ch",
      "Anna meinte gestern, dass die Lieferung morgen kommt.",
      [("Anna", "person")], "narr_first"),
    P("de_narr_first_2", "de_ch",
      "Laut Sophia ist der Termin auf nächsten Dienstag verschoben.",
      [("Sophia", "person")], "narr_first"),
    P("de_narr_first_3", "de_ch",
      "Ich habe Tobias gebeten, das Dokument durchzusehen.",
      [("Tobias", "person")], "narr_first"),

    # ---- narr_last
    P("de_narr_last_1", "de_ch",
      "Müller hat heute Morgen eine ausführliche Stellungnahme geschrieben.",
      [("Müller", "person")], "narr_last"),
    P("de_narr_last_2", "de_ch",
      "Die Beschwerde wurde von Bürgi eingereicht.",
      [("Bürgi", "person")], "narr_last"),
    P("de_narr_last_3", "de_ch",
      "Laut Aussage von Schmid liegt der Fehler beim Lieferanten.",
      [("Schmid", "person")], "narr_last"),

    # ---- sig_first
    P("de_sig_first_1", "de_ch",
      "Vielen Dank für die Bearbeitung.\n\nLiebe Grüsse,\nMarius",
      [("Marius", "person")], "sig_first"),
    P("de_sig_first_2", "de_ch",
      "Bei Fragen melde dich.\n\nGruss\nAnna",
      [("Anna", "person")], "sig_first"),
    P("de_sig_first_3", "de_ch",
      "Ich freue mich auf deine Rückmeldung.\n\nBeste Grüsse\nLukas",
      [("Lukas", "person")], "sig_first"),

    # ---- sig_full
    P("de_sig_full_1", "de_ch",
      "Freundliche Grüsse,\nLukas Brunner",
      [("Lukas Brunner", "person")], "sig_full"),
    P("de_sig_full_2", "de_ch",
      "Mit freundlichen Grüssen\nAnna-Lena Müller-Schmid",
      [("Anna-Lena Müller-Schmid", "person")], "sig_full"),

    # ---- title_full
    P("de_title_full_1", "de_ch",
      "Sehr geehrter Herr Müller, anbei das gewünschte Dokument.",
      [("Herr Müller", "person")], "title_full"),
    P("de_title_full_2", "de_ch",
      "Sehr geehrte Frau Dubois, vielen Dank für Ihre Anfrage.",
      [("Frau Dubois", "person")], "title_full"),
    P("de_title_full_3", "de_ch",
      "Sehr geehrter Herr Bundesrat Berset, wir bedanken uns.",
      [("Herr Bundesrat Berset", "person")], "title_full"),

    # ---- title_abbrev
    P("de_title_abbrev_1", "de_ch",
      "Hr. Müller hat mich heute Morgen kontaktiert.",
      [("Hr. Müller", "person")], "title_abbrev"),
    P("de_title_abbrev_2", "de_ch",
      "Dr. Schmidt hat den Befund bestätigt.",
      [("Dr. Schmidt", "person")], "title_abbrev"),
    P("de_title_abbrev_3", "de_ch",
      "Prof. Bürgi leitet die Sitzung am Donnerstag.",
      [("Prof. Bürgi", "person")], "title_abbrev"),
    P("de_title_abbrev_4", "de_ch",
      "Fr. Schneider ist diese Woche im Homeoffice.",
      [("Fr. Schneider", "person")], "title_abbrev"),

    # ---- title_firstlast
    P("de_title_firstlast_1", "de_ch",
      "Dr. Anna Müller hat den Bericht verfasst.",
      [("Dr. Anna Müller", "person")], "title_firstlast"),

    # ---- multitle (multi-title)
    P("de_multitle_1", "de_ch",
      "Dr. med. Anna Müller, MBA, übernimmt die Klinikleitung.",
      [("Dr. med. Anna Müller", "person")], "multitle"),

    # ---- last_first_fmt
    P("de_last_first_fmt_1", "de_ch",
      "Unterschrift: MÜLLER, Anna",
      [("MÜLLER, Anna", "person")], "last_first_fmt"),
    P("de_last_first_fmt_2", "de_ch",
      "Kontaktperson: Schmid, Lukas",
      [("Schmid, Lukas", "person")], "last_first_fmt"),

    # ---- initials
    P("de_initials_1", "de_ch",
      "A. Müller hat den Brief unterschrieben.",
      [("A. Müller", "person")], "initials"),
    P("de_initials_2", "de_ch",
      "Hr. M. Müller hat mich heute kontaktiert.",
      [("Hr. M. Müller", "person")], "initials"),

    # ---- compound_name
    P("de_compound_1", "de_ch",
      "Hans-Peter Bürgi-Stocker hat das Protokoll unterzeichnet.",
      [("Hans-Peter Bürgi-Stocker", "person")], "compound_name"),
    P("de_compound_2", "de_ch",
      "Marie-Theres von Wartburg ist die neue Direktorin.",
      [("Marie-Theres von Wartburg", "person")], "compound_name"),

    # ---- common_word_sur (the Bach class)
    P("de_commonword_1", "de_ch",
      "Herr Bach hat das Konzert dirigiert.",
      [("Herr Bach", "person")], "common_word_sur"),
    P("de_commonword_2", "de_ch",
      "Frau Berg ist die neue Schulleiterin.",
      [("Frau Berg", "person")], "common_word_sur"),
    P("de_commonword_3", "de_ch",
      "Dr. Stein hat den Bericht verfasst.",
      [("Dr. Stein", "person")], "common_word_sur"),
    P("de_commonword_4", "de_ch",
      "Hr. König unterstützt das Projekt finanziell.",
      [("Hr. König", "person")], "common_word_sur"),
    P("de_commonword_5", "de_ch",
      "Frau Fischer hat den Vertrag heute unterschrieben.",
      [("Frau Fischer", "person")], "common_word_sur"),
    P("de_commonword_6", "de_ch",
      "Wie Wolf bereits sagte, ist die Sitzung verschoben.",
      [("Wolf", "person")], "common_word_sur"),

    # ---- adv_neg_person (lookalikes that are NOT persons)
    P("de_adv_neg_person_1", "de_ch",
      "Am Bach steht ein alter Baum.",
      [], "adv_neg_person"),
    P("de_adv_neg_person_2", "de_ch",
      "Der Berg ist 2400m hoch.",
      [], "adv_neg_person"),
    P("de_adv_neg_person_3", "de_ch",
      "Das Wasser im See ist heute klar.",
      [], "adv_neg_person"),
    P("de_adv_neg_person_4", "de_ch",
      "Der Sommer war dieses Jahr besonders trocken.",
      [], "adv_neg_person"),

    # ---- date_dot
    P("de_date_dot_1", "de_ch",
      "Der Termin ist am 30.06.2026 um 14:00.",
      [("30.06.2026", "date")], "date_dot"),
    P("de_date_dot_2", "de_ch",
      "Geboren am 12.03.1985 in Bern.",
      [("12.03.1985", "date")], "date_dot"),

    # ---- date_slash
    P("de_date_slash_1", "de_ch",
      "Datum: 30/06/2026",
      [("30/06/2026", "date")], "date_slash"),

    # ---- date_spelled
    P("de_date_spelled_1", "de_ch",
      "Am 30. Juni 2026 läuft mein Vertrag aus.",
      [("30. Juni 2026", "date")], "date_spelled"),
    P("de_date_spelled_2", "de_ch",
      "Der 1. August ist ein Feiertag.",
      [("1. August", "date")], "date_spelled"),

    # ---- date_iso
    P("de_date_iso_1", "de_ch",
      "Letzte Änderung: 2026-06-30T14:00:00Z",
      [("2026-06-30", "date")], "date_iso"),

    # ---- date_modifier
    P("de_date_modifier_1", "de_ch",
      "Anmeldung bis Freitag, den 5. Mai erforderlich.",
      [("Freitag, den 5. Mai", "date")], "date_modifier"),
    P("de_date_modifier_2", "de_ch",
      "Gültig ab 1.1.2027 (neue Regelung).",
      [("1.1.2027", "date")], "date_modifier"),

    # ---- adv_neg_date
    P("de_adv_neg_date_1", "de_ch",
      "Zimmer 30.06 ist heute frei.",
      [], "adv_neg_date"),
    P("de_adv_neg_date_2", "de_ch",
      "Version 1.2.3 wird nächste Woche veröffentlicht.",
      [], "adv_neg_date"),

    # ---- addr_full
    P("de_addr_full_1", "de_ch",
      "Bitte senden Sie die Unterlagen an Bahnhofstrasse 10, 8001 Zürich.",
      [("Bahnhofstrasse 10, 8001 Zürich", "address")], "addr_full"),
    P("de_addr_full_2", "de_ch",
      "Adresse: Limmatquai 22, 8001 Zürich, Schweiz",
      [("Limmatquai 22, 8001 Zürich", "address")], "addr_full"),

    # ---- addr_street_only
    P("de_addr_street_1", "de_ch",
      "Treffpunkt: Bahnhofstrasse 10",
      [("Bahnhofstrasse 10", "address")], "addr_street_only"),

    # ---- addr_with_canton
    P("de_addr_canton_1", "de_ch",
      "Hauptsitz: Industriestrasse 5, 3008 Bern BE",
      [("Industriestrasse 5, 3008 Bern BE", "address")], "addr_with_canton"),

    # ---- addr_city_only (in private context)
    P("de_addr_city_1", "de_ch",
      "Lukas wohnt seit drei Jahren in Winterthur.",
      [("Lukas", "person"), ("Winterthur", "address")], "addr_city_only"),

    # ---- adv_neg_addr (city as nominal reference, not address)
    P("de_adv_neg_addr_1", "de_ch",
      "Zürich ist eine schöne Stadt.",
      [], "adv_neg_addr"),

    # ---- email_std
    P("de_email_std_1", "de_ch",
      "Bei Fragen schreibe an lukas.brunner@example.ch.",
      [("lukas.brunner@example.ch", "email")], "email_std"),

    # ---- email_special
    P("de_email_special_1", "de_ch",
      "Antwort an anna.müller+billing@firma-ag.ch erwartet.",
      [("anna.müller+billing@firma-ag.ch", "email")], "email_special"),

    # ---- adv_neg_email (template/docs)
    P("de_adv_neg_email_1", "de_ch",
      "Beispiel: ersetze max.mustermann@example.com mit deiner Adresse.",
      [], "adv_neg_email"),

    # ---- phone_intl
    P("de_phone_intl_1", "de_ch",
      "Telefon: +41 44 555 12 34",
      [("+41 44 555 12 34", "phone")], "phone_intl"),

    # ---- phone_national
    P("de_phone_national_1", "de_ch",
      "Erreichbar unter 044 555 12 34 ab 9 Uhr.",
      [("044 555 12 34", "phone")], "phone_national"),

    # ---- phone_compact
    P("de_phone_compact_1", "de_ch",
      "Notfall-Nummer: +41445551234",
      [("+41445551234", "phone")], "phone_compact"),

    # ---- adv_neg_phone (number, not phone)
    P("de_adv_neg_phone_1", "de_ch",
      "Bestellnummer: 044 555 1234 (intern).",
      [], "adv_neg_phone"),

    # ---- url_bare
    P("de_url_bare_1", "de_ch",
      "Weitere Informationen unter example.ch.",
      [("example.ch", "url")], "url_bare"),

    # ---- url_full
    P("de_url_full_1", "de_ch",
      "Login: https://app.example.ch/dashboard?ref=email",
      [("https://app.example.ch/dashboard?ref=email", "url")], "url_full"),

    # ---- iban_spaced
    P("de_iban_spaced_1", "de_ch",
      "Bitte überweisen auf IBAN CH93 0076 2011 6238 5295 7.",
      [("CH93 0076 2011 6238 5295 7", "account")], "iban_spaced"),

    # ---- iban_unspaced
    P("de_iban_unspaced_1", "de_ch",
      "Konto-IBAN: CH9300762011623852957",
      [("CH9300762011623852957", "account")], "iban_unspaced"),

    # ---- ahv
    P("de_ahv_1", "de_ch",
      "AHV-Nummer: 756.9217.0769.85",
      [("756.9217.0769.85", "account")], "ahv"),

    # ---- vat_che
    P("de_vat_1", "de_ch",
      "UID: CHE-123.456.789 MWST",
      [("CHE-123.456.789", "account")], "vat_che"),

    # ---- credit_card
    P("de_cc_1", "de_ch",
      "Kreditkarte: 4111 1111 1111 1111 (Visa)",
      [("4111 1111 1111 1111", "account")], "credit_card"),

    # ---- secret_openai
    P("de_secret_openai_1", "de_ch",
      "OPENAI_API_KEY=sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789AbCdEfGhIjKl",
      [("sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789AbCdEfGhIjKl", "secret")],
      "secret_openai"),

    # ---- secret_github
    P("de_secret_github_1", "de_ch",
      "git push mit Token ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789",
      [("ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789", "secret")], "secret_github"),

    # ---- secret_aws
    P("de_secret_aws_1", "de_ch",
      "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE",
      [("AKIAIOSFODNN7EXAMPLE", "secret")], "secret_aws"),

    # ---- secret_env_file
    P("de_secret_env_1", "de_ch",
      "# .env\nDB_PASSWORD=Sup3rSecret!2026\nLOG_LEVEL=debug",
      [("Sup3rSecret!2026", "secret")], "secret_env_file"),

    # ---- multi_email (the original site demo)
    P("de_multi_email_1", "de_ch",
      ("Hallo Team,\n\nmein Name ist Lukas Brunner. Bitte erreichen Sie mich "
       "unter +41 44 555 12 34 oder lukas.brunner@example.ch. Meine Adresse "
       "ist Bahnhofstrasse 10, 8001 Zürich.\n\nAm 30. Juni 2026 läuft mein "
       "Vertrag aus. Bitte überweisen Sie CHF 230.00 auf IBAN "
       "CH9300762011623852957.\n\nFreundliche Grüsse,\nLukas Brunner"),
      [("Lukas Brunner", "person"),
       ("+41 44 555 12 34", "phone"),
       ("lukas.brunner@example.ch", "email"),
       ("Bahnhofstrasse 10, 8001 Zürich", "address"),
       ("30. Juni 2026", "date"),
       ("CH9300762011623852957", "account"),
       ("Lukas Brunner", "person")],
      "multi_email"),
]


# ---------------------------------------------------------------------------
# FR_CH cases
# ---------------------------------------------------------------------------

_FR: list[Probe] = [
    # ---- greet_first
    P("fr_greet_first_1", "fr_ch", "Bonjour Jean, merci de votre message.",
      [("Jean", "person")], "greet_first"),
    P("fr_greet_first_2", "fr_ch", "Salut Sophie, ça va ?",
      [("Sophie", "person")], "greet_first"),
    P("fr_greet_first_3", "fr_ch", "Cher Antoine, je vous écris au sujet de…",
      [("Antoine", "person")], "greet_first"),
    P("fr_greet_first_4", "fr_ch", "Chère Mélanie, voici les pièces jointes.",
      [("Mélanie", "person")], "greet_first"),
    P("fr_greet_first_5", "fr_ch", "Coucou Léa, dis-moi quand tu es libre.",
      [("Léa", "person")], "greet_first"),

    # ---- greet_first_var
    P("fr_greet_first_var_1", "fr_ch",
      "Bonjour,\nJean,\nmerci pour votre retour rapide.",
      [("Jean", "person")], "greet_first_var"),
    P("fr_greet_first_var_2", "fr_ch", "Salut! Pierre, peux-tu m'aider ?",
      [("Pierre", "person")], "greet_first_var"),

    # ---- narr_first
    P("fr_narr_first_1", "fr_ch",
      "Jean a confirmé hier que le contrat sera signé demain.",
      [("Jean", "person")], "narr_first"),
    P("fr_narr_first_2", "fr_ch",
      "D'après Mélanie, la réunion est reportée à mardi.",
      [("Mélanie", "person")], "narr_first"),

    # ---- narr_last
    P("fr_narr_last_1", "fr_ch",
      "Dupont a signé le protocole ce matin.",
      [("Dupont", "person")], "narr_last"),
    P("fr_narr_last_2", "fr_ch",
      "La plainte a été déposée par Fonjallaz.",
      [("Fonjallaz", "person")], "narr_last"),

    # ---- sig_first
    P("fr_sig_first_1", "fr_ch",
      "Merci d'avance pour votre retour.\n\nCordialement,\nMélanie",
      [("Mélanie", "person")], "sig_first"),
    P("fr_sig_first_2", "fr_ch",
      "À votre disposition.\n\nBien à vous,\nSylvie",
      [("Sylvie", "person")], "sig_first"),

    # ---- sig_full
    P("fr_sig_full_1", "fr_ch",
      "Cordialement,\nJean-Marc Dubois-Perret",
      [("Jean-Marc Dubois-Perret", "person")], "sig_full"),

    # ---- title_full
    P("fr_title_full_1", "fr_ch",
      "Cher Monsieur Dupont, je vous remercie de votre courrier.",
      [("Monsieur Dupont", "person")], "title_full"),
    P("fr_title_full_2", "fr_ch",
      "Chère Madame Müller, veuillez trouver ci-joint le devis.",
      [("Madame Müller", "person")], "title_full"),

    # ---- title_abbrev
    P("fr_title_abbrev_1", "fr_ch",
      "M. Droz a confirmé sa présence à la réunion.",
      [("M. Droz", "person")], "title_abbrev"),
    P("fr_title_abbrev_2", "fr_ch",
      "Mme Paccot sera absente jusqu'à vendredi.",
      [("Mme Paccot", "person")], "title_abbrev"),
    P("fr_title_abbrev_3", "fr_ch",
      "Dr Bernard a signé le rapport médical.",
      [("Dr Bernard", "person")], "title_abbrev"),

    # ---- title_firstlast
    P("fr_title_firstlast_1", "fr_ch",
      "Dr Anne Dubois a publié l'étude.",
      [("Dr Anne Dubois", "person")], "title_firstlast"),

    # ---- last_first_fmt
    P("fr_last_first_fmt_1", "fr_ch",
      "Signataire: DUPONT, Jean",
      [("DUPONT, Jean", "person")], "last_first_fmt"),

    # ---- initials
    P("fr_initials_1", "fr_ch",
      "J. Dupont a confirmé par e-mail.",
      [("J. Dupont", "person")], "initials"),

    # ---- compound_name
    P("fr_compound_1", "fr_ch",
      "Marie-Claire Dubois-Perret a accepté la proposition.",
      [("Marie-Claire Dubois-Perret", "person")], "compound_name"),

    # ---- date_dot
    P("fr_date_dot_1", "fr_ch",
      "Échéance: 30.06.2026",
      [("30.06.2026", "date")], "date_dot"),

    # ---- date_slash
    P("fr_date_slash_1", "fr_ch",
      "Date de naissance: 12/03/1985",
      [("12/03/1985", "date")], "date_slash"),

    # ---- date_spelled
    P("fr_date_spelled_1", "fr_ch",
      "Le contrat expire le 30 juin 2026.",
      [("30 juin 2026", "date")], "date_spelled"),
    P("fr_date_spelled_2", "fr_ch",
      "Veuillez confirmer avant le 5 mai.",
      [("5 mai", "date")], "date_spelled"),

    # ---- date_iso
    P("fr_date_iso_1", "fr_ch",
      "Dernière modification : 2026-06-30",
      [("2026-06-30", "date")], "date_iso"),

    # ---- adv_neg_date
    P("fr_adv_neg_date_1", "fr_ch",
      "Version 2.1.4 publiée la semaine prochaine.",
      [], "adv_neg_date"),

    # ---- addr_full
    P("fr_addr_full_1", "fr_ch",
      "Envoyer à Rue du Mont-Blanc 15, 1201 Genève.",
      [("Rue du Mont-Blanc 15, 1201 Genève", "address")], "addr_full"),

    # ---- addr_street_only
    P("fr_addr_street_1", "fr_ch",
      "Rendez-vous: Avenue de la Gare 22",
      [("Avenue de la Gare 22", "address")], "addr_street_only"),

    # ---- addr_with_canton
    P("fr_addr_canton_1", "fr_ch",
      "Siège: Place de la Riponne 5, 1005 Lausanne VD",
      [("Place de la Riponne 5, 1005 Lausanne VD", "address")], "addr_with_canton"),

    # ---- addr_city_only
    P("fr_addr_city_1", "fr_ch",
      "Jean habite à Lausanne depuis trois ans.",
      [("Jean", "person"), ("Lausanne", "address")], "addr_city_only"),

    # ---- adv_neg_addr
    P("fr_adv_neg_addr_1", "fr_ch",
      "Lausanne est une belle ville universitaire.",
      [], "adv_neg_addr"),

    # ---- email_std
    P("fr_email_std_1", "fr_ch",
      "Pour toute question, écrivez à jean.dupont@example.ch.",
      [("jean.dupont@example.ch", "email")], "email_std"),

    # ---- email_special
    P("fr_email_special_1", "fr_ch",
      "Réponse à anne.müller+notif@société-sa.ch attendue.",
      [("anne.müller+notif@société-sa.ch", "email")], "email_special"),

    # ---- phone_intl
    P("fr_phone_intl_1", "fr_ch",
      "Téléphone: +41 22 555 12 34",
      [("+41 22 555 12 34", "phone")], "phone_intl"),

    # ---- phone_national
    P("fr_phone_national_1", "fr_ch",
      "Joignable au 022 555 12 34 dès 9h.",
      [("022 555 12 34", "phone")], "phone_national"),

    # ---- url_bare
    P("fr_url_bare_1", "fr_ch",
      "Plus d'informations sur example.ch.",
      [("example.ch", "url")], "url_bare"),

    # ---- url_full
    P("fr_url_full_1", "fr_ch",
      "Connexion : https://app.example.ch/portal?lang=fr",
      [("https://app.example.ch/portal?lang=fr", "url")], "url_full"),

    # ---- iban_spaced
    P("fr_iban_spaced_1", "fr_ch",
      "IBAN: CH93 0076 2011 6238 5295 7",
      [("CH93 0076 2011 6238 5295 7", "account")], "iban_spaced"),

    # ---- ahv
    P("fr_ahv_1", "fr_ch",
      "Numéro AVS: 756.9217.0769.85",
      [("756.9217.0769.85", "account")], "ahv"),

    # ---- vat_che
    P("fr_vat_1", "fr_ch",
      "IDE: CHE-123.456.789 TVA",
      [("CHE-123.456.789", "account")], "vat_che"),

    # ---- secret_openai
    P("fr_secret_openai_1", "fr_ch",
      "OPENAI_API_KEY=sk-proj-XyZ1234567890abcDeFgHiJkLmNoPqRsTuVwXyZ0123456",
      [("sk-proj-XyZ1234567890abcDeFgHiJkLmNoPqRsTuVwXyZ0123456", "secret")],
      "secret_openai"),

    # ---- multi_email
    P("fr_multi_email_1", "fr_ch",
      ("Bonjour Marie,\n\nVoici les coordonnées que vous m'avez demandées : "
       "Jean Dubois, téléphone +41 22 555 12 34, e-mail "
       "jean.dubois@example.ch, adresse Rue du Mont-Blanc 15, 1201 Genève.\n\n"
       "Cordialement,\nMarie Dupont"),
      [("Marie", "person"),
       ("Jean Dubois", "person"),
       ("+41 22 555 12 34", "phone"),
       ("jean.dubois@example.ch", "email"),
       ("Rue du Mont-Blanc 15, 1201 Genève", "address"),
       ("Marie Dupont", "person")],
      "multi_email"),
]


# ---------------------------------------------------------------------------
# IT_CH cases
# ---------------------------------------------------------------------------

_IT: list[Probe] = [
    # ---- greet_first
    P("it_greet_first_1", "it_ch", "Ciao Marco, grazie del tuo messaggio.",
      [("Marco", "person")], "greet_first"),
    P("it_greet_first_2", "it_ch", "Caro Giuseppe, ti allego il documento.",
      [("Giuseppe", "person")], "greet_first"),
    P("it_greet_first_3", "it_ch", "Cara Francesca, grazie per la risposta.",
      [("Francesca", "person")], "greet_first"),
    P("it_greet_first_4", "it_ch", "Buongiorno Laura, ho ricevuto la tua nota.",
      [("Laura", "person")], "greet_first"),

    # ---- narr_first
    P("it_narr_first_1", "it_ch",
      "Marco ha confermato ieri che il contratto sarà firmato domani.",
      [("Marco", "person")], "narr_first"),
    P("it_narr_first_2", "it_ch",
      "Secondo Laura, la riunione è rinviata a martedì.",
      [("Laura", "person")], "narr_first"),

    # ---- narr_last
    P("it_narr_last_1", "it_ch",
      "Rossi ha presentato il rapporto alla riunione di ieri.",
      [("Rossi", "person")], "narr_last"),
    P("it_narr_last_2", "it_ch",
      "Il reclamo è stato presentato da Bianchi.",
      [("Bianchi", "person")], "narr_last"),

    # ---- sig_first
    P("it_sig_first_1", "it_ch",
      "Grazie in anticipo per il riscontro.\n\nCordiali saluti,\nFrancesca",
      [("Francesca", "person")], "sig_first"),
    P("it_sig_first_2", "it_ch",
      "Resto a disposizione.\n\nSaluti,\nGiuseppe",
      [("Giuseppe", "person")], "sig_first"),

    # ---- sig_full
    P("it_sig_full_1", "it_ch",
      "Cordiali saluti,\nGiuseppe-Maria Rossi-Bianchi",
      [("Giuseppe-Maria Rossi-Bianchi", "person")], "sig_full"),

    # ---- title_full
    P("it_title_full_1", "it_ch",
      "Egregio Signor Rossi, La ringrazio per la lettera.",
      [("Signor Rossi", "person")], "title_full"),
    P("it_title_full_2", "it_ch",
      "Gentile Signora Müller, in allegato il preventivo.",
      [("Signora Müller", "person")], "title_full"),

    # ---- title_abbrev
    P("it_title_abbrev_1", "it_ch",
      "Sig. Rossi ha confermato la sua presenza.",
      [("Sig. Rossi", "person")], "title_abbrev"),
    P("it_title_abbrev_2", "it_ch",
      "Sig.ra Bianchi sarà assente fino a venerdì.",
      [("Sig.ra Bianchi", "person")], "title_abbrev"),
    P("it_title_abbrev_3", "it_ch",
      "Dott. Esposito ha firmato il rapporto.",
      [("Dott. Esposito", "person")], "title_abbrev"),

    # ---- title_firstlast
    P("it_title_firstlast_1", "it_ch",
      "Dott.ssa Anna Rossi ha guidato lo studio.",
      [("Dott.ssa Anna Rossi", "person")], "title_firstlast"),

    # ---- last_first_fmt
    P("it_last_first_fmt_1", "it_ch",
      "Firma: ROSSI, Marco",
      [("ROSSI, Marco", "person")], "last_first_fmt"),

    # ---- initials
    P("it_initials_1", "it_ch",
      "M. Rossi ha confermato per email.",
      [("M. Rossi", "person")], "initials"),

    # ---- compound_name
    P("it_compound_1", "it_ch",
      "Maria-Chiara Rossi-Bianchi ha vinto la causa.",
      [("Maria-Chiara Rossi-Bianchi", "person")], "compound_name"),

    # ---- date_dot
    P("it_date_dot_1", "it_ch",
      "Scadenza: 30.06.2026",
      [("30.06.2026", "date")], "date_dot"),

    # ---- date_slash
    P("it_date_slash_1", "it_ch",
      "Data di nascita: 12/03/1985",
      [("12/03/1985", "date")], "date_slash"),

    # ---- date_spelled
    P("it_date_spelled_1", "it_ch",
      "Il contratto scade il 30 giugno 2026.",
      [("30 giugno 2026", "date")], "date_spelled"),

    # ---- date_iso
    P("it_date_iso_1", "it_ch",
      "Ultima modifica: 2026-06-30",
      [("2026-06-30", "date")], "date_iso"),

    # ---- addr_full
    P("it_addr_full_1", "it_ch",
      "Spedire a Via Nassa 12, 6900 Lugano.",
      [("Via Nassa 12, 6900 Lugano", "address")], "addr_full"),

    # ---- addr_with_canton
    P("it_addr_canton_1", "it_ch",
      "Sede: Piazza Riforma 1, 6900 Lugano TI",
      [("Piazza Riforma 1, 6900 Lugano TI", "address")], "addr_with_canton"),

    # ---- addr_city_only
    P("it_addr_city_1", "it_ch",
      "Marco abita a Bellinzona da tre anni.",
      [("Marco", "person"), ("Bellinzona", "address")], "addr_city_only"),

    # ---- email_std
    P("it_email_std_1", "it_ch",
      "Per domande, scrivete a marco.rossi@example.ch.",
      [("marco.rossi@example.ch", "email")], "email_std"),

    # ---- phone_intl
    P("it_phone_intl_1", "it_ch",
      "Telefono: +41 91 555 12 34",
      [("+41 91 555 12 34", "phone")], "phone_intl"),

    # ---- phone_national
    P("it_phone_national_1", "it_ch",
      "Raggiungibile al 091 555 12 34.",
      [("091 555 12 34", "phone")], "phone_national"),

    # ---- url_bare
    P("it_url_bare_1", "it_ch",
      "Maggiori informazioni su example.ch.",
      [("example.ch", "url")], "url_bare"),

    # ---- iban_spaced
    P("it_iban_spaced_1", "it_ch",
      "IBAN: CH93 0076 2011 6238 5295 7",
      [("CH93 0076 2011 6238 5295 7", "account")], "iban_spaced"),

    # ---- ahv
    P("it_ahv_1", "it_ch",
      "Numero AVS: 756.9217.0769.85",
      [("756.9217.0769.85", "account")], "ahv"),

    # ---- secret_openai
    P("it_secret_openai_1", "it_ch",
      "OPENAI_API_KEY=sk-proj-QwErTyUiOpAsDfGhJkLzXcVbNm0123456789QwErTyUiOp",
      [("sk-proj-QwErTyUiOpAsDfGhJkLzXcVbNm0123456789QwErTyUiOp", "secret")],
      "secret_openai"),

    # ---- multi_email
    P("it_multi_email_1", "it_ch",
      ("Ciao Marco,\n\nti scrivo per confermare i miei dati: Anna Bianchi, "
       "telefono +41 91 555 12 34, e-mail anna.bianchi@example.ch, "
       "indirizzo Via Nassa 12, 6900 Lugano.\n\nCordiali saluti,\nAnna"),
      [("Marco", "person"),
       ("Anna Bianchi", "person"),
       ("+41 91 555 12 34", "phone"),
       ("anna.bianchi@example.ch", "email"),
       ("Via Nassa 12, 6900 Lugano", "address"),
       ("Anna", "person")],
      "multi_email"),
]


# ---------------------------------------------------------------------------
# RM cases
# ---------------------------------------------------------------------------

_RM: list[Probe] = [
    # ---- greet_first
    P("rm_greet_first_1", "rm", "Hai Selina, grazia per tia communicaziun.",
      [("Selina", "person")], "greet_first"),
    P("rm_greet_first_2", "rm", "Char Curdin, ti agiat per la rispusta.",
      [("Curdin", "person")], "greet_first"),
    P("rm_greet_first_3", "rm", "Bun di Andri, en agiunta chattas igl document.",
      [("Andri", "person")], "greet_first"),

    # ---- narr_first
    P("rm_narr_first_1", "rm",
      "Ladina ha confermau che la furniziun arriva damaun.",
      [("Ladina", "person")], "narr_first"),
    P("rm_narr_first_2", "rm",
      "Tenor Reto, la sesida vegn spustada a mardi.",
      [("Reto", "person")], "narr_first"),

    # ---- narr_last
    P("rm_narr_last_1", "rm",
      "Caduff ha suttascrit il protocoll quest mintga.",
      [("Caduff", "person")], "narr_last"),
    P("rm_narr_last_2", "rm",
      "La reclamaziun ei vegnida inoltrada da Camenisch.",
      [("Camenisch", "person")], "narr_last"),

    # ---- sig_first
    P("rm_sig_first_1", "rm",
      "Grazia per Vossa attenziun.\n\nCordials salids,\nCurdin",
      [("Curdin", "person")], "sig_first"),

    # ---- title_full
    P("rm_title_full_1", "rm",
      "Stimà Sgnr. Caduff, grazia per Vossa cuminonza.",
      [("Sgnr. Caduff", "person")], "title_full"),
    P("rm_title_full_2", "rm",
      "Stimada Sgnra. Tschuor, en agiunta chattas il document.",
      [("Sgnra. Tschuor", "person")], "title_full"),

    # ---- title_abbrev
    P("rm_title_abbrev_1", "rm",
      "Sgnr. Caduff ha confermau sia preschientscha.",
      [("Sgnr. Caduff", "person")], "title_abbrev"),
    P("rm_title_abbrev_2", "rm",
      "Sgnra. Capeder ei absenta entro venderdi.",
      [("Sgnra. Capeder", "person")], "title_abbrev"),

    # ---- date_dot
    P("rm_date_dot_1", "rm",
      "Termin: 30.06.2026 ad las 14:00.",
      [("30.06.2026", "date")], "date_dot"),

    # ---- date_spelled
    P("rm_date_spelled_1", "rm",
      "Igl contract finescha als 30 da zercladur 2026.",
      [("30 da zercladur 2026", "date")], "date_spelled"),

    # ---- addr_full
    P("rm_addr_full_1", "rm",
      "Mandar a Via Centrala 7, 7500 St. Murezzan.",
      [("Via Centrala 7, 7500 St. Murezzan", "address")], "addr_full"),

    # ---- email_std
    P("rm_email_std_1", "rm",
      "Per dumondas scriver ad andri.caduff@example.ch.",
      [("andri.caduff@example.ch", "email")], "email_std"),

    # ---- phone_intl
    P("rm_phone_intl_1", "rm",
      "Telefon: +41 81 555 12 34",
      [("+41 81 555 12 34", "phone")], "phone_intl"),

    # ---- iban_spaced
    P("rm_iban_1", "rm",
      "IBAN: CH93 0076 2011 6238 5295 7",
      [("CH93 0076 2011 6238 5295 7", "account")], "iban_spaced"),

    # ---- secret_openai
    P("rm_secret_openai_1", "rm",
      "Igl token API: sk-proj-AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPp0123456789",
      [("sk-proj-AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPp0123456789", "secret")],
      "secret_openai"),

    # ---- secret_env_file
    P("rm_secret_env_1", "rm",
      "# .env\nDB_PASSWORD=PassPrivat2026!\n# tegnair segira",
      [("PassPrivat2026!", "secret")], "secret_env_file"),
]


# ---------------------------------------------------------------------------
# EN cases (international Swiss-company context)
# ---------------------------------------------------------------------------

_EN: list[Probe] = [
    # ---- greet_first
    P("en_greet_first_1", "en", "Hi John, thanks for your message.",
      [("John", "person")], "greet_first"),
    P("en_greet_first_2", "en", "Hello Sarah, please find attached the contract.",
      [("Sarah", "person")], "greet_first"),
    P("en_greet_first_3", "en", "Dear Michael, I confirm our meeting on Tuesday.",
      [("Michael", "person")], "greet_first"),

    # ---- narr_first
    P("en_narr_first_1", "en",
      "Sarah confirmed yesterday that the delivery will arrive tomorrow.",
      [("Sarah", "person")], "narr_first"),

    # ---- narr_last
    P("en_narr_last_1", "en",
      "Smith signed the protocol this morning.",
      [("Smith", "person")], "narr_last"),

    # ---- sig_first
    P("en_sig_first_1", "en",
      "Looking forward to your reply.\n\nBest regards,\nMichael",
      [("Michael", "person")], "sig_first"),

    # ---- sig_full
    P("en_sig_full_1", "en",
      "Kind regards,\nMichael Brown",
      [("Michael Brown", "person")], "sig_full"),

    # ---- title_abbrev
    P("en_title_abbrev_1", "en",
      "Dr. Smith confirmed the diagnosis.",
      [("Dr. Smith", "person")], "title_abbrev"),

    # ---- title_firstlast
    P("en_title_firstlast_1", "en",
      "Prof. Anna Mueller leads the research group.",
      [("Prof. Anna Mueller", "person")], "title_firstlast"),

    # ---- date_iso
    P("en_date_iso_1", "en",
      "Last updated: 2026-06-30",
      [("2026-06-30", "date")], "date_iso"),

    # ---- email_std
    P("en_email_std_1", "en",
      "Please contact john.smith@example.ch for details.",
      [("john.smith@example.ch", "email")], "email_std"),

    # ---- phone_intl
    P("en_phone_intl_1", "en",
      "Phone: +41 44 555 12 34",
      [("+41 44 555 12 34", "phone")], "phone_intl"),

    # ---- iban_spaced
    P("en_iban_1", "en",
      "IBAN: CH93 0076 2011 6238 5295 7",
      [("CH93 0076 2011 6238 5295 7", "account")], "iban_spaced"),

    # ---- secret_openai
    P("en_secret_openai_1", "en",
      "OPENAI_API_KEY=sk-proj-ZzYyXxWwVvUuTtSs1234567890ZzYyXxWwVvUuTtSs0123",
      [("sk-proj-ZzYyXxWwVvUuTtSs1234567890ZzYyXxWwVvUuTtSs0123", "secret")],
      "secret_openai"),

    # ---- multi_email
    P("en_multi_email_1", "en",
      ("Hi Anna,\n\nMy contact details: John Smith, "
       "phone +41 44 555 12 34, email john.smith@example.ch, "
       "address Bahnhofstrasse 10, 8001 Zurich.\n\n"
       "Best regards,\nJohn Smith"),
      [("Anna", "person"),
       ("John Smith", "person"),
       ("+41 44 555 12 34", "phone"),
       ("john.smith@example.ch", "email"),
       ("Bahnhofstrasse 10, 8001 Zurich", "address"),
       ("John Smith", "person")],
      "multi_email"),
]


# ---------------------------------------------------------------------------
# Supplemental cases — handwritten to round out underrepresented patterns
# ---------------------------------------------------------------------------

_SUPP: list[Probe] = [
    # More adversarial negatives (model needs strong common-word anchors)
    P("de_adv_neg_person_5", "de_ch",
      "Der Berger Bauer hat heute den Markt besucht.",
      [], "adv_neg_person"),
    P("de_adv_neg_addr_2", "de_ch",
      "Die Stadt Bern hat ein neues Stadion gebaut.",
      [], "adv_neg_addr"),
    P("fr_adv_neg_person_1", "fr_ch",
      "Le boulanger du quartier ouvre à 6h.",
      [], "adv_neg_person"),
    P("it_adv_neg_person_1", "it_ch",
      "Il panettiere apre alle 6 del mattino.",
      [], "adv_neg_person"),
    P("de_adv_neg_email_2", "de_ch",
      "API-Beispiel: POST /users mit body { email: 'demo@example.com' }",
      [], "adv_neg_email"),
    P("de_adv_neg_url_1", "de_ch",
      "Siehe RFC 5321 (https://tools.ietf.org/html/rfc5321) für Details.",
      [("https://tools.ietf.org/html/rfc5321", "url")], "adv_neg_url"),
    P("de_adv_neg_account_1", "de_ch",
      "Bestellnummer ist CH-2026-00001234 (Pseudo-IBAN-Format).",
      [], "adv_neg_account"),

    # More common-word surnames in FR/IT (parity with DE)
    P("fr_commonword_1", "fr_ch",
      "Monsieur Roche a dirigé l'enquête.",
      [("Monsieur Roche", "person")], "common_word_sur"),
    P("it_commonword_1", "it_ch",
      "Signor Monte ha guidato il progetto.",
      [("Signor Monte", "person")], "common_word_sur"),

    # More RM patterns
    P("rm_initials_1", "rm",
      "A. Caduff ha suttascrit il document.",
      [("A. Caduff", "person")], "initials"),
    P("rm_addr_city_1", "rm",
      "Curdin abita a Cuera dapi trais onns.",
      [("Curdin", "person"), ("Cuera", "address")], "addr_city_only"),
    P("rm_phone_national_1", "rm",
      "Cuntactai a 081 555 12 34 da las 9 era.",
      [("081 555 12 34", "phone")], "phone_national"),
    P("rm_date_modifier_1", "rm",
      "Validitad fin venderdi, ils 5 da matg.",
      [("venderdi, ils 5 da matg", "date")], "date_modifier"),

    # Secret variety
    P("de_secret_aws_2", "de_ch",
      "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
      [("wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "secret")], "secret_aws"),
    P("de_secret_github_2", "de_ch",
      "Auth: github_pat_11ABCDEF0_aBcDeFgHiJkLmNoPqRsTuVwXyZ0123",
      [("github_pat_11ABCDEF0_aBcDeFgHiJkLmNoPqRsTuVwXyZ0123", "secret")],
      "secret_github"),

    # Form / KYC multi-PII
    P("de_multi_form_1", "de_ch",
      ("Formular zur Anmeldung\n"
       "Name: Anna Müller\n"
       "Geburtsdatum: 12.03.1985\n"
       "AHV-Nr: 756.9217.0769.85\n"
       "Adresse: Bahnhofstrasse 10, 8001 Zürich\n"
       "Telefon: +41 44 555 12 34\n"
       "E-Mail: anna.mueller@example.ch"),
      [("Anna Müller", "person"),
       ("12.03.1985", "date"),
       ("756.9217.0769.85", "account"),
       ("Bahnhofstrasse 10, 8001 Zürich", "address"),
       ("+41 44 555 12 34", "phone"),
       ("anna.mueller@example.ch", "email")],
      "multi_form"),
]


ALL_PROBES: list[Probe] = _DE + _FR + _IT + _RM + _EN + _SUPP


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _predict_spans(model: object, tokenizer: object, text: str, max_len: int,
                   ) -> list[tuple[int, int, str]]:
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


def _diff_spans(gold: list[tuple[int, int, str]],
                pred: list[tuple[int, int, str]]) -> dict:
    def overlaps(a: tuple[int, int, str], b: tuple[int, int, str]) -> bool:
        return a[2] == b[2] and not (a[1] <= b[0] or b[1] <= a[0])
    gold_set = set(gold); pred_set = set(pred)
    exact_tp = len(gold_set & pred_set)
    exact_fp = len(pred_set - gold_set)
    exact_fn = len(gold_set - pred_set)
    soft_tp = sum(1 for g in gold if any(overlaps(g, p) for p in pred))
    soft_fp = sum(1 for p in pred if not any(overlaps(g, p) for g in gold))
    soft_fn = len(gold) - soft_tp
    return {"exact_tp": exact_tp, "exact_fp": exact_fp, "exact_fn": exact_fn,
            "soft_tp": soft_tp, "soft_fp": soft_fp, "soft_fn": soft_fn,
            "verdict": _verdict(gold, soft_tp, soft_fp, soft_fn)}


def _verdict(gold: list[tuple], stp: int, sfp: int, sfn: int) -> str:
    if not gold and sfp == 0:
        return "PASS_NEG"  # adversarial-negative case correctly produced nothing
    if not gold and sfp > 0:
        return "FAIL_NEG"  # adversarial-negative case fired a false positive
    if sfn == 0 and sfp == 0:
        return "PASS"
    if stp == 0:
        return "FAIL"
    return "PARTIAL"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path("checkpoints/gheim-ch-v2"))
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--report-out", type=Path,
                    default=Path("eval/v3_probe_report.json"))
    ap.add_argument("--quiet", action="store_true",
                    help="Skip per-case stdout; just emit summary tables.")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    print(f"Loading model: {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(str(args.model), use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(str(args.model)).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    print(f"Running {len(ALL_PROBES)} probe cases…", flush=True)
    print()

    per_case: list[dict] = []
    per_tag: dict[str, Counter] = defaultdict(Counter)
    per_lang_tag: dict[tuple[str, str], Counter] = defaultdict(Counter)
    overall = Counter()

    for probe in ALL_PROBES:
        pred = _predict_spans(model, tok, probe.text, args.max_seq_length)
        d = _diff_spans(probe.expected, pred)
        verdict = d["verdict"]
        overall[verdict] += 1
        per_tag[probe.pattern_tag][verdict] += 1
        per_lang_tag[(probe.language, probe.pattern_tag)][verdict] += 1
        per_case.append({
            "case_id": probe.case_id, "language": probe.language,
            "pattern_tag": probe.pattern_tag, "text": probe.text,
            "expected": [list(x) for x in probe.expected],
            "pred": [list(x) for x in pred],
            **d,
        })
        if not args.quiet:
            v_icon = {"PASS": "✓", "PASS_NEG": "✓-", "PARTIAL": "~",
                      "FAIL": "✗", "FAIL_NEG": "✗+"}[verdict]
            print(f"  [{v_icon}] {probe.case_id:<28} [{probe.pattern_tag}]")

    # ---- summary ----
    print()
    print("=" * 70)
    print(f"OVERALL: {sum(overall.values())} cases")
    for k in ("PASS", "PASS_NEG", "PARTIAL", "FAIL", "FAIL_NEG"):
        if overall[k]:
            print(f"  {k:<10} {overall[k]:>4}  ({100*overall[k]/sum(overall.values()):5.1f}%)")
    print()
    pass_total = overall["PASS"] + overall["PASS_NEG"]
    print(f"  Perfect (PASS + PASS_NEG): {pass_total} / {sum(overall.values())} "
          f"({100*pass_total/sum(overall.values()):.1f}%)")
    print()

    print("=" * 70)
    print("Per pattern_tag breakdown:")
    print(f"  {'pattern_tag':<24} {'PASS':>5} {'PART':>5} {'FAIL':>5} {'TOTAL':>6}  fail-cases")
    for tag in sorted(per_tag):
        c = per_tag[tag]
        passes = c["PASS"] + c["PASS_NEG"]
        partial = c["PARTIAL"]
        fails = c["FAIL"] + c["FAIL_NEG"]
        total = passes + partial + fails
        # List 1-3 failing case IDs for quick triage
        fail_ids = [cc["case_id"] for cc in per_case
                    if cc["pattern_tag"] == tag and cc["verdict"] in ("FAIL", "FAIL_NEG")][:3]
        print(f"  {tag:<24} {passes:>5} {partial:>5} {fails:>5} {total:>6}  "
              f"{', '.join(fail_ids)}")
    print()

    print("=" * 70)
    print("Per (language × pattern_tag) — only cells with at least one failure:")
    print(f"  {'lang':<6} {'pattern_tag':<24} {'PASS':>5} {'PART':>5} {'FAIL':>5}")
    for (la, tag), c in sorted(per_lang_tag.items()):
        passes = c["PASS"] + c["PASS_NEG"]
        partial = c["PARTIAL"]
        fails = c["FAIL"] + c["FAIL_NEG"]
        if fails == 0:
            continue
        print(f"  {la:<6} {tag:<24} {passes:>5} {partial:>5} {fails:>5}")
    print()

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps({
        "model": str(args.model),
        "n_cases": len(ALL_PROBES),
        "overall": dict(overall),
        "per_tag": {k: dict(v) for k, v in per_tag.items()},
        "per_lang_tag": {f"{la}__{tag}": dict(c) for (la, tag), c in per_lang_tag.items()},
        "per_case": per_case,
    }, indent=2, ensure_ascii=False))
    print(f"Wrote {args.report_out}")


if __name__ == "__main__":
    main()
