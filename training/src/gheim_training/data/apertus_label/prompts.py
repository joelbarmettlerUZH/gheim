"""Prompt design for Apertus labeling of real Swiss text.

M6: 12 targeted few-shot demos covering the failure modes observed in v3
audit (anonymized parties wrongly tagged, title-case authors missed,
spelled-out dates missed, public figures missed, slash-coauthors). Each
output includes a brief ``rule`` field that maps the tag to one of the
prompt's named rules — cheap chain-of-thought that forces the small model
to commit to a justification, dramatically improving rule-following.

Output contract (M6):
  - Pure JSON array, no markdown
  - Each item: {"value": "<verbatim>", "label": "<one of three>", "rule": "<rule-id>"}
  - Empty array if no PII

The ``rule`` field is post-stripped by ``label.py`` so downstream code is
unchanged.

Apertus is asked for ONLY three categories: ``private_person``,
``private_address``, ``private_date``. The other five (email, phone,
account_number, url, secret) are covered by regex (``prefilter.py``).
"""
from __future__ import annotations

_RULES_BLOCK = (
    "You are a precise Swiss PII extractor. You operate on fragments of "
    "Swiss legal documents, parliamentary records, news articles, and "
    "administrative texts in German, French, Italian, or Romansh.\n\n"

    "Extract every span containing one of three categories. For each span "
    "output a `rule` field naming which specific rule made you tag it.\n\n"

    "## Person rules\n"
    "  - **P1-name**: a real person's first+last name ('Anna Müller', "
    "    'Pierre Dubois'), or a name with title ('Dr. Müller', 'Mme Dubois')\n"
    "  - **P2-author-citation**: a surname (alone or with first name/initial) "
    "    appearing in an academic or legal citation ('ROLAND PFÄFFLI', "
    "    'Thomas Rauscher', 'WIEGAND', 'PIERRE A. KARRER'). ALL-CAPS AND "
    "    title-case both qualify. Tag EVERY author you see, not just the "
    "    first.\n"
    "  - **P3-slash-coauthor**: each surname in a slash-separated co-author "
    "    citation ('STAEHELIN/GROLIMUND') is a SEPARATE span — tag "
    "    'STAEHELIN' and 'GROLIMUND' independently, never together.\n"
    "  - **P4-public-figure**: a real public figure mentioned in news, "
    "    commentary, or articles ('Justin Gatlin', 'Marilyn Monroe', "
    "    'Steven Spielberg'). Public figures ARE PII for our purposes.\n"
    "  - **P5-historical**: historical figures with or without title "
    "    ('Kg. Sigismund', 'Karl der Grosse').\n\n"
    "Person NEGATIVE rules — DO NOT tag any of these:\n"
    "  - Anonymized procedural references: 'X.', 'Y. AG', 'A.A.', 'B.X.', "
    "    'X. SA', 'C.', 'der Beschwerdeführer', 'die Beschwerdegegnerin'\n"
    "  - Generic role nouns alone: 'der Richter', 'le secrétaire', "
    "    'der Anwalt', 'le requérant'\n"
    "  - Organizations: 'Bundesgericht', 'UBS AG', 'SBB', 'Tribunal "
    "    fédéral', 'die Stiftung X'\n\n"

    "## Address rules\n"
    "  - **A1-postal**: a complete postal address with street/PLZ/city or "
    "    similar ('Bahnhofstrasse 12, 8001 Zürich')\n"
    "  - **A2-place-as-residence**: a place name explicitly used as where "
    "    someone lives ('M. Dupont, domicilié à Lausanne'; 'Hr. Müller "
    "    aus Niederweningen'). Context must show the place IS the residence.\n\n"
    "Address NEGATIVE rules — DO NOT tag:\n"
    "  - Place names in narrative or topical context ('die Schweiz hat...', "
    "    'das Bundesgericht in Lausanne', 'in Bern fand statt...')\n"
    "  - Court/institution locations ('Bundesgericht in Lausanne')\n"
    "  - Country/canton references ('Schweiz', 'Kanton Zürich')\n\n"

    "## Date rules\n"
    "  - **D1-numeric**: strict numeric date — DD.MM.YYYY, DD/MM/YYYY, "
    "    DD.MM.YY, YYYY-MM-DD ('12.03.1985', '1.1.2024', '22.8.2012')\n"
    "  - **D2-spelled**: spelled-out date with day/month/year — '5. Juni "
    "    2012', '3 mars 2023', '14 gennaio 2011', '2 juin 2010'. ALWAYS "
    "    extract just the date; if it's inside a citation like 'Urteil X "
    "    vom 8. Oktober 1997 E. 3a', tag only '8. Oktober 1997', NOT the "
    "    whole citation.\n"
    "  - **D3-month-year**: month name + 4-digit year — 'Januar 2024', "
    "    'Juni 2012', 'mars 2023'\n\n"
    "Date NEGATIVE rules — DO NOT tag:\n"
    "  - Year-only references ('im Jahr 1985', '2024', 'im Berichtsjahr 2023')\n"
    "  - Journal-issue references ('11/1993', '6/2011', 'AJP 12/2010')\n"
    "  - Procedural references ('Art. 12', 'Mo. 10.3195', 'Ziff. 2.2.11', "
    "    'BBl 2006 7001', 'BGE 138 III 443', '5A_264/2009')\n"
    "  - Times ('08:32 Uhr', '14h30')\n"
    "  - Pure section/subsection numbers ('E. 2', 'Ziff. 5.4.1')\n\n"

    "## Out-of-scope (NEVER tag — handled separately)\n"
    "Do NOT extract phone numbers, e-mails, URLs, IBANs, AHV numbers, VAT "
    "numbers, credit cards, API keys, or any other category.\n\n"

    "## Output format\n"
    "Output ONLY a JSON array. Each entry has THREE fields:\n"
    '  "value": the EXACT substring from the input (verbatim, same casing, '
    "    same punctuation, same whitespace)\n"
    '  "label": one of "private_person", "private_address", "private_date"\n'
    '  "rule": the rule-id that justifies this tag (e.g. "P2-author-citation", '
    '    "D2-spelled", "A1-postal")\n\n'
    "Do not include any explanation, markdown fence, or text before/after "
    "the JSON. If there is no PII, output the empty array `[]`.\n\n"
    "I will give you 12 demonstration examples first, then your input. "
    "Each input message has the form:\n"
    "Input:\n<<<\n<text>\n>>>"
)


# In the chat-template, rules go in the system prompt (once); user turns
# carry just `Input:\n<<<\n{chunk}\n>>>`. This keeps the prompt around
# 5k-6k tokens with 12 few-shot demos vs 15k+ if rules were repeated.
SYSTEM_PROMPT = _RULES_BLOCK


# 12 targeted few-shot demos. Each one targets a specific failure pattern
# observed in the v3 audit, plus baseline coverage. The "rule" field maps
# each span to one of the named rules in _USER_INSTRUCTIONS.
FEW_SHOT_EXAMPLES = (
    # 1. Baseline DE court chunk
    {
        "user": (
            "Die Beschwerdeführerin Anna Müller, geboren am 15.04.1968, "
            "wohnhaft an der Bahnhofstrasse 12 in 8001 Zürich, beantragt..."
        ),
        "assistant": (
            '[{"value":"Anna Müller","label":"private_person","rule":"P1-name"},'
            '{"value":"15.04.1968","label":"private_date","rule":"D1-numeric"},'
            '{"value":"Bahnhofstrasse 12 in 8001 Zürich","label":"private_address","rule":"A1-postal"}]'
        ),
    },
    # 2. Baseline FR
    {
        "user": (
            "Le recourant, M. Pierre Dubois, domicilié à Lausanne, a déposé "
            "son recours le 3 mars 2023 contre la décision de l'autorité..."
        ),
        "assistant": (
            '[{"value":"Pierre Dubois","label":"private_person","rule":"P1-name"},'
            '{"value":"Lausanne","label":"private_address","rule":"A2-place-as-residence"},'
            '{"value":"3 mars 2023","label":"private_date","rule":"D2-spelled"}]'
        ),
    },
    # 3. Baseline IT
    {
        "user": (
            "Considerato che il sig. Mario Bernasconi ha ricevuto la "
            "convocazione il 12.05.2024..."
        ),
        "assistant": (
            '[{"value":"Mario Bernasconi","label":"private_person","rule":"P1-name"},'
            '{"value":"12.05.2024","label":"private_date","rule":"D1-numeric"}]'
        ),
    },
    # 4. M6: ALL-CAPS + title-case authors in the SAME citation
    {
        "user": (
            "Vgl. ROLAND PFÄFFLI, Neuerungen im Immobiliarsachenrecht, BN "
            "11/1993 S. 455; sowie Thomas Rauscher in Europäisches Zivilprozess- "
            "und Kollisionsrecht, 4. Aufl., München 2011; STAEHELIN/GROLIMUND, "
            "Zivilprozessrecht, Zürich 2008, S. 240; ferner Alfred Koller, "
            "Schweizerisches Obligationenrecht."
        ),
        "assistant": (
            '[{"value":"ROLAND PFÄFFLI","label":"private_person","rule":"P2-author-citation"},'
            '{"value":"Thomas Rauscher","label":"private_person","rule":"P2-author-citation"},'
            '{"value":"STAEHELIN","label":"private_person","rule":"P3-slash-coauthor"},'
            '{"value":"GROLIMUND","label":"private_person","rule":"P3-slash-coauthor"},'
            '{"value":"Alfred Koller","label":"private_person","rule":"P2-author-citation"}]'
        ),
    },
    # 5. M6: anonymized parties — NEGATIVE; counter-example
    {
        "user": (
            "Im Verfahren X. AG gegen B.X. (5A_741/2011) hat das Bundesgericht "
            "am 13. Juni 2012 entschieden, dass A.A. nicht zur Beschwerde "
            "legitimiert ist. Die Beschwerdeführerin wird auf den Entscheid "
            "Y. SA verwiesen."
        ),
        "assistant": (
            '[{"value":"13. Juni 2012","label":"private_date","rule":"D2-spelled"}]'
        ),
    },
    # 6. M6: spelled-out dates buried in narrative
    {
        "user": (
            "Die ausserordentliche Generalversammlung vom 31. Juli 2013 wurde "
            "am 14 gennaio 2011 vorbereitet und am 2 juin 2010 erstmals "
            "diskutiert. Eine Folgeversammlung fand im Juni 2012 statt."
        ),
        "assistant": (
            '[{"value":"31. Juli 2013","label":"private_date","rule":"D2-spelled"},'
            '{"value":"14 gennaio 2011","label":"private_date","rule":"D2-spelled"},'
            '{"value":"2 juin 2010","label":"private_date","rule":"D2-spelled"},'
            '{"value":"Juni 2012","label":"private_date","rule":"D3-month-year"}]'
        ),
    },
    # 7. M6: public figure in news context
    {
        "user": (
            "Justin Gatlin beendet wegen einer Krankheit vorzeitig die Saison. "
            "Der Bronzemedaillen-Gewinner von London hat bereits Marilyn "
            "Monroe als Inspiration genannt."
        ),
        "assistant": (
            '[{"value":"Justin Gatlin","label":"private_person","rule":"P4-public-figure"},'
            '{"value":"Marilyn Monroe","label":"private_person","rule":"P4-public-figure"}]'
        ),
    },
    # 8. M6: counter-example — institutions, narrative places, year-only
    {
        "user": (
            "Die Schweiz hat im Jahr 2024 die Beziehungen zur EU vertieft. "
            "Das Bundesgericht in Lausanne und der Bundesrat haben sich "
            "in Bern getroffen. Die SBB betrieb im Berichtsjahr 2023 mehr "
            "Linien als je zuvor."
        ),
        "assistant": '[]',
    },
    # 9. M6: form/table layout with multiple PII
    {
        "user": (
            "Versicherter:\nName: Petra Vetterli\nGeburtsdatum: 03.07.1962\n"
            "Wohnsitz: Quellenstrasse 31, 8005 Zürich\n"
            "Beginn Versicherung: 01.01.2024"
        ),
        "assistant": (
            '[{"value":"Petra Vetterli","label":"private_person","rule":"P1-name"},'
            '{"value":"03.07.1962","label":"private_date","rule":"D1-numeric"},'
            '{"value":"Quellenstrasse 31, 8005 Zürich","label":"private_address","rule":"A1-postal"},'
            '{"value":"01.01.2024","label":"private_date","rule":"D1-numeric"}]'
        ),
    },
    # 10. M6: long citation with date — extract DATE, not whole citation
    {
        "user": (
            "Vgl. Urteil des Bundesgerichts 4C.93/1997 vom 8. Oktober 1997 "
            "E. 3a; ferner BGE 138 III 443 vom 13. Juni 2012, sowie arrêt "
            "5C.42/2002 du 29 septembre 2002."
        ),
        "assistant": (
            '[{"value":"8. Oktober 1997","label":"private_date","rule":"D2-spelled"},'
            '{"value":"13. Juni 2012","label":"private_date","rule":"D2-spelled"},'
            '{"value":"29 septembre 2002","label":"private_date","rule":"D2-spelled"}]'
        ),
    },
    # 11. M6: counter-example — journal refs, procedural IDs, times
    {
        "user": (
            "Wie in AJP 6/2011 S. 174 und ZSR 12/2010 dargelegt, ist die "
            "Frage in der Lehre umstritten (vgl. WIEGAND, op. cit. Ziff. 2.2.11; "
            "siehe auch Mo. 10.3195). Die Sitzung vom 22.8.2012 um 08:32 Uhr "
            "wurde verschoben."
        ),
        "assistant": (
            '[{"value":"WIEGAND","label":"private_person","rule":"P2-author-citation"},'
            '{"value":"22.8.2012","label":"private_date","rule":"D1-numeric"}]'
        ),
    },
    # 12. M6: historical figure with title + Romansh name
    {
        "user": (
            "Sezzida tar Andri Caduff, en preschientscha da Curdin Tschuor "
            "ed Annina Camenisch, ils 17 mars 2025 a Disentis."
        ),
        "assistant": (
            '[{"value":"Andri Caduff","label":"private_person","rule":"P1-name"},'
            '{"value":"Curdin Tschuor","label":"private_person","rule":"P1-name"},'
            '{"value":"Annina Camenisch","label":"private_person","rule":"P1-name"},'
            '{"value":"17 mars 2025","label":"private_date","rule":"D2-spelled"}]'
        ),
    },
)


def _user_msg(chunk_text: str) -> str:
    return f"Input:\n<<<\n{chunk_text}\n>>>"


def build_messages(chunk_text: str) -> list[dict]:
    """Chat-template messages: system (full rules) + N (user, assistant)
    few-shot demos + final user. The rules block is sent ONCE in the system
    prompt rather than repeated per turn, so total prompt ≈ 5-6k tokens
    with 12 demos instead of 15k+ if rules were embedded in every user msg."""
    msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEW_SHOT_EXAMPLES:
        msgs.append({"role": "user", "content": _user_msg(ex["user"])})
        msgs.append({"role": "assistant", "content": ex["assistant"]})
    msgs.append({"role": "user", "content": _user_msg(chunk_text)})
    return msgs
