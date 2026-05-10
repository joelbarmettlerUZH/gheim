"""Per-language prompt design for Gemma 4 26B-A4B labeling.

Each target language gets its own prompt (rules + few-shot examples
written in that language). Routing happens in the labeler based on the
chunk's detected language. Rationale: in our PoC, an English-only prompt
applied to German court text caused Gemma to default to ``{"spans":[]}``
on chunks that obviously contain dates and addresses — the model was
matching the empty negative-example pattern more strongly than the
positive ones because the rule text didn't share vocabulary with the
input. Language-matched prompts fix this.

Categories emitted: account_number, private_address, private_date,
private_email, private_person, private_phone, private_url, secret.

Output contract (all languages):
  - Pure JSON object: {"spans": [...]}
  - Each span: {"value": "<verbatim>", "label": "<one of 8>"}
  - Empty list ``{"spans": []}`` ONLY when zero PII spans are present.
    Empty is a meaningful signal — explicitly asked for, not the default.

The schema is enforced at decode time via vLLM's xgrammar backend, so
malformed outputs are impossible.
"""
from __future__ import annotations

from typing import Literal, get_args

from ..schema import Language

# Languages we have a fully-localised prompt for. RM falls back to DE
# because Gemma's RM grasp is weaker and the rules are easier to follow
# when expressed in a stronger language; the few-shot examples include
# RM content so the model still sees in-language demos.
PromptLang = Literal["de_ch", "fr_ch", "it_ch", "en", "rm"]
SUPPORTED_PROMPT_LANGS: tuple[PromptLang, ...] = get_args(PromptLang)


# ---------------------------------------------------------------------------
# Rule + output contract — language-specific.
#
# Each block describes the same 8 categories with rules written in the
# target language. They share the IDENTICAL output contract so the
# downstream parser is unchanged.
# ---------------------------------------------------------------------------

_DE_RULES = """Du bist ein präziser Schweizer PII-Extraktor. Du arbeitest mit \
Ausschnitten aus Schweizer Gerichtsentscheiden, Parlamentsprotokollen, \
Zeitungsartikeln und Verwaltungstexten in Deutsch.

Extrahiere jede Spanne mit personenbezogenen Daten. Für jede Spanne gibst du \
den wortgetreuen Teilstring und eine der 8 Kategorien aus. Wenn der Text \
keine PII enthält, gibst du eine leere Liste aus: das ist eine \
beabsichtigte und valide Antwort.

## Kategorie `private_person`
  - Reale Personennamen (Vor- + Nachname): "Anna Müller", "Dr. Müller"
  - Autoren in Zitaten — auch ALLE-GROSSBUCHSTABEN: "ROLAND PFÄFFLI", "WIEGAND"
  - Bei Schrägstrich-Coautoren: jeden Namen einzeln taggen \
("STAEHELIN/GROLIMUND" → "STAEHELIN", "GROLIMUND" als getrennte Spans)
  - Reale öffentliche Personen in News/Kommentaren
NEGATIV — NICHT taggen:
  - Anonymisierte Verfahrensbezeichnungen: "X.", "Y. AG", "A.A.", "B.X.", "C."
  - Generische Rollen: "der Beschwerdeführer", "die Beschwerdegegnerin", \
"der Richter", "der Anwalt"
  - Organisationen/Institutionen: "Bundesgericht", "UBS AG", "Konkursamt", "SBB"

## Kategorie `private_address`
  - Vollständige Postadressen: "Bahnhofstrasse 12, 8001 Zürich"
  - Ortsname als Wohnort einer Person: "Hr. Müller aus Niederweningen"
NEGATIV — NICHT taggen:
  - Ortsnamen im Erzählkontext: "in Bern fand statt", "die Schweiz hat..."
  - Behörden-/Gerichtsstandorte: "Bundesgericht in Lausanne"
  - Kantons-/Länderverweise: "Schweiz", "Kanton Zürich"
  - Institutionsnamen mit Ort: "Konkursamt Berner Jura-Seeland", \
"Aufsichtsbehörde Bern"

## Kategorie `private_date`
  - Numerisch: "12.03.1985", "1.1.2024", "2024-01-15"
  - Ausgeschrieben: "5. Juni 2012", "31. Juli 2013", "19. Juli 2010"
  - Monat + Jahr: "Januar 2024", "Juni 2012"
  - WICHTIG: In Zitaten wie "Urteil X vom 8. Oktober 1997 E. 3a" tagge NUR \
das Datum "8. Oktober 1997", nicht das ganze Zitat
NEGATIV — NICHT taggen:
  - Reine Jahreszahl: "im Jahr 1985", "2024", "im Berichtsjahr 2023"
  - Zeitschriften-/Heftnummern: "11/1993", "AJP 6/2011"
  - Verfahrens-/Aktenzeichen: "Art. 12", "Mo. 10.3195", "BGE 138 III 443", \
"5A_741/2011"
  - Uhrzeiten: "08:32 Uhr", "14h30"

## Kategorie `private_email`
  - Vollständige E-Mail-Adressen: "name@domain.ch"

## Kategorie `private_phone`
  - Schweizer Telefonnummern: "+41 79 555 12 34", "0041791234567", \
"079 123 45 67"
  - NEGATIV: Aktenzeichen, Artikelnummern

## Kategorie `private_url`
  - Vollständige URLs: "https://example.com/path", "www.example.ch"
  - IP-Adressen: "192.168.1.1"

## Kategorie `account_number`
  - IBAN: "CH93 0076 2011 6238 5295 7"
  - AHV-Nummer: "756.9217.0769.85" oder "7569217076985"
  - UID/MWST-Nummer: "CHE-123.456.789 MWST"
  - Kreditkartennummer (13–19 Ziffern)
  - Pass-/ID-/Führerausweisnummer: "Pass-Nr. ABC1234567"
  - Steuernummer
  - NEGATIV: Aktenzeichen, BGE-Verweise, Artikelnummern

## Kategorie `secret`
  - API-Tokens: "sk-...", "ghp_...", AWS-Schlüssel "AKIA..."
  - Explizit benannte Passwörter: "Passwort: hunter2", "PIN: 4321"
  - Kreditkarten-CVV (separat von der Kartennummer)
  - NEGATIV: zufällige Strings ohne Kontext (Hashes, UUIDs, Aktenzeichen)

## Ausgabeformat
Gib NUR ein JSON-Objekt aus, mit einem Feld `spans` (Liste). Jeder Eintrag:
  - `value`: der EXAKTE Teilstring aus dem Eingabetext (gleiche Schreibweise, \
gleiche Zeichensetzung, gleiche Leerzeichen)
  - `label`: eine der 8 Kategorien

Wenn keine PII vorhanden ist: `{"spans": []}` (das ist EINE beabsichtigte \
Antwort, nicht der Standardfall — gib sie nur aus, wenn du nach sorgfältiger \
Prüfung keine Spanne identifizieren konntest).

Ich gebe dir mehrere Beispiele und dann deine Eingabe. Eingabe-Format:
Input:
<<<
<text>
>>>"""

_FR_RULES = """Tu es un extracteur précis d'informations personnelles \
(PII) suisses. Tu travailles sur des extraits de décisions judiciaires \
suisses, procès-verbaux parlementaires, articles de presse et textes \
administratifs en français.

Extrais chaque empan contenant des données personnelles. Pour chaque empan, \
indique la sous-chaîne textuelle exacte et l'une des 8 catégories. Si le \
texte ne contient aucun PII, retourne une liste vide — c'est une réponse \
intentionnelle et valide.

## Catégorie `private_person`
  - Noms réels de personnes (prénom + nom) : "Pierre Dubois", "M. Dupont"
  - Auteurs cités, y compris en MAJUSCULES : "PIERRE A. KARRER", "WIEGAND"
  - Pour les co-auteurs séparés par des barres obliques, tagger chaque nom \
séparément
  - Personnalités publiques mentionnées dans la presse
NÉGATIF — ne PAS tagger :
  - Désignations anonymisées : "X.", "Y. AG", "A.A.", "C."
  - Rôles génériques : "le recourant", "la requérante", "le juge"
  - Organisations/institutions : "Tribunal fédéral", "UBS SA", "CFF"

## Catégorie `private_address`
  - Adresses postales complètes : "Rue du Rhône 12, 1204 Genève"
  - Nom de lieu identifiant la résidence : "M. Dupont, domicilié à Lausanne"
NÉGATIF :
  - Lieux dans un contexte narratif : "à Berne, le Conseil fédéral...", \
"en Suisse"
  - Lieux d'institutions : "Tribunal fédéral à Lausanne"
  - Cantons/pays seuls : "Suisse", "Canton de Vaud"

## Catégorie `private_date`
  - Numérique : "12.03.1985", "15/06/2024"
  - Écrite en toutes lettres : "3 mars 2023", "5 juin 2012"
  - Mois + année : "mars 2023", "janvier 2024"
  - IMPORTANT : dans une citation comme "arrêt 5C.42/2002 du 29 septembre 2002", \
tagger UNIQUEMENT "29 septembre 2002"
NÉGATIF :
  - Année seule : "en 1985", "2024"
  - Références de revue : "RJB 11/1993"
  - Références de procédure : "art. 12", "ATF 138 III 443"
  - Heures : "08h32", "14h30"

## Catégorie `private_email`
  - Adresses e-mail complètes : "nom@domaine.ch"

## Catégorie `private_phone`
  - Numéros suisses : "+41 22 555 12 34", "0041 22 555 12 34", "022 555 12 34"

## Catégorie `private_url`
  - URLs complètes : "https://exemple.com", "www.exemple.ch"
  - Adresses IP : "192.168.1.1"

## Catégorie `account_number`
  - IBAN : "CH93 0076 2011 6238 5295 7"
  - Numéro AVS : "756.9217.0769.85"
  - IDE/TVA : "CHE-123.456.789 TVA"
  - Numéro de carte de crédit (13–19 chiffres)
  - Numéro de passeport/CI/permis : "n° passeport ABC1234567"

## Catégorie `secret`
  - Tokens API : "sk-...", "ghp_..."
  - Mots de passe explicitement nommés : "mot de passe : hunter2"
  - Cryptogrammes CVV
  - NÉGATIF : chaînes aléatoires sans contexte

## Format de sortie
Retourne UNIQUEMENT un objet JSON avec un champ `spans`. Chaque entrée :
  - `value` : la sous-chaîne EXACTE de l'entrée
  - `label` : l'une des 8 catégories

S'il n'y a aucun PII : `{"spans": []}` — réponse intentionnelle, à utiliser \
seulement après vérification soigneuse.

Format d'entrée :
Input:
<<<
<text>
>>>"""

_IT_RULES = """Sei un estrattore preciso di informazioni personali (PII) \
svizzere. Lavori su frammenti di decisioni giudiziarie svizzere, verbali \
parlamentari, articoli di stampa e testi amministrativi in italiano.

Estrai ogni span contenente dati personali. Per ciascuno indica la \
sottostringa testuale esatta e una delle 8 categorie. Se il testo non \
contiene PII, restituisci una lista vuota — è una risposta intenzionale \
e valida.

## Categoria `private_person`
  - Nomi reali di persone (nome + cognome): "Mario Bernasconi", "Sig. Rossi"
  - Autori citati, anche in MAIUSCOLO: "MARIO ROSSI", "WIEGAND"
  - Per coautori con barra: ogni nome separatamente
  - Personalità pubbliche menzionate nei media
NEGATIVO — NON taggare:
  - Designazioni anonimizzate: "X.", "Y. SA", "A.A.", "C."
  - Ruoli generici: "il ricorrente", "la giudice", "l'avvocato"
  - Organizzazioni/istituzioni: "Tribunale federale", "UBS SA", "FFS"

## Categoria `private_address`
  - Indirizzi postali completi: "Via Nassa 12, 6900 Lugano"
  - Nome di luogo come residenza: "Sig. Bernasconi domiciliato a Bellinzona"
NEGATIVO:
  - Luoghi in contesto narrativo: "a Berna si è svolto..."
  - Luoghi di istituzioni: "Tribunale federale a Losanna"
  - Cantoni/paesi: "Svizzera", "Canton Ticino"

## Categoria `private_date`
  - Numerico: "12.03.1985", "15/06/2024"
  - In lettere: "14 gennaio 2011", "5 giugno 2012"
  - Mese + anno: "gennaio 2024"
  - IMPORTANTE: in citazioni come "sentenza X del 12 maggio 2024 cons. 3a", \
taggare SOLO "12 maggio 2024"
NEGATIVO:
  - Anno solo: "nel 1985", "2024"
  - Riferimenti a riviste: "AJP 11/1993"
  - Riferimenti procedurali: "art. 12", "DTF 138 III 443"
  - Orari: "08:32", "14h30"

## Categoria `private_email`
  - Indirizzi e-mail completi: "nome@dominio.ch"

## Categoria `private_phone`
  - Numeri svizzeri: "+41 91 555 12 34", "091 555 12 34"

## Categoria `private_url`
  - URL complete: "https://esempio.com", "www.esempio.ch"
  - Indirizzi IP: "192.168.1.1"

## Categoria `account_number`
  - IBAN: "CH93 0076 2011 6238 5295 7"
  - Numero AVS: "756.9217.0769.85"
  - IDI/IVA: "CHE-123.456.789 IVA"
  - Carta di credito (13–19 cifre)
  - Passaporto/Carta d'identità/Patente

## Categoria `secret`
  - Token API: "sk-...", "ghp_..."
  - Password esplicite: "password: hunter2"
  - CVV
  - NEGATIVO: stringhe casuali senza contesto

## Formato di output
Restituisci SOLO un oggetto JSON con un campo `spans`. Ogni voce:
  - `value`: la sottostringa ESATTA dall'input
  - `label`: una delle 8 categorie

Se nessun PII: `{"spans": []}` — risposta intenzionale, usata solo dopo \
attenta verifica.

Formato di input:
Input:
<<<
<text>
>>>"""

# RM prompt is in Romansh — Gemma's RM is weaker than DE/FR/IT but the
# few-shot examples are in RM, which carries the pattern.
_RM_RULES = """Ti es in extractur precis da datas persunalas (PII) svizras. \
Ti lavuras sin frammets da deciziuns giudizialas svizras, protocols \
parlamentaris e texts administrativs en rumantsch.

Extrai mintga span che cuntegna datas persunalas. Per mintga span dai la \
sutstring exacta e ina da las 8 categorias. Sche'l text na cuntegna nagina \
PII, datti ina glista vida — quai è ina resposta intenziunala e valida.

## Categoria `private_person`
  - Nums reals da persunas: "Andri Caduff", "Curdin Tschuor"
  - Auturs en citaziuns, era en MAIUSCLAS
NEGATIV — BETG taggar:
  - Designaziuns anonimisadas: "X.", "Y. AG", "A.A."
  - Rolas generalas: "il recurrent", "la dretgira"
  - Organisaziuns: "Tribunal federal"

## Categoria `private_address`
  - Adressas cumplettas: "Plazza Major 1, 7000 Cuira"
  - Lieu sco residenza
NEGATIV: lieus en text narrativ, sezs d'instituziuns

## Categoria `private_date`
  - Numeric: "12.03.1985"
  - Scrit: "17 mars 2025", "5 schaner 2024"
  - Mais + onn: "schaner 2024"
NEGATIV: onn sulet, referenzas procedialas

## Las autras categorias
  - `private_email`: adressas e-mail cumplettas
  - `private_phone`: numers da telefon svizzer
  - `private_url`: URL e adressas IP
  - `account_number`: IBAN, AHV, UID/MWST, nums da passaport, nums da \
carta da credit
  - `secret`: tokens API, pleds-clav explicits

## Format da return
Returna SOL in object JSON cun in champ `spans` cun glista da \
{value, label}. Sche nagina PII: `{"spans": []}`.

Format d'input:
Input:
<<<
<text>
>>>"""

_EN_RULES = """You are a precise PII extractor for English text. You operate \
on excerpts of articles, correspondence, support tickets, and similar \
content.

Extract every span containing personally-identifying information. For each \
span report the verbatim substring and one of the 8 categories. If the text \
contains no PII, return an empty list — this is an intentional, valid \
response, NOT the default.

## Category `private_person`
  - Real first+last names; with or without title
  - Authors in citations; ALL-CAPS qualifies
  - Slash-separated co-authors: tag each separately
  - Public figures
NEGATIVE: anonymised refs ("X.", "A.A."), generic roles ("the plaintiff", \
"the appellant"), organisations.

## Category `private_address`
  - Full postal addresses
  - Place name explicitly identifying residence ("residing at Lausanne")
NEGATIVE: places in narrative; institution locations; country/state alone.

## Category `private_date`
  - Numeric (DD.MM.YYYY, YYYY-MM-DD), spelled-out, month+year.
  - In citations like "Smith v. Jones, May 12, 2024 at p. 4", tag only "May 12, 2024".
NEGATIVE: year alone, journal refs, article numbers, times.

## Other categories
  - `private_email`: full email addresses
  - `private_phone`: phone numbers (any country)
  - `private_url`: URLs, IP addresses
  - `account_number`: IBAN, SSN, credit card, passport, ID, driver's license
  - `secret`: API tokens, named passwords, CVVs

## Output format
Output ONLY a JSON object with a `spans` field. Each entry:
  - `value`: EXACT substring from the input
  - `label`: one of the 8 categories

If no PII: `{"spans": []}` — intentional, only after careful checking.

Input format:
Input:
<<<
<text>
>>>"""


# ---------------------------------------------------------------------------
# Few-shot examples per language. Each language gets at least:
#   - 2-3 positive examples covering different PII categories
#   - 1 negative (no PII) example to anchor the empty-list output
#   - 1 mixed example with subtle/anonymised content
# ---------------------------------------------------------------------------

_DE_EXAMPLES = (
    {
        "user": ("Die Beschwerdeführerin Anna Müller, geboren am 15.04.1968, "
                 "wohnhaft an der Bahnhofstrasse 12 in 8001 Zürich, beantragt "
                 "die Aufhebung des Entscheids."),
        "assistant": (
            '{"spans":['
            '{"value":"Anna Müller","label":"private_person"},'
            '{"value":"15.04.1968","label":"private_date"},'
            '{"value":"Bahnhofstrasse 12 in 8001 Zürich","label":"private_address"}'
            ']}'
        ),
    },
    # Anonymised + dates — IMPORTANT: dates MUST still be tagged
    {
        "user": ("A.d Am 19. Juli 2010 gelangte X. an die Aufsichtsbehörde "
                 "und beschwerte sich über die im Konkursprotokoll vom "
                 "2. Juni 2010 aufgeführten ordentlichen Konkurskosten."),
        "assistant": (
            '{"spans":['
            '{"value":"19. Juli 2010","label":"private_date"},'
            '{"value":"2. Juni 2010","label":"private_date"}'
            ']}'
        ),
    },
    # Citations + ALL-CAPS authors
    {
        "user": ("Vgl. ROLAND PFÄFFLI, Neuerungen im Immobiliarsachenrecht, "
                 "BN 11/1993 S. 455; sowie Thomas Rauscher, München 2011; "
                 "STAEHELIN/GROLIMUND, Zivilprozessrecht, Zürich 2008."),
        "assistant": (
            '{"spans":['
            '{"value":"ROLAND PFÄFFLI","label":"private_person"},'
            '{"value":"Thomas Rauscher","label":"private_person"},'
            '{"value":"STAEHELIN","label":"private_person"},'
            '{"value":"GROLIMUND","label":"private_person"}'
            ']}'
        ),
    },
    # Form layout with structured PII
    {
        "user": ("Versicherter:\nName: Petra Vetterli\nGeburtsdatum: 03.07.1962\n"
                 "Wohnsitz: Quellenstrasse 31, 8005 Zürich\n"
                 "Telefon: +41 44 123 45 67\nE-Mail: petra.vetterli@bluewin.ch\n"
                 "AHV-Nr.: 756.9217.0769.85"),
        "assistant": (
            '{"spans":['
            '{"value":"Petra Vetterli","label":"private_person"},'
            '{"value":"03.07.1962","label":"private_date"},'
            '{"value":"Quellenstrasse 31, 8005 Zürich","label":"private_address"},'
            '{"value":"+41 44 123 45 67","label":"private_phone"},'
            '{"value":"petra.vetterli@bluewin.ch","label":"private_email"},'
            '{"value":"756.9217.0769.85","label":"account_number"}'
            ']}'
        ),
    },
    # Pure narrative, NO PII (anchor the empty case)
    {
        "user": ("Die Verhandlung wurde aufgrund formeller Mängel der "
                 "Beschwerde abgewiesen. Der Beschwerdegegnerin wurden die "
                 "Verfahrenskosten auferlegt; auf eine Parteientschädigung "
                 "wird verzichtet."),
        "assistant": '{"spans":[]}',
    },
    # Counter-example: institutions, narrative places, no PII
    {
        "user": ("Die Schweiz hat im Jahr 2024 die Beziehungen zur EU "
                 "vertieft. Das Bundesgericht in Lausanne und der Bundesrat "
                 "haben sich in Bern getroffen. Die SBB betrieb mehr Linien "
                 "als je zuvor."),
        "assistant": '{"spans":[]}',
    },
)

_FR_EXAMPLES = (
    {
        "user": ("Le recourant, M. Pierre Dubois, domicilié à Lausanne, a "
                 "déposé son recours le 3 mars 2023 contre la décision de "
                 "l'autorité."),
        "assistant": (
            '{"spans":['
            '{"value":"Pierre Dubois","label":"private_person"},'
            '{"value":"Lausanne","label":"private_address"},'
            '{"value":"3 mars 2023","label":"private_date"}'
            ']}'
        ),
    },
    # Citation + author + date extracted only
    {
        "user": ("Voir l'arrêt 5C.42/2002 du 29 septembre 2002 ; cf. "
                 "PIERRE A. KARRER, Le contrat international, Genève 2010."),
        "assistant": (
            '{"spans":['
            '{"value":"29 septembre 2002","label":"private_date"},'
            '{"value":"PIERRE A. KARRER","label":"private_person"}'
            ']}'
        ),
    },
    # Form with structured PII
    {
        "user": ("Bénéficiaire: Marie Petit\nDate de naissance: 12/06/1985\n"
                 "IBAN: CH93 0076 2011 6238 5295 7\n"
                 "E-mail: marie.petit@example.ch\n"
                 "Téléphone: +41 22 555 12 34"),
        "assistant": (
            '{"spans":['
            '{"value":"Marie Petit","label":"private_person"},'
            '{"value":"12/06/1985","label":"private_date"},'
            '{"value":"CH93 0076 2011 6238 5295 7","label":"account_number"},'
            '{"value":"marie.petit@example.ch","label":"private_email"},'
            '{"value":"+41 22 555 12 34","label":"private_phone"}'
            ']}'
        ),
    },
    # Negative
    {
        "user": ("Le tribunal a rejeté le recours pour vice de forme. La "
                 "partie défenderesse devra supporter les frais de la "
                 "procédure."),
        "assistant": '{"spans":[]}',
    },
    # Anonymised — only the date should be tagged
    {
        "user": ("Dans la procédure X. SA contre B. (5A_741/2011), le "
                 "Tribunal fédéral a statué le 13 juin 2012 que A.A. "
                 "n'avait pas qualité pour recourir."),
        "assistant": (
            '{"spans":['
            '{"value":"13 juin 2012","label":"private_date"}'
            ']}'
        ),
    },
)

_IT_EXAMPLES = (
    {
        "user": ("Il sig. Mario Bernasconi, domiciliato a Bellinzona, ha "
                 "presentato ricorso il 14 gennaio 2011 contro la "
                 "decisione dell'autorità."),
        "assistant": (
            '{"spans":['
            '{"value":"Mario Bernasconi","label":"private_person"},'
            '{"value":"Bellinzona","label":"private_address"},'
            '{"value":"14 gennaio 2011","label":"private_date"}'
            ']}'
        ),
    },
    # Form with structured PII
    {
        "user": ("Beneficiario: Anna Rossi\nData di nascita: 03.07.1962\n"
                 "Indirizzo: Via Nassa 12, 6900 Lugano\n"
                 "E-mail: anna.rossi@example.ch\n"
                 "Telefono: +41 91 555 12 34"),
        "assistant": (
            '{"spans":['
            '{"value":"Anna Rossi","label":"private_person"},'
            '{"value":"03.07.1962","label":"private_date"},'
            '{"value":"Via Nassa 12, 6900 Lugano","label":"private_address"},'
            '{"value":"anna.rossi@example.ch","label":"private_email"},'
            '{"value":"+41 91 555 12 34","label":"private_phone"}'
            ']}'
        ),
    },
    # Citation — date only
    {
        "user": ("Vedi sentenza 4C.93/1997 del 8 ottobre 1997 cons. 3a; "
                 "cfr. anche DTF 138 III 443."),
        "assistant": (
            '{"spans":['
            '{"value":"8 ottobre 1997","label":"private_date"}'
            ']}'
        ),
    },
    # Negative
    {
        "user": ("Il tribunale ha respinto il ricorso per vizio di forma. "
                 "Le spese processuali sono poste a carico della parte "
                 "ricorrente."),
        "assistant": '{"spans":[]}',
    },
    # Narrative places, year-only — empty
    {
        "user": ("Nel 2024 la Svizzera ha approfondito le relazioni con "
                 "l'UE. Il Tribunale federale a Losanna e il Consiglio "
                 "federale si sono incontrati a Berna."),
        "assistant": '{"spans":[]}',
    },
)

_RM_EXAMPLES = (
    {
        "user": ("Sezzida tar Andri Caduff, en preschientscha da Curdin "
                 "Tschuor ed Annina Camenisch, ils 17 mars 2025 a "
                 "Disentis. Contact: andri.caduff@gr.ch."),
        "assistant": (
            '{"spans":['
            '{"value":"Andri Caduff","label":"private_person"},'
            '{"value":"Curdin Tschuor","label":"private_person"},'
            '{"value":"Annina Camenisch","label":"private_person"},'
            '{"value":"17 mars 2025","label":"private_date"},'
            '{"value":"andri.caduff@gr.ch","label":"private_email"}'
            ']}'
        ),
    },
    # Form-style with structured PII
    {
        "user": ("Num: Gian-Andrea Caviezel\nData da naschientscha: "
                 "12.05.1980\nIBAN: CH93 0076 2011 6238 5295 7"),
        "assistant": (
            '{"spans":['
            '{"value":"Gian-Andrea Caviezel","label":"private_person"},'
            '{"value":"12.05.1980","label":"private_date"},'
            '{"value":"CH93 0076 2011 6238 5295 7","label":"account_number"}'
            ']}'
        ),
    },
    # Negative
    {
        "user": ("La dretgira ha refusà il recurs per motivs formals. Las "
                 "expensas dal process èn a quint da la part che ha "
                 "recurrì."),
        "assistant": '{"spans":[]}',
    },
)

_EN_EXAMPLES = (
    {
        "user": ("Dear Mr. John Smith, please confirm your appointment on "
                 "March 12, 2024 at our Boston office. Reply to "
                 "jsmith@example.com or call +1 617 555 0123."),
        "assistant": (
            '{"spans":['
            '{"value":"John Smith","label":"private_person"},'
            '{"value":"March 12, 2024","label":"private_date"},'
            '{"value":"jsmith@example.com","label":"private_email"},'
            '{"value":"+1 617 555 0123","label":"private_phone"}'
            ']}'
        ),
    },
    # Negative
    {
        "user": ("The court dismissed the appeal on procedural grounds. "
                 "Costs were assessed against the appellant. No further "
                 "submissions are scheduled."),
        "assistant": '{"spans":[]}',
    },
    # Structured PII
    {
        "user": ("Account holder: Jane Doe, SSN 123-45-6789, "
                 "card 4111 1111 1111 1111, expires 09/27."),
        "assistant": (
            '{"spans":['
            '{"value":"Jane Doe","label":"private_person"},'
            '{"value":"123-45-6789","label":"account_number"},'
            '{"value":"4111 1111 1111 1111","label":"account_number"}'
            ']}'
        ),
    },
)


_RULES_BY_LANG: dict[PromptLang, str] = {
    "de_ch": _DE_RULES,
    "fr_ch": _FR_RULES,
    "it_ch": _IT_RULES,
    "rm": _RM_RULES,
    "en": _EN_RULES,
}

_EXAMPLES_BY_LANG: dict[PromptLang, tuple[dict[str, str], ...]] = {
    "de_ch": _DE_EXAMPLES,
    "fr_ch": _FR_EXAMPLES,
    "it_ch": _IT_EXAMPLES,
    "rm": _RM_EXAMPLES,
    "en": _EN_EXAMPLES,
}


def _user_msg(chunk_text: str) -> str:
    return f"Input:\n<<<\n{chunk_text}\n>>>"


def _resolve_prompt_lang(language: Language) -> PromptLang:
    """Map a chunk's detected language to a prompt language.

    GSW (Swiss German) → de_ch prompt (no separate prompt; written GSW is
    close enough to standard German that the rules still apply).
    """
    if language == "gsw":
        return "de_ch"
    if language in SUPPORTED_PROMPT_LANGS:
        # mypy/ty narrows via the membership check
        return language  # type: ignore[return-value]
    # Unknown future language → fall back to English (most general rules).
    return "en"


def build_messages(chunk_text: str, language: Language = "de_ch") -> list[dict]:
    """Chat-template messages: system (rules in target language) + few-shot
    demos (in the same language) + final user.

    The rules block + demos run ~3-5k tokens total, leaving plenty of room
    inside Gemma 4's 8k context for the chunk text and the response.
    """
    lang = _resolve_prompt_lang(language)
    msgs: list[dict] = [{"role": "system", "content": _RULES_BY_LANG[lang]}]
    for ex in _EXAMPLES_BY_LANG[lang]:
        msgs.append({"role": "user", "content": _user_msg(ex["user"])})
        msgs.append({"role": "assistant", "content": ex["assistant"]})
    msgs.append({"role": "user", "content": _user_msg(chunk_text)})
    return msgs
