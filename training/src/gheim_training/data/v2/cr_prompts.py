"""Commercial-register-aware PII labelling prompts.

Specialised prompt for re-labelling Swiss commercial-register chunks.
Differs from the standard per-language Gemma prompt in two ways:

1. **Up-front context**: tells the model the chunk is from a Swiss
   commercial-register listing (Zefix, FineWeb-2 entries on
   newly-formed GmbHs, etc.) so it knows to expect the contact-block
   format rather than prose.
2. **Few-shot examples are commercial-register listings**: directors,
   board members, signatories with role suffixes — the exact format
   the v1 Gemma pass under-recalled.

The output schema is identical to the standard prompt so the existing
parse + verify pipeline (``gheim_training.data.gemma.labeler``) works
unchanged.
"""
from __future__ import annotations

from typing import Literal

from ..gemma.prompts import _user_msg
from ..schema import Language

PromptLang = Literal["de_ch", "fr_ch", "it_ch", "rm", "en"]


_DE_CR_RULES = """Du bist ein präziser Schweizer PII-Extraktor. Der folgende \
Auszug stammt aus einem Schweizer Handelsregistereintrag (Zefix-Eintrag, \
Firmenprofil oder ähnliches Verzeichnis). Solche Texte folgen einem \
festen Muster: «Personen mit Entscheidungsbefugnis: <Name> (<Rolle>) und \
<Name> (<Rolle>)…».

WICHTIG für diesen Texttyp: JEDE namentlich genannte natürliche Person \
ist PII und MUSS extrahiert werden — auch wenn ihre Rolle (Verwaltungsrat, \
Geschäftsführer, Gesellschafter, Präsident, Vorstand, Prokurist) nur \
generisch beschrieben wird. Der Eintrag ist gerade deswegen sensibel, \
weil er Personen mit ihren Rollen koppelt.

## Kategorie `private_person`
  - Vor- + Nachname jeder genannten Person: "Reto Rubeli", "Anna Müller-Brunner"
  - Doppelnamen, Bindestriche, mehrere Vornamen: "Marie-Claude Bossy", \
"Hans-Peter Schmid", "Raquel Alves de Oliveira Duarte"
  - Auch wenn nur ein Name auftritt: "Kollektivunterschrift zu zweien mit Müller"
  - NEGATIV: Firmennamen ("Rubeli Bau Holding GmbH"), generische Rollen \
ohne Namen ("der Verwaltungsrat")

## Kategorie `private_address`
  - Wohnort einer genannten Person: "Reto Rubeli, von Bern in Zürich" → \
"Bern" und "Zürich" als Adressen
  - Vollständige Postadressen: "Bahnhofstrasse 1, 8001 Zürich"
  - NEGATIV: Sitz der Firma ohne Personenbezug ("mit Sitz in Bern")

## Kategorie `account_number`
  - UID/MWST-Nummer der Firma: "CHE-442.474.400"
  - IBAN, AHV-Nummer, Pass-Nr.

Andere Kategorien (Datum, E-Mail, Telefon, URL, Geheimnis) wie üblich.

## Ausgabeformat
JSON `{"spans": [{"value": "<wortgetreuer Teilstring>", "label": "<Kategorie>"}, ...]}`. \
Leere Liste `{"spans": []}` nur wenn der Text wirklich keine PII enthält.
"""

_FR_CR_RULES = """Tu es un extracteur précis de PII suisses. Le texte suivant \
provient d'un extrait du registre du commerce suisse (Zefix, fiche d'entreprise \
ou liste similaire). Ces textes suivent un format fixe: «Personnes ayant le \
pouvoir de décision: <nom> (<rôle>) et <nom> (<rôle>)…».

IMPORTANT pour ce type de texte: TOUTE personne physique nommément \
mentionnée est une PII et DOIT être extraite — même quand son rôle \
(administrateur, directeur, gérant, président, membre du conseil) est \
décrit de façon générique. L'entrée est sensible précisément parce \
qu'elle associe des personnes à leurs rôles.

## Catégorie `private_person`
  - Prénom + nom de famille de chaque personne mentionnée: "François Roullet", \
"Karim Lassueur"
  - Noms composés, traits d'union, plusieurs prénoms: "Jean-Pierre Müller", \
"Marie-Claude Bossy"
  - NÉGATIF: noms d'entreprises ("Rubeli Bau Holding SA"), rôles génériques \
sans nom ("le conseil d'administration")

## Catégorie `private_address`
  - Domicile d'une personne nommée: "François Roullet, de Genève à Lausanne" → \
"Genève" et "Lausanne" comme adresses
  - Adresses postales complètes: "Avenue de la Gare 12, 1003 Lausanne"
  - NÉGATIF: siège de l'entreprise sans lien personnel

## Catégorie `account_number`
  - IDE/numéro TVA de l'entreprise: "CHE-442.474.400"
  - IBAN, numéro AVS, numéro de passeport

Autres catégories (date, e-mail, téléphone, URL, secret) comme d'habitude.

## Format de sortie
JSON `{"spans": [{"value": "<sous-chaîne textuelle>", "label": "<catégorie>"}, ...]}`. \
Liste vide `{"spans": []}` uniquement si le texte ne contient vraiment aucune PII.
"""

_IT_CR_RULES = """Sei un estrattore preciso di PII svizzere. Il testo seguente \
proviene da un estratto del registro di commercio svizzero (Zefix, scheda \
aziendale o elenco simile). Questi testi seguono un formato fisso: «Persone \
con potere decisionale: <nome> (<ruolo>) e <nome> (<ruolo>)…».

IMPORTANTE per questo tipo di testo: OGNI persona fisica menzionata per \
nome è PII e DEVE essere estratta — anche quando il suo ruolo \
(amministratore, direttore, presidente, gerente, membro del consiglio) \
è descritto in modo generico.

## Categoria `private_person`
  - Nome + cognome di ogni persona menzionata: "Claudio Marone", "Carlo Cattaneo"
  - Nomi composti, trattini, più nomi: "Marie-Claude Bossy"
  - NEGATIVO: nomi di aziende, ruoli generici senza nome ("il consiglio di \
amministrazione")

## Categoria `private_address`
  - Luogo di residenza di una persona nominata: "Claudio Marone, da Bellinzona, \
in Muralto" → "Bellinzona" e "Muralto" come indirizzi
  - Indirizzi postali completi
  - NEGATIVO: sede dell'azienda senza riferimento personale

## Categoria `account_number`
  - IDI/numero IVA dell'azienda: "CHE-442.474.400"
  - IBAN, numero AVS, numero di passaporto

Altre categorie (data, e-mail, telefono, URL, segreto) come al solito.

## Formato di output
JSON `{"spans": [{"value": "<sottostringa esatta>", "label": "<categoria>"}, ...]}`. \
Lista vuota `{"spans": []}` solo se il testo non contiene davvero PII.
"""

_EN_CR_RULES = """You are a precise Swiss PII extractor. The following text \
is from a Swiss commercial-register listing (Zefix entry, company profile, \
or similar directory). These texts follow a fixed pattern: "Persons \
authorised to take decisions: <name> (<role>) and <name> (<role>)…".

IMPORTANT for this text type: EVERY natural person named is PII and MUST \
be extracted — even when their role (board member, managing director, \
president, partner, signatory) is described generically. The entry is \
sensitive precisely because it pairs persons with their roles.

## Category `private_person`
  - First + last name of every person mentioned
  - Compound names, hyphens, multiple given names
  - NEGATIVE: company names, generic roles without a name ("the board")

## Category `private_address`
  - Place of residence of a named person
  - Full postal addresses
  - NEGATIVE: company seat without personal connection

## Category `account_number`
  - UID/VAT number of the company
  - IBAN, social-security number, passport number

Other categories (date, e-mail, phone, URL, secret) as usual.

## Output format
JSON `{"spans": [{"value": "<verbatim substring>", "label": "<category>"}, ...]}`. \
Empty list `{"spans": []}` only if the text really contains no PII.
"""

_RULES_BY_LANG: dict[PromptLang, str] = {
    "de_ch": _DE_CR_RULES,
    "fr_ch": _FR_CR_RULES,
    "it_ch": _IT_CR_RULES,
    "en": _EN_CR_RULES,
    "rm": _DE_CR_RULES,  # RM falls back to DE rules — same fallback as the standard prompt
}


# Few-shot demos for each language. Just one per language is enough —
# the rule text already does most of the work, the demo just shows the
# JSON output format.
_DE_DEMOS = [{
    "user": (
        "Das Unternehmen Beispiel Holding AG mit Sitz in Zürich wurde "
        "im Jahr 2020 gegründet. Personen mit Entscheidungsbefugnis: "
        "Anna Müller-Brunner (Verwaltungsratspräsidentin) und Hans Peter Schmid "
        "(Geschäftsführer). UID: CHE-123.456.789."
    ),
    "assistant": (
        '{"spans":['
        '{"value":"Anna Müller-Brunner","label":"private_person"},'
        '{"value":"Hans Peter Schmid","label":"private_person"},'
        '{"value":"CHE-123.456.789","label":"account_number"}'
        ']}'
    ),
}]
_FR_DEMOS = [{
    "user": (
        "La société Exemple Holding SA avec siège à Genève a été fondée "
        "en 2020. Personnes ayant le pouvoir de décision: Marie-Claude "
        "Bossy (présidente du conseil d'administration) et Jean-Pierre "
        "Roulet (directeur). IDE: CHE-987.654.321."
    ),
    "assistant": (
        '{"spans":['
        '{"value":"Marie-Claude Bossy","label":"private_person"},'
        '{"value":"Jean-Pierre Roulet","label":"private_person"},'
        '{"value":"CHE-987.654.321","label":"account_number"}'
        ']}'
    ),
}]
_IT_DEMOS = [{
    "user": (
        "La società Esempio Holding SA con sede a Lugano è stata fondata "
        "nel 2020. Persone con potere decisionale: Claudio Marone "
        "(presidente del consiglio di amministrazione) e Carlo Cattaneo "
        "(amministratore). IDI: CHE-555.123.456."
    ),
    "assistant": (
        '{"spans":['
        '{"value":"Claudio Marone","label":"private_person"},'
        '{"value":"Carlo Cattaneo","label":"private_person"},'
        '{"value":"CHE-555.123.456","label":"account_number"}'
        ']}'
    ),
}]
_EN_DEMOS = [{
    "user": (
        "The company Example Holding Ltd, headquartered in Zurich, was "
        "founded in 2020. Persons authorised to take decisions: Anna "
        "Mueller-Brunner (chair of the board) and John Peter Schmid "
        "(managing director). UID: CHE-123.456.789."
    ),
    "assistant": (
        '{"spans":['
        '{"value":"Anna Mueller-Brunner","label":"private_person"},'
        '{"value":"John Peter Schmid","label":"private_person"},'
        '{"value":"CHE-123.456.789","label":"account_number"}'
        ']}'
    ),
}]
_DEMOS_BY_LANG: dict[PromptLang, list[dict]] = {
    "de_ch": _DE_DEMOS,
    "fr_ch": _FR_DEMOS,
    "it_ch": _IT_DEMOS,
    "en": _EN_DEMOS,
    "rm": _DE_DEMOS,
}


def _resolve_lang(language: Language) -> PromptLang:
    if language in ("de_ch", "fr_ch", "it_ch", "en", "rm"):
        return language  # type: ignore[return-value]
    if language == "gsw":
        return "de_ch"
    return "en"


def build_cr_messages(chunk_text: str, language: Language = "de_ch") -> list[dict]:
    """Chat-template messages for the commercial-register-aware
    re-label pass. Output schema identical to the standard prompt."""
    lang = _resolve_lang(language)
    msgs: list[dict] = [{"role": "system", "content": _RULES_BY_LANG[lang]}]
    for ex in _DEMOS_BY_LANG[lang]:
        msgs.append({"role": "user", "content": _user_msg(ex["user"])})
        msgs.append({"role": "assistant", "content": ex["assistant"]})
    msgs.append({"role": "user", "content": _user_msg(chunk_text)})
    return msgs
