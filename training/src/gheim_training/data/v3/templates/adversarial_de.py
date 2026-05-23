"""German adversarial-negative fragments. Hand-written, 35 entries.
Each contains text that LOOKS like PII but isn't (in context). The model
must learn to suppress false positives on these patterns.

Output has ZERO spans per fragment — the [[PII:...]] markers are NOT
used because there is nothing to mark. These chunks teach the detector
the surrounding context that disambiguates placeholders, codes, and
role nouns from real PII."""
from ..core import Fragment

TEMPLATES: list[Fragment] = [
    # ---- email lookalikes (placeholders, example domains, role mailboxes) ----
    Fragment("adv_de_email_01", "de_ch",
             "Beispiel: ersetze noreply@example.com durch deine "
             "eigene Adresse, bevor du das Formular abschickst."),
    Fragment("adv_de_email_02", "de_ch",
             "Die API-Dokumentation zeigt user@example.com lediglich als "
             "Platzhalter, nicht als echten Kontakt."),
    Fragment("adv_de_email_03", "de_ch",
             "In der Mustervorlage steht max.mustermann@firma.tld nur "
             "als Beispieleintrag und wird beim Druck ersetzt."),
    Fragment("adv_de_email_04", "de_ch",
             "Setze für lokale Tests die Variable MAIL_FROM auf "
             "dev@localhost und niemals auf eine produktive Adresse."),
    Fragment("adv_de_email_05", "de_ch",
             "Anfragen bitte nicht an info@example.org senden — das "
             "ist eine reservierte Beispieldomain ohne Empfänger."),
    Fragment("adv_de_email_06", "de_ch",
             "Im Tutorial wird foo@bar.test mehrfach erwähnt, dient "
             "aber ausschliesslich zur Veranschaulichung der Syntax."),

    # ---- phone lookalikes (order numbers, product codes, version strings) ----
    Fragment("adv_de_phone_01", "de_ch",
             "Die interne Bestellnummer 044 555 1234 ist für die "
             "Logistik reserviert und keine Telefonnummer."),
    Fragment("adv_de_phone_02", "de_ch",
             "Produktcode 41 22 9876 erscheint auf der Verpackung, hat "
             "aber nichts mit einer Rufnummer zu tun."),
    Fragment("adv_de_phone_03", "de_ch",
             "Auftrag 079 0001 2345 wurde im neuen ERP-System angelegt "
             "und gehört zur Charge vom Vormittag."),
    Fragment("adv_de_phone_04", "de_ch",
             "Die Seriennummer SN 044 778 9012 steht auf der Rückseite "
             "des Gerätes neben dem Stromanschluss."),
    Fragment("adv_de_phone_05", "de_ch",
             "Modell 058 200 1100 wurde im Katalog als Auslaufmodell "
             "markiert und nicht mehr nachbestellt."),
    Fragment("adv_de_phone_06", "de_ch",
             "Im Lager ist Regalplatz 031 405 22 belegt; bitte vor "
             "der Einlagerung umsortieren."),

    # ---- date lookalikes (room numbers, version numbers, decimals) ----
    Fragment("adv_de_date_01", "de_ch",
             "Zimmer 30.06 liegt im dritten Stock und ist heute noch "
             "frei für eine spontane Besprechung."),
    Fragment("adv_de_date_02", "de_ch",
             "Version 1.2.3 der Bibliothek wird voraussichtlich "
             "nächste Woche freigegeben."),
    Fragment("adv_de_date_03", "de_ch",
             "Sitzungssaal 12.05 befindet sich im Erdgeschoss links "
             "neben dem Lift."),
    Fragment("adv_de_date_04", "de_ch",
             "Der Messwert 24.12 mg/l überschreitet den Grenzwert "
             "leicht und sollte erneut geprüft werden."),
    Fragment("adv_de_date_05", "de_ch",
             "Patch 03.11 behebt einen Speicherleck-Bug und ist in "
             "den Release-Notes ausführlich beschrieben."),

    # ---- account/IBAN lookalikes (reference codes, pseudo-IBANs) ----
    Fragment("adv_de_account_01", "de_ch",
             "Bestellnummer CH-2026-0001234 entspricht zwar formal "
             "einem IBAN-ähnlichen Muster, ist aber kein echtes Konto."),
    Fragment("adv_de_account_02", "de_ch",
             "Die Referenznummer 21 00000 00003 13947 143000 901 dient "
             "ausschliesslich der internen Buchungslogik."),
    Fragment("adv_de_account_03", "de_ch",
             "Vertragscode DE-AB-99-00-7766 wird in der Datenbank als "
             "technischer Schlüssel geführt, nicht als Bankverbindung."),
    Fragment("adv_de_account_04", "de_ch",
             "Im Pflichtenheft wird der Identifier LI91 0000 0000 "
             "0000 0000 0 als anonymisiertes Beispiel zitiert."),

    # ---- URL lookalikes (DOIs, RFCs, ISO/DIN standards) ----
    Fragment("adv_de_url_01", "de_ch",
             "Siehe Norm DIN EN ISO 9001:2015 für die formalen "
             "Anforderungen an das Qualitätsmanagement."),
    Fragment("adv_de_url_02", "de_ch",
             "Die DOI 10.1000/xyz123 verweist auf eine fiktive "
             "Publikation und dient nur zu Demozwecken."),
    Fragment("adv_de_url_03", "de_ch",
             "RFC 7231 beschreibt die Semantik der HTTP/1.1-Methoden "
             "und wird im Anhang ausführlich zitiert."),
    Fragment("adv_de_url_04", "de_ch",
             "NIST SP 800-53 listet die Sicherheitskontrollen auf, "
             "die im Audit nachgewiesen werden müssen."),

    # ---- person lookalikes (job titles / role nouns without specific name) ----
    Fragment("adv_de_person_01", "de_ch",
             "Der Bäcker des Quartiers öffnet jeden Morgen um sechs "
             "Uhr und schliesst spät am Nachmittag."),
    Fragment("adv_de_person_02", "de_ch",
             "Der Direktor hat die Sitzung kurzfristig abgesagt und "
             "auf nächste Woche verschoben."),
    Fragment("adv_de_person_03", "de_ch",
             "Ein erfahrener Lehrer kann diese Aufgabe in wenigen "
             "Minuten erklären, ohne lange auszuholen."),
    Fragment("adv_de_person_04", "de_ch",
             "Die Verkäuferin an der Kasse war ausgesprochen freundlich "
             "und hat geduldig alles erklärt."),
    Fragment("adv_de_person_05", "de_ch",
             "Der zuständige Sachbearbeiter meldet sich in den "
             "nächsten zwei Werktagen telefonisch zurück."),

    # ---- misc edge cases (CHF sums, year refs, brand names, decimals) ----
    Fragment("adv_de_misc_01", "de_ch",
             "Der Betrag von CHF 12'345.50 wurde fristgerecht "
             "überwiesen und im Hauptbuch korrekt verbucht."),
    Fragment("adv_de_misc_02", "de_ch",
             "Im Jahr 1984 erschien der gleichnamige Roman, der bis "
             "heute auf vielen Lehrplänen steht."),
    Fragment("adv_de_misc_03", "de_ch",
             "Die Konzentration liegt bei 3.14 Prozent, also leicht "
             "unter dem im Vorquartal gemessenen Wert."),
    Fragment("adv_de_misc_04", "de_ch",
             "Die Marke Migros bietet diese Woche eine Aktion auf "
             "regionale Bio-Produkte aus der Innerschweiz."),
    Fragment("adv_de_misc_05", "de_ch",
             "Im Geschäftsjahr 2024 wurde ein Umsatz von 8.7 Millionen "
             "Franken erwirtschaftet, ein leichter Rückgang."),
]
