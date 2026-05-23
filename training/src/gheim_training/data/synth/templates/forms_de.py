"""German form/list/CSV-style chunks. Hand-written, 20 entries.
Each presents PII in a structured layout (key:value, table, vCard, YAML, etc.)
rather than natural prose."""
from ..core import Fragment

TEMPLATES: list[Fragment] = [
    # --- 01: classic vertical key:value form ---
    Fragment("form_de_01", "de_ch",
             "Name: [[PII:full]]\n"
             "Telefon: [[PII:phone]]\n"
             "E-Mail: [[PII:email]]\n"
             "Adresse: [[PII:address]]\n"
             "Geburtsdatum: [[PII:date/dot]]"),

    # --- 02: single pipe-separated table row ---
    Fragment("form_de_02", "de_ch",
             "Kontaktperson | [[PII:title_first_last]] | "
             "[[PII:phone]] | [[PII:email]] | [[PII:url]]"),

    # --- 03: CSV with header line and one data row ---
    Fragment("form_de_03", "de_ch",
             "name,telefon,email,iban\n"
             "[[PII:full]],[[PII:phone]],[[PII:email]],[[PII:iban]]"),

    # --- 04: vCard-style block ---
    Fragment("form_de_04", "de_ch",
             "BEGIN:VCARD\n"
             "VERSION:3.0\n"
             "FN:[[PII:full]]\n"
             "TEL;TYPE=CELL:[[PII:phone]]\n"
             "EMAIL:[[PII:email]]\n"
             "ADR:;;[[PII:address]]\n"
             "END:VCARD"),

    # --- 05: YAML list-item entry ---
    Fragment("form_de_05", "de_ch",
             "- kunde:\n"
             "    name: [[PII:full]]\n"
             "    telefon: [[PII:phone]]\n"
             "    email: [[PII:email]]\n"
             "    geburtstag: [[PII:date/iso]]"),

    # --- 06: pseudo-JSON without quotes ---
    Fragment("form_de_06", "de_ch",
             "{ name: [[PII:full]], "
             "telefon: [[PII:phone]], "
             "email: [[PII:email]], "
             "ahv: [[PII:ahv]] }"),

    # --- 07: bullet list with dash labels ---
    Fragment("form_de_07", "de_ch",
             "Notizen zum Termin:\n"
             "- Name: [[PII:title_first_last]]\n"
             "- Telefon: [[PII:phone]]\n"
             "- Mail: [[PII:email]]\n"
             "- Wohnort: [[PII:address]]"),

    # --- 08: numbered enumeration ---
    Fragment("form_de_08", "de_ch",
             "Teilnehmerliste Workshop:\n"
             "1. [[PII:full]] – [[PII:email]]\n"
             "2. [[PII:full]] – [[PII:email]]\n"
             "3. [[PII:full]] – [[PII:email]]"),

    # --- 09: tab-separated columns ---
    Fragment("form_de_09", "de_ch",
             "Vorname\tNachname\tTelefon\tIBAN\n"
             "[[PII:first]]\t[[PII:last]]\t[[PII:phone]]\t[[PII:iban]]"),

    # --- 10: HTML-table-ish cells ---
    Fragment("form_de_10", "de_ch",
             "<tr><td>[[PII:full]]</td>"
             "<td>[[PII:phone]]</td>"
             "<td>[[PII:email]]</td>"
             "<td>[[PII:address]]</td></tr>"),

    # --- 11: formal letter header style ---
    Fragment("form_de_11", "de_ch",
             "Anrede: Sehr geehrte Damen und Herren\n"
             "Kunde: [[PII:title_first_last]]\n"
             "Kundennummer: [[PII:vat]]\n"
             "Konto: [[PII:iban]]\n"
             "Datum: [[PII:date/dot]]"),

    # --- 12: Excel paste with embedded tabs across one row ---
    Fragment("form_de_12", "de_ch",
             "[[PII:last_first]]\t[[PII:address]]\t"
             "[[PII:phone]]\t[[PII:email]]\t[[PII:date/dot]]"),

    # --- 13: markdown table (header + separator + row) ---
    Fragment("form_de_13", "de_ch",
             "| Feld     | Wert                |\n"
             "|----------|---------------------|\n"
             "| Name     | [[PII:full]]        |\n"
             "| Telefon  | [[PII:phone]]       |\n"
             "| E-Mail   | [[PII:email]]       |\n"
             "| AHV-Nr.  | [[PII:ahv]]         |"),

    # --- 14: Slack/Discord card with markdown emphasis ---
    Fragment("form_de_14", "de_ch",
             "Neuer Kontakt hinzugefügt:\n"
             "Name **[[PII:full]]**\n"
             "Telefon _[[PII:phone]]_\n"
             "E-Mail `[[PII:email]]`\n"
             "Profil <[[PII:url]]>"),

    # --- 15: INI-config block ---
    Fragment("form_de_15", "de_ch",
             "[kontakt]\n"
             "name=[[PII:full]]\n"
             "telefon=[[PII:phone]]\n"
             "email=[[PII:email]]\n"
             "iban=[[PII:iban]]\n"
             "geburtsdatum=[[PII:date/iso]]"),

    # --- 16: two-column letterhead, sender vs date ---
    Fragment("form_de_16", "de_ch",
             "Absender: [[PII:full]]\t\tDatum: [[PII:date/dot]]\n"
             "Adresse:  [[PII:address]]\t\tTel: [[PII:phone]]"),

    # --- 17: bilingual DE/EN receipt label ---
    Fragment("form_de_17", "de_ch",
             "Empfänger / Recipient: [[PII:full]]\n"
             "Anschrift / Address:   [[PII:address]]\n"
             "Telefon / Phone:       [[PII:phone]]\n"
             "MwSt / VAT:            [[PII:vat]]"),

    # --- 18: address block, postal layout ---
    Fragment("form_de_18", "de_ch",
             "[[PII:full]]\n"
             "c/o Firma Sägesser AG\n"
             "[[PII:address]]\n"
             "Tel. [[PII:phone]]\n"
             "[[PII:email]]"),

    # --- 19: bare mailing label / sticker — no key labels at all ---
    Fragment("form_de_19", "de_ch",
             "[[PII:full]]\n"
             "z. Hd. Personalabteilung\n"
             "[[PII:address]]\n"
             "Schweiz\n"
             "[[PII:phone]]\n"
             "[[PII:email]]"),

    # --- 20: cheque payee field with IBAN ---
    Fragment("form_de_20", "de_ch",
             "Bezahlen an: [[PII:full]]\n"
             "IBAN:        [[PII:iban]]\n"
             "Betrag fällig am: [[PII:date/dot]]\n"
             "Referenz Kreditkarte: [[PII:cc]]"),
]
