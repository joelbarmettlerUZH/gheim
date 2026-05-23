"""German email body fragments — banking/finance/payment domain. Hand-written, 20 entries."""
from ..core import Fragment

TEMPLATES: list[Fragment] = [
    Fragment("body_de_bank_01", "de_ch",
             "bitte überweisen Sie den ausstehenden Rechnungsbetrag auf unsere neue IBAN "
             "[[PII:iban]] bis spätestens [[PII:date]]."),
    Fragment("body_de_bank_02", "de_ch",
             "meine AHV-Nummer lautet [[PII:ahv]] – bitte korrigieren Sie den "
             "Eintrag im Kundenstamm von [[PII:full]] bei der nächsten Buchung."),
    Fragment("body_de_bank_03", "de_ch",
             "die Lastschrift über CHF 1'248.50 wurde heute Morgen erfolgreich "
             "ausgeführt; eine Bestätigung geht an [[PII:email]]."),
    Fragment("body_de_bank_04", "de_ch",
             "für die Mehrwertsteuerabrechnung benötigen wir noch Ihre UID-Nummer; "
             "gemäss Handelsregister wäre dies [[PII:vat]]."),
    Fragment("body_de_bank_05", "de_ch",
             "leider wurde Ihre Kreditkarte [[PII:cc]] beim Buchungsversuch "
             "abgelehnt – bitte rufen Sie uns unter [[PII:phone]] zurück."),
    Fragment("body_de_bank_06", "de_ch",
             "[[PII:title_last]] hat den Dauerauftrag auf IBAN [[PII:iban]] "
             "freigegeben; die erste Ausführung erfolgt am Monatsende."),
    Fragment("body_de_bank_07", "de_ch",
             "wir bestätigen den Eingang Ihrer Mahnung; eine Rückerstattung "
             "über den vollen Betrag erreicht Sie bis [[PII:date]]."),
    Fragment("body_de_bank_08", "de_ch",
             "die Kontoauflösung wurde eingeleitet – das Restguthaben transferieren "
             "wir an [[PII:full]], Restbestand laut Auszug CHF 87.20."),
    Fragment("body_de_bank_09", "de_ch",
             "für die Auslandüberweisung nach Frankfurt benötige ich dringend "
             "die SWIFT-Bestätigung; meine Telefonnummer ist [[PII:phone]]."),
    Fragment("body_de_bank_10", "de_ch",
             "anbei der Kontoauszug per [[PII:date]]; bitte gegenzeichnen und "
             "an [[PII:email]] retournieren."),
    Fragment("body_de_bank_11", "de_ch",
             "ich beantrage hiermit die sofortige Sperrung meiner Maestro-Karte "
             "(Kartennummer [[PII:cc]]) wegen Verlustverdachts."),
    Fragment("body_de_bank_12", "de_ch",
             "die Hypothekenrate wird ab kommendem Quartal vom neuen Konto "
             "[[PII:iban]] eingezogen – bisheriges Konto bitte deaktivieren."),
    Fragment("body_de_bank_13", "de_ch",
             "betreffend die AHV-Beiträge von [[PII:title_first_last]]: wir haben "
             "den Saldo mit [[PII:ahv]] abgeglichen, die Differenz wird kommende Woche korrigiert."),
    Fragment("body_de_bank_14", "de_ch",
             "Ihre Zahlung über CHF 3'400.00 ist seit zwei Wochen überfällig; wir "
             "bitten um Begleichung bis [[PII:date]], andernfalls erfolgt die Übergabe ans Inkasso."),
    Fragment("body_de_bank_15", "de_ch",
             "die Buchhaltung erreichen Sie unter [[PII:phone]] oder direkt per "
             "Mail; Anfragen zur UID [[PII:vat]] bitte schriftlich einreichen."),
    Fragment("body_de_bank_16", "de_ch",
             "der Devisenkurs wurde fixiert – Ihre Auszahlung erfolgt am [[PII:date]] "
             "auf das hinterlegte Konto [[PII:iban]] bei der Kantonalbank."),
    Fragment("body_de_bank_17", "de_ch",
             "die Rechnungsadresse für die Kreditkartenauszüge ändert sich per "
             "1.6. auf [[PII:address]]; bitte im System hinterlegen."),
    Fragment("body_de_bank_18", "de_ch",
             "anbei der Zahlungsbeleg für die Akontorechnung vom [[PII:date]]; "
             "Rückfragen richten Sie bitte direkt an [[PII:first]]."),
    Fragment("body_de_bank_19", "de_ch",
             "wir haben die Stornierung der Daueraufträge per [[PII:date]] vorgemerkt "
             "und werden eine schriftliche Bestätigung an [[PII:address]] zustellen."),
    Fragment("body_de_bank_20", "de_ch",
             "die Freigabe für den E-Banking-Zugang wurde erteilt – das "
             "Initialpasswort senden wir separat per SMS an [[PII:phone]]."),
]
