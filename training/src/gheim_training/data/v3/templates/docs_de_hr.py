"""German HR document templates — job offers, certificates, salary slips,
reference letters. Hand-written, 12 entries. Each is a full document."""
from ..core import Fragment

TEMPLATES: list[Fragment] = [
    # 1. Job offer (Stellenangebot)
    Fragment("doc_de_hr_01", "de_ch",
             "STELLENANGEBOT\n\n"
             "Sehr geehrte Frau [[PII:last]],\n\n"
             "wir freuen uns sehr, Ihnen nach dem ausführlichen Gespräch vom "
             "[[PII:date/dot]] die Position als Senior Software Engineer in "
             "unserem Engineering-Team in Zürich anbieten zu dürfen.\n\n"
             "Eintrittsdatum: [[PII:date/spelled]]\n"
             "Bruttojahreslohn: CHF 132'000.00 (inkl. 13. Monatslohn)\n"
             "Arbeitspensum: 100 %\n"
             "Arbeitsort: [[PII:address]]\n"
             "Probezeit: 3 Monate\n\n"
             "Wir bitten Sie, das beigelegte Vertragsdoppel bis spätestens "
             "[[PII:date/dot]] unterzeichnet zu retournieren. Bei Rückfragen "
             "erreichen Sie mich direkt unter [[PII:phone]] oder per Mail an "
             "[[PII:email]].\n\n"
             "Mit freundlichen Grüssen\n\n"
             "[[PII:title_first_last]]\n"
             "Head of People & Culture"),

    # 2. Employment certificate (Arbeitszeugnis)
    Fragment("doc_de_hr_02", "de_ch",
             "ARBEITSZEUGNIS\n\n"
             "Herr [[PII:title_first_last]], geboren am [[PII:date/dot]], "
             "war vom [[PII:date/spelled]] bis [[PII:date/spelled]] in unserem "
             "Unternehmen als Projektleiter Bau tätig.\n\n"
             "Zu seinen Hauptaufgaben gehörten die Leitung komplexer Hochbau-"
             "Projekte im Grossraum Bern, die Koordination mit Bauherren und "
             "Subunternehmern sowie die Termin- und Kostenkontrolle. Herr "
             "[[PII:last]] erfüllte seine Aufgaben stets zu unserer vollsten "
             "Zufriedenheit. Seine fachliche Kompetenz, sein "
             "Verhandlungsgeschick und sein ausgesprochen kollegiales "
             "Verhalten haben wir besonders geschätzt.\n\n"
             "Das Arbeitsverhältnis endet auf eigenen Wunsch. Wir bedauern "
             "seinen Weggang und danken ihm für die geleistete Arbeit.\n\n"
             "Bern, [[PII:date/dot]]\n\n"
             "[[PII:title_first_last]], Geschäftsführer"),

    # 3. Salary slip / payroll (Lohnabrechnung)
    Fragment("doc_de_hr_03", "de_ch",
             "LOHNABRECHNUNG\n\n"
             "Mitarbeiter:     [[PII:full]]\n"
             "Personalnummer:  P-04217\n"
             "AHV-Nummer:      [[PII:ahv]]\n"
             "Abrechnungsmonat: [[PII:date/short]]\n\n"
             "Bruttolohn:                  CHF 8'750.00\n"
             "Familienzulage:              CHF   230.00\n"
             "-----------------------------------------\n"
             "Bruttototal:                 CHF 8'980.00\n\n"
             "AHV/IV/EO (5.3 %):          CHF  -475.94\n"
             "ALV (1.1 %):                CHF   -98.78\n"
             "Pensionskasse:              CHF  -612.30\n"
             "NBU-Versicherung:           CHF   -71.84\n"
             "Quellensteuer:              CHF  -445.00\n"
             "-----------------------------------------\n"
             "Nettolohn:                   CHF 7'276.14\n\n"
             "Auszahlung am [[PII:date/dot]] auf IBAN [[PII:iban]]."),

    # 4. Reference letter (Referenzschreiben)
    Fragment("doc_de_hr_04", "de_ch",
             "REFERENZSCHREIBEN\n\n"
             "Hiermit bestätige ich gerne, dass Frau [[PII:title_first_last]] "
             "vom [[PII:date/spelled]] bis [[PII:date/spelled]] als Marketing "
             "Managerin in meinem Team gearbeitet hat.\n\n"
             "Frau [[PII:last]] hat in dieser Zeit massgeblich zum Erfolg "
             "unserer Marken-Neulancierung beigetragen. Sie verfügt über ein "
             "sehr fundiertes Verständnis für Markenführung, arbeitet "
             "äusserst strukturiert und besitzt eine ausgeprägte Fähigkeit, "
             "auch komplexe Stakeholder-Konstellationen souverän zu "
             "moderieren. Ihre Kreativität wie auch ihr analytisches Denken "
             "haben mich wiederholt beeindruckt.\n\n"
             "Ich empfehle Frau [[PII:last]] uneingeschränkt weiter und stehe "
             "für mündliche Auskünfte gerne unter [[PII:phone]] zur "
             "Verfügung.\n\n"
             "[[PII:title_first_last]]\n"
             "ehem. Marketing Director, [[PII:email]]"),

    # 5. Termination notice (Kündigungsschreiben)
    Fragment("doc_de_hr_05", "de_ch",
             "Einschreiben\n\n"
             "[[PII:title_first_last]]\n"
             "[[PII:address]]\n\n"
             "Luzern, [[PII:date/dot]]\n\n"
             "Kündigung des Arbeitsverhältnisses\n\n"
             "Sehr geehrte Frau [[PII:last]],\n\n"
             "nach reiflicher Überlegung müssen wir Ihnen mitteilen, dass wir "
             "Ihr Arbeitsverhältnis unter Einhaltung der vertraglichen "
             "Kündigungsfrist von drei Monaten per [[PII:date/spelled]] "
             "auflösen.\n\n"
             "Die Restferien (8.5 Tage) sind während der Kündigungsfrist zu "
             "beziehen. Ein qualifiziertes Arbeitszeugnis stellen wir Ihnen "
             "auf Wunsch gerne aus. Für ein persönliches Gespräch steht Ihnen "
             "Herr [[PII:title_last]] aus der HR-Abteilung unter "
             "[[PII:phone]] jederzeit zur Verfügung.\n\n"
             "Wir danken Ihnen für Ihr Engagement und wünschen Ihnen für "
             "Ihren weiteren beruflichen Weg alles Gute.\n\n"
             "Freundliche Grüsse\n\n"
             "[[PII:title_first_last]], CEO"),

    # 6. Sick leave certificate (Krankheits-Attest)
    Fragment("doc_de_hr_06", "de_ch",
             "ARZTZEUGNIS\n\n"
             "Dr. med. [[PII:title_first_last]]\n"
             "Praxis am Bahnhofplatz\n"
             "[[PII:address]]\n"
             "Tel. [[PII:phone]]\n\n"
             "Hiermit bestätige ich, dass mein/e Patient/in\n\n"
             "Name: [[PII:full]]\n"
             "Geburtsdatum: [[PII:date/dot]]\n\n"
             "aus medizinischen Gründen vom [[PII:date/dot]] bis "
             "voraussichtlich [[PII:date/dot]] zu 100 % arbeitsunfähig ist.\n\n"
             "Eine Verlängerung ist je nach Verlauf möglich; eine "
             "Nachkontrolle ist auf den [[PII:date/dot]] vereinbart.\n\n"
             "Zürich, [[PII:date/dot]]\n\n"
             "(Unterschrift und Stempel)"),

    # 7. Vacation request approval (Urlaubsbestätigung)
    Fragment("doc_de_hr_07", "de_ch",
             "URLAUBSBESTÄTIGUNG\n\n"
             "Sehr geehrte/r [[PII:title_last]]\n\n"
             "Hiermit bestätigen wir Ihren Ferienbezug für das laufende "
             "Geschäftsjahr wie folgt:\n\n"
             "Bezugsperiode:    [[PII:date/dot]] bis [[PII:date/dot]]\n"
             "Anzahl Tage:      14 Arbeitstage\n"
             "Restguthaben:     6.5 Tage (Stichtag 31.12.)\n"
             "Stellvertretung:  [[PII:full]]\n\n"
             "Wir bitten Sie, vor Abwesenheit eine Übergabenotiz im "
             "Team-Wiki zu hinterlegen und Ihre Out-of-Office-Meldung mit "
             "der Kontaktadresse [[PII:email]] zu aktivieren.\n\n"
             "Wir wünschen Ihnen erholsame Ferien.\n\n"
             "HR Team — [[PII:title_first_last]]"),

    # 8. Performance review (Mitarbeitergespräch-Protokoll)
    Fragment("doc_de_hr_08", "de_ch",
             "PROTOKOLL JAHRESGESPRÄCH\n\n"
             "Datum:       [[PII:date/dot]]\n"
             "Mitarbeitende: [[PII:full]]\n"
             "Vorgesetzte/r: [[PII:title_first_last]]\n"
             "Funktion:    Senior Data Analyst\n"
             "Eintritt:    [[PII:date/dot]]\n\n"
             "Rückblick: Die Leistungen im vergangenen Jahr lagen klar über "
             "den vereinbarten Zielen. Besonders hervorzuheben ist die "
             "erfolgreiche Einführung des neuen Reporting-Dashboards, das "
             "konzernweit ausgerollt wurde.\n\n"
             "Entwicklungsziele für das kommende Jahr:\n"
             "  - Vertiefung der Kompetenzen in MLOps\n"
             "  - Übernahme der fachlichen Führung des Junior-Teams\n"
             "  - Besuch des CAS Data Engineering (Start [[PII:date/spelled]])\n\n"
             "Gesamtbeurteilung: A (übertrifft die Erwartungen)\n\n"
             "Nächster Gesprächstermin: [[PII:date/dot]]"),

    # 9. Onboarding checklist (Onboarding-Liste)
    Fragment("doc_de_hr_09", "de_ch",
             "ONBOARDING-CHECKLISTE\n\n"
             "Neue/r Mitarbeitende/r: [[PII:full]]\n"
             "Eintrittsdatum:         [[PII:date/dot]]\n"
             "Funktion:               Junior Consultant\n"
             "Buddy:                  [[PII:title_first_last]]\n"
             "Vorgesetzte/r:          [[PII:title_first_last]]\n\n"
             "Vor dem ersten Tag:\n"
             "  [ ] Arbeitsvertrag retour erhalten\n"
             "  [ ] AHV-Nummer in SAP erfasst ([[PII:ahv]])\n"
             "  [ ] Lohnkonto IBAN [[PII:iban]] hinterlegt\n"
             "  [ ] Laptop, Badge und Mobile-Abo bestellt\n"
             "  [ ] Mailbox [[PII:email]] eingerichtet\n\n"
             "Erster Arbeitstag:\n"
             "  [ ] Empfang um 08:30 durch HR (Sitzungszimmer Matterhorn)\n"
             "  [ ] Hausführung & Sicherheitsinstruktion\n"
             "  [ ] Vorstellung im Team-Stand-up um 10:00\n\n"
             "Bei Fragen vor Eintritt: HR-Hotline [[PII:phone]]"),

    # 10. Pension statement (Pensionskassen-Ausweis)
    Fragment("doc_de_hr_10", "de_ch",
             "PERSÖNLICHER VORSORGEAUSWEIS\n\n"
             "Versicherte Person:  [[PII:full]]\n"
             "AHV-Nummer:          [[PII:ahv]]\n"
             "Geburtsdatum:        [[PII:date/dot]]\n"
             "Eintritt Vorsorge:   [[PII:date/dot]]\n"
             "Stichtag:            [[PII:date/dot]]\n\n"
             "Versicherter Jahreslohn:        CHF 102'400.00\n"
             "Koordinationsabzug:             CHF  25'725.00\n"
             "Versicherter Lohn BVG:          CHF  76'675.00\n\n"
             "Sparguthaben per Stichtag:      CHF 187'432.65\n"
             "  davon obligatorisch (Schub):  CHF 121'080.10\n"
             "  davon überobligatorisch:      CHF  66'352.55\n\n"
             "Voraussichtliche Altersrente ab 65:  CHF 38'714.00/Jahr\n"
             "Voraussichtliches Alterskapital:     CHF 612'500.00\n\n"
             "Rückfragen an die Vorsorgestiftung: [[PII:phone]] oder "
             "[[PII:email]]."),

    # 11. Work permit / residence permit (Arbeits-/Aufenthaltsbewilligung)
    Fragment("doc_de_hr_11", "de_ch",
             "ARBEITSBEWILLIGUNG B (EU/EFTA)\n\n"
             "Kanton St. Gallen — Migrationsamt\n\n"
             "Name, Vorname:    [[PII:last_first]]\n"
             "Geburtsdatum:     [[PII:date/dot]]\n"
             "Staatsangehörigkeit: Deutschland\n"
             "Wohnadresse CH:   [[PII:address]]\n\n"
             "Bewilligungstyp:  B EU/EFTA (Erwerbstätigkeit)\n"
             "Gültig von:       [[PII:date/dot]]\n"
             "Gültig bis:       [[PII:date/dot]]\n\n"
             "Arbeitgeber:      Helvetia Engineering AG, Rorschach\n"
             "Funktion:         Maschinenbau-Ingenieur (100 %)\n"
             "Eintrittsdatum:   [[PII:date/dot]]\n\n"
             "Der/Die Bewilligungsinhaber/in ist verpflichtet, jede Adress- "
             "oder Stellenänderung innert 14 Tagen dem Migrationsamt zu "
             "melden. Eine Verlängerung ist mindestens drei Monate vor "
             "Ablauf zu beantragen.\n\n"
             "Sachbearbeitung: [[PII:title_first_last]] — [[PII:phone]]"),

    # 12. Internal transfer letter (Versetzungsschreiben)
    Fragment("doc_de_hr_12", "de_ch",
             "INTERNE VERSETZUNG\n\n"
             "Sehr geehrter Herr [[PII:last]],\n\n"
             "wir freuen uns, Ihnen mitteilen zu können, dass die in unseren "
             "Gesprächen vom [[PII:date/dot]] besprochene interne Versetzung "
             "wie geplant umgesetzt wird.\n\n"
             "Bisherige Funktion: Spezialist Treasury, Hauptsitz Zürich\n"
             "Neue Funktion:      Head of Treasury Operations\n"
             "Neuer Arbeitsort:   [[PII:address]]\n"
             "Wirksam ab:         [[PII:date/spelled]]\n"
             "Neuer Vorgesetzter: [[PII:title_first_last]]\n\n"
             "Ihr Grundlohn erhöht sich per Versetzungsdatum auf CHF "
             "168'000.00 brutto pro Jahr; die übrigen Anstellungs-"
             "bedingungen bleiben unverändert. Eine angepasste Vertragsbeilage "
             "geht Ihnen separat zu.\n\n"
             "Für die Übergangsphase steht Ihnen die HR-Businesspartnerin "
             "Frau [[PII:title_last]] (Direktwahl [[PII:phone]], Mail "
             "[[PII:email]]) gerne zur Verfügung. Wir wünschen Ihnen für "
             "Ihre neue Aufgabe viel Erfolg und Freude.\n\n"
             "Freundliche Grüsse\n\n"
             "[[PII:title_first_last]]\n"
             "Chief Financial Officer"),
]
