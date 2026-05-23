"""German medical/insurance document templates — prescriptions, lab results,
admissions, insurance letters. Hand-written, 12 entries. Each is a full document
with PII fields."""
from ..core import Fragment

TEMPLATES: list[Fragment] = [
    # 1. Rezept (Prescription)
    Fragment("doc_de_med_01", "de_ch",
             "REZEPT\n\n"
             "Praxis Dr. med. Bircher · Kreuzplatz 4 · 8032 Zürich\n"
             "Tel. 044 252 18 90\n\n"
             "Patient: [[PII:full]]\n"
             "Geboren: [[PII:date/dot]]\n"
             "AHV-Nr.: [[PII:ahv]]\n"
             "Wohnhaft: [[PII:address]]\n\n"
             "Verschrieben:\n"
             "  - Amoxicillin 1 g, 1-0-1, während 7 Tagen\n"
             "  - Dafalgan 500 mg bei Bedarf, max. 4x täglich\n\n"
             "Bitte Medikamente in der Apotheke einlösen. Kontrolle in 10 Tagen.\n\n"
             "Zürich, [[PII:date]]\n"
             "[[PII:title_first_last]]"),

    # 2. Arztzeugnis / Krankheitsattest (Sick leave)
    Fragment("doc_de_med_02", "de_ch",
             "ARZTZEUGNIS\n\n"
             "Hiermit bestätige ich, dass\n\n"
             "    [[PII:full]], geboren am [[PII:date/spelled]],\n"
             "    wohnhaft [[PII:address]],\n\n"
             "aus gesundheitlichen Gründen vom [[PII:date/dot]] bis voraussichtlich "
             "[[PII:date/dot]] zu 100 % arbeitsunfähig ist.\n\n"
             "Eine Verlängerung wird nach erneuter Konsultation in Aussicht gestellt. "
             "Dieses Zeugnis ist ausschliesslich für die Vorlage beim Arbeitgeber bestimmt.\n\n"
             "Bern, [[PII:date]]\n\n"
             "[[PII:title_first_last]]\n"
             "FMH Allgemeine Innere Medizin"),

    # 3. Laborbefund (Lab result)
    Fragment("doc_de_med_03", "de_ch",
             "LABORBEFUND\n"
             "Auftraggeber: [[PII:title_last]]\n"
             "Patient/in: [[PII:full]] · Geb. [[PII:date/dot]]\n"
             "AHV: [[PII:ahv]]\n"
             "Probeneingang: [[PII:date/dot]], 09:42 Uhr\n"
             "Befund vom [[PII:date/dot]]\n\n"
             "Hämatologie:\n"
             "  Hämoglobin    142 g/l    (Norm 135-170)\n"
             "  Leukozyten    6.8 G/l    (Norm 4.0-10.0)\n"
             "  Thrombozyten  238 G/l    (Norm 150-400)\n\n"
             "Chemie:\n"
             "  Kreatinin     78 µmol/l  (Norm 60-110)\n"
             "  CRP           <3 mg/l    (Norm <5)\n\n"
             "Befund: Sämtliche Werte im Referenzbereich. Keine "
             "Auffälligkeiten.\n\n"
             "Rückfragen an Labor Risch, [[PII:phone]] oder [[PII:email]]."),

    # 4. Aufnahmeformular (Hospital admission)
    Fragment("doc_de_med_04", "de_ch",
             "STATIONÄRE AUFNAHME · Universitätsspital\n"
             "Aufnahmedatum: [[PII:date/dot]], 14:20 Uhr\n"
             "Klinik: Innere Medizin, Bettenstation C3\n\n"
             "PERSONALIEN\n"
             "Name, Vorname: [[PII:last_first]]\n"
             "Geburtsdatum:  [[PII:date/dot]]\n"
             "AHV-Nummer:    [[PII:ahv]]\n"
             "Adresse:       [[PII:address]]\n"
             "Telefon:       [[PII:phone]]\n"
             "E-Mail:        [[PII:email]]\n\n"
             "ANGEHÖRIGE / KONTAKTPERSON\n"
             "  [[PII:full]], erreichbar unter [[PII:phone]]\n\n"
             "Zuweisender Arzt: [[PII:title_first_last]]\n"
             "Einweisungsdiagnose: V.a. akute Cholezystitis\n\n"
             "Aufnahme bestätigt durch das Sekretariat."),

    # 5. Austrittsbericht (Discharge letter)
    Fragment("doc_de_med_05", "de_ch",
             "AUSTRITTSBERICHT\n\n"
             "Sehr geehrte Kollegin, sehr geehrter Kollege\n\n"
             "Wir berichten über unsere/n gemeinsame/n Patient/in\n\n"
             "    [[PII:full]], geboren [[PII:date/dot]]\n"
             "    [[PII:address]]\n\n"
             "Eintritt: [[PII:date/dot]] · Austritt: [[PII:date/dot]]\n\n"
             "Diagnosen:\n"
             "  1. Pneumonie linker Unterlappen, Erreger nicht nachgewiesen\n"
             "  2. Arterielle Hypertonie, medikamentös eingestellt\n\n"
             "Therapie: Antibiotische Behandlung mit Co-Amoxicillin i.v. während "
             "fünf Tagen, anschliessend orale Umstellung. Klinische und "
             "laborchemische Besserung im Verlauf.\n\n"
             "Procedere: Nachkontrolle in Ihrer Sprechstunde in zwei Wochen. "
             "Bei Bedarf Rücksprache mit dem unterzeichnenden Oberarzt unter "
             "[[PII:phone]].\n\n"
             "Mit freundlichen Grüssen\n"
             "[[PII:title_first_last]]"),

    # 6. Schadenmeldung Krankenkasse (Insurance claim notification)
    Fragment("doc_de_med_06", "de_ch",
             "SCHADENMELDUNG GRUNDVERSICHERUNG (KVG)\n\n"
             "Versicherte Person\n"
             "  Name:        [[PII:full]]\n"
             "  Geboren am:  [[PII:date/dot]]\n"
             "  AHV-Nummer:  [[PII:ahv]]\n"
             "  Adresse:     [[PII:address]]\n\n"
             "Ereignis\n"
             "  Datum:       [[PII:date/dot]]\n"
             "  Ort:         Wanderweg Bachalpsee, Grindelwald\n"
             "  Hergang:     Sturz beim Abstieg, Verdacht auf Bänderriss am rechten "
             "Sprunggelenk; Erstversorgung durch die Rega.\n\n"
             "Behandelnder Arzt: [[PII:title_first_last]], "
             "erreichbar unter [[PII:phone]]\n\n"
             "Bitte überweisen Sie allfällige Rückerstattungen auf das Konto "
             "[[PII:iban]] lautend auf die versicherte Person.\n\n"
             "[[PII:address]], [[PII:date]]"),

    # 7. Krankenkassenrechnung (Insurance invoice)
    Fragment("doc_de_med_07", "de_ch",
             "PRÄMIENRECHNUNG · 2. Quartal\n\n"
             "Versicherungsnehmer: [[PII:full]]\n"
             "Versichertennummer (AHV): [[PII:ahv]]\n"
             "Rechnungsadresse: [[PII:address]]\n\n"
             "Police Nr. KVG-7745-A · Fälligkeit [[PII:date/dot]]\n\n"
             "  Grundversicherung KVG (Franchise 2'500)   CHF   312.40\n"
             "  Zusatz Spital halbprivat                  CHF    84.10\n"
             "  Zahnpflege Komplett                       CHF    27.50\n"
             "  ----------------------------------------------------------\n"
             "  Total fällig                              CHF   424.00\n\n"
             "Bitte begleichen Sie den Betrag mittels beiliegendem QR-Einzahlungs"
             "schein oder per Überweisung auf IBAN [[PII:iban]] (Vermerk: "
             "Police KVG-7745-A).\n\n"
             "Bei Fragen erreichen Sie unsere Kundenbetreuung unter [[PII:phone]] "
             "oder [[PII:email]]."),

    # 8. Überweisung an Spezialist (Referral letter)
    Fragment("doc_de_med_08", "de_ch",
             "ÜBERWEISUNG ZUR FACHÄRZTLICHEN ABKLÄRUNG\n\n"
             "An: [[PII:title_first_last]], Kardiologie\n\n"
             "Patient/in:  [[PII:full]]\n"
             "Geboren:     [[PII:date/dot]]\n"
             "Adresse:     [[PII:address]]\n"
             "Telefon:     [[PII:phone]]\n"
             "AHV:         [[PII:ahv]]\n\n"
             "Sehr geehrte Frau Kollegin\n\n"
             "Ich überweise Ihnen oben genannte Person mit seit drei Monaten "
             "rezidivierenden Palpitationen und gelegentlicher Belastungsdyspnoe "
             "zur weiteren Abklärung (24-h-Holter, Echokardiographie).\n\n"
             "Aktuelle Medikation: Bisoprolol 2.5 mg 1-0-0, Lisinopril 10 mg 1-0-0.\n\n"
             "Für Rückfragen stehe ich Ihnen gerne zur Verfügung.\n\n"
             "Freundliche Grüsse\n"
             "[[PII:title_first_last]]\n"
             "[[PII:date/dot]]"),

    # 9. Impfausweis / Vaccination certificate
    Fragment("doc_de_med_09", "de_ch",
             "IMPFAUSWEIS · Auszug\n\n"
             "Name:        [[PII:full]]\n"
             "Geboren:     [[PII:date/dot]]\n"
             "AHV-Nummer:  [[PII:ahv]]\n\n"
             "Dokumentierte Impfungen:\n\n"
             "  Tetanus / Diphtherie    Boostrix      [[PII:date/dot]]\n"
             "  FSME (Encepur)          1. Dosis      [[PII:date/dot]]\n"
             "  FSME (Encepur)          2. Dosis      [[PII:date/dot]]\n"
             "  Influenza saisonal      Vaxigrip Tetra [[PII:date/dot]]\n"
             "  Hepatitis A+B           Twinrix Boost [[PII:date/dot]]\n\n"
             "Sämtliche Impfungen wurden in der Hausarztpraxis verabreicht und "
             "ins elektronische Patientendossier übertragen.\n\n"
             "Bestätigt: [[PII:title_last]], [[PII:date]]"),

    # 10. Anamnese-Formular (Medical history form)
    Fragment("doc_de_med_10", "de_ch",
             "ANAMNESE-FORMULAR · Erstkonsultation\n\n"
             "Vor dem ersten Termin bitte vollständig ausfüllen und unterschrieben "
             "mitbringen.\n\n"
             "PERSONALIEN\n"
             "  Name, Vorname:  [[PII:last_first]]\n"
             "  Geburtsdatum:   [[PII:date/dot]]\n"
             "  Wohnadresse:    [[PII:address]]\n"
             "  Telefon privat: [[PII:phone]]\n"
             "  E-Mail:         [[PII:email]]\n"
             "  AHV-Nummer:     [[PII:ahv]]\n\n"
             "HAUSARZT\n"
             "  Name: [[PII:title_first_last]]\n\n"
             "BEKANNTE ERKRANKUNGEN (bitte ankreuzen):\n"
             "  [ ] Diabetes      [ ] Asthma       [ ] Hypertonie\n"
             "  [ ] Allergien     [ ] Herzleiden   [ ] Schilddrüse\n\n"
             "Regelmässig eingenommene Medikamente:\n"
             "  _____________________________________________________\n\n"
             "Datum: [[PII:date/dot]]   Unterschrift: ___________________"),

    # 11. Police-Auszug (Insurance policy excerpt)
    Fragment("doc_de_med_11", "de_ch",
             "VERSICHERUNGSPOLICE · Auszug\n\n"
             "Police Nr.:    ZH-KVG-009842/3\n"
             "Gültig ab:     [[PII:date/dot]]\n"
             "Erstellt am:   [[PII:date/dot]]\n\n"
             "Versicherte Person\n"
             "  [[PII:full]]\n"
             "  geboren [[PII:date/dot]]\n"
             "  [[PII:address]]\n"
             "  AHV-Nr. [[PII:ahv]]\n\n"
             "Versicherungsdeckungen\n"
             "  · Grundversicherung KVG, Franchise CHF 1'500\n"
             "  · Hausarztmodell (HMO-Praxis Stadelhofen)\n"
             "  · Halbprivat Spital (schweizweit)\n"
             "  · Komplementärmedizin nach Liste KVG\n\n"
             "Prämienjahresbetrag: CHF 5'488.80 (zahlbar quartalsweise auf "
             "IBAN [[PII:iban]]).\n\n"
             "Kündigungsfrist gemäss Art. 7 KVG. Für Anpassungen kontaktieren "
             "Sie uns unter [[PII:phone]] oder [[PII:email]]."),

    # 12. Therapiebericht (Therapy report)
    Fragment("doc_de_med_12", "de_ch",
             "THERAPIEBERICHT · Physiotherapie\n\n"
             "Patient/in:   [[PII:full]]\n"
             "Geboren:      [[PII:date/dot]]\n"
             "AHV-Nummer:   [[PII:ahv]]\n"
             "Adresse:      [[PII:address]]\n"
             "Zuweisung:    [[PII:title_first_last]]\n"
             "Verordnung vom [[PII:date/dot]] (9 Sitzungen)\n\n"
             "Diagnose: Lumbovertebralsyndrom mit muskulärer Dysbalance.\n\n"
             "Verlauf der bisherigen 6 Sitzungen:\n"
             "  · Manuelle Therapie der LWS, Mobilisation L4/L5\n"
             "  · Stabilisationstraining nach Janda\n"
             "  · Heimprogramm mit Theraband, 3x wöchentlich\n\n"
             "Aktueller Status: Schmerzen bei Belastung von initial NRS 7 auf "
             "NRS 3 reduziert. Beweglichkeit in Flexion deutlich verbessert.\n\n"
             "Procedere: Abschluss der verbleibenden 3 Sitzungen, danach Übergabe "
             "an Medizinische Trainingstherapie. Rückmeldung an Hausärztin "
             "schriftlich nach Abschluss.\n\n"
             "Winterthur, [[PII:date/dot]]\n"
             "Praxis erreichbar unter [[PII:phone]]"),
]
