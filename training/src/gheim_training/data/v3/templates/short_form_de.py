"""German short-form narrative fragments. Hand-written, 30 entries.
NOT email greetings/signatures — mid-document narrative, news-style,
note-style, or conversational snippets where a name appears in context."""
from ..core import Fragment

TEMPLATES: list[Fragment] = [
    # --- bare last name, subject position ---
    Fragment("sf_de_01", "de_ch",
             "[[PII:last]] hat heute Morgen das Protokoll unterschrieben."),
    Fragment("sf_de_02", "de_ch",
             "Laut [[PII:last]] beginnt die nächste Sitzung pünktlich um 14:00."),
    Fragment("sf_de_03", "de_ch",
             "Gestern Abend rief [[PII:last]] noch kurz aus dem Büro an, "
             "alles sei im Griff."),
    # --- bare first name, conversational ---
    Fragment("sf_de_04", "de_ch",
             "Hast du [[PII:first]] heute zufällig im Gang gesehen?"),
    Fragment("sf_de_05", "de_ch",
             "Ich glaube, [[PII:first]] meinte gestern etwas Ähnliches "
             "beim Mittagessen."),
    Fragment("sf_de_06", "de_ch",
             "Sag [[PII:first]] bitte, dass die Akten oben auf dem "
             "Schreibtisch liegen."),
    # --- bare first name, object position ---
    Fragment("sf_de_07", "de_ch",
             "Ich habe [[PII:first]] schon dreimal angerufen, aber niemand "
             "geht ans Telefon."),
    # --- genitive, last name ---
    Fragment("sf_de_08", "de_ch",
             "Die Aussage [[PII:last]]s vor dem Untersuchungsausschuss "
             "sorgt weiter für Diskussionen."),
    Fragment("sf_de_09", "de_ch",
             "[[PII:first]]s Bericht aus dem Spital klang ehrlich gesagt "
             "nicht besonders ermutigend."),
    # --- news-style relative clause, full name ---
    Fragment("sf_de_10", "de_ch",
             "[[PII:full]], der seit 2018 die Abteilung leitet, kündigte "
             "gestern überraschend seinen Rücktritt an."),
    Fragment("sf_de_11", "de_ch",
             "Wie der Gemeinderat bestätigte, wird [[PII:full]] das neue "
             "Quartierprojekt ab Januar verantworten."),
    # --- title + last name, formal narrative ---
    Fragment("sf_de_12", "de_ch",
             "[[PII:title_last]] erklärte vor versammelter Presse, der "
             "Entscheid sei einstimmig gefallen."),
    Fragment("sf_de_13", "de_ch",
             "Im Vorzimmer wartete bereits [[PII:title_last]] auf das "
             "vereinbarte Gespräch."),
    # --- two names in one snippet ---
    Fragment("sf_de_14", "de_ch",
             "[[PII:first]] und [[PII:first]] haben sich am Bahnhof "
             "verabredet, um gemeinsam ins Konzert zu fahren."),
    Fragment("sf_de_15", "de_ch",
             "An der Sitzung nahmen neben [[PII:last]] auch [[PII:full]] "
             "und mehrere Mitarbeitende des Stadtbauamts teil."),
    # --- name + date ---
    Fragment("sf_de_16", "de_ch",
             "[[PII:full]] wurde am [[PII:date/dot]] in Schaffhausen geboren "
             "und lebt heute mit ihrer Familie in Winterthur."),
    Fragment("sf_de_17", "de_ch",
             "Seit dem [[PII:date/spelled]] arbeitet [[PII:last]] wieder "
             "Vollzeit in der Filiale Bern."),
    # --- name + address ---
    Fragment("sf_de_18", "de_ch",
             "Die Firma plant einen Umzug nach [[PII:address]]; [[PII:full]] "
             "leitet das gesamte Projekt."),
    Fragment("sf_de_19", "de_ch",
             "Zeugen sahen [[PII:last]] kurz vor Mitternacht in der Nähe "
             "der Liegenschaft an der [[PII:address]]."),
    # --- name + phone ---
    Fragment("sf_de_20", "de_ch",
             "Für Auskünfte zum Programm steht [[PII:title_last]] unter "
             "[[PII:phone/natl_slash]] zur Verfügung."),
    Fragment("sf_de_21", "de_ch",
             "Wenn du nicht durchkommst, probier es bei [[PII:first]] "
             "direkt auf dem Handy: [[PII:phone]]."),
    # --- name + email ---
    Fragment("sf_de_22", "de_ch",
             "Rückfragen zur Anmeldung beantwortet [[PII:full]] gerne per "
             "Mail an [[PII:email]]."),
    # --- name + IBAN ---
    Fragment("sf_de_23", "de_ch",
             "Die Spende von [[PII:full]] in Höhe von CHF 500 ist heute "
             "auf dem Konto [[PII:iban]] eingegangen."),
    # --- name + url ---
    Fragment("sf_de_24", "de_ch",
             "Das vollständige Interview mit [[PII:full]] ist seit gestern "
             "auf [[PII:url]] abrufbar."),
    # --- prepositional, conversational with last name ---
    Fragment("sf_de_25", "de_ch",
             "Mit [[PII:last]] habe ich noch nie ein vernünftiges Wort "
             "gewechselt, ehrlich gesagt."),
    # --- last-name-first formal listing ---
    Fragment("sf_de_26", "de_ch",
             "Auf der Teilnehmerliste fand ich auch [[PII:last_first]], "
             "was mich ziemlich überrascht hat."),
    # --- initial + last narrative ---
    Fragment("sf_de_27", "de_ch",
             "Im Begleitschreiben war als Kontaktperson [[PII:first_initial]] "
             "vermerkt, ohne weitere Angaben."),
    # --- short note style, bare last name ---
    Fragment("sf_de_28", "de_ch",
             "Notiz an mich selbst: [[PII:last]] morgen unbedingt zurückrufen, "
             "Sache ist dringend."),
    # --- gossipy conversational, first name ---
    Fragment("sf_de_29", "de_ch",
             "Stell dir vor, [[PII:first]] hat tatsächlich gekündigt – und "
             "zwar von einem Tag auf den anderen."),
    # --- long narrative, full + date + address ---
    Fragment("sf_de_30", "de_ch",
             "[[PII:full]] zog am [[PII:date/dot]] nach [[PII:address]] und "
             "eröffnete dort wenige Wochen später eine kleine Buchhandlung."),
]
