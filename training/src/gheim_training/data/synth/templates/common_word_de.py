"""German common-word surname disambiguation fragments. Hand-written, 80 entries.
40 person-positive + 40 noun-negative for 8 common-word surnames (Bach, Berg,
Stein, Fischer, Wolf, König, Sommer, Schneider). Forces the model to learn
context-sensitive disambiguation.

Person fragments use the literal surname inline (no [[PII:...]] marker) so the
model learns the specific token in person contexts. Negative fragments use the
same word in its common-noun meaning with NO PII span.

The companion SPANS dict declares which substring(s) of each positive fragment
are private_person spans; negative fragments have no entry.
"""
from ..core import Fragment

TEMPLATES: list[Fragment] = [
    # ============================================================ BACH
    # ---- POSITIVE: Bach (the person) ----
    Fragment("cw_de_bach_pos_01", "de_ch",
             "Herr Bach hat das gestrige Konzert in der Tonhalle "
             "souverän dirigiert."),
    Fragment("cw_de_bach_pos_02", "de_ch",
             "Anna Bach unterzeichnete den Mietvertrag noch am selben "
             "Nachmittag im Notariat."),
    Fragment("cw_de_bach_pos_03", "de_ch",
             "Frau Bach wird morgen die Begrüssungsrede an der "
             "Generalversammlung halten."),
    Fragment("cw_de_bach_pos_04", "de_ch",
             "Wir haben Lukas Bach aus dem Marketingteam für das "
             "Projekt nominiert."),
    Fragment("cw_de_bach_pos_05", "de_ch",
             "Dr. Bach wird Sie nach der Sprechstunde persönlich "
             "über die Befunde informieren."),
    # ---- NEGATIVE: Bach (the creek) ----
    Fragment("cw_de_bach_neg_01", "de_ch",
             "Am Bach steht ein alter Nussbaum, unter dem wir früher "
             "im Schatten gepicknickt haben."),
    Fragment("cw_de_bach_neg_02", "de_ch",
             "Der Bach fliesst hinter dem Hof entlang und mündet "
             "unterhalb der Mühle in die Reuss."),
    Fragment("cw_de_bach_neg_03", "de_ch",
             "Nach dem Gewitter war der Bach so angeschwollen, dass "
             "die untere Brücke kurzzeitig gesperrt werden musste."),
    Fragment("cw_de_bach_neg_04", "de_ch",
             "Im Sommer sammeln die Kinder Steine am Bach und bauen "
             "kleine Staudämme aus Ästen und Kies."),
    Fragment("cw_de_bach_neg_05", "de_ch",
             "Entlang des Bachs verläuft ein schmaler Wanderweg bis "
             "zur alten Sägerei am Waldrand."),

    # ============================================================ BERG
    # ---- POSITIVE: Berg (the person) ----
    Fragment("cw_de_berg_pos_01", "de_ch",
             "Herr Berg hat heute Morgen die Schlüssel im Sekretariat "
             "abgeholt und ist bereits unterwegs."),
    Fragment("cw_de_berg_pos_02", "de_ch",
             "Sandra Berg leitet seit drei Jahren unsere Filiale in "
             "Aarau und kennt jeden Stammkunden persönlich."),
    Fragment("cw_de_berg_pos_03", "de_ch",
             "Frau Berg möchte den Termin gerne auf nächsten Dienstag "
             "verschieben, falls das für Sie passt."),
    Fragment("cw_de_berg_pos_04", "de_ch",
             "Ich habe gerade kurz mit Berg telefoniert; er kommt "
             "etwa zwanzig Minuten später."),
    Fragment("cw_de_berg_pos_05", "de_ch",
             "Professor Berg hat seine Vorlesung diese Woche "
             "ausnahmsweise in den Hörsaal B verlegt."),
    # ---- NEGATIVE: Berg (the mountain) ----
    Fragment("cw_de_berg_neg_01", "de_ch",
             "Vom Tal aus sieht der Berg ganz harmlos aus, aber der "
             "Aufstieg dauert volle sechs Stunden."),
    Fragment("cw_de_berg_neg_02", "de_ch",
             "Wir wollten ursprünglich auf den Berg steigen, mussten "
             "die Tour aber wegen Nebel abbrechen."),
    Fragment("cw_de_berg_neg_03", "de_ch",
             "Hinter dem Dorf erhebt sich ein bewaldeter Berg mit "
             "einer kleinen Kapelle auf dem Gipfel."),
    Fragment("cw_de_berg_neg_04", "de_ch",
             "Im Winter ist der Berg meist tief verschneit und nur "
             "über die präparierte Pistenseite erreichbar."),
    Fragment("cw_de_berg_neg_05", "de_ch",
             "Ein Aktenstapel türmte sich wie ein kleiner Berg auf "
             "seinem Schreibtisch und drohte umzufallen."),

    # ============================================================ STEIN
    # ---- POSITIVE: Stein (the person) ----
    Fragment("cw_de_stein_pos_01", "de_ch",
             "Herr Stein hat den überarbeiteten Entwurf gestern Abend "
             "noch per Mail an die Geschäftsleitung geschickt."),
    Fragment("cw_de_stein_pos_02", "de_ch",
             "Miriam Stein wird das Eröffnungsreferat halten und "
             "anschliessend die Podiumsdiskussion moderieren."),
    Fragment("cw_de_stein_pos_03", "de_ch",
             "Frau Stein bittet darum, ihr die Unterlagen bis "
             "Freitagmittag in Kopie zukommen zu lassen."),
    Fragment("cw_de_stein_pos_04", "de_ch",
             "Mein Kollege Stein arbeitet derzeit an einer Studie "
             "über regionale Sprachvarietäten in der Ostschweiz."),
    Fragment("cw_de_stein_pos_05", "de_ch",
             "Wir freuen uns, Daniel Stein als neuen Bereichsleiter "
             "im Stiftungsrat begrüssen zu dürfen."),
    # ---- NEGATIVE: Stein (the stone) ----
    Fragment("cw_de_stein_neg_01", "de_ch",
             "Sie hob einen flachen Stein vom Strand auf und liess "
             "ihn übers Wasser hüpfen."),
    Fragment("cw_de_stein_neg_02", "de_ch",
             "Ein schwerer Stein versperrte den Wanderweg, sodass "
             "wir einen Umweg durch das Gebüsch nehmen mussten."),
    Fragment("cw_de_stein_neg_03", "de_ch",
             "Beim Pflügen stiess der Bauer auf einen ungewöhnlich "
             "grossen Stein direkt unter der Erdoberfläche."),
    Fragment("cw_de_stein_neg_04", "de_ch",
             "Der Brunnen war aus rohem Stein gemauert und stand "
             "seit über zweihundert Jahren am Dorfplatz."),
    Fragment("cw_de_stein_neg_05", "de_ch",
             "Mir fiel ein Stein vom Herzen, als ich die gute "
             "Nachricht vom Spital erhielt."),

    # ============================================================ FISCHER
    # ---- POSITIVE: Fischer (the person) ----
    Fragment("cw_de_fischer_pos_01", "de_ch",
             "Herr Fischer hat die Offerte mit dem Lieferanten neu "
             "verhandelt und einen besseren Stückpreis erzielt."),
    Fragment("cw_de_fischer_pos_02", "de_ch",
             "Petra Fischer übernimmt ab Juni die Leitung der "
             "Personalabteilung in unserer Niederlassung."),
    Fragment("cw_de_fischer_pos_03", "de_ch",
             "Frau Fischer wird Sie an der Réception abholen und "
             "zum Sitzungszimmer im dritten Stock begleiten."),
    Fragment("cw_de_fischer_pos_04", "de_ch",
             "Ich habe Fischer in den Verteiler aufgenommen, damit "
             "er den Sitzungsbericht direkt erhält."),
    Fragment("cw_de_fischer_pos_05", "de_ch",
             "Dr. med. Fischer empfiehlt eine zweite Untersuchung "
             "in vier bis sechs Wochen."),
    # ---- NEGATIVE: Fischer (the profession) ----
    Fragment("cw_de_fischer_neg_01", "de_ch",
             "Frühmorgens fährt der Fischer mit seinem Kahn hinaus "
             "und kehrt erst gegen Mittag wieder zurück."),
    Fragment("cw_de_fischer_neg_02", "de_ch",
             "Am Bodensee leben heute nur noch wenige Fischer von "
             "ihrem traditionellen Handwerk."),
    Fragment("cw_de_fischer_neg_03", "de_ch",
             "Ein erfahrener Fischer erkennt schon am Wellenbild, "
             "wo sich die Schwärme aufhalten."),
    Fragment("cw_de_fischer_neg_04", "de_ch",
             "Die Fischer am See klagen über sinkende Fangerträge "
             "und steigende Konzessionsgebühren."),
    Fragment("cw_de_fischer_neg_05", "de_ch",
             "Im Hafen reparierte ein älterer Fischer seine Netze "
             "und unterhielt sich dabei mit den Touristen."),

    # ============================================================ WOLF
    # ---- POSITIVE: Wolf (the person) ----
    Fragment("cw_de_wolf_pos_01", "de_ch",
             "Herr Wolf hat die Buchhaltung für das letzte Quartal "
             "bereits vollständig abgeschlossen."),
    Fragment("cw_de_wolf_pos_02", "de_ch",
             "Claudia Wolf wird Sie am Eingang in Empfang nehmen und "
             "Ihnen das Gebäude kurz zeigen."),
    Fragment("cw_de_wolf_pos_03", "de_ch",
             "Frau Wolf hat heute Vormittag bereits zweimal angerufen, "
             "Sie möchten bitte zurückrufen."),
    Fragment("cw_de_wolf_pos_04", "de_ch",
             "Mit Markus Wolf haben wir einen erfahrenen Architekten "
             "für die Sanierung gewinnen können."),
    Fragment("cw_de_wolf_pos_05", "de_ch",
             "Anwalt Wolf hat das Schreiben durchgesehen und einige "
             "Formulierungen vorsichtshalber angepasst."),
    # ---- NEGATIVE: Wolf (the animal) ----
    Fragment("cw_de_wolf_neg_01", "de_ch",
             "Im Calandagebiet wurde diese Woche erneut ein Wolf "
             "auf einer Wildkamera dokumentiert."),
    Fragment("cw_de_wolf_neg_02", "de_ch",
             "Der Wolf scheue grundsätzlich den Menschen, beruhigte "
             "die Wildhüterin die besorgten Bauern."),
    Fragment("cw_de_wolf_neg_03", "de_ch",
             "Ein einzelner Wolf kann ohne Herdenschutz erheblichen "
             "Schaden an einer ungesicherten Schafherde anrichten."),
    Fragment("cw_de_wolf_neg_04", "de_ch",
             "Im Bündner Hinterland wurde vor zwei Jahren ein "
             "ausgewachsener Wolf gesichtet und fotografisch belegt."),
    Fragment("cw_de_wolf_neg_05", "de_ch",
             "In dem alten Märchen verschlingt der Wolf die Grossmutter "
             "und legt sich anschliessend in ihr Bett."),

    # ============================================================ KÖNIG
    # ---- POSITIVE: König (the person) ----
    Fragment("cw_de_koenig_pos_01", "de_ch",
             "Herr König hat die Ausschreibungsunterlagen heute "
             "Morgen termingerecht beim Kanton eingereicht."),
    Fragment("cw_de_koenig_pos_02", "de_ch",
             "Verena König wird ab nächstem Monat die Abteilung "
             "Kommunikation interimistisch leiten."),
    Fragment("cw_de_koenig_pos_03", "de_ch",
             "Frau König bittet um Verschiebung der Sitzung wegen "
             "eines kurzfristigen externen Termins."),
    Fragment("cw_de_koenig_pos_04", "de_ch",
             "Ich habe gestern lange mit Stefan König gesprochen, "
             "er sieht die Lage etwas anders als wir."),
    Fragment("cw_de_koenig_pos_05", "de_ch",
             "Pfarrer König wird die Trauerfeier am Samstag in der "
             "reformierten Kirche halten."),
    # ---- NEGATIVE: König (the monarch) ----
    Fragment("cw_de_koenig_neg_01", "de_ch",
             "Der König von Spanien empfing die Delegation zu einer "
             "kurzen Audienz im Palast von Madrid."),
    Fragment("cw_de_koenig_neg_02", "de_ch",
             "Im Mittelalter herrschte hier ein König, der die Burg "
             "über dem Tal errichten liess."),
    Fragment("cw_de_koenig_neg_03", "de_ch",
             "Beim Schachspiel ist es das oberste Ziel, den König "
             "des Gegners matt zu setzen."),
    Fragment("cw_de_koenig_neg_04", "de_ch",
             "Die Geschichte erzählt von einem König, der sein "
             "halbes Reich verlor, weil er der Eitelkeit verfiel."),
    Fragment("cw_de_koenig_neg_05", "de_ch",
             "In der Schweiz gibt es seit jeher keinen König, "
             "sondern einen jährlich wechselnden Bundespräsidenten."),

    # ============================================================ SOMMER
    # ---- POSITIVE: Sommer (the person) ----
    Fragment("cw_de_sommer_pos_01", "de_ch",
             "Herr Sommer hat sich für die nächste Sitzung "
             "entschuldigt; er ist auf einer Geschäftsreise in Wien."),
    Fragment("cw_de_sommer_pos_02", "de_ch",
             "Nadja Sommer betreut bei uns sämtliche Kundenanfragen "
             "aus der Westschweiz und der Romandie."),
    Fragment("cw_de_sommer_pos_03", "de_ch",
             "Frau Sommer wird Ihnen die Vertragsunterlagen am "
             "Montagmorgen persönlich überbringen."),
    Fragment("cw_de_sommer_pos_04", "de_ch",
             "Ich habe Sommer aus der Rechtsabteilung gebeten, den "
             "Entwurf vor der Unterschrift gegenzulesen."),
    Fragment("cw_de_sommer_pos_05", "de_ch",
             "Dr. Sommer wird im Anschluss an die Versammlung ein "
             "kurzes Referat zur Bilanz halten."),
    # ---- NEGATIVE: Sommer (the season) ----
    Fragment("cw_de_sommer_neg_01", "de_ch",
             "Im Sommer wandern wir am liebsten in den Voralpen, "
             "weil dort die Temperaturen erträglich bleiben."),
    Fragment("cw_de_sommer_neg_02", "de_ch",
             "Der vergangene Sommer war einer der trockensten seit "
             "Beginn der systematischen Wetteraufzeichnungen."),
    Fragment("cw_de_sommer_neg_03", "de_ch",
             "Letzten Sommer hatten wir wochenlang über dreissig "
             "Grad und kaum einen Tropfen Regen."),
    Fragment("cw_de_sommer_neg_04", "de_ch",
             "Im Sommer ist das Restaurant am See bis spät in die "
             "Nacht hinein gut besucht."),
    Fragment("cw_de_sommer_neg_05", "de_ch",
             "Wir verbringen jeden Sommer zwei Wochen im Tessin, "
             "meist in einem kleinen Rustico bei Locarno."),

    # ============================================================ SCHNEIDER
    # ---- POSITIVE: Schneider (the person) ----
    Fragment("cw_de_schneider_pos_01", "de_ch",
             "Herr Schneider hat die Garantieverlängerung schriftlich "
             "bestätigt und die Kopie ins Dossier gelegt."),
    Fragment("cw_de_schneider_pos_02", "de_ch",
             "Lisa Schneider wird das Eröffnungswort an der "
             "Jubiläumsfeier in Olten halten."),
    Fragment("cw_de_schneider_pos_03", "de_ch",
             "Frau Schneider hat sich heute krankgemeldet; die "
             "Vertretung übernimmt vorübergehend Frau Meier."),
    Fragment("cw_de_schneider_pos_04", "de_ch",
             "Mit Felix Schneider haben wir einen ausgewiesenen "
             "Steuerexperten in den Verwaltungsrat geholt."),
    Fragment("cw_de_schneider_pos_05", "de_ch",
             "Notar Schneider hat den beglaubigten Auszug am "
             "Freitag persönlich vorbeigebracht."),
    # ---- NEGATIVE: Schneider (the tailor) ----
    Fragment("cw_de_schneider_neg_01", "de_ch",
             "Der Schneider in der Altstadt fertigt seit über vierzig "
             "Jahren Anzüge nach Mass."),
    Fragment("cw_de_schneider_neg_02", "de_ch",
             "Ein erfahrener Schneider kann eine ungefütterte Jacke "
             "innerhalb weniger Tage komplett umnähen."),
    Fragment("cw_de_schneider_neg_03", "de_ch",
             "Sie liess das Brautkleid bei einem renommierten "
             "Schneider in Zürich anfertigen."),
    Fragment("cw_de_schneider_neg_04", "de_ch",
             "Der Schneider nahm gerade Mass, als der nächste "
             "Kunde bereits ungeduldig im Vorzimmer wartete."),
    Fragment("cw_de_schneider_neg_05", "de_ch",
             "Im Quartier gab es früher noch einen kleinen Schneider, "
             "der auch Reparaturen für die Nachbarschaft übernahm."),
]


# Parallel SPANS dict: for each positive template_id, declare the
# substring(s) inside the rendered text that are private_person spans.
# Negative templates have NO entry here.
SPANS: dict[str, list[tuple[str, str]]] = {
    # Bach
    "cw_de_bach_pos_01": [("Herr Bach", "person")],
    "cw_de_bach_pos_02": [("Anna Bach", "person")],
    "cw_de_bach_pos_03": [("Frau Bach", "person")],
    "cw_de_bach_pos_04": [("Lukas Bach", "person")],
    "cw_de_bach_pos_05": [("Dr. Bach", "person")],
    # Berg
    "cw_de_berg_pos_01": [("Herr Berg", "person")],
    "cw_de_berg_pos_02": [("Sandra Berg", "person")],
    "cw_de_berg_pos_03": [("Frau Berg", "person")],
    "cw_de_berg_pos_04": [("Berg", "person")],
    "cw_de_berg_pos_05": [("Professor Berg", "person")],
    # Stein
    "cw_de_stein_pos_01": [("Herr Stein", "person")],
    "cw_de_stein_pos_02": [("Miriam Stein", "person")],
    "cw_de_stein_pos_03": [("Frau Stein", "person")],
    "cw_de_stein_pos_04": [("Stein", "person")],
    "cw_de_stein_pos_05": [("Daniel Stein", "person")],
    # Fischer
    "cw_de_fischer_pos_01": [("Herr Fischer", "person")],
    "cw_de_fischer_pos_02": [("Petra Fischer", "person")],
    "cw_de_fischer_pos_03": [("Frau Fischer", "person")],
    "cw_de_fischer_pos_04": [("Fischer", "person")],
    "cw_de_fischer_pos_05": [("Dr. med. Fischer", "person")],
    # Wolf
    "cw_de_wolf_pos_01": [("Herr Wolf", "person")],
    "cw_de_wolf_pos_02": [("Claudia Wolf", "person")],
    "cw_de_wolf_pos_03": [("Frau Wolf", "person")],
    "cw_de_wolf_pos_04": [("Markus Wolf", "person")],
    "cw_de_wolf_pos_05": [("Anwalt Wolf", "person")],
    # König
    "cw_de_koenig_pos_01": [("Herr König", "person")],
    "cw_de_koenig_pos_02": [("Verena König", "person")],
    "cw_de_koenig_pos_03": [("Frau König", "person")],
    "cw_de_koenig_pos_04": [("Stefan König", "person")],
    "cw_de_koenig_pos_05": [("Pfarrer König", "person")],
    # Sommer
    "cw_de_sommer_pos_01": [("Herr Sommer", "person")],
    "cw_de_sommer_pos_02": [("Nadja Sommer", "person")],
    "cw_de_sommer_pos_03": [("Frau Sommer", "person")],
    "cw_de_sommer_pos_04": [("Sommer", "person")],
    "cw_de_sommer_pos_05": [("Dr. Sommer", "person")],
    # Schneider
    "cw_de_schneider_pos_01": [("Herr Schneider", "person")],
    "cw_de_schneider_pos_02": [("Lisa Schneider", "person")],
    "cw_de_schneider_pos_03": [("Frau Schneider", "person")],
    "cw_de_schneider_pos_04": [("Felix Schneider", "person")],
    "cw_de_schneider_pos_05": [("Notar Schneider", "person")],
}
