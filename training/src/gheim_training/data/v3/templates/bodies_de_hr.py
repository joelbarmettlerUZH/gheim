"""German email body fragments — HR/admin domain. Hand-written, 20 entries."""
from ..core import Fragment

TEMPLATES: list[Fragment] = [
    Fragment("body_de_hr_01", "de_ch",
             "anbei sende ich Ihnen meine Bewerbungsunterlagen für die ausgeschriebene Stelle. "
             "Mein frühestmögliches Eintrittsdatum wäre der [[PII:date]]."),
    Fragment("body_de_hr_02", "de_ch",
             "Frau [[PII:last]] ist ab dem [[PII:date]] in Mutterschaftsurlaub. "
             "Die Stellvertretung übernimmt in dieser Zeit Herr [[PII:last]]."),
    Fragment("body_de_hr_03", "de_ch",
             "ich melde mich heute krank und werde voraussichtlich bis zum [[PII:date]] ausfallen. "
             "Das Arztzeugnis sende ich Ihnen umgehend nach."),
    Fragment("body_de_hr_04", "de_ch",
             "hiermit kündige ich mein Arbeitsverhältnis fristgerecht per [[PII:date]]. "
             "Für die gute Zusammenarbeit danke ich Ihnen herzlich."),
    Fragment("body_de_hr_05", "de_ch",
             "die Vertragsänderung von [[PII:full]] tritt rückwirkend per [[PII:date/spelled]] in Kraft. "
             "Bitte signieren Sie das beigelegte Dokument bis Ende Woche."),
    Fragment("body_de_hr_06", "de_ch",
             "für das Onboarding unserer neuen Mitarbeiterin [[PII:full]] benötige ich noch ihre Wohnadresse. "
             "Gemäss HR-Bogen lautet diese: [[PII:address]]."),
    Fragment("body_de_hr_07", "de_ch",
             "anbei finden Sie das Arbeitszeugnis von [[PII:title_last]]. "
             "Bitte prüfen Sie den Entwurf und melden Sie sich bei Rückfragen unter [[PII:phone]]."),
    Fragment("body_de_hr_08", "de_ch",
             "die jährliche Lohnerhöhung von [[PII:first]] wurde per [[PII:date]] genehmigt. "
             "Die neue Auszahlung erfolgt ab nächstem Monat auf das hinterlegte Konto."),
    Fragment("body_de_hr_09", "de_ch",
             "für die Aktualisierung der Personalakte fehlt uns noch Ihre AHV-Nummer. "
             "Bitte senden Sie diese (Format [[PII:ahv]]) per Antwort an mich zurück."),
    Fragment("body_de_hr_10", "de_ch",
             "wir freuen uns, Ihnen das Praktikum ab dem [[PII:date]] anbieten zu können. "
             "Ihr Praktikumsbetreuer wird Herr [[PII:title_last]] sein."),
    Fragment("body_de_hr_11", "de_ch",
             "die nächste Weiterbildung «Führung kompakt» findet am [[PII:date/spelled]] statt. "
             "Die Anmeldung ist über [[PII:url]] noch bis Freitag möglich."),
    Fragment("body_de_hr_12", "de_ch",
             "[[PII:full]] erreicht das ordentliche Pensionierungsalter am [[PII:date]]. "
             "Wir planen die Abschiedsfeier in der Woche davor."),
    Fragment("body_de_hr_13", "de_ch",
             "im Rahmen der internen Versetzung wechselt [[PII:first]] per [[PII:date]] in unser Team in Bern. "
             "Die neue Funktion lautet «Senior Specialist Compliance»."),
    Fragment("body_de_hr_14", "de_ch",
             "ich möchte beantragen, ab dem [[PII:date]] auf 80 % zu reduzieren. "
             "Gerne bespreche ich die Details persönlich mit meiner Vorgesetzten."),
    Fragment("body_de_hr_15", "de_ch",
             "die Home-Office-Vereinbarung mit [[PII:title_last]] ist unterschrieben und gilt ab [[PII:date]]. "
             "Eine Kopie liegt in der Personalakte."),
    Fragment("body_de_hr_16", "de_ch",
             "für die Auszahlung des 13. Monatslohns benötigt die Personalabteilung Ihre aktuelle IBAN. "
             "Bisher hinterlegt ist [[PII:iban]] — bitte bestätigen oder korrigieren Sie diese Angabe."),
    Fragment("body_de_hr_17", "de_ch",
             "ich bitte um Bewilligung meines unbezahlten Urlaubs vom [[PII:date]] bis [[PII:date]]. "
             "Eine Übergaberegelung mit dem Team habe ich bereits ausgearbeitet."),
    Fragment("body_de_hr_18", "de_ch",
             "die Bewerbung von [[PII:full]] (geboren am [[PII:date]]) liegt nun vollständig vor. "
             "Ich schlage ein erstes Interview in der nächsten Woche vor."),
    Fragment("body_de_hr_19", "de_ch",
             "für die Verlängerung der Aufenthaltsbewilligung benötige ich eine Arbeitgeberbestätigung. "
             "Diese sollte bitte direkt an [[PII:email]] gesendet werden."),
    Fragment("body_de_hr_20", "de_ch",
             "Herr [[PII:title_last]] hat seine Kündigung zurückgezogen. "
             "Der Vertrag läuft also unverändert weiter, neuer Stichtag für das Mitarbeitergespräch ist der [[PII:date]]."),
]
