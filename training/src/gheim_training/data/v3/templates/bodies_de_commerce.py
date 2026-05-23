"""German email body fragments — commerce/ordering domain. Hand-written, 25 entries."""
from ..core import Fragment

TEMPLATES: list[Fragment] = [
    Fragment("body_de_com_01", "de_ch",
             "ich möchte meine Bestellung Nr. 48217 vom [[PII:date/dot]] leider stornieren. "
             "Bei Rückfragen erreichen Sie mich unter [[PII:phone]] oder [[PII:email]]."),
    Fragment("body_de_com_02", "de_ch",
             "die Lieferung an [[PII:address]] ist heute Vormittag wohlbehalten "
             "eingetroffen – herzlichen Dank für die rasche Bearbeitung!"),
    Fragment("body_de_com_03", "de_ch",
             "vielen Dank für Ihre Bestellung. Den Versand haben wir für "
             "[[PII:date/spelled]] eingeplant; eine Tracking-Nummer folgt per Mail."),
    Fragment("body_de_com_04", "de_ch",
             "anbei finden Sie die Rechnung über CHF 1'248.50. Ich bitte um "
             "Überweisung auf unser Konto [[PII:iban]] bis spätestens [[PII:date/iso]]."),
    Fragment("body_de_com_05", "de_ch",
             "leider passt mir der bestellte Pullover nicht – könnten Sie mir bitte "
             "ein Retourenetikett an [[PII:email]] senden? Danke im Voraus."),
    Fragment("body_de_com_06", "de_ch",
             "bezüglich Ihrer Anfrage vom [[PII:date/dot]]: Der Artikel ist ab "
             "Mitte nächster Woche wieder lagernd. Soll ich Sie unter [[PII:phone/natl_slash]] "
             "informieren, sobald er eingetroffen ist?"),
    Fragment("body_de_com_07", "de_ch",
             "die Zahlung von [[PII:full]] für Bestellung #99214 ist heute "
             "eingegangen, vielen Dank für die schnelle Erledigung."),
    Fragment("body_de_com_08", "de_ch",
             "könnten Sie meine Lieferadresse bitte auf [[PII:address]] ändern? "
             "Die Bestellung wurde gestern unter der Nummer 7741-B aufgegeben."),
    Fragment("body_de_com_09", "de_ch",
             "ich möchte das Geschenk bitte direkt versenden lassen – Empfängerin "
             "ist [[PII:full]], die Karte soll handschriftlich beigelegt werden."),
    Fragment("body_de_com_10", "de_ch",
             "Ihre Garantieverlängerung wurde aktiviert. Den Vertrag im PDF-Format "
             "finden Sie auf [[PII:url]] unter Ihrem Kundenkonto."),
    Fragment("body_de_com_11", "de_ch",
             "anbei das Foto der beschädigten Verpackung. Ich bitte um einen "
             "Ersatzartikel oder Gutschrift auf meine Karte [[PII:cc]]."),
    Fragment("body_de_com_12", "de_ch",
             "wir haben Ihre Reklamation erhalten und werden uns innert drei "
             "Werktagen telefonisch bei Ihnen melden, am besten unter [[PII:phone/intl_paren]]."),
    Fragment("body_de_com_13", "de_ch",
             "die Sonderanfertigung für [[PII:title_last]] ist abholbereit – "
             "unsere Werkstatt ist Mo–Fr von 09:00 bis 18:00 geöffnet."),
    Fragment("body_de_com_14", "de_ch",
             "kurze Frage: Ist die Espressomaschine vom Modell EM-220 noch "
             "verfügbar? Ich würde am [[PII:date/spelled]] gerne vorbeikommen."),
    Fragment("body_de_com_15", "de_ch",
             "vielen Dank für Ihren Einkauf! Die Rechnung über die acht Stühle "
             "geht Ihnen separat per Post an [[PII:address]] zu."),
    Fragment("body_de_com_16", "de_ch",
             "leider mussten wir Ihre Bestellung am [[PII:date/dot]] mangels "
             "Lager retournieren. Bitte teilen Sie mir auf [[PII:email]] mit, "
             "ob wir vormerken sollen."),
    Fragment("body_de_com_17", "de_ch",
             "die Gutschrift in Höhe von CHF 89.– wurde soeben auf das Konto "
             "[[PII:iban]] überwiesen; eingebucht in den nächsten 1–2 Werktagen."),
    Fragment("body_de_com_18", "de_ch",
             "darf ich Sie bitten, mir bis [[PII:date/iso]] noch die "
             "Mengenangaben zu bestätigen? Ohne Rückmeldung reservieren wir nur die Hälfte."),
    Fragment("body_de_com_19", "de_ch",
             "der Kurier von Planzer wird Ihre Sendung morgen zwischen 08:00 "
             "und 12:00 anliefern. Bitte stellen Sie sicher, dass jemand unter "
             "[[PII:phone]] erreichbar ist."),
    Fragment("body_de_com_20", "de_ch",
             "wir freuen uns, dass Sie sich für unseren Onlineshop entschieden "
             "haben. Ihre Zugangsdaten haben wir an [[PII:email]] gesendet."),
    Fragment("body_de_com_21", "de_ch",
             "im Anhang finden Sie das aktualisierte Angebot. Die Lieferung "
             "erfolgt frei Haus an [[PII:address]], gültig bis [[PII:date/dot]]."),
    Fragment("body_de_com_22", "de_ch",
             "[[PII:title_last]], bezüglich Ihrer Retoure vom letzten Donnerstag: "
             "Der Eingang ist bestätigt, die Rückerstattung folgt in den nächsten Tagen."),
    Fragment("body_de_com_23", "de_ch",
             "die bestellten Visitenkarten sind in Produktion; voraussichtlicher "
             "Versand ist der [[PII:date/spelled]]. Korrekturbögen sehen Sie unter [[PII:url]]."),
    Fragment("body_de_com_24", "de_ch",
             "für die Geschenkverpackung bräuchten wir noch die Empfängeradresse. "
             "Bitte senden Sie diese an [[PII:email]] oder telefonisch an [[PII:phone/natl_slash]]."),
    Fragment("body_de_com_25", "de_ch",
             "ich beanstande die Lieferung vom [[PII:date/iso]]: zwei der vier "
             "Weingläser sind zerbrochen angekommen. Vorgangsnummer R-2024-0813."),
]
