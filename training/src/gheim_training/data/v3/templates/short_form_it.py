"""Italian short-form narrative fragments. Hand-written, 25 entries.
NOT email greetings/signatures — mid-document narrative, news-style,
conversational snippets."""
from ..core import Fragment

TEMPLATES: list[Fragment] = [
    # full name, subject position, news-style
    Fragment("sf_it_01", "it_ch",
             "[[PII:full]] ha firmato il protocollo stamattina."),
    # prepositional phrase, last name
    Fragment("sf_it_02", "it_ch",
             "Secondo [[PII:last]], la seduta inizia alle 14 in punto."),
    # conversational question with title + last
    Fragment("sf_it_03", "it_ch",
             "Hai visto se [[PII:title_last]] è già passato dall'ufficio stamattina?"),
    # past-tense conversational, first name as object
    Fragment("sf_it_04", "it_ch",
             "Ho chiamato [[PII:first]] ieri sera ma non ha risposto al telefono."),
    # news-style with full name + apposition
    Fragment("sf_it_05", "it_ch",
             "[[PII:full]], direttore dal 2020, ha confermato la fusione "
             "durante una conferenza a Lugano."),
    # quoted speech, title + last
    Fragment("sf_it_06", "it_ch",
             "«Stiamo valutando ogni ipotesi», ha dichiarato [[PII:title_last]] "
             "ai cronisti presenti."),
    # full name in passive/agent position
    Fragment("sf_it_07", "it_ch",
             "Il rapporto cita brevemente [[PII:full]] fra i testimoni "
             "ascoltati dal magistrato."),
    # short note for someone, bare first
    Fragment("sf_it_08", "it_ch",
             "Promemoria per [[PII:first]]: richiamare il cliente prima di mezzogiorno."),
    # casual encounter, full name
    Fragment("sf_it_09", "it_ch",
             "Incrociato [[PII:full]] al mercato di Bellinzona stamattina, sta benone."),
    # full name + address
    Fragment("sf_it_10", "it_ch",
             "Il nuovo studio di [[PII:full]] si trova ora in [[PII:address]]."),
    # name + date (birthday note)
    Fragment("sf_it_11", "it_ch",
             "Compleanno di [[PII:first]] il [[PII:date]] — ricordarsi della torta."),
    # journalistic with role + last name
    Fragment("sf_it_12", "it_ch",
             "Il commissario [[PII:last]] per ora si rifiuta di commentare il caso."),
    # full name + phone, contact note
    Fragment("sf_it_13", "it_ch",
             "Per qualsiasi questione logistica contattare [[PII:full]] al [[PII:phone]]."),
    # title + last with absence/travel
    Fragment("sf_it_14", "it_ch",
             "Consiglio rinviato: [[PII:title_last]] è in trasferta all'estero "
             "fino al [[PII:date]]."),
    # casual schedule with first
    Fragment("sf_it_15", "it_ch",
             "Colazione con [[PII:first]] domattina alle 8, al bar dietro l'ufficio."),
    # parenthetical email after name
    Fragment("sf_it_16", "it_ch",
             "Il dossier trasmesso da [[PII:full]] ([[PII:email]]) "
             "attende ancora la convalida dell'ufficio legale."),
    # list with dashes, bare first
    Fragment("sf_it_17", "it_ch",
             "Tre studenti — fra cui [[PII:first]] — hanno conseguito "
             "la lode quest'anno."),
    # banking note with IBAN + full name
    Fragment("sf_it_18", "it_ch",
             "Onorari versati stamane sul conto [[PII:iban]] intestato a [[PII:full]]."),
    # informal social plan, first
    Fragment("sf_it_19", "it_ch",
             "Si cena da [[PII:first]] venerdì, non scordarti il vino bianco."),
    # interview refusal, title + first + last
    Fragment("sf_it_20", "it_ch",
             "Interpellato sulla polemica, [[PII:title_first_last]] ha preferito non rispondere."),
    # delivery address + last name attention
    Fragment("sf_it_21", "it_ch",
             "Indirizzo di consegna confermato: [[PII:address]], "
             "all'attenzione di [[PII:last]]."),
    # worried first-person note about first
    Fragment("sf_it_22", "it_ch",
             "[[PII:first]] mi ha mandato un messaggio strano verso mezzanotte, "
             "comincio a preoccuparmi."),
    # news bio sentence, full + address
    Fragment("sf_it_23", "it_ch",
             "Il cronista riporta che [[PII:full]], 47 anni, abita "
             "in [[PII:address]] dalla nascita."),
    # appointment with title_last + date
    Fragment("sf_it_24", "it_ch",
             "Appuntamento fissato con [[PII:title_last]] il [[PII:date]], "
             "studio del dott. Bernasconi."),
    # forwarding request to first + email
    Fragment("sf_it_25", "it_ch",
             "Puoi inoltrarlo a [[PII:first]]? Il suo indirizzo è [[PII:email]]."),
]
