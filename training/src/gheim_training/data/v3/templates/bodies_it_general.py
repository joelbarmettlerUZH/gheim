"""Italian email body fragments — general correspondence. Hand-written, 25 entries."""
from ..core import Fragment

TEMPLATES: list[Fragment] = [
    # --- commerce / ordering (6) ---
    Fragment("body_it_gen_01", "it_ch",
             "in allegato trova la conferma d'ordine relativa al Suo acquisto del [[PII:date]]. "
             "La merce verrà spedita all'indirizzo [[PII:address]] nei prossimi tre giorni lavorativi."),
    Fragment("body_it_gen_02", "it_ch",
             "La informo che il prodotto da Lei richiesto è nuovamente disponibile a magazzino. "
             "Per procedere con l'ordine può rispondere a questo messaggio entro il [[PII:date]]."),
    Fragment("body_it_gen_03", "it_ch",
             "Confermiamo l'avvenuta spedizione del pacco n. 778-IT-04 in data odierna. "
             "Il tracciamento è consultabile sul sito [[PII:url]] inserendo il codice riportato in fattura."),
    Fragment("body_it_gen_04", "it_ch",
             "Con riferimento al Suo reclamo, abbiamo trasmesso la pratica al collega [[PII:title_last]], "
             "che La ricontatterà al numero [[PII:phone]] per concordare la sostituzione dell'articolo difettoso."),
    Fragment("body_it_gen_05", "it_ch",
             "Le ricordiamo che l'offerta promozionale riservata alla Sua azienda scade il [[PII:date]]. "
             "Per maggiori dettagli può consultare il catalogo aggiornato su [[PII:url]]."),
    Fragment("body_it_gen_06", "it_ch",
             "La merce ordinata risulta purtroppo esaurita; Le proponiamo un articolo equivalente al medesimo prezzo. "
             "Resto a disposizione allo [[PII:phone]] per concordare la sostituzione."),

    # --- banking / payment (6) ---
    Fragment("body_it_gen_07", "it_ch",
             "Le confermiamo l'avvenuta ricezione del bonifico effettuato sul conto [[PII:iban]]. "
             "La quietanza ufficiale Le verrà recapitata per posta entro il [[PII:date]]."),
    Fragment("body_it_gen_08", "it_ch",
             "La presente per sollecitare il saldo della fattura n. 2031/A, ancora aperta da oltre sessanta giorni. "
             "La preghiamo di provvedere al pagamento sul conto [[PII:iban]] al più presto."),
    Fragment("body_it_gen_09", "it_ch",
             "Per motivi di sicurezza la Sua vecchia carta con numero [[PII:cc]] è stata disattivata. "
             "La nuova tessera Le verrà spedita all'indirizzo registrato in archivio."),
    Fragment("body_it_gen_10", "it_ch",
             "Vorrei segnalarLe che il numero di partita IVA da indicare in fattura è [[PII:vat]]. "
             "La prego di aggiornare di conseguenza i dati nel Suo gestionale."),
    Fragment("body_it_gen_11", "it_ch",
             "Il rimborso da Lei richiesto, pari a CHF 845.30, è stato autorizzato in data [[PII:date]]. "
             "L'accredito comparirà sul conto [[PII:iban]] entro cinque giorni lavorativi."),
    Fragment("body_it_gen_12", "it_ch",
             "Come da nostri accordi telefonici, riepilogo le coordinate per il versamento della cauzione: "
             "IBAN [[PII:iban]], causale 'deposito locazione', importo CHF 2'400.-."),

    # --- HR / admin (6) ---
    Fragment("body_it_gen_13", "it_ch",
             "Con riferimento alla Sua candidatura, La invitiamo a un colloquio conoscitivo il [[PII:date]] "
             "presso la nostra sede di [[PII:address]]. La preghiamo di confermare la presenza."),
    Fragment("body_it_gen_14", "it_ch",
             "Per completare la pratica di assunzione abbiamo ancora bisogno del Suo numero AVS ([[PII:ahv]]) "
             "e di una copia del permesso di soggiorno aggiornata."),
    Fragment("body_it_gen_15", "it_ch",
             "Le comunico che a partire dal [[PII:date]] la nuova responsabile delle risorse umane sarà [[PII:title_first_last]]. "
             "Per qualsiasi questione contrattuale potrà rivolgersi direttamente a lei."),
    Fragment("body_it_gen_16", "it_ch",
             "Le scrivo per informarLa che la Sua domanda di congedo è stata accolta dal direttore [[PII:title_last]]. "
             "Buone vacanze e a presto in ufficio."),
    Fragment("body_it_gen_17", "it_ch",
             "Vorrei ricordarLe che la formazione obbligatoria sulla protezione dei dati si terrà online il [[PII:date]]; "
             "il link di collegamento è già stato inviato all'indirizzo [[PII:email]]."),
    Fragment("body_it_gen_18", "it_ch",
             "In seguito alle dimissioni di [[PII:title_first_last]], il Suo nuovo superiore diretto sarà il sottoscritto. "
             "Resto a disposizione per qualsiasi chiarimento."),

    # --- scheduling / meetings (4) ---
    Fragment("body_it_gen_19", "it_ch",
             "La riunione di coordinamento è confermata per il [[PII:date]] alle ore 14:30 nel mio ufficio in [[PII:address]]. "
             "La prego di portare con sé la documentazione aggiornata."),
    Fragment("body_it_gen_20", "it_ch",
             "Vorrei proporLe uno scambio di vedute la prossima settimana. "
             "Sarebbe disponibile per un caffè con me e [[PII:first]] nella mattinata di giovedì?"),
    Fragment("body_it_gen_21", "it_ch",
             "Purtroppo devo posticipare il nostro incontro previsto domani. "
             "Possiamo risentirci al [[PII:phone]] per concordare una nuova data?"),
    Fragment("body_it_gen_22", "it_ch",
             "Le confermo l'appuntamento del [[PII:date]] alle 10:00; "
             "ci troverà al secondo piano, lo stabile è quello accanto a [[PII:address]]."),

    # --- general correspondence (3) ---
    Fragment("body_it_gen_23", "it_ch",
             "Approfitto di questo messaggio per porgerLe i miei più sentiti auguri di buone feste, "
             "estesi a [[PII:first]] e a tutta la Sua famiglia."),
    Fragment("body_it_gen_24", "it_ch",
             "Mi permetto di presentarmi: sono [[PII:full]], nuovo referente per la regione del Mendrisiotto. "
             "Spero di poterLa incontrare di persona molto presto."),
    Fragment("body_it_gen_25", "it_ch",
             "La ringrazio sentitamente per la pronta risposta. "
             "Per qualsiasi ulteriore chiarimento può scrivermi a [[PII:email]] oppure telefonarmi al [[PII:phone]]."),
]
