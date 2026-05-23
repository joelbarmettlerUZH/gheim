"""Italian email signature fragments. Hand-written, 15 entries.
Use \\n for line breaks within the signature."""
from ..core import Fragment

TEMPLATES: list[Fragment] = [
    Fragment("sig_it_01", "it_ch", "Cordiali saluti,\n[[PII:first]]"),
    Fragment("sig_it_02", "it_ch",
             "Distinti saluti\n[[PII:full]]\n[[PII:phone]] | [[PII:email]]"),
    Fragment("sig_it_03", "it_ch", "A presto!\n[[PII:first]]"),
    Fragment("sig_it_04", "it_ch",
             "Con i migliori saluti\n[[PII:title_first_last]]\n"
             "Direttore Generale\n[[PII:phone]]\n[[PII:email]]\n[[PII:address]]"),
    Fragment("sig_it_05", "it_ch",
             "Cordialmente\n[[PII:full]]\nTel. [[PII:phone]]"),
    Fragment("sig_it_06", "it_ch",
             "Buona giornata,\n[[PII:first]]\n[[PII:email]]"),
    Fragment("sig_it_07", "it_ch",
             "In attesa di un Suo riscontro, porgo distinti saluti.\n\n"
             "[[PII:title_first_last]]"),
    Fragment("sig_it_08", "it_ch",
             "Saluti\n[[PII:first]] -- [[PII:phone]]"),
    Fragment("sig_it_09", "it_ch",
             "Grazie e cordiali saluti\n[[PII:full]]\n[[PII:email]]"),
    Fragment("sig_it_10", "it_ch",
             "Un caro saluto da Lugano,\n[[PII:first]]"),
    Fragment("sig_it_11", "it_ch",
             "Cordiali saluti\nSig. [[PII:last]]"),
    Fragment("sig_it_12", "it_ch",
             "Con i migliori saluti\n[[PII:full]]\n"
             "[[PII:address]]\nM [[PII:phone]] · [[PII:url]]"),
    Fragment("sig_it_13", "it_ch",
             "Distinti saluti\n[[PII:title_first_last]]\n"
             "Responsabile Vendite | [[PII:phone]] | [[PII:email]]"),
    Fragment("sig_it_14", "it_ch",
             "Resto a disposizione per qualsiasi chiarimento.\n\n"
             "Cordiali saluti\n[[PII:full]]\nDiretto: [[PII:phone]]\n"
             "E-mail: [[PII:email]]\n[[PII:url]]"),
    Fragment("sig_it_15", "it_ch",
             "Ciao,\n[[PII:first]]"),
]
