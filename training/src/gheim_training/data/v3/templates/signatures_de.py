"""German email signature fragments. Hand-written, 25 entries.
Use \\n for line breaks within the signature."""
from ..core import Fragment

TEMPLATES: list[Fragment] = [
    Fragment("sig_de_01", "de_ch", "Liebe Grüsse,\n[[PII:first]]"),
    Fragment("sig_de_02", "de_ch",
             "Freundliche Grüsse\n[[PII:full]]\n[[PII:phone]] | [[PII:email]]"),
    Fragment("sig_de_03", "de_ch", "Gruss\n[[PII:first]]"),
    Fragment("sig_de_04", "de_ch",
             "Mit freundlichen Grüssen\n[[PII:title_first_last]]\n"
             "Geschäftsführer\n[[PII:phone]]\n[[PII:email]]\n[[PII:address]]"),
    Fragment("sig_de_05", "de_ch", "Beste Grüsse aus Zürich\n[[PII:first]]"),
    Fragment("sig_de_06", "de_ch",
             "Herzlich,\n[[PII:full]]\nT [[PII:phone]]"),
    Fragment("sig_de_07", "de_ch", "Bis bald!\n[[PII:first]]"),
    Fragment("sig_de_08", "de_ch",
             "Viele Grüsse\n[[PII:full]]\n[[PII:email]]"),
    Fragment("sig_de_09", "de_ch",
             "Mit besten Grüssen\n[[PII:title_first_last]]\nRechtsanwalt\n"
             "Tel. [[PII:phone]] · [[PII:url]]"),
    Fragment("sig_de_10", "de_ch", "Tschüss\n[[PII:first]]"),
    Fragment("sig_de_11", "de_ch",
             "Gerne stehe ich für Rückfragen zur Verfügung.\n\n"
             "Freundliche Grüsse\n[[PII:full]]"),
    Fragment("sig_de_12", "de_ch",
             "Beste Grüsse\n[[PII:first]]\n--\nMobile: [[PII:phone]]"),
    Fragment("sig_de_13", "de_ch",
             "Mit freundlichen Grüssen\nHerr [[PII:last]]"),
    Fragment("sig_de_14", "de_ch",
             "Liebe Grüsse\n[[PII:full]]\n"
             "[[PII:address]]\n[[PII:phone]]\n[[PII:email]]\n[[PII:url]]"),
    Fragment("sig_de_15", "de_ch",
             "Gruss & Dank\n[[PII:first]]\n[[PII:email]]"),
    Fragment("sig_de_16", "de_ch",
             "Herzliche Grüsse\n[[PII:title_first_last]]\n"
             "Leiterin Personal | Tel [[PII:phone]]"),
    Fragment("sig_de_17", "de_ch", "Schöne Woche noch,\n[[PII:first]]"),
    Fragment("sig_de_18", "de_ch",
             "Mit kollegialen Grüssen\n[[PII:full]]\n"
             "Direkt: [[PII:phone]]\nE-Mail: [[PII:email]]"),
    Fragment("sig_de_19", "de_ch",
             "Freundliche Grüsse aus dem Homeoffice\n[[PII:first]]"),
    Fragment("sig_de_20", "de_ch",
             "Hochachtungsvoll\n[[PII:title_first_last]]"),
    Fragment("sig_de_21", "de_ch",
             "Danke und Gruss\n[[PII:first]]\n"
             "(erreichbar unter [[PII:phone]])"),
    Fragment("sig_de_22", "de_ch",
             "Sonnige Grüsse\n[[PII:full]]\n[[PII:email]] · [[PII:phone]]"),
    Fragment("sig_de_23", "de_ch", "Cheers\n[[PII:first]] // [[PII:email]]"),
    Fragment("sig_de_24", "de_ch",
             "Bis dann, [[PII:first]]"),
    Fragment("sig_de_25", "de_ch",
             "Mit freundlichen Grüssen\n[[PII:title_first_last]]\n"
             "Senior Consultant\n[[PII:address]]\n"
             "M [[PII:phone]] | [[PII:email]] | [[PII:url]]"),
]
