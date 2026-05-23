"""French email signature fragments. Hand-written, 15 entries.
Swiss-French business style. Use \\n for line breaks."""
from ..core import Fragment

TEMPLATES: list[Fragment] = [
    Fragment("sig_fr_01", "fr_ch", "Cordialement,\n[[PII:first]]"),
    Fragment("sig_fr_02", "fr_ch",
             "Bien à vous,\n[[PII:full]]\n[[PII:phone]] | [[PII:email]]"),
    Fragment("sig_fr_03", "fr_ch", "À bientôt,\n[[PII:first]]"),
    Fragment("sig_fr_04", "fr_ch",
             "Veuillez agréer, Madame, Monsieur, mes salutations distinguées.\n\n"
             "[[PII:title_first_last]]\nDirectrice Générale\n"
             "[[PII:phone]]\n[[PII:email]]\n[[PII:address]]"),
    Fragment("sig_fr_05", "fr_ch",
             "Sincères salutations\n[[PII:full]]\nTél. [[PII:phone]]"),
    Fragment("sig_fr_06", "fr_ch",
             "Belle journée,\n[[PII:first]]\n[[PII:email]]"),
    Fragment("sig_fr_07", "fr_ch",
             "Dans l'attente de votre retour, je vous prie d'agréer "
             "mes meilleures salutations.\n\n[[PII:title_first_last]]"),
    Fragment("sig_fr_08", "fr_ch",
             "Salutations\n[[PII:first]] -- [[PII:phone]]"),
    Fragment("sig_fr_09", "fr_ch",
             "Merci d'avance et meilleures salutations\n[[PII:full]]\n[[PII:email]]"),
    Fragment("sig_fr_10", "fr_ch",
             "Amitiés depuis Lausanne,\n[[PII:first]]"),
    Fragment("sig_fr_11", "fr_ch",
             "Meilleures salutations\nM. [[PII:last]]"),
    Fragment("sig_fr_12", "fr_ch",
             "Bien cordialement\n[[PII:full]]\n"
             "[[PII:address]]\nM [[PII:phone]] · [[PII:url]]"),
    Fragment("sig_fr_13", "fr_ch",
             "Salutations distinguées\n[[PII:title_first_last]]\n"
             "Responsable des Ventes | [[PII:phone]] | [[PII:email]]"),
    Fragment("sig_fr_14", "fr_ch",
             "Je reste à votre disposition pour tout complément d'information.\n\n"
             "Avec mes meilleurs sentiments,\n[[PII:full]]\n"
             "Ligne directe: [[PII:phone]]\nCourriel: [[PII:email]]\n[[PII:url]]"),
    Fragment("sig_fr_15", "fr_ch",
             "Bonne fin de semaine,\n[[PII:first]]"),
]
