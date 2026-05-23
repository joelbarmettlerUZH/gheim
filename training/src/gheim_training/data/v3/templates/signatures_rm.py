"""Romansh email signature fragments. Hand-written, 10 entries.
Use \\n for line breaks within the signature."""
from ..core import Fragment

TEMPLATES: list[Fragment] = [
    Fragment("sig_rm_01", "rm", "Cordials salids,\n[[PII:first]]"),
    Fragment("sig_rm_02", "rm",
             "Salids\n[[PII:full]]\n[[PII:phone]] | [[PII:email]]"),
    Fragment("sig_rm_03", "rm", "Grazia e bun di\n[[PII:first]]"),
    Fragment("sig_rm_04", "rm",
             "Cordialmain\n[[PII:title_first_last]]\n"
             "Directur\n[[PII:phone]]\n[[PII:email]]\n[[PII:address]]"),
    Fragment("sig_rm_05", "rm",
             "Cun bun saluts da Cuira\n[[PII:first]]"),
    Fragment("sig_rm_06", "rm",
             "A revair,\n[[PII:full]]\nTel. [[PII:phone]]"),
    Fragment("sig_rm_07", "rm",
             "In attais Vossa risposta\n[[PII:title_first_last]]\n"
             "Advocat | [[PII:url]]"),
    Fragment("sig_rm_08", "rm",
             "Salids e grazia fitg\n[[PII:first]] – [[PII:email]]"),
    Fragment("sig_rm_09", "rm",
             "Cordials salids\nSgnr. [[PII:last]]"),
    Fragment("sig_rm_10", "rm",
             "Cordialmain Vossa\n[[PII:full]]\n"
             "[[PII:address]]\n[[PII:phone]] · [[PII:email]] · [[PII:url]]"),
]
