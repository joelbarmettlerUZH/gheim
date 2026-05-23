"""English email signature fragments. Hand-written, 10 entries.
International Swiss-company business style. Use \\n for line breaks."""
from ..core import Fragment

TEMPLATES: list[Fragment] = [
    Fragment("sig_en_01", "en", "Best regards,\n[[PII:first]]"),
    Fragment("sig_en_02", "en",
             "Kind regards\n[[PII:full]]\n[[PII:phone]] | [[PII:email]]"),
    Fragment("sig_en_03", "en", "Cheers,\n[[PII:first]]"),
    Fragment("sig_en_04", "en",
             "Yours sincerely,\n[[PII:title_first_last]]\n"
             "Chief Financial Officer\n[[PII:phone]]\n[[PII:email]]\n"
             "[[PII:address]]"),
    Fragment("sig_en_05", "en", "Thanks!\n[[PII:first]]"),
    Fragment("sig_en_06", "en",
             "Warmly,\n[[PII:full]]\nM +41 [[PII:phone]]"),
    Fragment("sig_en_07", "en",
             "Yours faithfully,\nMr [[PII:last]]"),
    Fragment("sig_en_08", "en",
             "All the best\n[[PII:title_first_last]] · Head of Operations\n"
             "[[PII:email]] | [[PII:url]]"),
    Fragment("sig_en_09", "en",
             "Sincerely,\n[[PII:full]]\n[[PII:email]]"),
    Fragment("sig_en_10", "en",
             "Please do not hesitate to get in touch should you have "
             "any further questions.\n\nRegards\n[[PII:first]]"),
]
