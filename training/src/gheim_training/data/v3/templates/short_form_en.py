"""English short-form narrative fragments. Hand-written, 15 entries.
International Swiss-business context. NOT email greetings/signatures —
mid-document narrative, news-style, note-style, or conversational
snippets. Mix of US and UK conventions."""
from ..core import Fragment

TEMPLATES: list[Fragment] = [
    # --- bare first name, conversational ---
    Fragment("sf_en_01", "en",
             "Have you seen [[PII:first]] in the office today? "
             "I needed to hand over the Geneva file before lunch."),
    Fragment("sf_en_02", "en",
             "I tried calling [[PII:first]] twice this morning, "
             "but it went straight to voicemail both times."),
    Fragment("sf_en_03", "en",
             "Tell [[PII:first]] the audit folder is sitting on her desk "
             "next to the printer."),

    # --- bare last name, formal narrative / subject ---
    Fragment("sf_en_04", "en",
             "[[PII:last]] signed off on the revised vendor agreement "
             "shortly before the quarterly close."),
    Fragment("sf_en_05", "en",
             "According to [[PII:last]], the Zurich office will reopen "
             "on Tuesday once the renovation is finished."),

    # --- news-style with full name + role ---
    Fragment("sf_en_06", "en",
             "[[PII:full]], a managing partner at the firm since 2019, "
             "confirmed the relocation to Lausanne earlier today."),
    Fragment("sf_en_07", "en",
             "In a brief statement to reporters, [[PII:full]] denied any "
             "involvement in the procurement irregularities."),

    # --- title + last, formal ---
    Fragment("sf_en_08", "en",
             "[[PII:title_last]] is expected to address the shareholders "
             "at the AGM in Basel next month."),

    # --- multi-PII: name + date ---
    Fragment("sf_en_09", "en",
             "[[PII:full]] joined the Berne office on [[PII:date]] "
             "and has since led three cross-border investigations."),

    # --- multi-PII: name + phone ---
    Fragment("sf_en_10", "en",
             "For technical queries about the rollout, please reach "
             "[[PII:first]] directly on [[PII:phone]]."),

    # --- multi-PII: name + email, note style ---
    Fragment("sf_en_11", "en",
             "Quick note -- [[PII:last]] asked that all PO confirmations "
             "be cc'd to [[PII:email]] going forward."),

    # --- multi-PII: name + address, news style ---
    Fragment("sf_en_12", "en",
             "The package was eventually re-routed to [[PII:full]] at "
             "[[PII:address]] after a week-long delay at customs."),

    # --- multi-PII: name + url ---
    Fragment("sf_en_13", "en",
             "A full transcript of the interview with [[PII:full]] is "
             "now available at [[PII:url]]."),

    # --- two names, conversational ---
    Fragment("sf_en_14", "en",
             "[[PII:first]] and [[PII:first]] are flying in from London "
             "on Thursday for the strategy workshop."),

    # --- last-name-first listing, narrative ---
    Fragment("sf_en_15", "en",
             "The witness list also included [[PII:last_first]], which "
             "frankly came as a surprise to everyone in the room."),
]
