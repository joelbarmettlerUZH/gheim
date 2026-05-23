"""English email body fragments — international Swiss-company correspondence.
Hand-written, 20 entries. Mix US and UK English."""
from ..core import Fragment

TEMPLATES: list[Fragment] = [
    # --- commerce / ordering (5) ---
    Fragment("body_en_gen_01", "en",
             "please find attached the signed purchase order. "
             "Feel free to reach me at [[PII:phone]] should any clarifications be needed."),
    Fragment("body_en_gen_02", "en",
             "we have shipped your order #88421 to [[PII:address]] this morning; "
             "tracking details will follow under separate cover."),
    Fragment("body_en_gen_03", "en",
             "could you kindly confirm the delivery window for the Zurich consignment? "
             "Our warehouse manager [[PII:first]] needs to plan the dock by [[PII:date/iso]]."),
    Fragment("body_en_gen_04", "en",
             "the revised quotation totalling CHF 12'480 has been uploaded to [[PII:url]]; "
             "please countersign within five working days."),
    Fragment("body_en_gen_05", "en",
             "unfortunately two of the four crates arrived damaged. "
             "We have escalated the case to [[PII:title_first_last]] for immediate replacement."),

    # --- banking / payment (5) ---
    Fragment("body_en_gen_06", "en",
             "kindly remit the outstanding balance of CHF 4'320.50 to our Swiss IBAN "
             "[[PII:iban]] no later than [[PII:date]]."),
    Fragment("body_en_gen_07", "en",
             "your refund has been processed today and should reach the account on file within three business days."),
    Fragment("body_en_gen_08", "en",
             "for the wire transfer please use the reference INV-2026-0418 and quote our VAT number [[PII:vat]] "
             "to avoid customs delays."),
    Fragment("body_en_gen_09", "en",
             "we have noticed an unrecognised charge on card ending [[PII:cc]]; "
             "could you call our fraud desk on [[PII:phone/intl_spaced]] at your earliest convenience?"),
    Fragment("body_en_gen_10", "en",
             "the credit note in favour of [[PII:full]] has been booked against the original invoice "
             "and is reflected in this month's statement."),

    # --- HR / admin (5) ---
    Fragment("body_en_gen_11", "en",
             "to finalise the onboarding paperwork, please send your AHV number along with a scanned passport copy to [[PII:email]]."),
    Fragment("body_en_gen_12", "en",
             "we are delighted to confirm your start date as [[PII:date/spelled]]; "
             "[[PII:title_last]] will meet you at reception at 09:00."),
    Fragment("body_en_gen_13", "en",
             "kindly note that [[PII:first]] is on parental leave until further notice -- "
             "for urgent HR matters please contact the duty officer."),
    Fragment("body_en_gen_14", "en",
             "the updated employee handbook has been published internally; "
             "you can review it under your personal profile and acknowledge receipt by month-end."),
    Fragment("body_en_gen_15", "en",
             "please be advised that the new home address we have on record for you is [[PII:address]]; "
             "let me know if any detail is incorrect."),

    # --- scheduling (3) ---
    Fragment("body_en_gen_16", "en",
             "the steering committee will reconvene on [[PII:date/dot]] at 14:00 CET via the usual Teams link."),
    Fragment("body_en_gen_17", "en",
             "would [[PII:date]] suit you for a thirty-minute catch-up? "
             "If the morning slot does not work, my assistant can offer alternatives."),
    Fragment("body_en_gen_18", "en",
             "I have provisionally booked the boardroom for our visit with [[PII:title_initial_last]]; "
             "kindly confirm attendance so I can organise catering."),

    # --- general (2) ---
    Fragment("body_en_gen_19", "en",
             "thank you for the warm welcome in Basel last week -- the team at [[PII:url]] truly enjoyed the factory tour."),
    Fragment("body_en_gen_20", "en",
             "I am out of the office travelling between sites; for anything time-sensitive please reach me on [[PII:phone]] "
             "or contact [[PII:first]] who has full visibility on my files."),
]
