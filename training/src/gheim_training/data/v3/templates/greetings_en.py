"""English email greeting fragments. Hand-written, 15 entries.
International Swiss-company correspondence style."""
from ..core import Fragment

TEMPLATES: list[Fragment] = [
    Fragment("greet_en_01", "en", "Hi [[PII:first]], thanks for your message."),
    Fragment("greet_en_02", "en", "Dear [[PII:title_last]],"),
    Fragment("greet_en_03", "en", "Hello [[PII:first]] -- hope you're having a good week."),
    Fragment("greet_en_04", "en", "Dear [[PII:title_last]]:"),
    Fragment("greet_en_05", "en", "Good morning [[PII:first]],\nquick follow-up on yesterday's call."),
    Fragment("greet_en_06", "en", "Hey [[PII:first]]!"),
    Fragment("greet_en_07", "en", "To whom it may concern,\nattn: [[PII:title_first_last]]"),
    Fragment("greet_en_08", "en", "Greetings [[PII:title_first_last]],"),
    Fragment("greet_en_09", "en", "Good afternoon, [[PII:first]] -- circling back on the proposal."),
    Fragment("greet_en_10", "en", "Dear Sir or Madam,\nFAO [[PII:title_last]]"),
    Fragment("greet_en_11", "en", "Hi [[PII:full]], lovely to e-meet you."),
    Fragment("greet_en_12", "en", "Dear [[PII:title_last]];\nfurther to our exchange last week,"),
    Fragment("greet_en_13", "en", "Morning [[PII:first]]."),
    Fragment("greet_en_14", "en", "Hello everyone, and a special hello to [[PII:first]],"),
    Fragment("greet_en_15", "en", "Dear Ms [[PII:last]]\n\nI trust this email finds you well."),
]
