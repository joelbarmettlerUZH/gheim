"""Romansh email greeting fragments. Hand-written, 20 entries. RM is data-thin; we
use short, cognate-heavy greetings rather than complex idioms (no native review
available)."""
from ..core import Fragment

TEMPLATES: list[Fragment] = [
    Fragment("greet_rm_01", "rm", "Hai [[PII:first]], grazia per tia communicaziun."),
    Fragment("greet_rm_02", "rm", "Bun di Sgnr. [[PII:last]],"),
    Fragment("greet_rm_03", "rm", "Char [[PII:first]],"),
    Fragment("greet_rm_04", "rm", "Stimada Sgnra. [[PII:last]]"),
    Fragment("greet_rm_05", "rm", "Salida [[PII:first]]!"),
    Fragment("greet_rm_06", "rm", "Bun saira [[PII:first]] – in cuort messadi."),
    Fragment("greet_rm_07", "rm", "Stimà [[PII:title_last]],\n"),
    Fragment("greet_rm_08", "rm", "Hai [[PII:full]], co vas?"),
    Fragment("greet_rm_09", "rm", "Chara [[PII:first]], grazia fitg."),
    Fragment("greet_rm_10", "rm", "A: [[PII:title_first_last]]"),
    Fragment("greet_rm_11", "rm", "Bun di chara [[PII:first]],"),
    Fragment("greet_rm_12", "rm", "Stimada [[PII:title_last]]!"),
    Fragment("greet_rm_13", "rm", "Salida Sgnr. [[PII:last]] – in pitschen rumitg."),
    Fragment("greet_rm_14", "rm", "Hai [[PII:first]]\n\ngrazia per tia risposta."),
    Fragment("greet_rm_15", "rm", "Bun di [[PII:title_first_last]],"),
    Fragment("greet_rm_16", "rm", "Char Sgnr. [[PII:last]]"),
    Fragment("greet_rm_17", "rm", "Salida a tuts, e spezialmain a [[PII:first]]"),
    Fragment("greet_rm_18", "rm", "Stimà [[PII:title_first_last]]:"),
    Fragment("greet_rm_19", "rm", "Hai [[PII:first]], in salid da Cuira."),
    Fragment("greet_rm_20", "rm", "Chara Sgnra. [[PII:last]],"),
]
