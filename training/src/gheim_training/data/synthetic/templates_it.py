"""Swiss Italian (IT-CH) templates."""
from __future__ import annotations

import random
from collections.abc import Callable

from . import faker_ch as F


def _const(value: str) -> Callable[[], str]:
    return lambda: value


def _bank_email() -> dict[str, Callable[[], str]]:
    name = F.name_it()
    return {
        "name": _const(name),
        "email": _const(F.email_ch(name)),
        "iban": F.iban_ch,
        "date": F.date_it,
    }


def _hr_letter() -> dict[str, Callable[[], str]]:
    name = F.name_it()
    return {
        "name": _const(name),
        "name2": F.name_it,
        "dob": F.date_it,
        "addr": F.address_it,
        "start": F.date_it,
        "end": F.date_it,
        "ahv": F.ahv,
        "phone": F.phone_ch,
    }


def _kyc_form() -> dict[str, Callable[[], str]]:
    name = F.name_it()
    return {
        "name": _const(name),
        "dob": F.date_it,
        "addr": F.address_it,
        "phone": F.phone_ch,
        "email": _const(F.email_ch(name)),
        "ahv": F.ahv,
        "iban": F.iban_ch,
        "uid": F.vat_che,
    }


def _support_ticket() -> dict[str, Callable[[], str]]:
    name = F.name_it()
    return {
        "name": _const(name),
        "client_id": F.iban_ch,
        "email": _const(F.email_ch(name)),
        "phone": F.phone_ch,
        "url": F.url_ch,
    }


def _secret_leak() -> dict[str, Callable[[], str]]:
    name = F.name_it()
    return {
        "name": _const(name),
        "email": _const(F.email_ch(name)),
        "secret": F.secret_token,
    }


TEMPLATES: list[tuple[str, str, Callable[[], dict[str, Callable[[], str]]]]] = [
    (
        "bank_email_v1",
        "Gentile {{name:private_person}}\n\n"
        "Confermiamo la ricezione del Suo versamento del {{date:private_date}} "
        "sul conto {{iban:account_number}}. Per qualsiasi domanda, ci contatti "
        "all'indirizzo {{email:private_email}}.\n\nCordiali saluti\nUBS",
        _bank_email,
    ),
    (
        "hr_letter_v1",
        "Certificato di lavoro di {{name:private_person}}, nato/a il {{dob:private_date}}, "
        "domiciliato/a in {{addr:private_address}}.\n\n"
        "Sig./Sig.ra {{name2:private_person}} è stato/a impiegato/a presso di noi dal "
        "{{start:private_date}} al {{end:private_date}}. N. AVS: {{ahv:account_number}}. "
        "Reperibile al {{phone:private_phone}}.",
        _hr_letter,
    ),
    (
        "kyc_form_v1",
        "Modulo di identificazione (LRD)\n\n"
        "Nome completo: {{name:private_person}}\n"
        "Data di nascita: {{dob:private_date}}\n"
        "Indirizzo: {{addr:private_address}}\n"
        "Telefono: {{phone:private_phone}}\n"
        "E-mail: {{email:private_email}}\n"
        "N. AVS: {{ahv:account_number}}\n"
        "IBAN: {{iban:account_number}}\n"
        "N. IDI: {{uid:account_number}}",
        _kyc_form,
    ),
    (
        "support_ticket_v1",
        "Buongiorno\n\nMi chiamo {{name:private_person}} e ho un problema con "
        "il mio conto. Il mio numero cliente è {{client_id:account_number}}, "
        "registrato come {{email:private_email}}. Vi prego di richiamarmi al "
        "{{phone:private_phone}}. Maggiori informazioni: {{url:private_url}}.\n\n"
        "Cordialmente",
        _support_ticket,
    ),
    (
        "secret_leak_v1",
        "Nota: nell'e-mail di {{name:private_person}} ({{email:private_email}}) "
        "è stata accidentalmente divulgata la chiave API {{secret:secret}}. "
        "Si prega di revocarla immediatamente.",
        _secret_leak,
    ),
]


def random_template() -> tuple[str, str, Callable[[], dict[str, Callable[[], str]]]]:
    return random.choice(TEMPLATES)
