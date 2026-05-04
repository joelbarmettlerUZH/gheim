"""Standard German (DE-CH) templates.

Each entry is (template_id, template_string, fillers_factory). The factory
returns a fresh dict of slot → callable on each call so per-render randomness
works. ``_consistent`` ties two slot values together (e.g. name → email derived
from name) so a single render is internally coherent.
"""
from __future__ import annotations

import random
from collections.abc import Callable

from . import faker_ch as F


def _const(value: str) -> Callable[[], str]:
    return lambda: value


def _bank_email() -> dict[str, Callable[[], str]]:
    name = F.name_de()
    return {
        "name": _const(name),
        "email": _const(F.email_ch(name)),
        "iban": F.iban_ch,
        "amount_date": F.date_de,
    }


def _hr_letter() -> dict[str, Callable[[], str]]:
    name = F.name_de()
    return {
        "name": _const(name),
        "name2": F.name_de,
        "dob": F.date_de,
        "addr": F.address_de,
        "start": F.date_de,
        "end": F.date_de,
        "ahv": F.ahv,
        "phone": F.phone_ch,
    }


def _kyc_form() -> dict[str, Callable[[], str]]:
    name = F.name_de()
    return {
        "name": _const(name),
        "dob": F.date_de,
        "addr": F.address_de,
        "phone": F.phone_ch,
        "email": _const(F.email_ch(name)),
        "ahv": F.ahv,
        "iban": F.iban_ch,
        "uid": F.vat_che,
    }


def _doctor_note() -> dict[str, Callable[[], str]]:
    name = F.name_de()
    return {
        "name": _const(name),
        "dob": F.date_de,
        "addr": F.address_de,
        "visit": F.date_de,
        "email": _const(F.email_ch(name)),
        "phone": F.phone_ch,
    }


def _support_ticket() -> dict[str, Callable[[], str]]:
    name = F.name_de()
    return {
        "name": _const(name),
        "kundennr": F.iban_ch,
        "email": _const(F.email_ch(name)),
        "phone": F.phone_ch,
        "url": F.url_ch,
    }


def _secret_leak() -> dict[str, Callable[[], str]]:
    name = F.name_de()
    return {
        "name": _const(name),
        "email": _const(F.email_ch(name)),
        "secret": F.secret_token,
    }


TEMPLATES: list[tuple[str, str, Callable[[], dict[str, Callable[[], str]]]]] = [
    (
        "bank_email_v1",
        "Sehr geehrte/r {{name:private_person}}\n\n"
        "Wir bestätigen den Eingang Ihrer Zahlung vom {{amount_date:private_date}} "
        "auf das Konto {{iban:account_number}}. Bei Rückfragen erreichen Sie uns "
        "unter {{email:private_email}}.\n\nFreundliche Grüsse\nIhre UBS",
        _bank_email,
    ),
    (
        "hr_letter_v1",
        "Arbeitszeugnis für {{name:private_person}}, geboren am {{dob:private_date}}, "
        "wohnhaft an {{addr:private_address}}.\n\n"
        "Frau/Herr {{name2:private_person}} war vom {{start:private_date}} "
        "bis {{end:private_date}} bei uns angestellt. AHV-Nummer: "
        "{{ahv:account_number}}. Telefonisch erreichbar unter {{phone:private_phone}}.",
        _hr_letter,
    ),
    (
        "kyc_form_v1",
        "Identifikationsformular gemäss GwG\n\n"
        "Vollständiger Name: {{name:private_person}}\n"
        "Geburtsdatum: {{dob:private_date}}\n"
        "Adresse: {{addr:private_address}}\n"
        "Telefon: {{phone:private_phone}}\n"
        "E-Mail: {{email:private_email}}\n"
        "AHV-Nr.: {{ahv:account_number}}\n"
        "IBAN: {{iban:account_number}}\n"
        "UID-Nr.: {{uid:account_number}}",
        _kyc_form,
    ),
    (
        "doctor_note_v1",
        "Arztbericht\n\nPatient: {{name:private_person}}, {{dob:private_date}}\n"
        "Wohnhaft: {{addr:private_address}}\n"
        "Konsultation am {{visit:private_date}}.\n"
        "Befund: Patient klagt über Kopfschmerzen seit drei Tagen. "
        "Rückfragen an {{email:private_email}} oder telefonisch unter {{phone:private_phone}}.",
        _doctor_note,
    ),
    (
        "support_ticket_v1",
        "Hallo Support\n\nIch heisse {{name:private_person}} und habe ein Problem "
        "mit meinem Konto. Meine Kundennummer ist {{kundennr:account_number}}, "
        "registriert auf {{email:private_email}}. Bitte rufen Sie mich unter "
        "{{phone:private_phone}} zurück. Mehr Infos: {{url:private_url}}.\n\n"
        "Freundliche Grüsse",
        _support_ticket,
    ),
    (
        "secret_leak_v1",
        "Hinweis: in der Mail von {{name:private_person}} ({{email:private_email}}) "
        "war versehentlich der API-Schlüssel {{secret:secret}} enthalten. "
        "Bitte umgehend rotieren.",
        _secret_leak,
    ),
]


def random_template() -> tuple[str, str, Callable[[], dict[str, Callable[[], str]]]]:
    return random.choice(TEMPLATES)
