"""Swiss French (FR-CH) templates."""
from __future__ import annotations

import random
from collections.abc import Callable

from .. import faker_ch as F


def _const(value: str) -> Callable[[], str]:
    return lambda: value


def _bank_email() -> dict[str, Callable[[], str]]:
    name = F.name_fr()
    return {
        "name": _const(name),
        "email": _const(F.email_ch(name)),
        "iban": F.iban_ch,
        "date": F.date_fr,
    }


def _hr_letter() -> dict[str, Callable[[], str]]:
    name = F.name_fr()
    return {
        "name": _const(name),
        "name2": F.name_fr,
        "dob": F.date_fr,
        "addr": F.address_fr,
        "start": F.date_fr,
        "end": F.date_fr,
        "ahv": F.ahv,
        "phone": F.phone_ch,
    }


def _kyc_form() -> dict[str, Callable[[], str]]:
    name = F.name_fr()
    return {
        "name": _const(name),
        "dob": F.date_fr,
        "addr": F.address_fr,
        "phone": F.phone_ch,
        "email": _const(F.email_ch(name)),
        "ahv": F.ahv,
        "iban": F.iban_ch,
        "uid": F.vat_che,
    }


def _support_ticket() -> dict[str, Callable[[], str]]:
    name = F.name_fr()
    return {
        "name": _const(name),
        "client_id": F.iban_ch,
        "email": _const(F.email_ch(name)),
        "phone": F.phone_ch,
        "url": F.url_ch,
    }


def _secret_leak() -> dict[str, Callable[[], str]]:
    name = F.name_fr()
    return {
        "name": _const(name),
        "email": _const(F.email_ch(name)),
        "secret": F.secret_token,
    }


TEMPLATES: list[tuple[str, str, Callable[[], dict[str, Callable[[], str]]]]] = [
    (
        "bank_email_v1",
        "Madame, Monsieur {{name:private_person}}\n\n"
        "Nous accusons réception de votre versement du {{date:private_date}} "
        "sur le compte {{iban:account_number}}. Pour toute question, contactez-nous "
        "à {{email:private_email}}.\n\nMeilleures salutations\nUBS",
        _bank_email,
    ),
    (
        "hr_letter_v1",
        "Certificat de travail de {{name:private_person}}, né(e) le {{dob:private_date}}, "
        "domicilié(e) à {{addr:private_address}}.\n\n"
        "M./Mme {{name2:private_person}} a été employé(e) chez nous du {{start:private_date}} "
        "au {{end:private_date}}. N° AVS : {{ahv:account_number}}. "
        "Joignable au {{phone:private_phone}}.",
        _hr_letter,
    ),
    (
        "kyc_form_v1",
        "Formulaire d'identification (LBA)\n\n"
        "Nom complet : {{name:private_person}}\n"
        "Date de naissance : {{dob:private_date}}\n"
        "Adresse : {{addr:private_address}}\n"
        "Téléphone : {{phone:private_phone}}\n"
        "E-mail : {{email:private_email}}\n"
        "N° AVS : {{ahv:account_number}}\n"
        "IBAN : {{iban:account_number}}\n"
        "N° IDE : {{uid:account_number}}",
        _kyc_form,
    ),
    (
        "support_ticket_v1",
        "Bonjour\n\nJe m'appelle {{name:private_person}} et j'ai un problème "
        "avec mon compte. Mon numéro client est {{client_id:account_number}}, "
        "enregistré sous {{email:private_email}}. Merci de me rappeler au "
        "{{phone:private_phone}}. Plus d'infos : {{url:private_url}}.\n\n"
        "Cordialement",
        _support_ticket,
    ),
    (
        "secret_leak_v1",
        "Note : dans le courriel de {{name:private_person}} ({{email:private_email}}), "
        "la clé API {{secret:secret}} a été divulguée par erreur. "
        "Veuillez la révoquer immédiatement.",
        _secret_leak,
    ),
]


def random_template() -> tuple[str, str, Callable[[], dict[str, Callable[[], str]]]]:
    return random.choice(TEMPLATES)
