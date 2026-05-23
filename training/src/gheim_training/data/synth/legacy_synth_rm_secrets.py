"""Synthetic Romansh-context chunks with secret slots.

The v2_balanced dataset has just **one** RM × secret span in the entire
171k corpus (Layer 9 was never instructed to generate secrets, Layer 1
has no RM templates). This module fills that gap with 500-800 templated
RM technical-context chunks, each with one or two secret values inserted
at deterministic offsets.

Approach is pure-template, not LLM-generated:

- RM templates use short technical sentences with high-cognate
  vocabulary (cf. German `Token`, `API`, `Konfiguration` / Italian
  `chiave`, `password` / English `bearer`). This keeps native-quality
  RM out of the requirement loop — we don't have a Romansh reviewer.
- Secret values come from ``faker_ch.secret_token()`` which already
  produces format-correct OpenAI / GitHub / AWS / Slack / JWT tokens.
- One template = one chunk = one (start, end, label=secret) span. Two
  template variants insert a 2nd PII (a Faker name or generic phone)
  to add multi-PII chunks.

Output is Layer 9 schema:
``{text, spans, language, source, template_id, meta}``
written to ``data/layer_rm_secrets.jsonl``. The V2-9 balancer treats it
as a third synthetic source (``synthetic_rm_secrets``) so the per-cell
caps apply normally.

Run
---
    uv run python -m gheim_training.data.synth.legacy_synth_rm_secrets
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ..synth.faker_ch import name_rm, phone_ch, secret_token, seed_all

OUT_PATH = Path("data/layer_rm_secrets.jsonl")
DEFAULT_N = 800
SEED = 20260523

# Templates with exactly one {secret} slot each. Some also include
# {name} or {phone} to produce multi-PII chunks (matches v2's 35.5%
# multi-PII target). Each template is short enough that the secret
# regex catalogue would catch most insertions — but the model also
# needs to see the surrounding RM context.
_TEMPLATES: tuple[tuple[str, str, str | None], ...] = (
    # (template_id, body, secondary_slot)
    ("rm_secret_api_token_v1",
     "Igl token API per acceder al servetsch: {secret}. Tegnair segira.",
     None),
    ("rm_secret_falla_auth_v1",
     "Falla d'autenticaziun: la clav '{secret}' ei sbagliada. Empruvai puspei.",
     None),
    ("rm_secret_config_env_v1",
     "Configurar API_KEY={secret} en igl agen file de configuraziun.",
     None),
    ("rm_secret_db_password_v1",
     "Per acceder a la database, dovrar: PASSWORD={secret}",
     None),
    ("rm_secret_github_pat_v1",
     "GitHub Personal Access Token: {secret} - na divulgar betg!",
     None),
    ("rm_secret_aws_v1",
     "AWS_SECRET_ACCESS_KEY = {secret} # tegnair segira en igl vault",
     None),
    ("rm_secret_openai_v1",
     "OpenAI API key: {secret}. Quei ei privat - na cundivider mai.",
     None),
    ("rm_secret_stripe_v1",
     "Stripe webhook secret: {secret} (privat, na committer en git)",
     None),
    ("rm_secret_oauth_bearer_v1",
     "Bearer-Token per OAuth: Authorization: Bearer {secret}",
     None),
    ("rm_secret_curl_v1",
     "curl -H 'Authorization: Bearer {secret}' https://api.example.ch/v1/users",
     None),
    ("rm_secret_user_generated_v1",
     "L'usuari {name} ha generau ina nova clav d'API: {secret}",
     "name"),
    ("rm_secret_email_share_v1",
     "Cara {name}, igl token sgnir ei: {secret}. Empruvai cun quei valur.",
     "name"),
    ("rm_secret_support_ticket_v1",
     "Ticket de support da {name}: l'API-token {secret} dat ina falla 401.",
     "name"),
    ("rm_secret_kyc_form_v1",
     "Numer da telefon: {phone}. Token d'autenticaziun: {secret}",
     "phone"),
    ("rm_secret_admin_alert_v1",
     "ALERTA admin: nova clav generada {secret} - validar avon utilisar.",
     None),
    ("rm_secret_test_env_v1",
     "Pe igl ambient de test (sandbox), dovrar: STRIPE_KEY={secret}",
     None),
    ("rm_secret_jwt_decoded_v1",
     "JWT decodau invalid: {secret} - signatura n'ei betg validada.",
     None),
    ("rm_secret_slack_webhook_v1",
     "Slack webhook URL contegna igl secret: {secret}",
     None),
    ("rm_secret_rotate_v1",
     "Rotaziun da clavs planisada per damaun. Veglia clav: {secret} (suspendida)",
     None),
    ("rm_secret_env_file_v1",
     "# .env (na committer en git)\nAPI_KEY={secret}\nLOG_LEVEL=debug",
     None),
)


def _render(template_body: str, secondary_slot: str | None,
            ) -> tuple[str, list[dict]]:
    """Render one template with deterministic secret + optional 2nd PII.
    Returns (text, [span dicts])."""
    secret = secret_token()
    spans: list[dict] = []

    if secondary_slot == "name":
        name = name_rm()
        text = template_body.format(secret=secret, name=name)
        # name span
        n_start = text.find(name)
        if n_start >= 0:
            spans.append({"start": n_start, "end": n_start + len(name),
                          "label": "private_person"})
    elif secondary_slot == "phone":
        phone = phone_ch()
        text = template_body.format(secret=secret, phone=phone)
        p_start = text.find(phone)
        if p_start >= 0:
            spans.append({"start": p_start, "end": p_start + len(phone),
                          "label": "private_phone"})
    else:
        text = template_body.format(secret=secret)

    # secret span — find AFTER substitution so we land on the rendered offset
    s_start = text.find(secret)
    if s_start < 0:
        raise RuntimeError(
            f"secret value not found in rendered template: {text!r} / {secret!r}"
        )
    spans.append({"start": s_start, "end": s_start + len(secret),
                  "label": "secret"})

    # Sort spans by start for deterministic order.
    spans.sort(key=lambda s: s["start"])
    return text, spans


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    seed_all(args.seed)
    rng = random.Random(args.seed)
    print(f"Generating {args.n} RM × secret chunks "
          f"(seed={args.seed}) → {args.out}")

    n_written = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for i in range(args.n):
            tpl_id, body, secondary = rng.choice(_TEMPLATES)
            text, spans = _render(body, secondary)
            rec = {
                "id": f"rm_secret_{i:05d}",
                "text": text,
                "spans": spans,
                "language": "rm",
                "source": "synthetic_rm_secrets",
                "template_id": tpl_id,
                "meta": {"template_id": tpl_id},
            }
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")
            n_written += 1

    print(f"Wrote {n_written:,} chunks to {args.out}")


if __name__ == "__main__":
    main()
