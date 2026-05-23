"""Romansh-context fragments with secret slots — closes the rare
(rm × secret) cell that the LLM-labelled corpus would otherwise leave
unmeasurable. 20 hand-written templates using short cognate-heavy RM
prose (API tokens, env configs, OAuth bearers, etc.) with one secret
marker each; a handful also embed a co-occurring person or phone."""
from ..core import Fragment

TEMPLATES: list[Fragment] = [
    Fragment("rm_secret_api_token", "rm",
             "Igl token API per acceder al servetsch: [[PII:secret]]. Tegnair segira."),
    Fragment("rm_secret_falla_auth", "rm",
             "Falla d'autenticaziun: la clav '[[PII:secret]]' ei sbagliada. Empruvai puspei."),
    Fragment("rm_secret_config_env", "rm",
             "Configurar API_KEY=[[PII:secret]] en igl agen file de configuraziun."),
    Fragment("rm_secret_db_password", "rm",
             "Per acceder a la database, dovrar: PASSWORD=[[PII:secret]]"),
    Fragment("rm_secret_github_pat", "rm",
             "GitHub Personal Access Token: [[PII:secret]] - na divulgar betg!"),
    Fragment("rm_secret_aws", "rm",
             "AWS_SECRET_ACCESS_KEY = [[PII:secret]] # tegnair segira en igl vault"),
    Fragment("rm_secret_openai", "rm",
             "OpenAI API key: [[PII:secret]]. Quei ei privat - na cundivider mai."),
    Fragment("rm_secret_stripe", "rm",
             "Stripe webhook secret: [[PII:secret]] (privat, na committer en git)"),
    Fragment("rm_secret_oauth_bearer", "rm",
             "Bearer-Token per OAuth: Authorization: Bearer [[PII:secret]]"),
    Fragment("rm_secret_curl", "rm",
             "curl -H 'Authorization: Bearer [[PII:secret]]' https://api.example.ch/v1/users"),
    Fragment("rm_secret_user_generated", "rm",
             "L'usuari [[PII:full]] ha generau ina nova clav d'API: [[PII:secret]]"),
    Fragment("rm_secret_email_share", "rm",
             "Cara [[PII:first]], igl token sgnir ei: [[PII:secret]]. Empruvai cun quei valur."),
    Fragment("rm_secret_support_ticket", "rm",
             "Ticket de support da [[PII:full]]: l'API-token [[PII:secret]] dat ina falla 401."),
    Fragment("rm_secret_kyc_form", "rm",
             "Numer da telefon: [[PII:phone]]. Token d'autenticaziun: [[PII:secret]]"),
    Fragment("rm_secret_admin_alert", "rm",
             "ALERTA admin: nova clav generada [[PII:secret]] - validar avon utilisar."),
    Fragment("rm_secret_test_env", "rm",
             "Pe igl ambient de test (sandbox), dovrar: STRIPE_KEY=[[PII:secret]]"),
    Fragment("rm_secret_jwt_decoded", "rm",
             "JWT decodau invalid: [[PII:secret]] - signatura n'ei betg validada."),
    Fragment("rm_secret_slack_webhook", "rm",
             "Slack webhook URL contegna igl secret: [[PII:secret]]"),
    Fragment("rm_secret_rotate", "rm",
             "Rotaziun da clavs planisada per damaun. Veglia clav: [[PII:secret]] (suspendida)"),
    Fragment("rm_secret_env_file", "rm",
             "# .env (na committer en git)\nAPI_KEY=[[PII:secret]]\nLOG_LEVEL=debug"),
]
