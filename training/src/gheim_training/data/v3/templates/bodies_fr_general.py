"""French email body fragments — general correspondence. Hand-written, 25 entries."""
from ..core import Fragment

TEMPLATES: list[Fragment] = [
    Fragment("body_fr_gen_01", "fr_ch",
             "Veuillez trouver ci-joint la facture relative à votre commande du [[PII:date]]. "
             "Merci de procéder au règlement sur le compte [[PII:iban]] d'ici la fin du mois."),
    Fragment("body_fr_gen_02", "fr_ch",
             "Suite à notre échange téléphonique, je vous confirme que la livraison aura lieu à "
             "[[PII:address]] le [[PII:date]] entre 9h et 11h."),
    Fragment("body_fr_gen_03", "fr_ch",
             "Comme convenu, je transmets votre dossier de candidature à [[PII:title_last]] "
             "qui reviendra vers vous dans les jours qui viennent."),
    Fragment("body_fr_gen_04", "fr_ch",
             "Nous avons bien reçu votre paiement de CHF 1'250.- effectué le [[PII:date]]. "
             "Un reçu officiel vous sera envoyé sous peu par voie postale."),
    Fragment("body_fr_gen_05", "fr_ch",
             "Je me permets de vous relancer concernant la facture impayée. "
             "Vous pouvez me joindre directement au [[PII:phone]] pour discuter d'un éventuel arrangement."),
    Fragment("body_fr_gen_06", "fr_ch",
             "Permettez-moi de me présenter : je suis [[PII:full]], "
             "nouvelle responsable du département marketing depuis le [[PII:date]]."),
    Fragment("body_fr_gen_07", "fr_ch",
             "Votre rendez-vous avec notre conseiller est confirmé pour le [[PII:date]] à 14h30, "
             "dans nos bureaux situés au [[PII:address]]."),
    Fragment("body_fr_gen_08", "fr_ch",
             "Concernant votre contrat de travail, veuillez nous communiquer votre numéro AVS "
             "([[PII:ahv]]) afin que nous puissions finaliser votre inscription."),
    Fragment("body_fr_gen_09", "fr_ch",
             "Toutes nos coordonnées bancaires figurent désormais sur notre site [[PII:url]]. "
             "N'oubliez pas de mentionner notre numéro TVA [[PII:vat]] sur vos prochaines factures."),
    Fragment("body_fr_gen_10", "fr_ch",
             "Je profite de ce message pour vous présenter mes meilleurs vœux pour cette nouvelle année. "
             "Au plaisir de vous recroiser bientôt."),
    Fragment("body_fr_gen_11", "fr_ch",
             "Suite à votre demande, j'ai bien transmis votre dossier à [[PII:title_first_last]]. "
             "Elle prendra contact avec vous à l'adresse [[PII:email]] dans les meilleurs délais."),
    Fragment("body_fr_gen_12", "fr_ch",
             "Nous avons le plaisir de vous informer que votre commande n°487-2024 a été expédiée ce matin. "
             "Le suivi est disponible sur [[PII:url]]."),
    Fragment("body_fr_gen_13", "fr_ch",
             "Je vous écris pour vous proposer une rencontre afin de discuter du projet. "
             "Seriez-vous disponible le [[PII:date]] en fin d'après-midi ?"),
    Fragment("body_fr_gen_14", "fr_ch",
             "N'hésitez pas à me contacter au [[PII:phone]] ou par courriel à [[PII:email]] "
             "si vous avez la moindre question concernant votre dossier."),
    Fragment("body_fr_gen_15", "fr_ch",
             "Veuillez noter que nos bureaux seront exceptionnellement fermés à partir du [[PII:date]] "
             "pour cause de travaux de rénovation."),
    Fragment("body_fr_gen_16", "fr_ch",
             "Pour régulariser le paiement de votre cotisation annuelle, "
             "nous vous prions de bien vouloir effectuer un virement sur le compte [[PII:iban]]."),
    Fragment("body_fr_gen_17", "fr_ch",
             "Je tenais personnellement à vous remercier pour la qualité de votre intervention lors du séminaire. "
             "[[PII:first]] et moi-même avons beaucoup apprécié votre exposé."),
    Fragment("body_fr_gen_18", "fr_ch",
             "Suite à la démission de [[PII:title_last]], nous recherchons activement un remplaçant. "
             "Si vous connaissez quelqu'un d'intéressé, merci de me l'indiquer."),
    Fragment("body_fr_gen_19", "fr_ch",
             "Pour des raisons de sécurité, nous avons procédé au remplacement de votre carte. "
             "L'ancien numéro [[PII:cc]] a été désactivé ce matin à 8h."),
    Fragment("body_fr_gen_20", "fr_ch",
             "Comme convenu hier, je vous fais parvenir l'adresse de la salle de réunion : "
             "[[PII:address]]. Le code d'accès vous sera communiqué sur place."),
    Fragment("body_fr_gen_21", "fr_ch",
             "Votre demande de remboursement a été acceptée. Le montant sera crédité sur votre compte "
             "[[PII:iban]] dans un délai maximum de cinq jours ouvrables."),
    Fragment("body_fr_gen_22", "fr_ch",
             "Pourriez-vous me confirmer la date de votre prochaine visite ? "
             "[[PII:first]] souhaiterait organiser un déjeuner d'équipe à cette occasion."),
    Fragment("body_fr_gen_23", "fr_ch",
             "Le contrat est prêt à être signé. Je vous propose de passer le récupérer au "
             "[[PII:address]] dès que cela vous arrange, idéalement avant le [[PII:date]]."),
    Fragment("body_fr_gen_24", "fr_ch",
             "Nous accusons réception de votre dossier complet. Notre service vous contactera prochainement "
             "au [[PII:phone]] pour fixer un entretien d'évaluation."),
    Fragment("body_fr_gen_25", "fr_ch",
             "En l'absence de [[PII:title_first_last]] jusqu'au [[PII:date]], "
             "je reste votre interlocutrice privilégiée pour toute question urgente."),
]
