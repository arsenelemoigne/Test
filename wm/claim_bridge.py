"""
Du contrat-creance au contrat parametrique : les vecteurs CALCULES.

parametric.py attend, pour chaque redaction d'un point, un effet sur notre
vecteur et un effet sur le leur. Jusqu'ici ces effets etaient DECLARES par un
modele de langage - et rien ne les calibrait (rho = 0,22, p = 0,28).

Ici ils sont calcules : chaque redaction est un changement de Terms, et son
effet est l'ecart de gain que le moteur de scenarios lui attribue, pour chaque
partie, en dollars. Deux dimensions seulement, toutes deux en dollars :

  revenue    ecart de gain moyen
  tail_risk  ecart de queue (CVaR5 - moyenne) : ce que la redaction fait
             dans les mauvais mondes, au-dela de ce qu'elle fait en moyenne

Le reste de la pile - derive, ratios, boucle de valeur, simulateur de
negociation - tourne sans changement. Seule l'origine des chiffres change :
un LLM ne les invente plus, un programme les calcule a partir de termes
qu'un juriste peut verifier contre le texte en trente secondes.
"""

from __future__ import annotations

import statistics

from . import claim as C
from . import parametric as pm

# Les points du dossier Veridian / Halcyon, avec pour chacun les redactions du
# template (reference), du memo (contre-position) et du markup. Chaque
# redaction est un changement de Terms - verifiable, pas une opinion.
VERIDIAN = [
    ("users", "Modele d'utilisateurs", "1.15", [
        ("named_1500", "1 500 Named Users (template)", {}),
        ("named_2500", "2 500 Named Users a 3 168 000 $ (memo)", dict(users=2500, fee_y1=3_168_000)),
        ("unlimited", "Authorized Users illimites, 2 640 000 $ (markup)", dict(users=0)),
    ]),
    ("escalator", "Escalateur tarifaire", "4.2", [
        ("fixed4", "4 % fixe compose (template)", {}),
        ("cpi_3_5", "CPI-U plancher 3 % plafond 5 % (memo)", dict(escalator="cpi", esc_floor=0.03, esc_cap=0.05)),
        ("cpi_cap2", "CPI-U plafonne a 2 % (markup)", dict(escalator="cpi", esc_floor=0.0, esc_cap=0.02)),
    ]),
    ("billing", "Facturation", "4.3", [
        ("q_adv_30", "trimestriel d'avance, net 30 (template)", {}),
        ("q_adv_45", "trimestriel d'avance, net 45 (memo)", dict(payment_days=45)),
        ("s_arr_60", "semestriel echu, net 60 (markup)", dict(billing="semiannual_arrears", payment_days=60)),
    ]),
    ("cap", "Plafond de responsabilite", "11.1", [
        ("m12", "12 mois de redevances glissantes (template)", {}),
        ("m18", "18 mois de redevances glissantes (memo)", dict(cap_months=18)),
        ("x2_total", "2x le total des redevances du terme (markup)", dict(cap_basis="total_term", cap_mult=2.0)),
    ]),
    ("ip_remedy", "Indemnite PI : recours exclusif 10.2", "10.2", [
        ("keep", "10.2 conserve : modifier / remplacer / licence (template, memo)", {}),
        ("delete", "10.2 supprime, indemnite monetaire hors plafond (markup)", dict(sole_remedy_ip=False, ip_uncapped=True)),
    ]),
    ("conseq", "Carve-outs des dommages indirects", "11.2", [
        ("none", "aucun carve-out (template)", {}),
        ("conf_data", "confidentialite + securite des donnees (memo)", dict(carve_confidentiality=True, carve_data=True, data_uncapped=True)),
        ("all4", "confidentialite, PI, donnees, perte d'exploitation (markup)", dict(carve_confidentiality=True, carve_ip=True, carve_data=True, carve_downtime_profits=True, data_uncapped=True)),
    ]),
    ("sla", "Niveau de service et credits", "sched. C", [
        ("none", "pas de SLA (template)", {}),
        ("sla999", "99,9 %, 2 % par 0,1 %, plafond 10 % (memo)", dict(sla=True, sla_target=0.999, credit_per_01=0.02, credit_cap=0.10)),
        ("sla9995", "99,95 %, 5 % par 0,1 %, sans plafond (markup)", dict(sla=True, sla_target=0.9995, credit_per_01=0.05, credit_cap=None)),
    ]),
    ("chronic", "Resiliation pour defaillance chronique", "sched. C", [
        ("none", "aucune (template)", {}),
        ("c98_3of12", "98 % sur 3 mois de 12, 60 j de cure (memo)", dict(chronic=True, chronic_threshold=0.98, chronic_months=3, chronic_window=12)),
        ("c99_any", "99 % un seul mois (markup)", dict(chronic=True, chronic_threshold=0.99, chronic_months=1, chronic_window=1)),
    ]),
    ("warranty", "Periode de garantie", "7.2", [
        ("d90", "90 jours (template)", {}),
        ("m12", "12 mois (memo)", dict(warranty_days=365)),
        ("m18", "18 mois (markup)", dict(warranty_days=548)),
    ]),
    ("tfc", "Resiliation pour convenance du licencie", "9.3", [
        ("none", "aucune (template)", {}),
        ("after24_180_50", "apres 24 mois, 180 j, 50 % des redevances restantes (memo)", dict(tfc_licensee=True, tfc_after_months=24, tfc_notice_days=180, tfc_fee_frac=0.5)),
        ("any_90_free", "a tout moment, 90 j, sans frais (markup)", dict(tfc_licensee=True, tfc_after_months=0, tfc_notice_days=90, tfc_fee_frac=0.0)),
    ]),
    ("law", "Droit applicable et for", "13", [
        ("de_court", "Delaware, tribunaux (template)", {}),
        ("ny_court", "New York, tribunaux (memo : acceptable)", dict(law="NY")),
        ("tx_arb", "Texas, arbitrage AAA (markup)", dict(law="TX", forum="arbitration")),
    ]),
    ("audit", "Droit d'audit", "12", [
        ("m12", "une fois par 12 mois (template)", {}),
        ("m18", "une fois par 18 mois (memo)", dict(audit_interval_months=18)),
        ("none", "supprime (markup)", dict(audit=False)),
    ]),
    ("escrow", "Depot du code source", "14.12", [
        ("none", "aucun (template)", {}),
        ("no_breach", "escrow, sans declencheur de manquement (memo)", dict(escrow=True, escrow_breach_trigger=False)),
        ("breach", "escrow, liberation sur manquement (markup)", dict(escrow=True, escrow_breach_trigger=True)),
    ]),
    ("mfc", "Clause du client le plus favorise", "14.14", [
        ("none", "aucune (template, memo : walk-away)", {}),
        ("mfc", "MFC (markup)", dict(mfc=True)),
    ]),
]

# quelle redaction chaque position retient
THEIRS = {"users": "unlimited", "escalator": "cpi_cap2", "billing": "s_arr_60", "cap": "x2_total",
          "ip_remedy": "delete", "conseq": "all4", "sla": "sla9995", "chronic": "c99_any",
          "warranty": "m18", "tfc": "any_90_free", "law": "tx_arb", "audit": "none",
          "escrow": "breach", "mfc": "mfc"}
COUNTER = {"users": "named_2500", "escalator": "cpi_3_5", "billing": "q_adv_45", "cap": "m18",
           "ip_remedy": "keep", "conseq": "conf_data", "sla": "sla999", "chronic": "c98_3of12",
           "warranty": "m12", "tfc": "after24_180_50", "law": "ny_court", "audit": "m18",
           "escrow": "no_breach", "mfc": "none"}


def _delta(base: C.Terms, changes: dict, n: int, seed: int) -> tuple[float, float, float, float]:
    """(licensor mean, licensor tail, licensee mean, licensee tail) de la
    redaction par rapport a la reference, en dollars. Memes tirages des deux
    cotes (meme seed) : la difference est celle des termes, pas du hasard."""
    A = C.simulate(base, n, seed)
    Bp = C.simulate(base.with_(**changes), n, seed)
    dl = [b.licensor - a.licensor for a, b in zip(A, Bp)]
    dr = [b.licensee - a.licensee for a, b in zip(A, Bp)]
    sl, sr = C.stats(dl), C.stats(dr)
    return (sl["mean"], sl["cvar5"] - sl["mean"], sr["mean"], sr["cvar5"] - sr["mean"])


def build(base: C.Terms = C.TEMPLATE, n: int = 600, seed: int = 0,
          spec=VERIDIAN, theirs=THEIRS) -> pm.Contract:
    """Le contrat parametrique dont les vecteurs sont calcules par claim.py.

    Chaque redaction est evaluee SEULE, la reference etant `base` : c'est un
    modele additif au premier ordre. Les couplages (ex. plafond x carve-out)
    existent dans le moteur - evaluate() les calcule exactement sur une
    assignation complete - mais ne sont pas projetes ici ; la valeur d'une
    assignation complete se demande a claim.evaluate, pas a la somme.
    """
    params = []
    for pid, name, section, options in spec:
        opts = []
        for oid, text, changes in options:
            if not changes:
                opts.append(pm.Option(id=oid, text=text))
                continue
            lm, lt, rm, rt = _delta(base, changes, n, seed)
            opts.append(pm.Option(id=oid, text=text,
                                  ours={"revenue": lm, "tail_risk": lt},
                                  theirs={"revenue": rm, "tail_risk": rt}))
        params.append(pm.Parameter(id=pid, name=name, section=section, options=opts,
                                   ours=options[0][0], theirs=theirs[pid]))
    return pm.Contract(params)


# Couche 3 : le poids de la queue est l'aversion au risque. A 1,0 on compte
# le pire vingtieme des mondes autant que la moyenne ; a 0 on est neutre au
# risque. 0,25 est un defaut modere, et c'est le SEUL reglage propre a vous.
WEIGHTS = {"revenue": 1.0, "tail_risk": 0.25, "control": 0.0, "flexibility": 0.0,
           "admin": 0.0, "enforceability": 0.0}


def terms_of(assignment: dict, base: C.Terms = C.TEMPLATE, spec=VERIDIAN) -> C.Terms:
    """L'assignation complete, rendue comme Terms - pour l'evaluer exactement."""
    t = base
    for pid, name, section, options in spec:
        oid = assignment.get(pid, options[0][0])
        for o, _, changes in options:
            if o == oid and changes:
                t = t.with_(**changes)
    return t
