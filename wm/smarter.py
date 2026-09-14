"""
L'elicitation par RANGS, et non par nombres.

Pourquoi. Notre modele de valeur est declare : on demande a un modele de
langage COMBIEN vaut chaque redaction, sur six axes. Il ne se calibre pas -
correlation de rang avec la fermete du mandat de +0,01 sur un contrat, +0,30
sur l'autre, +0,15 combine (p = 0,30). Ce n'est pas une surprise : sur les
memes repondants, la ponderation directe produit un ecart de 1,4 entre
l'attribut le plus et le moins important la ou un choix discret en produit
14,9 (PLOS ONE 2023, PMC10381030). La ponderation directe ecrase tout vers
l'uniforme. La litterature LLM rapporte la meme pathologie : on renonce aux
comparaisons d'importance parce que "les modeles rendent des sorties presque
identiques" (arXiv:2601.05267) ; en revanche le jugement PAR PAIRES est plus
stable que la notation directe, pour les modeles comme pour les humains
(arXiv:2403.16950).

SMARTER (Edwards & Barron 1994) ne demande qu'un CLASSEMENT et applique les
poids ROC (rank order centroid) :

    w_i = (1/n) * somme_{k=i}^{n} 1/k

Rapporte a environ 98 % de la performance d'une elicitation complete, pour la
seule information qu'un modele produit de facon stable. Ici deux rangs
suffisent : celui des POINTS entre eux, et celui des REDACTIONS a l'interieur
de chaque point.

Ce que cela abandonne, et pourquoi ce n'est pas cher. Le vecteur a six
dimensions disparait au profit d'une utilite scalaire. Mais il etait deja
degenere : sur le second contrat, tail_risk portait 78 % de la derive
ponderee, et le rapport le signalait - "les six axes se comportent presque
comme un seul". On perd une multidimensionnalite que rien ne validait, on
gagne une echelle dont la construction se verifie.

Le domaine (quels points, quelles redactions) est REPRIS du modele declare.
Seules les VALEURS changent, donc la comparaison entre les deux elicitations
ne porte que sur ce qu'on veut comparer.

CE QUE LA MESURE A DONNE - et elle contredit ce qui precede.
------------------------------------------------------------------
Contrat 1 (licence Veridian, 26 points, GLM-4.6), correlation de rang avec les
dollars calcules par le moteur de scenarios, sur les 10 points que les deux
modeles partagent :

    elicitation directe (nombres)   rho = +0,81   (n = 10, p = 0,005)
    elicitation par rangs (ROC)     rho = +0,52   (n = 10, p = 0,128)

Le remede est moins bon que le mal suppose. Et le modele par rangs echoue en
plus ses propres controles la ou le modele declare les passait : il fait
demander a l'adversaire 4 redactions contre son propre interet, et rend 15
redactions sur 52 meilleures pour nous que les notres - ce qui voudrait dire
que nous avons envoye un modele sous-optimal.

Pourquoi, vraisemblablement. ROC transforme un RANG en poids, donc impose une
decroissance geometrique : ici le premier point pese 100 fois le dernier, par
construction et non par mesure. Quand les ecarts reels sont plus plats que
cela, la forme imposee est plus fausse que les nombres bruts. Et deux
classements independants (le notre, le leur) n'ont aucune raison d'etre
coherents entre eux : rien ne les rattache a une echelle commune, alors qu'une
elicitation en nombres, meme mal calibree, est ancree des deux cotes sur la
meme redaction de reference.

Ce module reste dans l'arbre parce que le resultat negatif est un resultat :
il ferme une piste que la litterature recommandait, et il le fait avec un
chiffre. Il ne doit pas servir de source de valeur par defaut.
"""

from __future__ import annotations

import json
import re

from . import parametric as pm

# Une seule dimension : l'utilite. Les autres restent a zero pour que le reste
# de la pile (derive, ratios, boucle de valeur, simulateur) tourne inchange.
DIM = "revenue"
ECHELLE = 1000.0        # l'unite est arbitraire ; seuls les rapports comptent


def roc(rang: int, n: int) -> float:
    """Poids ROC d'un attribut classe `rang` (1 = le plus important) sur n."""
    return sum(1.0 / k for k in range(rang, n + 1)) / n


def _menu(c: pm.Contract) -> str:
    L = []
    for p in c.params.values():
        L.append(f"{p.id}  {p.name}  [section {p.section}]")
        for o in p.options:
            tag = "  (le modele)" if o.id == p.ours else (
                  "  (le markup)" if o.id == p.theirs else "")
            L.append(f"    {o.id}: {o.text[:150]}{tag}")
        L.append("")
    return "\n".join(L)


RANK_PROMPT = """You are counsel to {who} in a contract negotiation with {other}.
Below are the points in dispute. For each point, several drafting options exist.

Do TWO rankings. Give NO numbers, NO scores, NO percentages — only orderings.

1. "points": rank ALL {n} points from most to least important to {who},
   by the identifiers given. Most important first. Every point appears exactly
   once. Importance means: how much {who}'s position would be damaged by
   losing this point entirely, compared with the others.

2. "options": for EACH point, rank its drafting options from best to worst for
   {who}. Best first. Every option of that point appears exactly once.

{mandate}

POINTS AND OPTIONS
{menu}

Return ONLY JSON:
{{"points": ["<id>", "<id>", ...],
  "options": {{"<point id>": ["<option id>", "<option id>", ...], ...}}}}"""


MANDATE_OURS = """The client's own negotiation mandate is your guide to importance:
what it calls firm or non-negotiable matters more than what it leaves to
judgement. Use it; do not restate it."""

MANDATE_THEIRS = """You have not seen the other side's mandate and must not guess at
it. Rank from {who}'s own commercial interest, reading the drafting options for
what they do to {who}."""


def parse_ranks(raw: str, c: pm.Contract) -> tuple[dict, dict, list[str]]:
    """(rang des points, rang des options par point, motifs de rejet)."""
    spans, pile, dans, esc = [], [], False, False
    for i, ch in enumerate(raw or ""):
        if dans:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                dans = False
            continue
        if ch == '"':
            dans = True
        elif ch == "{":
            pile.append(i)
        elif ch == "}" and pile:
            spans.append(raw[pile.pop():i + 1])
    d = None
    for s in sorted(spans, key=len, reverse=True):
        try:
            cand = json.loads(s)
        except ValueError:
            continue
        if isinstance(cand, dict) and "points" in cand:
            d = cand
            break
    if d is None:
        return {}, {}, ["aucun objet JSON portant 'points'"]

    why = []
    pts = [str(x) for x in (d.get("points") or []) if str(x) in c.params]
    manque = [pid for pid in c.params if pid not in pts]
    if manque:
        # Un point omis n'est pas neutre : il recevrait le poids le plus faible
        # sans que personne l'ait decide. On le place en queue et on le dit.
        why.append(f"{len(manque)} points absents du classement, mis en queue : "
                   f"{', '.join(manque[:8])}")
        pts += manque
    rang_pt = {pid: i + 1 for i, pid in enumerate(pts)}

    rang_op = {}
    for pid, p in c.params.items():
        ordre = [str(x) for x in ((d.get("options") or {}).get(pid) or [])]
        connus = [o.id for o in p.options]
        ordre = [x for x in ordre if x in connus]
        absents = [x for x in connus if x not in ordre]
        if absents:
            why.append(f"{pid}: {len(absents)} redactions absentes, mises en queue")
            ordre += absents
        rang_op[pid] = {oid: i + 1 for i, oid in enumerate(ordre)}
    return rang_pt, rang_op, why


def contract_from_ranks(c: pm.Contract, rp_ours: dict, ro_ours: dict,
                        rp_theirs: dict, ro_theirs: dict) -> pm.Contract:
    """Le meme domaine, des valeurs derivees des rangs.

    u(option) = w_point x v_option, ou w vient des poids ROC du rang du point
    et v est lineaire dans le rang de la redaction : 1 pour la meilleure, 0
    pour la pire. La valeur est ensuite RE-BASEE sur notre redaction de
    reference, pour que le modele et le markup se comparent comme ailleurs.
    """
    n = len(c.params)
    params = []
    for pid, p in c.params.items():
        m = len(p.options)

        def val(rangs, rp):
            w = roc(rp.get(pid, n), n)
            out = {}
            for o in p.options:
                r = rangs.get(pid, {}).get(o.id, m)
                v = 1.0 if m == 1 else (m - r) / (m - 1)
                out[o.id] = w * v * ECHELLE
            return out

        vo, vt = val(ro_ours, rp_ours), val(ro_theirs, rp_theirs)
        base_o, base_t = vo[p.ours], vt[p.ours]
        opts = [pm.Option(id=o.id, text=o.text,
                          ours={DIM: vo[o.id] - base_o},
                          theirs={DIM: vt[o.id] - base_t})
                for o in p.options]
        params.append(pm.Parameter(id=p.id, name=p.name, section=p.section,
                                   options=opts, ours=p.ours, theirs=p.theirs))
    return pm.Contract(params)


WEIGHTS = {d: (1.0 if d == DIM else 0.0) for d in pm.DIMENSIONS}


def elicit(c: pm.Contract, llm, us: str, them: str, mandate: str = "",
           raw_out: list | None = None) -> tuple[pm.Contract, list[str]]:
    """Deux appels : notre classement, puis le leur depuis leur siege."""
    why = []
    out = {}
    for cote, who, other, mand in (("ours", us, them, MANDATE_OURS),
                                   ("theirs", them, us, MANDATE_THEIRS.format(who=them))):
        prompt = RANK_PROMPT.format(who=who, other=other, n=len(c.params),
                                    mandate=(mand + ("\n\n" + mandate if mandate and
                                                     cote == "ours" else "")),
                                    menu=_menu(c))
        raw = llm(prompt, max_tokens=8000) or ""
        if raw_out is not None:
            raw_out.append(raw)
        rp, ro, w = parse_ranks(raw, c)
        out[cote] = (rp, ro)
        why += [f"[{cote}] {x}" for x in w]
    return (contract_from_ranks(c, out["ours"][0], out["ours"][1],
                                out["theirs"][0], out["theirs"][1]), why)


def report(c: pm.Contract, rangs_ours: dict | None = None) -> str:
    """Ce que les rangs ont produit, dans l'ordre de l'importance elicitee."""
    L = ["LE CONTRAT, VALORISE PAR RANGS (SMARTER / poids ROC)", "=" * 78, "",
         f"{len(c.params)} points. u(redaction) = poids ROC du point x position "
         f"de la redaction.", ""]
    lignes = []
    for pid, p in c.params.items():
        d = abs(pm.total(p.option(p.theirs).vec(), WEIGHTS)
                - pm.total(p.option(p.ours).vec(), WEIGHTS))
        lignes.append((d, p.name, pid))
    L.append(f"{'point':<44}{'ecart modele -> markup':>24}")
    L.append("-" * 78)
    for d, nom, pid in sorted(lignes, reverse=True):
        L.append(f"{nom[:43]:<44}{d:>24.1f}")
    L += ["", "Les poids ROC decroissent vite : le premier point pese environ "
          f"{roc(1, len(c.params)) / roc(len(c.params), len(c.params)):.0f} fois le dernier.",
          "C'est voulu - la ponderation directe, elle, ecrase tout vers l'uniforme."]
    return "\n".join(L)
