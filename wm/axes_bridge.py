"""
Brancher le profil ORDINAL sur la moitie du contrat que les dollars n'atteignent pas.

Le probleme, pose franchement. claim.py chiffre 14 points sur 26 : ceux qui
ont des montants. Les 12 autres - definitions, charge de la preuve, seuils de
materialite, discretion, cession - n'ont pas de numeraire, et c'est exactement
pour eux qu'axes.py existe : il classe chaque modification par FONCTION
juridique et par sens, sans jamais la valoriser. Jusqu'ici il n'etait branche
sur rien, et ces 12 points restaient valorises par le seul vecteur declare.

Ce que fait ce fichier, et ce qu'il ne fait pas. Il ne fabrique pas un
numeraire : la litterature est nette la-dessus, aucune echelle cardinale des
clauses commerciales ne se justifie par elle-meme, et toutes celles que
publient les produits du marche sont declarees par leur auteur. Il fait
l'inverse : il CALIBRE l'echelle ordinale sur la moitie du contrat ou les deux
mesures coexistent.

    sur les 14 points chiffrables : on a un score ordinal ET des dollars
    on ajuste un seul coefficient    dollars ~= alpha * ordinal
    on applique alpha aux 12 autres

Un parametre, ajuste sur 14 observations, avec son R2 et sa correlation de
rang affiches. Si l'ajustement est mauvais, l'extrapolation n'est pas fondee
et le code le dit au lieu de rendre un nombre.

Ce que l'ordinal ne peut pas porter : la QUEUE. Un score de fonction juridique
ne dit rien de ce qui arrive dans les mauvais mondes, la ou claim.py separe la
moyenne du CVaR. Les points calibres n'ont donc qu'une dimension renseignee,
et c'est une perte reelle, pas un detail de mise en forme.

La FORME a l'interieur d'un point vient du modele declare. Les axes ne
decrivent que le passage modele -> markup ; ils ne disent rien de la redaction
intermediaire du memo. Or le vecteur declare, lui, est calibre en RANG contre
les dollars calcules (rho = +0,81, n = 10, p = 0,005) : il est donc legitime
pour ordonner les redactions A L'INTERIEUR d'un point. On garde sa forme et on
lui impose l'echelle que la calibration donne au point. Chacun fait ce pour
quoi il a ete teste.
"""

from __future__ import annotations

import math

from . import axes as AX
from . import claim_bridge as CB
from . import parametric as pm

# Notre client a envoye le modele ; dans ce dossier c'est le concedant. Un
# mouvement au profit du licencie joue donc contre nous.
NOUS = "licensor"
_W = {"minor": 1.0, "moderate": 2.0, "major": 4.0}


def _sec(x: str) -> str:
    return str(x or "").split("(")[0].strip()


def score(moves, section: str, nous: str = NOUS) -> float:
    """Le net ordinal d'une section : somme des poids signes.

    Positif = le markup nous renforce sur cette section. Negatif = il nous
    affaiblit. Les mouvements ambigus et neutres comptent zero - c'est
    volontaire : axes.py les signale au lieu de les deviner, et deviner ici
    reviendrait a defaire ce choix.
    """
    s, cible = 0.0, _sec(section)
    for m in moves:
        if _sec(getattr(m, "section", "")) != cible:
            continue
        f = getattr(m, "favours", "")
        if f == nous:
            s += _W.get(getattr(m, "magnitude", ""), 0.0)
        elif f in AX.PARTIES:
            s -= _W.get(getattr(m, "magnitude", ""), 0.0)
    return s


def calibre(moves, n: int = 800, seed: int = 0, nous: str = NOUS) -> dict:
    """Ajuste dollars ~= alpha * ordinal sur les points ou les deux existent."""
    calc = CB.build(n=n, seed=seed)
    obs = []
    for pid, nom, sec, _opts in CB.VERIDIAN:
        p = calc.params[pid]
        d = (pm.total(p.option(p.theirs).vec(), CB.WEIGHTS)
             - pm.total(p.option(p.ours).vec(), CB.WEIGHTS))
        s = score(moves, sec, nous)
        obs.append((pid, nom, sec, s, d))
    xs = [o[3] for o in obs]
    ys = [o[4] for o in obs]
    den = sum(x * x for x in xs)
    alpha = (sum(x * y for x, y in zip(xs, ys)) / den) if den else 0.0
    # R2 d'une droite par l'origine, et correlation de rang
    ss_res = sum((y - alpha * x) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum(y * y for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    rho, p_rho = _spearman(xs, ys)
    return {"alpha": alpha, "r2": r2, "rho": rho, "p": p_rho, "obs": obs, "n": len(obs)}


def _spearman(xs, ys):
    n = len(xs)
    if n < 4:
        return None, None

    def rangs(v):
        ordre = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[ordre[j + 1]] == v[ordre[i]]:
                j += 1
            moy = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[ordre[k]] = moy
            i = j + 1
        return r

    a, b = rangs(xs), rangs(ys)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    if den == 0:
        return None, None
    rho = num / den
    t = abs(rho) * ((n - 2) / max(1e-12, 1 - rho ** 2)) ** 0.5
    from .tools_stats import p_student
    return rho, p_student(t, n - 2)


def build(elic: pm.Contract, moves, cal: dict, sec_of: dict, n: int = 800,
          seed: int = 0, nous: str = NOUS) -> tuple[pm.Contract, list[dict]]:
    """Le contrat fusionne : dollars la ou on sait calculer, ordinal calibre ailleurs.

    sec_of : identifiant de point elicite -> numero de section, pour apparier
    les deux modeles. C'est la seule adresse qu'ils partagent.
    """
    calc = CB.build(n=n, seed=seed)
    par_sec = {_sec(sec): pid for pid, _, sec, _ in CB.VERIDIAN}
    alpha = cal["alpha"]
    params, journal = [], []
    for pid, p in elic.params.items():
        sec = _sec(sec_of.get(pid, ""))
        jumeau = par_sec.get(sec)
        if jumeau and jumeau in calc.params:
            # CALCULE : on prend le point du moteur tel quel, en gardant les
            # identifiants du modele elicite pour que le reste de la pile suive
            q = calc.params[jumeau]
            # APPARIER PAR ROLE, PAS PAR POSITION. Les deux modeles n'ont pas
            # le meme nombre de redactions par point : le modele elicite en a
            # deux ici, le moteur trois. Prendre la i-eme de chaque cote
            # faisait correspondre le markup a la position de repli du memo -
            # un point entier value avec les chiffres d'une autre redaction.
            # Ce qui se correspond, c'est notre modele avec notre modele et
            # leur markup avec leur markup ; le reste se range entre les deux.
            milieu = [o for o in q.options if o.id not in (q.ours, q.theirs)]
            opts, k = [], 0
            for o in p.options:
                if o.id == p.ours:
                    src = q.option(q.ours)
                elif o.id == p.theirs:
                    src = q.option(q.theirs)
                elif k < len(milieu):
                    src = milieu[k]
                    k += 1
                else:
                    # aucune redaction intermediaire en face : on place la
                    # notre a mi-chemin, et le journal le dira
                    a_, b_ = q.option(q.ours), q.option(q.theirs)
                    src = pm.Option(id=o.id, text=o.text,
                                    ours={d: (a_.ours.get(d, 0.0) + b_.ours.get(d, 0.0)) / 2
                                          for d in pm.DIMENSIONS},
                                    theirs={d: (a_.theirs.get(d, 0.0) + b_.theirs.get(d, 0.0)) / 2
                                            for d in pm.DIMENSIONS})
                opts.append(pm.Option(id=o.id, text=o.text,
                                      ours=dict(src.ours), theirs=dict(src.theirs)))
            journal.append({"id": pid, "nom": p.name, "section": sec, "source": "calcule",
                            "ordinal": score(moves, sec, nous),
                            "dollars": pm.total(
                                next(o for o in opts if o.id == p.theirs).vec(),
                                CB.WEIGHTS)})
            params.append(pm.Parameter(id=p.id, name=p.name, section=p.section,
                                       options=opts, ours=p.ours, theirs=p.theirs))
            continue
        # ORDINAL CALIBRE : l'echelle du point vient de alpha x son score ; la
        # forme a l'interieur du point vient du vecteur declare, qui est
        # calibre en rang et sert donc a ordonner les redactions entre elles.
        s = score(moves, sec, nous)
        cible = alpha * s
        base_o = pm.total(p.option(p.ours).vec())
        base_t = pm.total(p.option(p.theirs).vec())
        ecart = base_t - base_o
        opts = []
        for o in p.options:
            if o.id == p.ours:
                f = 0.0
            elif abs(ecart) > 1e-9:
                f = (pm.total(o.vec()) - base_o) / ecart
            else:
                f = 0.0 if o.id == p.ours else (1.0 if o.id == p.theirs else 0.5)
            # eux : on garde la structure non-somme-nulle du modele declare,
            # rapportee a la meme echelle
            g = 0.0
            if abs(ecart) > 1e-9:
                bo = pm.total(p.option(p.ours).vec("theirs"))
                g = (pm.total(o.vec("theirs")) - bo) / abs(ecart)
            opts.append(pm.Option(id=o.id, text=o.text,
                                  ours={"revenue": f * cible},
                                  theirs={"revenue": -g * cible if cible else 0.0}))
        journal.append({"id": pid, "nom": p.name, "section": sec,
                        "source": "axes x alpha" if s else "axes (score nul)",
                        "ordinal": s, "dollars": cible})
        params.append(pm.Parameter(id=p.id, name=p.name, section=p.section,
                                   options=opts, ours=p.ours, theirs=p.theirs))
    return pm.Contract(params), journal


def report(cal: dict, journal: list[dict]) -> str:
    a, r2, rho, p = cal["alpha"], cal["r2"], cal["rho"], cal["p"]
    L = ["LE PROFIL ORDINAL, CALIBRE SUR LA MOITIE CHIFFRABLE DU CONTRAT", "=" * 84, "",
         "L'ajustement : dollars ~= alpha x (net ordinal de la section),",
         f"sur les {cal['n']} points ou les deux mesures existent.", "",
         f"{'point':<40}{'ordinal':>10}{'calcule $':>14}{'ajuste $':>14}",
         "-" * 84]
    for pid, nom, sec, s, d in sorted(cal["obs"], key=lambda o: o[4]):
        L.append(f"{nom[:39]:<40}{s:>10.0f}{d/1e6:>13.2f}M{a*s/1e6:>13.2f}M")
    L += ["-" * 84,
          f"alpha = {a/1e6:.3f} M$ par point de net ordinal",
          f"R2 = {r2:+.2f}" + (f"   rho = {rho:+.2f} (p ~ {p:.3f})" if rho is not None else ""),
          ""]
    fiable = (rho is not None and p is not None and p <= 0.10 and r2 > 0.2)
    if fiable:
        L += ["  L'echelle ordinale suit les dollars sur la moitie ou l'on peut verifier.",
              "  L'extrapolation aux points non chiffrables repose sur UN parametre ajuste",
              "  sur 14 observations - c'est peu, et c'est verifiable ; ce n'est pas une",
              "  echelle inventee."]
    else:
        L += ["  L'AJUSTEMENT NE TIENT PAS. Le score ordinal ne suit pas les dollars la ou",
              "  l'on peut comparer, donc rien n'autorise a l'extrapoler la ou l'on ne peut",
              "  pas. Les points non chiffrables gardent leur vecteur declare, et le dire",
              "  vaut mieux que de rendre un nombre calibre sur rien.",
              "  Causes a regarder dans l'ordre : la classification des modifications",
              "  (axes --report), l'appariement par section, et le signe de NOUS."]
    L += ["", "LE CONTRAT FUSIONNE, POINT PAR POINT", "-" * 84,
          f"{'point':<40}{'source':<16}{'ordinal':>9}{'valeur $':>14}"]
    for r in sorted(journal, key=lambda r: r["dollars"]):
        L.append(f"{r['nom'][:39]:<40}{r['source']:<16}{r['ordinal']:>9.0f}"
                 f"{r['dollars']/1e6:>13.2f}M")
    nc = sum(1 for r in journal if r["source"] == "calcule")
    L += ["-" * 84,
          f"{nc} points calcules, {len(journal) - nc} calibres sur le profil ordinal.",
          "",
          "Les points calibres n'ont PAS de dimension de queue : un score de fonction",
          "juridique ne dit rien de ce qui arrive dans les mauvais mondes. Une politique",
          "qui pondere le risque de queue ne verra donc rien sur ces points-la."]
    return "\n".join(L)
