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
    """Normalise un numero de section pour pouvoir en comparer deux.

    Les deux modeles ne les ecrivent pas pareil : le moteur dit "sched. C" la
    ou l'extracteur dit "C.1", "13" la ou il dit "13.1", "7.2" la ou il dit
    "7.2(a)". Une egalite de chaines rate tout cela.
    """
    t = str(x or "").lower().split("(")[0]
    for m in ("schedule", "sched.", "sched", "section", "art.", "annexe"):
        t = t.replace(m, " ")
    return t.strip(" .:")


def _couvre(point: str, move: str) -> bool:
    """La section `move` tombe-t-elle sous la section `point` ?"""
    a, b = _sec(point), _sec(move)
    return bool(a) and (b == a or b.startswith(a + "."))


def repartit(moves, sections) -> dict[str, list]:
    """Chaque mouvement va au point LE PLUS SPECIFIQUE qui le couvre.

    C'est ce qui manquait, et c'est ce qui a fait echouer la premiere
    calibration : "sched. C" ne s'appariait a aucun "C.1", "13" a aucun
    "13.1", et cinq des quatorze points chiffrables sortaient donc avec un
    score de zero - dont la resiliation pour defaillance chronique, le plus
    gros poste du markup a -4,7 M$. Une correlation calculee sur cinq zeros
    plantes ne mesurait rien.

    Le plus specifique, et pas tous ceux qui couvrent : sinon un mouvement en
    13.1 compterait a la fois pour "loi applicable" et pour "13", et le meme
    poids serait compte deux fois. Chaque mouvement va a un point et un seul -
    c'est une partition, et c'est ce qui rend la somme des scores lisible.
    """
    out = {s: [] for s in sections}
    for m in moves:
        ms = getattr(m, "section", "")
        cands = [s for s in sections if _couvre(s, ms)]
        if not cands:
            continue
        out[max(cands, key=lambda s: len(_sec(s)))].append(m)
    return out


def _net(moves, nous: str = NOUS) -> float:
    s = 0.0
    for m in moves:
        f = getattr(m, "favours", "")
        if f == nous:
            s += _W.get(getattr(m, "magnitude", ""), 0.0)
        elif f in AX.PARTIES:
            s -= _W.get(getattr(m, "magnitude", ""), 0.0)
    return s


def score(moves, section: str, nous: str = NOUS) -> float:
    """Le net ordinal d'une section, prise seule (elle et ses sous-sections).

    Positif = le markup nous renforce. Les mouvements ambigus et neutres
    comptent zero - c'est volontaire : axes.py les signale au lieu de les
    deviner, et les compter reviendrait a defaire ce choix.
    """
    return _net([m for m in moves if _couvre(section, getattr(m, "section", ""))], nous)


def calibre(moves, n: int = 800, seed: int = 0, nous: str = NOUS) -> dict:
    """Ajuste dollars ~= alpha * ordinal sur les points ou les deux existent."""
    calc = CB.build(n=n, seed=seed)
    # une partition sur les sections du MOTEUR : chaque mouvement compte une
    # fois et une seule, pour le point le plus specifique qui le couvre
    part = repartit(moves, [sec for _, _, sec, _ in CB.VERIDIAN])
    obs, absents = [], []
    for pid, nom, sec, _opts in CB.VERIDIAN:
        p = calc.params[pid]
        d = (pm.total(p.option(p.theirs).vec(), CB.WEIGHTS)
             - pm.total(p.option(p.ours).vec(), CB.WEIGHTS))
        ligne = (pid, nom, sec, _net(part[sec], nous), d)
        # UN ZERO MESURE N'EST PAS UNE DONNEE MANQUANTE. Une section dont tous
        # les mouvements sont ambigus ou neutres vaut vraiment zero. Une
        # section a laquelle AUCUN mouvement n'a ete attribue n'a pas ete
        # mesuree - l'extracteur a range la modification ailleurs, ou il n'y en
        # avait pas. La faire entrer dans la regression avec un zero plante
        # revient a affirmer une observation qu'on n'a pas.
        (obs if part[sec] else absents).append(ligne)
    xs = [o[3] for o in obs]
    ys = [o[4] for o in obs]
    den = sum(x * x for x in xs)
    alpha = (sum(x * y for x, y in zip(xs, ys)) / den) if den else 0.0
    # R2 d'une droite par l'origine, et correlation de rang
    ss_res = sum((y - alpha * x) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum(y * y for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    rho, p_rho = _spearman(xs, ys)
    # exploratoire : quel AXE, s'il en est un, suit les dollars. Neuf axes sur
    # une douzaine de points ne se teste pas serieusement ; cela sert a voir si
    # la structure est la, pas a conclure.
    par_axe = {}
    for a in AX.AXES:
        xa = [_net([m for m in part[o[2]] if m.axis == a], nous) for o in obs]
        if len(set(xa)) > 2:
            par_axe[a] = _spearman(xa, ys)
    return {"alpha": alpha, "r2": r2, "rho": rho, "p": p_rho, "obs": obs,
            "n": len(obs), "absents": absents, "par_axe": par_axe}


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
    # une partition sur les sections du modele ELICITE, distincte de celle du
    # moteur : chaque partition est interne a son modele, et chacune fait
    # tomber chaque mouvement dans exactement un point.
    part = repartit(moves, [_sec(sec_of.get(pid, "")) for pid in elic.params])
    net_de = {pid: _net(part.get(_sec(sec_of.get(pid, "")), []), nous)
              for pid in elic.params}
    params, journal, deja = [], [], set()
    for pid, p in elic.params.items():
        sec = _sec(sec_of.get(pid, ""))
        jumeau = par_sec.get(sec)
        if jumeau in deja:
            # deux points elicites tombent sur le meme point du moteur (la
            # garantie et la garantie de conformite legale sont toutes deux en
            # 7.2) : le second recevrait les memes dollars et les compterait
            # une seconde fois. Il repasse par l'echelle ordinale.
            jumeau = None
        if jumeau and jumeau in calc.params:
            deja.add(jumeau)
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
                            "ordinal": net_de.get(pid, 0.0),
                            "dollars": pm.total(
                                next(o for o in opts if o.id == p.theirs).vec(),
                                CB.WEIGHTS)})
            params.append(pm.Parameter(id=p.id, name=p.name, section=p.section,
                                       options=opts, ours=p.ours, theirs=p.theirs))
            continue
        # ORDINAL CALIBRE : l'echelle du point vient de alpha x son score ; la
        # forme a l'interieur du point vient du vecteur declare, qui est
        # calibre en rang et sert donc a ordonner les redactions entre elles.
        s = net_de.get(pid, 0.0)
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
    for pid, nom, sec, s, d in sorted(cal.get("absents") or [], key=lambda o: o[4]):
        L.append(f"{nom[:39]:<40}{'aucun':>10}{d/1e6:>13.2f}M{'-':>14}"
                 f"   section {sec} : non apparie, hors regression")
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
    if cal.get("par_axe"):
        L += ["", "PAR AXE - lequel, s'il en est un, suit les dollars ? (EXPLORATOIRE :",
              f"neuf axes sur {cal['n']} points ne se teste pas serieusement)", "-" * 84]
        for ax, (r, pv) in sorted(cal["par_axe"].items(),
                                  key=lambda kv: -(abs(kv[1][0]) if kv[1][0] else 0)):
            if r is not None:
                L.append(f"  {ax:<16}rho = {r:+.2f}   (p ~ {pv:.3f})")
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


def triangle(elic: pm.Contract, moves, sec_of: dict, cal: dict, nous: str = NOUS) -> str:
    """Les trois mesures, deux a deux, sur les MEMES points.

    On dispose de trois avis sur la meme chose : ce que le moteur CALCULE en
    dollars, ce qu'un modele de langage DECLARE comme vecteur, et ce que la
    classification par fonction juridique rend en ORDINAL. Deux a deux, cela
    fait trois correlations, et elles doivent tenir ensemble :

        si declare suit les dollars, et que l'ordinal ne les suit pas,
        alors declare ne doit pas suivre l'ordinal non plus.

    Si les trois etaient positives, l'une des mesures serait de trop. Si
    declare suivait l'ordinal sans suivre les dollars, le vecteur declare ne
    serait qu'une reformulation de l'elaboration juridique - et son rho de
    +0,81 contre les dollars serait l'accident. Ce controle ne coute rien et
    il est le seul qui puisse attraper cela.
    """
    calc = CB.build(n=800)
    par_sec = {_sec(sec): pid for pid, _, sec, _ in CB.VERIDIAN}
    part = repartit(moves, [sec for _, _, sec, _ in CB.VERIDIAN])
    D, O, C, noms = [], [], [], []
    for pid_e, p_ in elic.params.items():
        sec = _sec(sec_of.get(pid_e, ""))
        j = par_sec.get(sec)
        if not j or not part.get(next((s2 for _, _, s2, _ in CB.VERIDIAN
                                       if _sec(s2) == sec), "")):
            continue
        q = calc.params[j]
        sec_m = next(s2 for _, _, s2, _ in CB.VERIDIAN if _sec(s2) == sec)
        C.append(pm.total(q.option(q.theirs).vec(), CB.WEIGHTS)
                 - pm.total(q.option(q.ours).vec(), CB.WEIGHTS))
        D.append(pm.total(p_.option(p_.theirs).vec())
                 - pm.total(p_.option(p_.ours).vec()))
        O.append(_net(part[sec_m], nous))
        noms.append(p_.name)
    L = ["", "LES TROIS MESURES, DEUX A DEUX", "-" * 84,
         f"sur les {len(C)} points ou les trois existent"]
    if len(C) < 4:
        return "\n".join(L + ["  trop peu de points apparies pour correler."])
    for nom, a, b in (("calcule $  vs  declare (LLM)", C, D),
                      ("calcule $  vs  ordinal (axes)", C, O),
                      ("declare    vs  ordinal (axes)", D, O)):
        r, pv = _spearman(a, b)
        L.append(f"  {nom:<34}rho = {r:+.2f}   (p ~ {pv:.3f})" if r is not None
                 else f"  {nom:<34}indefini")
    L += ["",
          "  Lecture : si le declare suit les dollars et que l'ordinal ne les suit pas,",
          "  le declare ne doit pas suivre l'ordinal. Les trois correlations ne peuvent",
          "  pas etre positives ensemble - si elles l'etaient, une des mesures serait",
          "  de trop, et ce serait probablement celle qu'on croit la plus solide."]
    return "\n".join(L) + "\n" + quadrants(noms, C, O)


def quadrants(noms, dollars, ordinal) -> str:
    """Croiser les deux mesures au lieu de les convertir l'une en l'autre.

    Le pont a echoue parce que l'elaboration juridique et le poids financier
    vont en sens inverse. Mais deux mesures qui divergent sont plus
    informatives qu'une seule : c'est leur DESACCORD qui designe les clauses
    interessantes.

      cher ET lourd juridiquement   le markup attaque franchement ; tout le
                                    monde le voit, y compris l'adversaire
      pas cher, lourd juridiquement de la procedure. Il ecrit beaucoup pour
                                    peu d'argent - c'est la monnaie d'echange
                                    la moins chere qu'on puisse lui rendre
      CHER, LEGER juridiquement     une phrase qui coute des millions et qui
                                    ne ressemble a rien dans le redline.
                                    C'est ce que Scott, Choi & Gulati appellent
                                    une MINE (41 Yale J. Reg. 307, 2024) :
                                    l'ecart le plus dangereux est celui qui ne
                                    se voit pas.
      ni l'un ni l'autre            a accepter sans discuter

    La coupure est la MEDIANE de chaque mesure : rien ici ne justifie un seuil
    absolu, et une mediane se defend - elle dit "dans la moitie haute de CE
    markup", pas "au-dessus d'un chiffre que j'ai choisi".
    """
    import statistics as st
    if len(noms) < 4:
        return ""
    md, mo = st.median([abs(x) for x in dollars]), st.median([abs(x) for x in ordinal])
    cases = {"mine": [], "franc": [], "procedure": [], "mineur": []}
    for n_, d_, o_ in zip(noms, dollars, ordinal):
        cher, lourd = abs(d_) > md, abs(o_) > mo
        cases[("franc" if lourd else "mine") if cher
              else ("procedure" if lourd else "mineur")].append((n_, d_, o_))
    L = ["", "LES DEUX MESURES CROISEES, PLUTOT QUE CONVERTIES", "-" * 84,
         f"coupures : {md/1e6:.2f} M$ et {mo:.0f} points de net ordinal (medianes)", ""]
    for cle, titre in (("mine", "MINES - cher, et juridiquement discret. A regarder en premier."),
                       ("franc", "ATTAQUE FRANCHE - cher et visible. L'adversaire sait ce qu'il demande."),
                       ("procedure", "PROCEDURE - beaucoup d'ecriture, peu d'argent. Monnaie d'echange."),
                       ("mineur", "MINEUR - ni l'un ni l'autre.")):
        if not cases[cle]:
            continue
        L.append(titre)
        for n_, d_, o_ in sorted(cases[cle], key=lambda r: r[1]):
            L.append(f"    {n_[:46]:<48}{d_/1e6:>8.2f} M${o_:>8.0f}")
        L.append("")
    L += ["C'est la seule chose que le pont ordinal, en echouant, a rendue possible :",
          "si les deux mesures allaient dans le meme sens, les croiser n'apprendrait rien."]
    return "\n".join(L)
