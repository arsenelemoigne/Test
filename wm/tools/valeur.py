"""Le modele parametrique, confronte a un run reel. Aucun appel API.

    python3 wm/tools/valeur.py [run] [--tous]

Trois questions, dans l'ordre ou elles se posent - et la premiere commande les
deux autres :

  1. COUVERTURE. La contre-proposition designe-t-elle des redactions que le
     domaine connait ? Si un modele repond "autre" partout, le bloc de valeur
     est calcule sur presque rien et son pourcentage est decoratif.
  2. VALEUR. Le bloc que la boucle rend au modele, sur les vraies decisions.
  3. REPRESENTATIVITE. Le modele donne-t-il plus de poids aux points que le
     mandat declare fermes qu'aux autres ? Il n'existe pas de verite terrain
     pour ces vecteurs - ils sont declares par un LLM, pas mesures. Mais s'ils
     n'ont AUCUN rapport avec ce que le client dit tenir, ils ne representent
     pas ce contrat-ci, et la question est tranchee dans le mauvais sens.
"""
import json
import pathlib
import sys
import math
import statistics

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from wm import parametric, taskctx                                  # noqa: E402
from wm.abstraction import Decision, Disposition                    # noqa: E402


def charge_decisions(d: pathlib.Path):
    rows = json.loads((d / "decisions.json").read_text())
    out = []
    for r in rows:
        x = Decision(issue_id=r["issue_id"],
                     disposition=Disposition(r["disposition"]),
                     counter=r.get("counter", ""), rationale=r.get("rationale", ""))
        x.option_id = str(r.get("option_id", "") or "")
        out.append(x)
    return out


# La FERMETE d'un point, lue dans la NATURE de ses limites, pas dans leur
# nombre. Sur le second contrat, 22 points sur 24 en portaient exactement une :
# la variable de reference etait quasi constante et une correlation de rang n'y
# mesurait rien - le rho de 0,13 ne disait pas que le modele est mal calibre,
# il disait que le test etait aveugle.
#   3  forbid_accept          la position ne peut pas etre acceptee : walk-away
#   2  forbid/require_phrase  une formule precise est imposee ou interdite
#   1  max/min quantity, money  une fourchette, negociable a l'interieur
#   0  aucune limite
_RANG = {"forbid_accept": 3, "forbid_phrase": 2, "require_phrase": 2,
         "max_quantity": 1, "min_quantity": 1, "max_money": 1,
         "require_quantity": 1, "require_money": 1}


def _fermete(iss) -> float:
    ls = getattr(iss, "limits", None) or []
    return float(max((_RANG.get(l.get("kind"), 0) for l in ls), default=0))


# La loi de Student vit dans wm/tools_stats.py : deux copies d'une meme formule
# finissent toujours par diverger, et c'est elle qui a corrige un p de 0,001 en
# 0,005 sur le resultat principal.
from wm.tools_stats import p_student                            # noqa: E402


def spearman(xs, ys):
    """Sans scipy. Rend None si un rang est indefini."""
    n = len(xs)
    if n < 4:
        return None

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

    rx, ry = rangs(xs), rangs(ys)
    if len(set(rx)) < 2 or len(set(ry)) < 2:
        return None
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None


def _rho_fermete(task_dir) -> tuple[float | None, int, dict]:
    """(rho, n, distribution de fermete) pour une tache, sans aucun run."""
    import os
    prev = os.environ.get("WM_TASK_DIR")
    os.environ["WM_TASK_DIR"] = str(task_dir)
    try:
        f = pathlib.Path(task_dir) / "parametric.json"
        if not f.exists():
            return None, 0, {}
        c = parametric.load(f.read_text())
        g = taskctx.gen_issues()
        iss = {i.id: i for i in g} if g else {}
        amp, fer = [], []
        for pid, prm in c.params.items():
            try:
                a = parametric.total(prm.option(prm.ours).vec())
                b = parametric.total(prm.option(prm.theirs).vec())
            except KeyError:
                continue
            amp.append(abs(b - a))
            fer.append(_fermete(iss.get(pid)))
        from collections import Counter
        return spearman(amp, fer), len(amp), dict(sorted(Counter(fer).items()))
    finally:
        if prev is None:
            os.environ.pop("WM_TASK_DIR", None)
        else:
            os.environ["WM_TASK_DIR"] = prev


def cmd_pool() -> int:
    """La calibration sur TOUS les contrats encodes, combinee.

    A n = 24 un rho de 0,30 n'est pas distinguable de zero ; deux contrats
    independants qui vont dans le meme sens le sont peut-etre. Les rho sont
    combines par la transformation z de Fisher, ponderes par n - 3 : c'est la
    facon standard d'agreger des correlations, et elle ne suppose pas que les
    deux contrats aient la meme echelle de valeur.
    """
    base = pathlib.Path(__file__).resolve().parents[1] / "tasks"
    rows = []
    for d in sorted(base.iterdir()):
        if not (d / "parametric.json").exists():
            continue
        rho, n, rep = _rho_fermete(d)
        rows.append((d.name, rho, n, rep))
    dflt = pathlib.Path(__file__).resolve().parents[1] / "task"
    if (dflt / "parametric.json").exists():
        rows.append(("task (DSA)",) + _rho_fermete(dflt))
    if not rows:
        print("aucun contrat encode (parametric.json) sous wm/tasks/")
        return 1

    print("=" * 74)
    print("CALIBRATION COMBINEE - l'ampleur prêtee a un point suit-elle la")
    print("fermete du mandat sur ce point ?")
    print("=" * 74)
    num = den = 0.0
    for nom, rho, n, rep in rows:
        dist = " ".join(f"{int(k)}:{v}" for k, v in rep.items())
        if rho is None or n < 4:
            print(f"  {nom[:46]:<48} n={n:<4} rho indefini   [{dist}]")
            continue
        z = 0.5 * math.log((1 + rho) / (1 - rho))
        num += (n - 3) * z
        den += (n - 3)
        print(f"  {nom[:46]:<48} n={n:<4} rho {rho:+.2f}      [{dist}]")
    if den <= 0:
        print("\npas assez de points pour combiner.")
        return 0
    z = num / den
    rho_c = (math.exp(2 * z) - 1) / (math.exp(2 * z) + 1)
    se = 1 / math.sqrt(den)
    p = math.erfc(abs(z / se) / math.sqrt(2))
    print("-" * 74)
    print(f"  {'COMBINE (Fisher z, ponderation n-3)':<48} n={int(den) + 3 * len(rows):<4} "
          f"rho {rho_c:+.2f}   p ~ {p:.3f}")
    print()
    if p <= 0.05:
        print("  La concordance avec le mandat est etablie a ce seuil. Elle reste une")
        print("  CONCORDANCE : le mandat a servi a l'elicitation, donc les deux ne sont")
        print("  pas independants. Elle ne dit pas que les dollars sont justes, elle dit")
        print("  que le modele ne range pas les points au hasard.")
    else:
        print("  Toujours pas distinguable de zero. Il faut soit d'autres contrats, soit")
        print("  une elicitation qui ne demande pas des nombres - des comparaisons par")
        print("  paires, par exemple, dont la coherence se mesure.")
    return 0


def cmd_calib() -> int:
    """Le vrai test de calibration : le vecteur DECLARE contre les dollars CALCULES.

    Le test de fermete compare une magnitude a une categorie juridique, et ne
    sait pas distinguer "le modele est mal calibre" de "fermete et magnitude
    sont deux choses differentes" - un walk-away sur qui paie l'audit est ferme
    et petit, un plafond a 34 M$ n'est qu'une fourchette et il est enorme. La
    remarque vaut contre mon propre test, et elle le disqualifie comme preuve.

    Ici les deux grandeurs sont de meme nature : ce que l'elicitation DIT que
    coute une redaction, et ce que le moteur de scenarios CALCULE qu'elle
    coute, sur les points ou les deux existent. Rien n'y est categorie. Si
    elles ne se suivent pas, l'une des deux se trompe, et ce n'est pas
    ambigu.
    """
    import os
    from wm import claim_bridge, smarter
    T = pathlib.Path(__file__).resolve().parents[1] / "tasks" / \
        "license-agreement-first-turn-redline-scenario-04"
    f = T / "parametric.json"
    if not f.exists():
        print(f"{f} n'existe pas - lance `model --elicit` sur ce contrat.")
        return 1
    os.environ["WM_TASK_DIR"] = str(T)
    elic = parametric.load(f.read_text())
    fs = T / "parametric_smarter.json"
    rang = parametric.load(fs.read_text()) if fs.exists() else None
    g = taskctx.gen_issues() or []
    sec_of = {i.id: str(getattr(i, "section", "")).strip() for i in g}
    nom_of = {i.id: i.name for i in g}

    calc = claim_bridge.build(n=800)
    # les deux modeles nomment leurs points differemment : on apparie par
    # numero de section, la seule adresse qu'ils partagent
    par_sec = {}
    for pid, prm in elic.params.items():
        sec = sec_of.get(pid, "").split("(")[0].strip()
        if sec:
            par_sec.setdefault(sec, []).append(pid)

    def ecart(c, pid, w=None):
        p_ = c.params[pid]
        return abs(parametric.total(p_.option(p_.theirs).vec(), w)
                   - parametric.total(p_.option(p_.ours).vec(), w))

    lignes = []
    for pid, nom, sec, _ in claim_bridge.VERIDIAN:
        base = sec.split("(")[0].strip()
        cands = par_sec.get(base) or []
        if not cands:
            continue
        try:
            d_dec = ecart(elic, cands[0])
            d_cal = ecart(calc, pid, claim_bridge.WEIGHTS)
            d_rng = (ecart(rang, cands[0], smarter.WEIGHTS)
                     if rang is not None and cands[0] in rang.params else None)
        except KeyError:
            continue
        lignes.append((nom, nom_of.get(cands[0], cands[0]), d_dec, d_cal, d_rng,
                       cands[0]))

    # Deux points du moteur peuvent tomber sur la meme section - le niveau de
    # service et la resiliation pour defaillance chronique sont tous deux
    # annexe C. Ils recoivent alors le MEME point elicite, et le compter deux
    # fois gonflerait n avec une observation qui n'en est pas une. On garde le
    # premier et on le dit.
    vus, uniques, doublons = set(), [], []
    for r in lignes:
        (doublons if r[5] in vus else uniques).append(r)
        vus.add(r[5])

    print("=" * 92)
    print("CALIBRATION : ce que l'elicitation DIT, contre ce que les scenarios CALCULENT")
    print("=" * 92)
    if len(uniques) < 4:
        print(f"seulement {len(uniques)} points apparies - pas assez pour correler.")
        return 0
    a_rang = rang is not None and any(r[4] is not None for r in lignes)
    print(f"{'point (moteur)':<32}{'point (elicite)':<28}{'declare':>10}"
          + (f"{'rangs':>10}" if a_rang else "") + f"{'calcule $':>14}")
    print("-" * 92)
    dbl = {id(r) for r in doublons}
    for r in sorted(lignes, key=lambda r: -r[3]):
        a, b, dd, dc, dr = r[0], r[1], r[2], r[3], r[4]
        col = ""
        if a_rang:
            col = f"{dr:>10.0f}" if dr is not None else f"{'-':>10}"
        marque = "  (meme section, non compte)" if id(r) in dbl else ""
        print(f"{a[:31]:<32}{b[:27]:<28}{dd:>10.0f}{col}{dc/1e6:>13.2f}M{marque}")
    print("-" * 92)
    if doublons:
        print(f"{len(doublons)} point(s) du moteur partagent une section avec un autre "
              f"et sont exclus\nde la correlation : "
              + ", ".join(x[0][:28] for x in doublons))

    lignes = uniques
    cal = [r[3] for r in lignes]

    def rapporte(nom, xs, ys):
        rho = spearman(xs, ys)
        if rho is None:
            print(f"{nom:<26} rho indefini (trop peu de variation)")
            return None, None
        t = abs(rho) * ((len(xs) - 2) / max(1e-12, 1 - rho ** 2)) ** 0.5
        pval = p_student(t, len(xs) - 2)
        print(f"{nom:<26} rho = {rho:+.2f}   (n={len(xs)}, p ~ {pval:.3f})")
        return rho, pval

    print("correlation de rang avec les dollars calcules :")
    r_dec, p_dec = rapporte("  elicitation directe", [r[2] for r in lignes], cal)
    r_rng = p_rng = None
    if a_rang:
        pairs = [(r[4], r[3]) for r in lignes if r[4] is not None]
        r_rng, p_rng = rapporte("  elicitation par rangs", [x for x, _ in pairs],
                                [y for _, y in pairs])
    print()

    def verdict(r, p, quoi):
        if r is None:
            return
        if p <= 0.05:
            print(f"  {quoi} suit le moteur en RANG. Elle n'est pas calibree en")
            print("  niveau, mais la negociation par ratios n'a besoin que de l'ordre.")
        else:
            print(f"  {quoi} ne suit pas le moteur. Comme le moteur, lui, se")
            print("  derive de termes verifiables contre le texte, c'est l'elicitation")
            print("  qui est en cause : elle ne mesure pas ce qu'elle pretend mesurer.")

    verdict(r_dec, p_dec, "L'elicitation directe")
    if r_rng is None:
        print()
        print("  (pas de modele par rangs : `python -m wm.run model --smarter` en construit")
        print("   un sur le MEME domaine, et cette table gagne une colonne.)")
        return 0
    print()
    verdict(r_rng, p_rng, "L'elicitation par rangs")
    print()
    d = r_rng - r_dec
    print(f"  ecart rangs - direct : {d:+.2f}")
    if abs(d) < 0.15:
        print(f"  Les deux elicitations se valent sur ce contrat. A n = {len(lignes)}")
        print("  points, un ecart de cette taille ne se distingue pas du bruit : ne pas")
        print("  conclure que les rangs n'apportent rien, conclure que ce test ne le")
        print("  montre pas.")
    elif d > 0:
        print("  Les rangs sont mieux calibres. Reste la question qui compte : est-ce")
        print("  que cela se voit en negociation ? `neg --contract smarter` contre")
        print("  `neg --contract elicited`, meme simulateur, memes adversaires, le dit.")
    else:
        print("  Les rangs sont MOINS bien calibres que les nombres declares. Le remede")
        print("  que propose la litterature ne marche pas ici : le dire, et garder le")
        print("  moteur de scenarios comme source de valeur partout ou il s'applique.")
    return 0

def main() -> int:
    if "--calib" in sys.argv:
        return cmd_calib()
    if "--pool" in sys.argv:
        return cmd_pool()
    runs = taskctx.runs_dir()
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    cands = [d for d in sorted(runs.glob("*__*__seed*"))
             if (d / "decisions.json").exists()]
    if args:
        cands = [d for d in cands if args[0] in d.name]
    elif "--tous" not in sys.argv:
        # Un run qui ne porte aucun option_id ne peut rien valoriser : il date
        # d'avant le champ, ou le modele l'a ignore. Le choisir par defaut
        # n'apprend que cela, donc on prefere un run qui en porte.
        def porte_option(d):
            try:
                return any(r.get("option_id")
                           for r in json.loads((d / "decisions.json").read_text()))
            except Exception:                                   # noqa: BLE001
                return False
        b = [d for d in cands if d.name.startswith("B")]
        pool = [d for d in b if porte_option(d)] or b or cands
        # L'absence de run n'est pas une raison de ne rien dire : la
        # representativite du modele ne demande aucun run, et c'est justement
        # ce qu'on veut savoir d'un contrat qu'on vient d'encoder. Le message
        # renvoyait a une section "ci-dessus" qui n'avait pas ete imprimee.
        cands = [pool[-1]] if pool else []

    f = taskctx.task_dir() / "parametric.json"
    if not f.exists():
        print(f"{f} n'existe pas - lance `python -m wm.run model --elicit`")
        return 1
    c = parametric.load(f.read_text())

    # gen_issues(), pas issues() : issues() rend des Issue adaptes pour le
    # prompt, qui ont perdu le champ `limits`. La correlation aurait alors
    # porte sur une liste vide et se serait declaree incalculable.
    g = taskctx.gen_issues()
    issues = {i.id: i for i in g} if g else {}

    # --- 3. REPRESENTATIVITE : independante du run, calculee une fois -------
    print("=" * 74)
    print("REPRESENTATIVITE DU MODELE")
    print("=" * 74)
    print(f"{len(c.params)} variables modelisees ; la liste des points en compte "
          f"{len(issues) if issues else '?'}.")
    manquants = [i for i in issues if i not in c.params] if issues else []
    if manquants:
        print(f"{len(manquants)} points n'ont AUCUNE variable : "
              f"{', '.join(sorted(manquants)[:10])}")
        print("  Sur ceux-la le modele economique n'a rien a dire, quoi qu'il "
              "affiche par ailleurs.")

    ampleur, ferme, noms = [], [], []
    # La FERMETE d'un point, lue dans la NATURE de ses limites, pas dans leur
    # nombre. Sur le second contrat, 22 points sur 24 en portaient exactement
    # une : la variable de reference etait quasi constante et une correlation
    # de rang n'y mesurait rien - le rho de 0,13 ne disait pas que le modele
    # est mal calibre, il disait que le test etait aveugle.
    #   3  forbid_accept      la position ne peut pas etre acceptee : walk-away
    #   2  forbid/require_phrase  une formule precise est imposee ou interdite
    #   1  max/min quantity, money  une fourchette, negociable a l'interieur
    #   0  aucune limite

    for pid, p in c.params.items():
        try:
            v_nous = parametric.total(p.option(p.ours).vec())
            v_eux = parametric.total(p.option(p.theirs).vec())
        except KeyError:
            continue
        ampleur.append(abs(v_eux - v_nous))
        ferme.append(_fermete(issues.get(pid)))
        noms.append(p.name)

    if ampleur and any(ferme):
        rho = spearman(ampleur, ferme)
        from collections import Counter as _C
        rep = _C(ferme)
        print(f"\nFermete du mandat par point (3 = walk-away, 2 = formule imposee, "
              f"1 = fourchette, 0 = libre) :")
        print("  " + ", ".join(f"{k:.0f} -> {v} points" for k, v in sorted(rep.items())))
        if max(rep.values()) > 0.8 * len(ferme):
            print("  QUASI CONSTANTE : une correlation de rang n'y mesurerait rien.")
        print(f"\nCorrelation de rang entre l'ampleur que le modele prete a un "
              f"point\net la fermete du mandat sur ce point : ", end="")
        if rho is None:
            print("indefinie (trop peu de variation)")
        else:
            n = len(ampleur)
            t = abs(rho) * ((n - 2) / max(1e-12, 1 - rho ** 2)) ** 0.5
            # approximation normale du test de Student, suffisante ici : elle
            # sert a dire "ce rho n'est pas distinguable de zero", pas a publier.
            import math
            pval = p_student(t, n - 2)
            print(f"rho = {rho:+.2f}  (n={n}, p ~ {pval:.2f})")
            if pval > 0.05:
                print("  A ce n, ce rho n'est PAS distinguable de zero. La")
                print("  concordance avec le mandat n'est ni etablie ni refutee :")
                print("  le modele pourrait aussi bien ranger les points au hasard.")
            if rho < 0.2:
                print("  Le modele ne met PAS son poids la ou le client dit tenir.")
                print("  Il decrit peut-etre un contrat, mais pas les priorites")
                print("  de celui-ci - c'est le sens le plus exigeant de "
                      "'representatif',")
                print("  et il n'est pas atteint.")
            else:
                print("  Le modele pese davantage les points que le mandat "
                      "encadre.")
                print("  C'est une concordance, pas une validation : le mandat "
                      "a servi")
                print("  a l'elicitation, donc les deux ne sont pas independants.")
    else:
        print("\nPas de limites lisibles : la concordance avec le mandat n'est "
              "pas calculable.")

    if ampleur:
        par = sorted(zip(noms, ampleur), key=lambda kv: -kv[1])
        print(f"\nLes 5 points ou le modele voit le plus d'ecart entre les deux "
              f"redactions :")
        for n, a in par[:5]:
            print(f"  {a:>9.0f}  {n[:56]}")
        nuls = [n for n, a in par if a < 1e-9]
        if nuls:
            print(f"\n{len(nuls)} variables ont un ecart NUL entre notre redaction "
                  f"et la leur :")
            print(f"  {', '.join(nuls[:6])}")
            print("  Le modele dit qu'y ceder est gratuit. A verifier a la main.")

    # --- 1 et 2 : par run --------------------------------------------------
    if not cands:
        print()
        print("=" * 74)
        print(f"AUCUN RUN DE CONTRE-PROPOSITION dans {runs}")
        print("=" * 74)
        print("Le modele ci-dessus existe, mais rien ne l'a encore utilise : ni")
        print("couverture ni valeur ne sont calculables. Pour en produire un :")
        print("  python -m wm.run trial B1 z-ai/glm-4.6 1")
        print("Le simulateur de negociation, lui, n'a pas besoin de run prealable :")
        print("  python -m wm.run neg --policies algo,llm_raw,llm_value --n 6 --rounds 6 \\")
        print("      --model z-ai/glm-4.6")
        return 0
    for d in cands:
        print()
        print("=" * 74)
        print(f"RUN {d.name}")
        print("=" * 74)
        dec = charge_decisions(d)
        a, hors = parametric.assignment_from(dec, c)
        connus = [x for x in dec if x.issue_id in c.params]
        designes = [x for x in connus
                    if any(o.id == getattr(x, "option_id", "")
                           for o in c.params[x.issue_id].options)]
        print(f"decisions              : {len(dec)}")
        print(f"  dont une variable    : {len(connus)}")
        print(f"  dont une redaction   : {len(designes)}"
              f"   <-- seules celles-ci sont valorisees")
        if hors:
            print(f"  redaction inconnue   : {len(hors)}  ({', '.join(hors[:6])})")
        if connus:
            part = len(designes) / len(connus)
            print(f"\ncouverture : {part:.0%} des points modelises portent une "
                  f"redaction du domaine.")
            if part < 0.5:
                print("  SOUS LA MOITIE. Le bloc ci-dessous est calcule sur une "
                      "minorite des")
                print("  points : son pourcentage de valeur recuperee ne decrit "
                      "pas la reponse.")
        print()
        print(parametric.value_feedback(dec, c))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
