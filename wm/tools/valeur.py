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


def main() -> int:
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
    RANG = {"forbid_accept": 3, "forbid_phrase": 2, "require_phrase": 2,
            "max_quantity": 1, "min_quantity": 1, "max_money": 1,
            "require_quantity": 1, "require_money": 1}

    def fermete(iss) -> float:
        ls = getattr(iss, "limits", None) or []
        return float(max((RANG.get(l.get("kind"), 0) for l in ls), default=0))

    for pid, p in c.params.items():
        try:
            v_nous = parametric.total(p.option(p.ours).vec())
            v_eux = parametric.total(p.option(p.theirs).vec())
        except KeyError:
            continue
        ampleur.append(abs(v_eux - v_nous))
        ferme.append(fermete(issues.get(pid)))
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
            pval = math.erfc(t / (2 ** 0.5))
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
