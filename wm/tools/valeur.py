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
        b = [d for d in cands if d.name.startswith("B")]
        cands = [b[-1]] if b else cands[-1:]
    if not cands:
        print(f"aucun run avec decisions.json dans {runs}")
        return 1

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
    for pid, p in c.params.items():
        try:
            v_nous = parametric.total(p.option(p.ours).vec())
            v_eux = parametric.total(p.option(p.theirs).vec())
        except KeyError:
            continue
        ampleur.append(abs(v_eux - v_nous))
        ferme.append(float(len(getattr(issues.get(pid), "limits", []) or [])))
        noms.append(p.name)

    if ampleur and any(ferme):
        rho = spearman(ampleur, ferme)
        print(f"\nCorrelation de rang entre l'ampleur que le modele prete a un "
              f"point\net le nombre de limites que le mandat y attache : ", end="")
        if rho is None:
            print("indefinie (trop peu de variation)")
        else:
            print(f"rho = {rho:+.2f}  (n={len(ampleur)})")
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
