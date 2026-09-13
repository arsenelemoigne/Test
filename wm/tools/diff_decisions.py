"""Compare ce que deux arms DECIDENT, a modele identique.

    python3 wm/tools/diff_decisions.py A4 B1 [modele]

La rubrique note si chaque point a ete traite. Elle ne dit pas si deux arms ont
pris la meme position, et c'est pourtant la seule facon de voir si une
representation change la negociation plutot que la note.

LES DEUX ARMS DOIVENT TOURNER SUR LE MEME MODELE. Une premiere version prenait
le premier run trouve par ordre alphabetique et comparait glm-4.5-air a
glm-4.6 : les ecarts mesures etaient alors ceux des deux modeles, pas ceux des
deux arms, et rien dans la sortie ne le disait.
"""
import json, sys, pathlib
from collections import defaultdict

ROOT = pathlib.Path("wm/runs")


def inventaire():
    """arm -> modele -> liste de dossiers."""
    out = defaultdict(lambda: defaultdict(list))
    for d in sorted(ROOT.rglob("*__*__seed*")):
        if not (d / "decisions.json").exists():
            continue
        cond, modele, _ = d.name.split("__", 2)
        out[cond][modele].append(d)
    return out


def charge(d):
    return {x["issue_id"]: x for x in json.loads((d / "decisions.json").read_text())}


def mots(s):
    return set(str(s).lower().split())


inv = inventaire()
a_name, b_name = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("A4", "B1")
voulu = sys.argv[3] if len(sys.argv) > 3 else None

communs_modeles = sorted(set(inv.get(a_name, {})) & set(inv.get(b_name, {})))
if voulu:
    communs_modeles = [m for m in communs_modeles if voulu in m]

if not communs_modeles:
    print(f"AUCUN MODELE COMMUN entre {a_name} et {b_name}.")
    print(f"  {a_name} tourne sur : {', '.join(sorted(inv.get(a_name, {}))) or 'rien'}")
    print(f"  {b_name} tourne sur : {', '.join(sorted(inv.get(b_name, {}))) or 'rien'}")
    print("\nComparer deux arms sur deux modeles differents mesure l'ecart entre")
    print("les modeles, pas entre les arms. Relance l'arm manquant sur le modele")
    print("de l'autre avant de comparer.")
    raise SystemExit(1)

modele = communs_modeles[-1]
A, B = inv[a_name][modele][0], inv[b_name][modele][0]
a, b = charge(A), charge(B)
print(f"modele commun : {modele}")
print(f"  {a_name}: {A.relative_to(ROOT)}")
print(f"  {b_name}: {B.relative_to(ROOT)}")
if len(communs_modeles) > 1:
    print(f"  (aussi disponible : {', '.join(communs_modeles[:-1])})")
print()

communs = [k for k in a if k in b]
diff = [k for k in communs if a[k]["disposition"] != b[k]["disposition"]]
print(f"{len(diff)}/{len(communs)} points ou la DISPOSITION differe")
for k in diff:
    print(f"  {k}  {a_name}={a[k]['disposition']:<8} {b_name}={b[k]['disposition']}")

ecart = []
for k in communs:
    if a[k]["disposition"] != b[k]["disposition"]:
        continue
    x, y = mots(a[k].get("counter", "")), mots(b[k].get("counter", ""))
    if x or y:
        j = len(x & y) / max(1, len(x | y))
        if j < 0.5:
            ecart.append((k, j))
print(f"\n{len(ecart)}/{len(communs)} points ou la disposition est la MEME mais")
print("la contre-proposition differe nettement (moins de 50% de mots communs) :")
for k, j in sorted(ecart, key=lambda t: t[1])[:8]:
    print(f"  {k}  recouvrement {j:.0%}")
    print(f"      {a_name}: {a[k].get('counter','')[:96]}")
    print(f"      {b_name}: {b[k].get('counter','')[:96]}")
