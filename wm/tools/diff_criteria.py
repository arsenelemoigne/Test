"""Quels criteres chaque arm a-t-il rates, et lesquels seul l'un d'eux rate ?

    python3 wm/tools/diff_criteria.py A0 B1 [modele]

Une note globale dit qu'un arm est moins bon. Elle ne dit pas SUR QUOI, et c'est
la seule question qui permette de distinguer "il lui manquait le contexte" de
"il raisonne moins bien". Le juge ecrit un verdict et une justification par
critere : il suffit de les lire.
"""
import json, sys, pathlib
from collections import defaultdict

ROOT = pathlib.Path("wm/runs")


def inventaire():
    out = defaultdict(lambda: defaultdict(list))
    for d in sorted(ROOT.rglob("*__*__seed*")):
        if list(d.glob("scores__*.json")):
            cond, modele, _ = d.name.split("__", 2)
            out[cond][modele].append(d)
    return out


def scores(d):
    f = sorted(d.glob("scores__*.json"))[0]
    s = json.loads(f.read_text())
    rows = s.get("criteria") or s.get("results") or []
    return {r["id"]: r for r in rows if isinstance(r, dict) and "id" in r}


def rate(r):
    return str(r.get("verdict", "")).lower() not in ("pass", "true", "yes")


inv = inventaire()
a_name, b_name = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("A0", "B1")
voulu = sys.argv[3] if len(sys.argv) > 3 else None
commun = sorted(set(inv.get(a_name, {})) & set(inv.get(b_name, {})))
if voulu:
    commun = [m for m in commun if voulu in m]
if not commun:
    print(f"AUCUN MODELE COMMUN entre {a_name} et {b_name}.")
    for n in (a_name, b_name):
        print(f"  {n} : {', '.join(sorted(inv.get(n, {}))) or 'aucun run note'}")
    raise SystemExit(1)

m = commun[-1]
A, B = inv[a_name][m][0], inv[b_name][m][0]
a, b = scores(A), scores(B)
if not a or not b:
    print(f"scores illisibles : {a_name}={len(a)} criteres, {b_name}={len(b)}")
    raise SystemExit(1)

ids = [k for k in a if k in b]
ra = {k for k in ids if rate(a[k])}
rb = {k for k in ids if rate(b[k])}
print(f"modele commun : {m}   ({len(ids)} criteres)")
print(f"  {a_name} rate {len(ra)}   {b_name} rate {len(rb)}\n")

seul_b = sorted(rb - ra)
print(f"=== {len(seul_b)} criteres que SEUL {b_name} rate " 
      f"(c'est ce qui explique l'ecart de note)")
for k in seul_b:
    print(f"\n  {b[k].get('title','')[:96]}")
    print(f"      {b_name} : {str(b[k].get('reasoning',''))[:200]}")

seul_a = sorted(ra - rb)
if seul_a:
    print(f"\n=== {len(seul_a)} criteres que seul {a_name} rate")
    for k in seul_a:
        print(f"  {a[k].get('title','')[:96]}")

deux = sorted(ra & rb)
if deux:
    print(f"\n=== {len(deux)} rates par les deux (le probleme n'est ni l'un ni l'autre)")
    for k in deux:
        print(f"  {a[k].get('title','')[:96]}")
