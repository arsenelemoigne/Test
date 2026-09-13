"""Compare les dispositions de deux arms, ou qu'ils se trouvent."""
import json, sys, pathlib

ROOT = pathlib.Path("wm/runs")


def trouve(cond, motif="z-ai"):
    hits = [d for d in ROOT.rglob(f"{cond}__*__seed*")
            if motif in d.name and (d / "decisions.json").exists()]
    return sorted(hits)


def charge(d):
    return {x["issue_id"]: x for x in json.loads((d / "decisions.json").read_text())}


a_name, b_name = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("A4", "B1")
A, B = trouve(a_name), trouve(b_name)
if not A or not B:
    print(f"introuvable : {a_name}={len(A)} runs, {b_name}={len(B)} runs")
    print("\nce qui existe :")
    for d in sorted(ROOT.rglob("*__*__seed*")):
        if (d / "decisions.json").exists():
            print(f"   {d.relative_to(ROOT)}")
    raise SystemExit(1)

a, b = charge(A[0]), charge(B[0])
print(f"{a_name}: {A[0].relative_to(ROOT)}")
print(f"{b_name}: {B[0].relative_to(ROOT)}\n")

communs = [k for k in a if k in b]
diff = [k for k in communs if a[k]["disposition"] != b[k]["disposition"]]
print(f"{len(diff)}/{len(communs)} points ou la DISPOSITION differe")
for k in diff:
    print(f"  {k}  {a_name}={a[k]['disposition']:<8} {b_name}={b[k]['disposition']}")

# une disposition identique peut cacher une contre-proposition tres differente
def mots(s):
    return set(str(s).lower().split())

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
