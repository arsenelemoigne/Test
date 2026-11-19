"""Per-task thinking budgets for the challenge score (consult30 priority 3 on Dev, user 18 Nov).

From the forced early-stop curves of the trained model (budget_curves.py, absolute budgets) and the ORIGINAL's natural
answers on the same questions, search every combination of one budget per task (or the natural length) and report the
combination that saves the most under the official score at a given accuracy:
    saving   = 1 - geometric mean over svamp / mbpp / ifeval / bfcl of (tokens / original tokens)
    accuracy = mean over the same four of (accuracy - original accuracy); bfcl pools its three parts with weights 400/200/240.
Constraint levels: the trained model's own natural Dev accuracy change (same accuracy), and that minus --slack points
(Dev reads accuracy ~3 points high, so compare against the natural model on Dev, not against the full-test numbers).

    python eff_allocate.py --base runs/eff/curves-base/curves.json --model runs/eff/curves-od5c-best/curves.json --out runs/eff/allocate.txt
"""
import argparse
import itertools
import json
import math
from pathlib import Path

BFCL_W = {"bfcl_simple_python": 400, "bfcl_multiple": 200, "bfcl_irrelevance": 240}
SINGLE = ("svamp", "mbpp", "ifeval")
PARTS = SINGLE + tuple(BFCL_W)


def score(base, choice):
    """choice: part -> {"acc", "tokens"}. Returns (saving %, accuracy change in points)."""
    logs, dacc = [], []
    for b in SINGLE:
        logs.append(math.log(choice[b]["tokens"] / base[b]["tokens"]))
        dacc.append(choice[b]["acc"] - base[b]["acc"])
    w = sum(BFCL_W.values())
    t1 = sum(BFCL_W[p] * choice[p]["tokens"] for p in BFCL_W) / w
    t0 = sum(BFCL_W[p] * base[p]["tokens"] for p in BFCL_W) / w
    a1 = sum(BFCL_W[p] * choice[p]["acc"] for p in BFCL_W) / w
    a0 = sum(BFCL_W[p] * base[p]["acc"] for p in BFCL_W) / w
    logs.append(math.log(t1 / t0))
    dacc.append(a1 - a0)
    return 100 * (1 - math.exp(sum(logs) / 4)), sum(dacc) / 4


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True, help="curves.json of the original (its 'natural' row is the reference)")
    ap.add_argument("--model", required=True, help="curves.json of the trained model")
    ap.add_argument("--slack", type=float, default=0.22, help="extra accuracy points allowed (-1.00 vs -0.78 on the full test)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base = {p: v["natural"] for p, v in json.loads(Path(args.base).read_text())["curves"].items()}
    cur = json.loads(Path(args.model).read_text())["curves"]
    opts = {p: [("natural", cur[p]["natural"])] + [(n, v) for n, v in cur[p].items() if n != "natural" and v.get("kind") == "b"]
            for p in PARTS}
    nat = score(base, {p: cur[p]["natural"] for p in PARTS})
    best = {}
    for combo in itertools.product(*(opts[p] for p in PARTS)):
        choice = {p: c[1] for p, c in zip(PARTS, combo)}
        s, a = score(base, choice)
        for label, floor in (("same accuracy", nat[1]), (f"accuracy - {args.slack}", nat[1] - args.slack)):
            if a >= floor - 1e-9 and (label not in best or s > best[label][0]):
                best[label] = (s, a, {p: c[0] for p, c in zip(PARTS, combo)})
    lines = [f"natural trained model on Dev: saved {nat[0]:.1f}% / accuracy {nat[1]:+.2f} vs the original's natural answers"]
    for label, (s, a, ch) in best.items():
        lines.append(f"best per-task budgets at {label}: saved {s:.1f}% / {a:+.2f}  (gain {s - nat[0]:+.1f} points of saving)")
        lines.append("   " + ", ".join(f"{p} {ch[p]}" for p in PARTS))
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
