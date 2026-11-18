"""Question-level change between two models on the full test (user 18 Nov; consult30's corrected measure). CPU, light.

Per question q: k0, k1 = right answers of 8 (original, trained). Per task and overall:
  - net drift      = sum_q (k1 - k0) / 8   question-equivalents (and in points of the task's accuracy);
  - excess churn   = sum_q ( |k0 - k1| / 8 - E0[|K0 - K1| / 8] ), where E0 is the exact hypergeometric null: if both models
                     were the same, the m = k0 + k1 right answers would be split at random between the two groups of 8;
                     movement beyond what rerunning the same model produces, in both directions together;
  - Pass@8         = share of questions right at least once in 8; 8/8 = share right all 8 times (G-Pass@8);
  - 95% bootstrap CIs over questions.
A NOISE CONTROL compares the original with itself (answers 0-3 against 4-7, k out of 4, same formula): it should be ~0.

python churn_metrics.py --records pack/per_answer --base base --new od5c16 --out runs/churn
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from math import comb
from pathlib import Path

TYPES = ["svamp", "mbpp", "ifeval", "bfcl_simple_python", "bfcl_multiple", "bfcl_irrelevance"]
NAME = {"svamp": "Math", "mbpp": "Code", "ifeval": "Instructions", "bfcl_simple_python": "One function",
        "bfcl_multiple": "Several functions", "bfcl_irrelevance": "No function fits", "all": "All"}


def null_abs(m, n):
    """E0[|K0 - K1|] / n when m rights are split at random into two groups of n (hypergeometric)."""
    tot = comb(2 * n, n)
    e = 0.0
    for k0 in range(max(0, m - n), min(n, m) + 1):
        e += comb(m, k0) * comb(2 * n - m, n - k0) / tot * abs(k0 - (m - k0))
    return e / n


def read(path):
    per = defaultdict(lambda: [0] * 8)
    part = {}
    for r in csv.DictReader(open(path)):
        per[r["id"]][int(r["sample"])] = int(r["correct"])
        part[r["id"]] = r["dataset"]
    return per, part


def stats(pairs, n):
    """pairs: list of (k0, k1) out of n. Returns dict of sums."""
    drift = sum(k1 - k0 for k0, k1 in pairs) / n
    churn = sum(abs(k0 - k1) / n - null_abs(k0 + k1, n) for k0, k1 in pairs)
    raw = sum(abs(k0 - k1) / n for k0, k1 in pairs)
    return drift, churn, raw


def boot(pairs, n, f, reps=1000, seed=0):
    rnd = random.Random(seed)
    vals = []
    for _ in range(reps):
        smp = [pairs[rnd.randrange(len(pairs))] for _ in pairs]
        vals.append(f(smp, n))
    vals.sort()
    return vals[int(0.025 * reps)], vals[int(0.975 * reps) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--base", default="base")
    ap.add_argument("--new", nargs="+", default=["od5c16"])
    ap.add_argument("--out", default="runs/churn")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    b, part = read(Path(args.records) / f"{args.base}.csv")
    res, lines = {}, []
    # noise control: the original against itself, 4 vs 4
    ctrl = defaultdict(list)
    for q, v in b.items():
        pr = (sum(v[:4]), sum(v[4:]))
        ctrl[part[q]].append(pr)
        ctrl["all"].append(pr)
    lines.append("== noise control: the original, answers 0-3 vs 4-7 (k of 4); excess churn should be ~0")
    for t in TYPES + ["all"]:
        d, c, raw = stats(ctrl[t], 4)
        lo, hi = boot(ctrl[t], 4, lambda s, n: stats(s, n)[1])
        res[f"control|{t}"] = {"questions": len(ctrl[t]), "drift_q": d, "excess_churn_q": c, "excess_churn_ci": [lo, hi], "raw_churn_q": raw}
        lines.append(f"   {NAME[t]:18s} n={len(ctrl[t]):4d}  net drift {d:+7.1f} q  excess churn {c:+7.1f} q [{lo:+.1f}, {hi:+.1f}]")
    for name in args.new:
        nw, _ = read(Path(args.records) / f"{name}.csv")
        groups = defaultdict(list)
        p8 = defaultdict(lambda: [0, 0, 0, 0])  # pass@8 base, pass@8 new, 8/8 base, 8/8 new
        for q in b:
            if q not in nw:
                continue
            k0, k1 = sum(b[q]), sum(nw[q])
            for t in (part[q], "all"):
                groups[t].append((k0, k1))
                p8[t][0] += k0 > 0; p8[t][1] += k1 > 0; p8[t][2] += k0 == 8; p8[t][3] += k1 == 8
        lines.append(f"\n== {args.base} -> {name} on the full test (question-equivalents q; points = per 100 questions of the task)")
        for t in TYPES + ["all"]:
            g = groups[t]
            N = len(g)
            d, c, raw = stats(g, 8)
            dlo, dhi = boot(g, 8, lambda s, n: stats(s, n)[0])
            clo, chi = boot(g, 8, lambda s, n: stats(s, n)[1])
            P = p8[t]
            res[f"{name}|{t}"] = {"questions": N, "drift_q": d, "drift_pts": 100 * d / N, "drift_ci_q": [dlo, dhi],
                                  "excess_churn_q": c, "excess_churn_pts": 100 * c / N, "excess_churn_ci_q": [clo, chi],
                                  "raw_churn_q": raw, "pass8": [100 * P[0] / N, 100 * P[1] / N], "all8": [100 * P[2] / N, 100 * P[3] / N]}
            lines.append(f"   {NAME[t]:18s} n={N:4d}  net drift {d:+7.1f} q ({100 * d / N:+5.2f} pts) [{dlo:+.1f}, {dhi:+.1f}]  "
                         f"excess churn {c:+7.1f} q ({100 * c / N:+5.2f} pts) [{clo:+.1f}, {chi:+.1f}]  "
                         f"Pass@8 {100 * P[0] / N:5.1f} -> {100 * P[1] / N:5.1f}  8/8 {100 * P[2] / N:5.1f} -> {100 * P[3] / N:5.1f}")
    (out / "churn.json").write_text(json.dumps(res, indent=1))
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
