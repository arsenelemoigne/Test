"""Strict "intelligence density" measure (17 Nov, user: half credit for 8/8 -> 4/8 is not thinking better).

Two separate questions, never traded against each other:

A. Did the model lose anything for real?  (verdict per type)
   Per question, the original and the candidate each answered 8 times: k0, k1 right. Under the null "same model"
   the 16 answers are exchangeable, so given m = k0 + k1 the original's count is hypergeometric(16, m, 8). The
   chance-level loss of that question is E_null[max(0, k0 - k1)] / 8 (exact, no assumption that 8/8 means p = 1).
   Excess loss = observed loss - chance loss, summed over questions (unbiased in aggregate; not clipped).
   Reported in accuracy points per type, with a paired bootstrap 95% CI over questions. A type PASSES if the
   upper bound of its excess loss is <= --margin points (an equivalence test: "no loss we can detect beyond 0.5 pt").

B. What does it cost to match the original's accuracy?  (the saving)
   A question the candidate now fails more often than the original must be re-asked to the original on that
   share of attempts: cost(q) = T1(q) + L(q) * T0(q), L(q) = the chance-corrected loss rate of question q.
   Saving_b = 1 - sum cost / sum T0 per benchmark; overall = the official aggregation (4 benchmarks, geometric
   mean of token ratios, BFCL pooled 400/200/240). A question that drops 8/8 -> 4/8 saves nothing unless its new
   answer is under about half the original's length; one that drops to 0/8 always costs more than the original.

Input: per-answer CSVs (id,dataset,sample,correct,output_tokens,...), one per model, same 2,759 x 8 questions.
"""
import argparse, csv, json, math
from collections import defaultdict
from pathlib import Path

import numpy as np

BENCH = {"svamp": "svamp", "mbpp": "mbpp", "ifeval": "ifeval",
         "bfcl_simple_python": "bfcl", "bfcl_multiple": "bfcl", "bfcl_irrelevance": "bfcl"}
TYPES = ["svamp", "mbpp", "ifeval", "bfcl_simple_python", "bfcl_multiple", "bfcl_irrelevance"]
NAMES = {"svamp": "math", "mbpp": "code", "ifeval": "instructions", "bfcl_simple_python": "one function",
         "bfcl_multiple": "several functions", "bfcl_irrelevance": "no function fits"}


def load(path):
    k, t, n, ds = defaultdict(int), defaultdict(float), defaultdict(int), {}
    with open(path) as f:
        for r in csv.DictReader(f):
            q = r["id"]
            k[q] += int(r["correct"])
            t[q] += float(r["output_tokens"])
            n[q] += 1
            ds[q] = r["dataset"]
    return {q: (k[q], t[q] / n[q], n[q], ds[q]) for q in k}


def chance_loss_table(K=8):
    """E_null[max(0, k0 - k1)] / K given m = k0 + k1 right among 2K exchangeable answers."""
    out = np.zeros(2 * K + 1)
    for m in range(2 * K + 1):
        tot, e = 0.0, 0.0
        for k0 in range(max(0, m - K), min(K, m) + 1):
            w = math.comb(K, k0) * math.comb(K, m - k0)
            tot += w
            e += w * max(0, k0 - (m - k0)) / K
        out[m] = e / tot
    return out


def score(base, cand, B=1000, margin=0.5, seed=0):
    tab = chance_loss_table()
    rng = np.random.default_rng(seed)
    per = {}
    for ty in TYPES:
        qs = [q for q in base if base[q][3] == ty and q in cand]
        k0 = np.array([base[q][0] for q in qs]); k1 = np.array([cand[q][0] for q in qs])
        t0 = np.array([base[q][1] for q in qs]); t1 = np.array([cand[q][1] for q in qs])
        obs = np.maximum(0, k0 - k1) / 8
        gain = np.maximum(0, k1 - k0) / 8
        ch = tab[k0 + k1]
        L = obs - ch                                  # chance-corrected loss rate per question (unbiased, unclipped)
        G = gain - ch                                 # same for gains (null is symmetric)
        cost = t1 + np.clip(L, 0, None) * t0          # clipped only for the cost (no negative fallback)
        idx = rng.integers(0, len(qs), size=(B, len(qs)))
        exl_bs = 100 * L[idx].mean(1)
        per[ty] = dict(n=len(qs), acc0=100 * k0.mean() / 8, acc1=100 * k1.mean() / 8,
                       excess_loss=100 * L.mean(), excess_loss_hi=float(np.percentile(exl_bs, 97.5)),
                       excess_loss_lo=float(np.percentile(exl_bs, 2.5)), excess_gain=100 * G.mean(),
                       q_lost=float(L.sum()), q_gained=float(G.sum()),
                       raw_saved=100 * (1 - t1.sum() / t0.sum()), strict_saved=100 * (1 - cost.sum() / t0.sum()),
                       _t0=t0, _t1=t1, _cost=cost, _L=L, _idx=idx)
        per[ty]["passes"] = per[ty]["excess_loss_hi"] <= margin
    # official aggregation: 4 benchmarks, BFCL pooled over its 840 questions; geometric mean of token ratios
    bench = defaultdict(lambda: [0.0, 0.0, 0.0])
    for ty in TYPES:
        b = bench[BENCH[ty]]
        b[0] += per[ty]["_t0"].sum(); b[1] += per[ty]["_t1"].sum(); b[2] += per[ty]["_cost"].sum()
    raw = 100 * (1 - math.exp(sum(math.log(v[1] / v[0]) for v in bench.values()) / 4))
    strict = 100 * (1 - math.exp(sum(math.log(v[2] / v[0]) for v in bench.values()) / 4))
    for ty in TYPES:
        for k in ("_t0", "_t1", "_cost", "_L", "_idx"):
            per[ty].pop(k)
    return dict(raw_saved=raw, strict_saved=strict, passes_all=all(per[t]["passes"] for t in TYPES),
                types_failed=[NAMES[t] for t in TYPES if not per[t]["passes"]],
                q_lost=sum(per[t]["q_lost"] for t in TYPES), per=per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="reviews/consult29_density/pack/per_answer")
    ap.add_argument("--base", default="base")
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--out", default="density_strict.json")
    a = ap.parse_args()
    base = load(Path(a.dir) / f"{a.base}.csv")
    res = {}
    for f in sorted(Path(a.dir).glob("*.csv")):
        if f.stem == a.base:
            continue
        res[f.stem] = score(base, load(f), margin=a.margin)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"{'model':10s} {'raw%':>6s} {'strict%':>8s} {'q lost':>7s}  types failing (excess loss > {a.margin} pt, 95% upper bound)")
    for m, r in sorted(res.items(), key=lambda kv: -kv[1]["strict_saved"]):
        print(f"{m:10s} {r['raw_saved']:6.1f} {r['strict_saved']:8.1f} {r['q_lost']:7.0f}  {', '.join(r['types_failed']) or 'none: PASSES'}")
    print("\nper type: excess loss pts [95% CI] / strict saving %")
    for m, r in sorted(res.items(), key=lambda kv: -kv[1]["strict_saved"]):
        print(m, " | ".join(f"{NAMES[t]} {p['excess_loss']:+.1f} [{p['excess_loss_lo']:+.1f},{p['excess_loss_hi']:+.1f}] / {p['strict_saved']:.0f}"
                            for t, p in r["per"].items()))


if __name__ == "__main__":
    main()
