"""Per-type thinking / accuracy stats and length-band cuts of trained models on the Screen 720, against the original.

Each records file is a 720 x 4 evaluation (runs/mvtf/s4-*/records.jsonl). Bands: each part's questions split into five
fifths by the ORIGINAL model's mean thinking on them; per band, thinking removed (%) and accuracy change (points).

    python s4_metrics.py --base s4/base.jsonl --models "bidax6=s4/bidax6-300.jsonl" ... --out s4/s4_metrics.json
"""
import argparse
import json
import statistics
from collections import defaultdict

PARTS = ["svamp", "mbpp", "ifeval", "bfcl_simple_python", "bfcl_multiple", "bfcl_irrelevance"]


def load(path):
    q = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        yes = lambda v: v is True or v == "True"
        q[(r["dataset"], r["id"])].append((1 if yes(r["correct"]) else 0, int(r.get("reasoning_tokens") or 0),
                                           int(r["output_tokens"]), yes(r["truncated"])))
    return q


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", required=True)
    parser.add_argument("--models", nargs="+", required=True, help="name=records.jsonl")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    base = load(args.base)
    models = {m.split("=", 1)[0]: load(m.split("=", 1)[1]) for m in args.models}
    stats, bands = {}, {}
    for name, m in [("original", base)] + list(models.items()):
        stats[name] = {}
        for p in PARTS:
            ans = [a for k in base if k[0] == p for a in m[k]]
            n, right, think = len(ans), sum(a[0] for a in ans), sum(a[1] for a in ans)
            stats[name][p] = {"acc": 100 * right / n, "think": think / n, "per_right": think / max(1, right),
                              "wrong_tok_share": 100 * sum(a[1] for a in ans if not a[0]) / max(1, think),
                              "wrong_ans_share": 100 * (n - right) / n, "capped": 100 * sum(a[3] for a in ans) / n}
    for p in PARTS:
        ks = sorted([k for k in base if k[0] == p], key=lambda k: statistics.fmean(a[1] for a in base[k]))
        n = len(ks)
        for name, m in models.items():
            row = []
            for b in range(5):
                kk = ks[b * n // 5:(b + 1) * n // 5]
                bt = sum(a[1] for k in kk for a in base[k])
                mt = sum(a[1] for k in kk for a in m[k])
                ba = sum(a[0] for k in kk for a in base[k])
                ma = sum(a[0] for k in kk for a in m[k])
                na = sum(len(base[k]) for k in kk)
                row.append([round(100 * (1 - mt / max(1, bt))), round(100 * (ma - ba) / na, 1),
                            round(statistics.fmean(a[1] for a in base[kk[0]])),
                            round(statistics.fmean(a[1] for a in base[kk[-1]])), len(kk)])
            bands.setdefault(name, {})[p] = row
    json.dump({"stats": stats, "bands": bands}, open(args.out, "w"))
    for name in models:
        print(name, " | ".join(f"{p[:8]} {stats[name][p]['acc'] - stats['original'][p]['acc']:+.1f}" for p in PARTS))


if __name__ == "__main__":
    main()
