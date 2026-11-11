"""Transfer check (11 Nov): a model against the original Qwen on benchmarks never trained on, per set: accuracy (right = graded
correct and not cut at the token cap), mean tokens, tokens saved, paired bootstrap 95% interval of the accuracy change.

    python transfer_score.py --base runs/bon/transfer-base/rollouts.jsonl --cand runs/bon/transfer-rpL60/rollouts.jsonl --name rpL60
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load(path):
    out = defaultdict(lambda: defaultdict(list))
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["source"]][r["id"]].append((bool(r["correct"]) and not r["truncated"], r["output_tokens"],
                                              bool(r["truncated"]) or not r.get("closed_thinking", True)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    base, cand = load(args.base), load(args.cand)
    rng = random.Random(0)
    res, lines = {}, [f"== transfer {args.name} (x4, vs the original):"]
    for src in sorted(base):
        ids = [q for q in base[src] if q in cand[src]]
        m = lambda d, q, i: sum(x[i] for x in d[src][q]) / len(d[src][q])  # noqa: E731
        a0 = sum(m(base, q, 0) for q in ids) / len(ids); a1 = sum(m(cand, q, 0) for q in ids) / len(ids)
        t0 = sum(m(base, q, 1) for q in ids) / len(ids); t1 = sum(m(cand, q, 1) for q in ids) / len(ids)
        c0 = sum(m(base, q, 2) for q in ids) / len(ids); c1 = sum(m(cand, q, 2) for q in ids) / len(ids)
        d = [m(cand, q, 0) - m(base, q, 0) for q in ids]
        boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(1000))
        res[src] = {"n": len(ids), "base_acc": 100 * a0, "acc": 100 * a1, "d_acc": 100 * (a1 - a0),
                    "d_acc_ci": [100 * boots[25], 100 * boots[974]], "base_tokens": t0, "tokens": t1, "saved": 100 * (1 - t1 / t0),
                    "base_runaway": 100 * c0, "runaway": 100 * c1}
        r = res[src]
        lines.append(f"   {src:13s} {r['n']:3d} q | saved {r['saved']:+5.1f}% ({t0:.0f} -> {t1:.0f} tokens) | accuracy {r['base_acc']:.1f} -> "
                     f"{r['acc']:.1f} ({r['d_acc']:+.1f} [{r['d_acc_ci'][0]:+.1f}, {r['d_acc_ci'][1]:+.1f}]) | runaway {r['base_runaway']:.1f} -> {r['runaway']:.1f}%")
    print("\n".join(lines), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps({"name": args.name, "sets": res}, indent=1))


if __name__ == "__main__":
    main()
