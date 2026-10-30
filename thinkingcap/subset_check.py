"""Are our small question sets representative of the full evaluation? (user, 30 Oct: "compare each of our eval samples to
the big final eval to see if we picked a harder or shorter sample")

Input: the full evaluation (2,759 questions x 8) of the original Qwen and of some candidates. Every small set we used
(choosing, questions_40, the refsets, fresh) is a subset of those questions, so its numbers can be recomputed from the
full runs' answers (all 8 samples per question):
  1. the original Qwen on each set vs on the whole evaluation, per part: accuracy and mean output tokens (harder? longer?)
  2. each candidate's headline (tokens saved / accuracy change, BFCL pooled as compare.py pools it) on each set vs on the
     whole evaluation
  3. the spread of a RANDOM draw of 40 questions per part (--draws times): 5th-95th percentile and where each set sits
     in it (its percentile). A set outside 5-95 is an unusual draw.

    python subset_check.py --base runs/mvtf/full-base --cands runs/mvtf/full-lpc-l2s50e-soup-100 \\
        --sets questions_val20261023 questions_40 questions_ref20261024 questions_ref20261025 questions_fresh20261022 \\
        --out runs/mvtf/full-subsets
"""

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

PARTS = ("svamp", "mbpp", "ifeval", "bfcl_simple_python", "bfcl_multiple", "bfcl_irrelevance")
BENCH = {"svamp": "svamp", "mbpp": "mbpp", "ifeval": "ifeval", "bfcl_simple_python": "bfcl", "bfcl_multiple": "bfcl",
         "bfcl_irrelevance": "bfcl"}


def read(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def per_question(run):
    """(dataset, id) -> (mean correct, mean output tokens) over all samples."""
    acc = defaultdict(list)
    for r in read(Path(run) / "records.jsonl"):
        acc[(r["dataset"], r["id"])].append((float(bool(r["correct"])), r["output_tokens"]))
    return {k: (statistics.fmean(c for c, _ in v), statistics.fmean(t for _, t in v)) for k, v in acc.items()}


def headline(base, cand, keys):
    """compare.py's average on the questions `keys`: per benchmark (BFCL parts pooled by question), accuracy change and
    token ratio; then the mean of the changes and 1 - geometric mean of the ratios."""
    groups = defaultdict(list)
    for k in keys:
        if k in base and k in cand:
            groups[BENCH[k[0]]].append(k)
    deltas, ratios = [], []
    for ks in groups.values():
        deltas.append(statistics.fmean(cand[k][0] for k in ks) - statistics.fmean(base[k][0] for k in ks))
        ratios.append(sum(cand[k][1] for k in ks) / sum(base[k][1] for k in ks))
    return 1 - math.exp(sum(map(math.log, ratios)) / len(ratios)), 100 * statistics.fmean(deltas)


def part_stats(run, keys, part):
    ks = [k for k in keys if k[0] == part and k in run]
    return 100 * statistics.fmean(run[k][0] for k in ks), statistics.fmean(run[k][1] for k in ks)


def pct(value, dist):
    return 100 * sum(d <= value for d in dist) / len(dist)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", required=True)
    parser.add_argument("--cands", nargs="*", default=[])
    parser.add_argument("--sets", nargs="+", required=True)
    parser.add_argument("--per-part", type=int, default=40)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    base = per_question(args.base)
    cands = {Path(c).name: per_question(c) for c in args.cands}
    every = sorted(base)
    by_part = defaultdict(list)
    for k in every:
        by_part[k[0]].append(k)
    sets = {Path(s).name: [(q["dataset"], q["id"]) for q in read(Path(s) / "questions.jsonl")] for s in args.sets}
    draws = [[k for p in PARTS for k in rng.sample(by_part[p], min(args.per_part, len(by_part[p])))]
             for _ in range(args.draws)]
    lines = [f"subset check: full evaluation {len(every)} questions; random draws {args.draws} x {args.per_part} per part",
             "", "1. ORIGINAL QWEN per part: accuracy % / mean output tokens (pXX = percentile among random draws)"]
    rand = {p: [part_stats(base, d, p) for d in draws] for p in PARTS}
    lines.append("   full evaluation: " + "; ".join(f"{p} {a:.1f} / {t:.0f}" for p in PARTS
                                                    for a, t in [part_stats(base, every, p)]))
    lines.append("   random draw 5-95%: " + "; ".join(
        f"{p} acc [{sorted(a for a, _ in rand[p])[len(draws) // 20]:.1f}, {sorted(a for a, _ in rand[p])[-len(draws) // 20]:.1f}]"
        f" tok [{sorted(t for _, t in rand[p])[len(draws) // 20]:.0f}, {sorted(t for _, t in rand[p])[-len(draws) // 20]:.0f}]"
        for p in PARTS))
    report = {"base": {}, "cands": {}}
    for name, keys in sets.items():
        cells = []
        for p in PARTS:
            a, t = part_stats(base, keys, p)
            cells.append(f"{p} {a:.1f} (p{pct(a, [x for x, _ in rand[p]]):.0f}) / {t:.0f} (p{pct(t, [x for _, x in rand[p]]):.0f})")
            report["base"].setdefault(name, {})[p] = {"acc": a, "tokens": t}
        lines.append(f"   {name}: " + "; ".join(cells))
    for cname, cand in cands.items():
        full_s, full_a = headline(base, cand, every)
        dist = [headline(base, cand, d) for d in draws]
        ds, da = sorted(s for s, _ in dist), sorted(a for _, a in dist)
        lines += ["", f"2-3. {cname}: FULL evaluation tokens saved {full_s:+.1%}, accuracy {full_a:+.2f} pts",
                  f"   random 240-draw 5-95%: saved [{ds[len(ds) // 20]:+.1%}, {ds[-len(ds) // 20]:+.1%}], "
                  f"accuracy [{da[len(da) // 20]:+.2f}, {da[-len(da) // 20]:+.2f}]  (sd {statistics.pstdev(da):.2f} pts)"]
        report["cands"][cname] = {"full": [full_s, full_a], "draw_sd_acc": statistics.pstdev(da)}
        for name, keys in sets.items():
            s, a = headline(base, cand, keys)
            lines.append(f"   {name}: saved {s:+.1%} (p{pct(s, ds):.0f}), accuracy {a:+.2f} (p{pct(a, da):.0f}); "
                         f"gap to full {100 * (s - full_s):+.1f} pts saved, {a - full_a:+.2f} pts accuracy")
            report["cands"][cname][name] = [s, a]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
