"""How far can the token count fall on each benchmark? Ceilings computed from saved evaluations.

Reads `records.jsonl` written by evaluate.py (one line per reply: dataset, id, output_tokens,
reasoning_tokens, correct) and reports, per benchmark group (BFCL subsets pooled, as compare.py
does) and averaged the way compare.py averages (geometric mean of token ratios):

  split       mean reply length, thinking tokens, answer tokens, share of thinking
  no-think    keep only the answer tokens: the upper bound (not allowed: the model must think)
  think-K     answer + at most K thinking tokens (a minimal think)
  best-seen   per question, the shortest *correct* reply seen in any of the given runs
              (questions no run solved keep the base length): what some model already does

    python ceiling.py runs/eval_base runs/eval_ablA runs/eval_blendCA50 runs/eval_P1 --think 32,64
"""

import argparse
import json
import math
from pathlib import Path

GROUPS = {"bfcl_irrelevance": "bfcl", "bfcl_multiple": "bfcl", "bfcl_simple_python": "bfcl"}


def load(run):
    rows = [json.loads(line) for line in (Path(run) / "records.jsonl").read_text().splitlines() if line.strip()]
    for row in rows:
        row["group"] = GROUPS.get(row["dataset"], row["dataset"])
    return rows


def ratio_summary(base_rows, new_tokens):
    """{group: tokens saved}, plus the geometric-mean average over groups."""
    groups = sorted({r["group"] for r in base_rows})
    saved = {}
    for group in groups:
        rows = [r for r in base_rows if r["group"] == group]
        base = sum(r["output_tokens"] for r in rows)
        new = sum(new_tokens(r) for r in rows)
        saved[group] = 1 - new / base
    saved["average"] = 1 - math.exp(sum(math.log(1 - saved[g]) for g in groups) / len(groups))
    return saved


def best_seen(runs):
    """{(dataset, id): shortest correct reply length across all runs}, None if never solved."""
    best = {}
    for rows in runs:
        for r in rows:
            key = (r["dataset"], r["id"])
            if r.get("correct"):
                best[key] = min(best.get(key) or r["output_tokens"], r["output_tokens"])
            else:
                best.setdefault(key, None)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base", help="evaluation folder of the untouched model")
    parser.add_argument("others", nargs="*", help="evaluation folders of other models, for best-seen")
    parser.add_argument("--think", default="32,64", help="minimal think budgets to report")
    args = parser.parse_args()
    base = load(args.base)
    runs = [base] + [load(o) for o in args.others]

    print("split (base)")
    for group in sorted({r["group"] for r in base}):
        rows = [r for r in base if r["group"] == group]
        out = sum(r["output_tokens"] for r in rows) / len(rows)
        think = sum(r["reasoning_tokens"] for r in rows) / len(rows)
        acc = sum(bool(r.get("correct")) for r in rows) / len(rows)
        print(f"  {group:10s} reply {out:7.0f}  thinking {think:7.0f}  answer {out - think:6.0f}  "
              f"thinking share {think / out:5.1%}  accuracy {acc:5.1%}")

    def show(name, saved):
        parts = "  ".join(f"{g} {v:+.1%}" for g, v in saved.items() if g != "average")
        print(f"  {name:12s} average {saved['average']:+.1%}   ({parts})")

    print("ceilings (tokens saved vs base)")
    show("no-think", ratio_summary(base, lambda r: r["output_tokens"] - r["reasoning_tokens"]))
    for k in [int(x) for x in args.think.split(",")]:
        show(f"think-{k}", ratio_summary(
            base, lambda r, k=k: r["output_tokens"] - r["reasoning_tokens"] + min(r["reasoning_tokens"], k)))
    if args.others:
        best = best_seen(runs)
        show("best-seen", ratio_summary(base, lambda r: best[(r["dataset"], r["id"])] or r["output_tokens"]))
        solved = sum(v is not None for v in best.values()) / len(best)
        print(f"  best-seen: {solved:.1%} of questions solved by at least one run "
              f"({len(args.others) + 1} runs); unsolved ones keep the base length")


if __name__ == "__main__":
    main()
