"""Did training shorten the answers it never taught? Thinking length on the fresh set x8, right vs wrong.

For every scored model (runs/mvtf/f8-*/records.jsonl), per benchmark: mean thinking tokens of its right answers, of its
wrong answers, and of the answers to questions the ORIGINAL model never solves (0 of 8 in f8-base), each vs f8-base's own.
No model ever saw a pair from an unsolved question, so any shortening there is spill-over.

    python wrong_lengths.py --root runs/mvtf
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def records(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default="runs/mvtf")
    parser.add_argument("--out", default="runs/mvtf/wrong_lengths.txt")
    args = parser.parse_args()
    root = Path(args.root)
    base = records(root / "f8-base" / "records.jsonl")
    solved = defaultdict(int)
    for r in base:
        solved[(r["dataset"], r["id"])] += bool(r["correct"])
    never = {k for k, n in solved.items() if n == 0}

    def groups(rows):
        out = defaultdict(list)  # (dataset, group) -> thinking tokens
        for r in rows:
            d = r["dataset"].replace("bfcl_", "bfcl ")
            for g in ("all", "right" if r["correct"] else "wrong") + (("never solved",) if (r["dataset"], r["id"]) in never else ()):
                out[(d, g)].append(r["reasoning_tokens"])
                out[("ALL", g)].append(r["reasoning_tokens"])
        return out

    ref = groups(base)
    datasets = sorted({d for d, _ in ref if d != "ALL"}) + ["ALL"]
    share = {d: len(ref[(d, "never solved")]) / max(1, len(ref[(d, "all")])) for d in datasets}
    lines = ["Thinking tokens on the fresh set x8: base mean -> model mean (change). 'never solved' = questions f8-base gets",
             "0/8 (" + ", ".join(f"{d} {share[d]:.0%}" for d in datasets) + " of answers); no training pair ever came from them.",
             f"f8-base: " + "; ".join(f"{d} right {sum(ref[(d, 'right')]) / max(1, len(ref[(d, 'right')])):.0f} / wrong "
                                      f"{sum(ref[(d, 'wrong')]) / max(1, len(ref[(d, 'wrong')])):.0f} / never "
                                      f"{sum(ref[(d, 'never solved')]) / max(1, len(ref[(d, 'never solved')])):.0f}" for d in datasets)]
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    for run in sorted(root.glob("f8-*/records.jsonl")):
        name = run.parent.name[3:]
        if name == "base":
            continue
        g = groups(records(run))
        cells = []
        for d in datasets:
            parts = []
            for k in ("right", "wrong", "never solved"):
                a, b = mean(ref[(d, k)]), mean(g[(d, k)])
                parts.append(f"{k} {1 - b / a:+.0%}" if a == a and b == b and a else f"{k} -")
            cells.append(f"{d}: " + ", ".join(parts))
        lines.append(f"{name}: shorter by  " + " | ".join(cells))
    text = "\n".join(lines)
    Path(args.out).write_text(text + "\n")
    print(text, flush=True)


if __name__ == "__main__":
    main()
