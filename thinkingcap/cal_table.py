"""Confidence vs real outcome per data type and cut point, for one or more cal_cuts.py runs (same cuts).

For each source and cut fraction: mean confidence (teacher-forced answer probability), mean real success (right
re-answers / 20) and the share of cuts right 20/20. Reads cuts.jsonl, verdicts.jsonl, confidence.jsonl.

    python cal_table.py runs/bon/pairs-rc1/cal runs/bon/calcheck-rc1s30 runs/bon/calcheck-rc1rl40 > cal_table.txt
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

FRACTIONS = [0.01, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8]


def read(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def main():
    for folder in map(Path, sys.argv[1:]):
        cuts = read(folder / "cuts.jsonl")
        verdict = {v["key"]: v for v in read(folder / "verdicts.jsonl")}
        conf = {c["key"]: c["confidence"] for c in read(folder / "confidence.jsonl")}
        cell = defaultdict(list)
        for k, c in enumerate(cuts):
            source = c["row"]["id"].rsplit("-", 1)[0]
            v = verdict[k]
            cell[(source, FRACTIONS[k % len(FRACTIONS)])].append((conf[k], v["right"] / max(1, v["of"]), v["right"] == v["of"]))
        print(f"== {folder}: conf = mean confidence, real = mean share right of 20 re-answers, 20/20 = share of cuts all right")
        print(f"{'source':<16}" + "".join(f"{f:>16.0%}" for f in FRACTIONS))
        for source in sorted({s for s, _ in cell}):
            row = []
            for f in FRACTIONS:
                xs = cell[(source, f)]
                c = sum(x[0] for x in xs) / len(xs)
                r = sum(x[1] for x in xs) / len(xs)
                a = sum(x[2] for x in xs) / len(xs)
                row.append(f"{c:.2f}/{r:.2f}/{a:.0%}")
            print(f"{source:<16}" + "".join(f"{x:>16}" for x in row))
        print()


if __name__ == "__main__":
    main()
