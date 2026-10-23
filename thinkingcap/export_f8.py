"""Compact export of every fresh-set x8 run, for the "where we shaved" visual.

For each runs/mvtf/f8-*/records.jsonl: per question (the 240 fresh questions), the mean generated tokens of its 8
answers and how many of them were right. One small JSON: {"q": [[id, part], ...], "m": {run: [[tokens, right], ...]}}
with questions in one fixed order. Prints the record fields first, so a changed format shows at once.

    python export_f8.py --out runs/viz/f8_summary.txt
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def pick(row, *names):
    for n in names:
        if n in row:
            return row[n]
    raise KeyError(f"none of {names} in {sorted(row)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", default="runs/mvtf")
    parser.add_argument("--out", default="runs/viz/f8_summary.txt")
    args = parser.parse_args()
    runs = sorted(p for p in Path(args.runs).glob("f8-*") if (p / "records.jsonl").exists())
    first = json.loads((runs[0] / "records.jsonl").open().readline())
    print("record fields:", {k: (v if k != "text" else "...") for k, v in first.items()}, flush=True)
    order, parts, models = None, {}, {}
    for run in runs:
        per = defaultdict(lambda: [0, 0, 0])
        for line in (run / "records.jsonl").open():
            if not line.strip():
                continue
            r = json.loads(line)
            qid = str(pick(r, "id", "question_id", "qid"))
            parts[qid] = pick(r, "dataset", "benchmark", "subset")
            tokens = pick(r, "output_tokens", "tokens", "num_tokens", "completion_tokens", "generated_tokens")
            right = pick(r, "correct", "is_correct", "passed", "score")
            p = per[qid]
            p[0] += tokens
            p[1] += bool(right)
            p[2] += 1
        if order is None:
            order = sorted(per, key=lambda q: (parts[q], q))
        if set(per) != set(order):
            print(f"skip {run.name}: other questions ({len(per)})", flush=True)
            continue
        models[run.name.removeprefix("f8-")] = [[round(per[q][0] / per[q][2]), per[q][1], per[q][2]] for q in order]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"q": [[q, parts[q]] for q in order], "m": models}, separators=(",", ":")))
    print(f"{len(models)} runs x {len(order)} questions -> {out} ({out.stat().st_size // 1024} KB): "
          + ", ".join(models), flush=True)


if __name__ == "__main__":
    main()
