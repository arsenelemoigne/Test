"""Write a smaller evaluation selection: the first N questions of every benchmark.

The full evaluation (2,759 questions x 8 samples) takes days on a free T4. This keeps the
first N questions of each benchmark in the order utils/eval/benchmarks.py shuffled them,
the same way its smoke selection does, with their seed_index unchanged, so a question
draws the generations it would draw in a full run. Score the base model and your
checkpoint on the same subset; the numbers are a development signal, not the leaderboard.

    python utils/eval/benchmarks.py                              # writes questions/
    python subset_questions.py --per-dataset 40 --out questions_40
    python evaluate.py --questions questions_40 --samples 2 --out runs/base
"""

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--questions", default="questions", help="directory from benchmarks.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-dataset", type=int, required=True)
    parser.add_argument("--skip", type=int, default=0,
                        help="leave out the first SKIP questions of every benchmark: a fresh held-out subset")
    parser.add_argument("--datasets", default="", help="comma-separated datasets to keep (default: all), e.g. mbpp")
    args = parser.parse_args()

    source, out = Path(args.questions), Path(args.out)
    shutil.copytree(source, out, dirs_exist_ok=True)  # graders' data files come along
    rows = [json.loads(line) for line in (source / "questions.jsonl").read_text().splitlines() if line]
    seen, taken, kept = Counter(), Counter(), []
    only = set(filter(None, args.datasets.split(",")))
    rows = [row for row in rows if not only or row["dataset"] in only]
    for row in rows:
        seen[row["dataset"]] += 1
        if seen[row["dataset"]] > args.skip and taken[row["dataset"]] < args.per_dataset:
            taken[row["dataset"]] += 1
            kept.append(row)
    (out / "questions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in kept))

    manifest = json.loads((out / "manifest.json").read_text())
    manifest |= {"counts": dict(taken), "total": len(kept),
                 "cohort_sha256": hashlib.sha256((out / "questions.jsonl").read_bytes()).hexdigest(),
                 "subset_of": manifest["cohort_sha256"], "per_dataset": args.per_dataset, "skip": args.skip}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{len(kept)} questions: {dict(taken)}")


if __name__ == "__main__":
    main()
