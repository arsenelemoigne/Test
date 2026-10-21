"""Write a smaller evaluation selection: the first N questions of every benchmark.

The full evaluation (2,759 questions x 8 samples) takes days on a free T4. This keeps the
first N questions of each benchmark in the order utils/eval/benchmarks.py shuffled them,
the same way its smoke selection does, with their seed_index unchanged, so a question
draws the generations it would draw in a full run. Score the base model and your
checkpoint on the same subset; the numbers are a development signal, not the leaderboard.

    python utils/eval/benchmarks.py                              # writes questions/
    python subset_questions.py --per-dataset 40 --out questions_40
    python evaluate.py --questions questions_40 --samples 2 --out runs/base

With --seed, the N questions per benchmark are drawn at random instead (seeded, so the same set
can be rebuilt), leaving out every question of --exclude: a fresh test set nothing was tuned on.

    python subset_questions.py --per-dataset 40 --seed 20261022 --exclude questions_40 --out questions_fresh
"""

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--questions", default="questions", help="directory from benchmarks.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-dataset", type=int, required=True)
    parser.add_argument("--seed", type=int, help="draw at random with this seed instead of taking the first N")
    parser.add_argument("--exclude", help="a selection directory whose questions are never drawn")
    args = parser.parse_args()

    source, out = Path(args.questions), Path(args.out)
    shutil.copytree(source, out, dirs_exist_ok=True)  # graders' data files come along
    rows = [json.loads(line) for line in (source / "questions.jsonl").read_text().splitlines() if line]
    taken, kept = Counter(), []
    if args.seed is not None:
        excluded = set()
        if args.exclude:
            excluded = {json.loads(l)["id"] for l in (Path(args.exclude) / "questions.jsonl").read_text().splitlines() if l}
        pool = [row for row in rows if row["id"] not in excluded]
        picked = set()
        for dataset in sorted({row["dataset"] for row in pool}):
            ids = [row["id"] for row in pool if row["dataset"] == dataset]
            picked |= set(random.Random(f"{args.seed}-{dataset}").sample(ids, min(args.per_dataset, len(ids))))
        rows = [row for row in pool if row["id"] in picked]  # kept in the evaluation's own order
    for row in rows:
        if taken[row["dataset"]] < args.per_dataset:
            taken[row["dataset"]] += 1
            kept.append(row)
    (out / "questions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in kept))

    manifest = json.loads((out / "manifest.json").read_text())
    manifest |= {"counts": dict(taken), "total": len(kept),
                 "cohort_sha256": hashlib.sha256((out / "questions.jsonl").read_bytes()).hexdigest(),
                 "subset_of": manifest["cohort_sha256"], "per_dataset": args.per_dataset,
                 "draw_seed": args.seed, "excluded": args.exclude}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{len(kept)} questions: {dict(taken)}")


if __name__ == "__main__":
    main()
