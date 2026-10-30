"""A screen set matched to the FULL evaluation (user, 30 Oct: "a better screen set"): --per-part questions per part
(default 120 = 720 in all), drawn so the original Qwen's behaviour on them matches its behaviour on all 2,759 questions.

From the original Qwen's full run (8 answers per question), each question gets a solve rate and a mean output length.
Inside each part the questions fall in cells (solve rate: never / below half / half or more / always, x output-length
quartiles of that part); every cell gets the share of the part it holds in the full evaluation (largest remainder), and
its questions are drawn at random. Questions of --exclude (the fresh set, the cross-check set) are never drawn, so those
stay independent. The selection is checked against the full evaluation, per part: accuracy and mean output tokens.

    python screen_set.py --full-base runs/mvtf/full-base --questions questions \\
        --exclude questions_fresh20261022 questions_ref20261024 --out questions_screen720
"""

import argparse
import hashlib
import json
import random
import shutil
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def read(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def solve_bin(s):
    return 0 if s == 0 else 3 if s == 1 else 1 if s < 0.5 else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--full-base", required=True)
    parser.add_argument("--questions", default="questions")
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--per-part", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    answers = defaultdict(list)
    for r in read(Path(args.full_base) / "records.jsonl"):
        answers[r["id"]].append((float(bool(r["correct"])), r["output_tokens"]))
    stats = {q: (statistics.fmean(c for c, _ in v), statistics.fmean(t for _, t in v)) for q, v in answers.items()}
    excluded = {q["id"] for e in args.exclude for q in read(Path(e) / "questions.jsonl")}
    rows = read(Path(args.questions) / "questions.jsonl")
    by_part = defaultdict(list)
    for row in rows:
        if row["id"] in stats:
            by_part[row["dataset"]].append(row["id"])
    picked, lines = set(), [f"screen set: {args.per_part} per part, matched to the full evaluation's original-Qwen cells"]
    for part, ids in sorted(by_part.items()):
        lengths = sorted(stats[q][1] for q in ids)
        edges = [lengths[k * len(lengths) // 4] for k in (1, 2, 3)]
        cell = {q: (solve_bin(stats[q][0]), sum(stats[q][1] >= e for e in edges)) for q in ids}
        share = Counter(cell.values())
        want = {c: args.per_part * n / len(ids) for c, n in share.items()}
        take = {c: int(w) for c, w in want.items()}
        for c in sorted(want, key=lambda c: want[c] - take[c], reverse=True)[:args.per_part - sum(take.values())]:
            take[c] += 1
        chosen = []
        for c, n in take.items():
            free = [q for q in ids if cell[q] == c and q not in excluded]
            if len(free) < n:
                print(f"  {part}: cell {c} has {len(free)} free questions for {n}", flush=True)
            chosen += rng.sample(free, min(n, len(free)))
        picked |= set(chosen)
        full_acc, full_tok = (100 * statistics.fmean(stats[q][0] for q in ids), statistics.fmean(stats[q][1] for q in ids))
        acc, tok = 100 * statistics.fmean(stats[q][0] for q in chosen), statistics.fmean(stats[q][1] for q in chosen)
        lines.append(f"  {part}: {len(chosen)} questions; accuracy {acc:.1f} vs full {full_acc:.1f}; "
                     f"mean output tokens {tok:.0f} vs full {full_tok:.0f} ({tok / full_tok - 1:+.1%})")
    source, out = Path(args.questions), Path(args.out)
    shutil.copytree(source, out, dirs_exist_ok=True)  # graders' data files come along
    kept = [row for row in rows if row["id"] in picked]  # the evaluation's own order, seed_index unchanged
    (out / "questions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in kept))
    manifest = json.loads((out / "manifest.json").read_text())
    manifest |= {"counts": dict(Counter(r["dataset"] for r in kept)), "total": len(kept),
                 "cohort_sha256": hashlib.sha256((out / "questions.jsonl").read_bytes()).hexdigest(),
                 "subset_of": manifest["cohort_sha256"], "per_dataset": args.per_part, "draw_seed": args.seed,
                 "excluded": args.exclude, "matched_to": str(args.full_base)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "screen_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
