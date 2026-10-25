"""Balanced DPO pairs from the model's own graded answers: one clean round from the raw model.

Per question (all its sampled answers, graded with the evaluation's graders; a capped or unclosed answer is wrong):
  chosen   = the shortest right, finished answer;
  "guard"  : chosen over the LONGEST wrong or truncated answer (protects accuracy; wrong answers are often long loops);
  "shorten": chosen over the longest right answer when it is at least --ratio times longer (drives the compression).
Per data type, shorten pairs are capped at the number of guard pairs: shorten pairs alone over-compress and collapse
accuracy, and a type the model already gets right (math) would otherwise get almost only shorten pairs.

    python balanced_pairs.py --rollouts runs/bon/probe-clean1/rollouts.jsonl --out runs/bon/pairs-clean1
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def build(groups, ratio, seed=0):
    """groups: {question id: [rollouts]} -> (pairs, counts per (source, kind)); pure, no I/O."""
    guard, shorten = defaultdict(list), defaultdict(list)
    for qid, rows in groups.items():
        ok = lambda r: r["correct"] and r["closed_thinking"] and not r["truncated"]
        right = sorted((r for r in rows if ok(r)), key=lambda r: r["output_tokens"])
        wrong = sorted((r for r in rows if not ok(r)), key=lambda r: -r["output_tokens"])
        if not right:
            continue
        chosen, src = right[0], rows[0]["source"]
        base = {"id": qid, "source": src, "prompt_ids": chosen["prompt_ids"], "chosen_ids": chosen["output_ids"]}
        if wrong:
            guard[src].append(base | {"kind": "guard", "rejected_ids": wrong[0]["output_ids"]})
        if len(right) > 1 and right[-1]["output_tokens"] >= ratio * chosen["output_tokens"]:
            shorten[src].append(base | {"kind": "shorten", "rejected_ids": right[-1]["output_ids"]})
    rng, pairs, counts = random.Random(seed), [], {}
    for src in sorted(set(guard) | set(shorten)):
        rng.shuffle(shorten[src])
        kept = shorten[src][:len(guard[src])]
        pairs += guard[src] + kept
        counts[(src, "guard")], counts[(src, "shorten")] = len(guard[src]), len(kept)
    rng.shuffle(pairs)
    return pairs, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ratio", type=float, default=1.5)
    parser.add_argument("--drop-sources", default="mbpp_train")
    args = parser.parse_args()
    dropped = set(args.drop_sources.split(",")) if args.drop_sources else set()
    groups = defaultdict(list)
    for line in Path(args.rollouts).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["source"] not in dropped:
                groups[r["id"]].append(r)
    pairs, counts = build(groups, args.ratio)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pairs.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pairs))
    text = "\n".join([f"{len(pairs)} balanced pairs from {len(groups)} questions"] +
                     [f"  {s} / {k}: {n}" for (s, k), n in sorted(counts.items())])
    (out / "summary.txt").write_text(text + "\n")
    print(text, flush=True)


if __name__ == "__main__":
    main()
