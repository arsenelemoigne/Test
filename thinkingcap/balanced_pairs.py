"""Balanced DPO pairs from the model's own graded answers: one clean round from the raw model.

Per question (all its sampled answers, graded with the evaluation's graders; a capped or unclosed answer is wrong):
  chosen   = the shortest right, finished answer;
  "guard"  : chosen over the LONGEST wrong or truncated answer (protects accuracy; wrong answers are often long loops);
  "shorten": chosen over the longest right answer when it is at least --ratio times longer (drives the compression).
Per data type, shorten pairs are capped at the number of guard pairs: shorten pairs alone over-compress and collapse
accuracy, and a type the model already gets right (math) would otherwise get almost only shorten pairs.
--protect (sources, or source/category such as toolace/multiple): those questions get no length signal at all: no
shorten and no guard pair, only "right": a right answer beats a wrong one of similar length (within --right-band), so
they learn to be right without being pushed shorter (math, where thinking is dense; tool calls choosing among functions).

    python balanced_pairs.py --rollouts runs/bon/probe-clean1/rollouts.jsonl --out runs/bon/pairs-clean1
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def build(groups, ratio, seed=0, protect=(), band=0.25):
    """groups: {question id: [rollouts]} -> (pairs, counts per (source, kind)); pure, no I/O."""
    guard, shorten, right_pairs = defaultdict(list), defaultdict(list), defaultdict(list)
    for qid, rows in groups.items():
        ok = lambda r: r["correct"] and r["closed_thinking"] and not r["truncated"]
        right = sorted((r for r in rows if ok(r)), key=lambda r: r["output_tokens"])
        wrong = sorted((r for r in rows if not ok(r)), key=lambda r: -r["output_tokens"])
        if not right:
            continue
        chosen, src = right[0], rows[0]["source"]
        if src in protect or f"{src}/{rows[0].get('category')}" in protect:
            close = [(abs(r["output_tokens"] - w["output_tokens"]), r, w) for r in right for w in wrong
                     if abs(r["output_tokens"] - w["output_tokens"]) <= band * max(r["output_tokens"], w["output_tokens"])]
            if close:
                _, r, w = min(close, key=lambda t: t[0])
                right_pairs[src].append({"id": qid, "source": src, "kind": "right", "prompt_ids": r["prompt_ids"],
                                         "chosen_ids": r["output_ids"], "rejected_ids": w["output_ids"]})
            continue
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
    for src in sorted(right_pairs):
        pairs += right_pairs[src]
        counts[(src, "right")] = len(right_pairs[src])
    rng.shuffle(pairs)
    return pairs, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ratio", type=float, default=1.5)
    parser.add_argument("--drop-sources", default="mbpp_train")
    parser.add_argument("--protect", default="", help="sources or source/category given only similar-length right>wrong pairs")
    parser.add_argument("--right-band", type=float, default=0.25)
    args = parser.parse_args()
    dropped = set(args.drop_sources.split(",")) if args.drop_sources else set()
    groups = defaultdict(list)
    for line in Path(args.rollouts).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["source"] not in dropped:
                groups[r["id"]].append(r)
    protect = set(args.protect.split(",")) if args.protect else set()
    pairs, counts = build(groups, args.ratio, protect=protect, band=args.right_band)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pairs.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pairs))
    text = "\n".join([f"{len(pairs)} balanced pairs from {len(groups)} questions"] +
                     [f"  {s} / {k}: {n}" for (s, k), n in sorted(counts.items())])
    (out / "summary.txt").write_text(text + "\n")
    print(text, flush=True)


if __name__ == "__main__":
    main()
