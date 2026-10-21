"""Chain-of-Draft code rows: terse but complete code thinking, taught without the instruction.

The current model answers each code task twice as it normally does and 8 times with a Chain-of-Draft
instruction added (every step and every check, at most a few words each). Per task, the passing
replies (draft and normal pooled, as Munkhbat et al. 2025 do) that still think (--min-thinking tokens),
show a real check (a trace or test of the code) and do not copy the prompt's exemplars are ranked by
length; the one at --quantile (not the very shortest: over-compression hurts small models, TokenSkip)
becomes an SFT row under the ORIGINAL prompt, so the brevity ends up in the model, not in the prompt.

Code-only change: replay rows (the same model's own correct replies to math, tool and instruction
questions, unchanged) are mixed in, one per code row, so training holds the other tasks in place.

    python cod_select.py --cod runs/cod/probe-cod/rollouts.jsonl --normal runs/cod/probe-normal/rollouts.jsonl \\
        --replay runs/cod/probe-replay/rollouts.jsonl --out runs/cod
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from probe_curve import read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cod", required=True, help="rollouts under the Chain-of-Draft instruction")
    parser.add_argument("--normal", required=True, help="rollouts of the same tasks under their original prompt")
    parser.add_argument("--replay", required=True, help="the same model's rollouts on the other kinds of question")
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-thinking", type=int, default=20, help="a kept reply must still think")
    parser.add_argument("--replay-ratio", type=float, default=1.0, help="replay rows per code row")
    parser.add_argument("--quantile", type=float, default=0.25,
                        help="rank of the kept reply among a task's passing replies, 0 = the shortest")
    parser.add_argument("--exemplar-names", default="count_vowels,second_largest,chunk",
                        help="functions of the prompt's exemplars: a reply that mentions one copied them")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    leak = re.compile(r"\b(" + "|".join(map(re.escape, args.exemplar_names.split(","))) + r")\b")

    cod, normal = group(read_jsonl(Path(args.cod))), group(read_jsonl(Path(args.normal)))
    rows, lines = [], []
    by_source = defaultdict(lambda: defaultdict(list))
    for qid, replies in cod.items():
        if qid not in normal:
            continue
        original = normal[qid][0]
        passed = [r for r in replies + normal[qid] if r["correct"] and r["closed_thinking"]
                  and len(r["thinking_ids"]) >= args.min_thinking and CHECK.search(r["thinking"])
                  and not leak.search(r["thinking"])]
        stats = by_source[original["source"]]
        stats["cod_correct"] += [r["correct"] for r in replies]
        stats["normal_correct"] += [r["correct"] for r in normal[qid]]
        stats["cod_tokens"] += [r["output_tokens"] for r in replies if r["correct"]]
        stats["normal_tokens"] += [r["output_tokens"] for r in normal[qid] if r["correct"]]
        if not passed:
            continue
        ranked = sorted(passed, key=lambda r: r["output_tokens"])
        best = ranked[int(args.quantile * (len(ranked) - 1))]
        stats["kept_draft"].append(best in replies)
        stats["kept_tokens"].append(best["output_tokens"])
        rows.append({"id": qid, "source": original["source"], "kind": "cod",
                     "prompt_ids": original["prompt_ids"], "completion_ids": best["output_ids"],
                     "thinking_tokens": len(best["thinking_ids"])})

    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    lines.append("Chain-of-Draft code rows (correct share / mean tokens of correct replies)")
    for source, s in sorted(by_source.items()):
        lines.append(f"  {source:<12} normal {mean(s['normal_correct']):.0%} / {mean(s['normal_tokens']):.0f}   "
                     f"CoD {mean(s['cod_correct']):.0%} / {mean(s['cod_tokens']):.0f}   "
                     f"kept {len(s['kept_tokens'])} rows, mean {mean(s['kept_tokens']):.0f} tokens, "
                     f"{mean(s['kept_draft']):.0%} of them drafts")

    replay = [r for r in read_jsonl(Path(args.replay)) if r["correct"] and r["closed_thinking"]]
    first = {}
    for r in replay:  # one reply per question
        first.setdefault(r["id"], r)
    pool = list(first.values())
    random.Random(args.seed).shuffle(pool)
    keep = pool[:round(args.replay_ratio * len(rows))]
    rows += [{"id": r["id"], "source": r["source"], "kind": "replay", "prompt_ids": r["prompt_ids"],
              "completion_ids": r["output_ids"], "thinking_tokens": len(r["thinking_ids"])} for r in keep]
    counts = defaultdict(int)
    for r in keep:
        counts[r["source"]] += 1
    lines.append(f"replay rows (unchanged own replies): {len(keep)} {dict(counts)}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "sft.jsonl", rows)
    # question texts for contamination.py (one id -> text file over all three rollout sets)
    texts = {}
    for path in (args.normal, args.replay):
        for r in read_jsonl(Path(path)):
            texts.setdefault(r["id"], r["question"])
    write_jsonl(out / "questions.jsonl", [{"id": k, "question": v} for k, v in texts.items()])
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


# a real check in the thinking: a trace, a test or an assert run through the code, with its result
CHECK = re.compile(r"\b(trace|tracing|test|testing|check|checking|assert|verify)\b|->|=>|\u2192", re.IGNORECASE)


def group(rollouts):
    grouped = defaultdict(list)
    for r in rollouts:
        grouped[r["id"]].append(r)
    return grouped


if __name__ == "__main__":
    main()
