"""Code "stopped too early" pairs: the model's own correct code replies against its own early-cut failures.

For each correct, finished code reply of the current model (its own rollouts), the thinking is cut
at --fractions of its tokens and the model answers --samples times from each cut (check_vllm.py,
one batch, the model's adapter). Where a cut yields a wrong answer, the pair is

    hard   the full reply (right)   beats   that cut + its wrong answer

using the latest such cut: the most tempting place to stop. These pairs teach code, specifically,
not to stop thinking too early (as on-policy truncation oracles do; Lightning McQwen3, ThoughtFold).

    python code_pairs.py --rollouts runs/bon/probe-r3/rollouts.jsonl --keep runs/bon/select-r3/sft.jsonl \\
        --adapter adapters/start --out runs/bon/code-r3
"""

import argparse
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from probe_curve import read_jsonl, write_jsonl
from prune_traces import GRADE_FIELDS, thinking_text


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter", help="the model that wrote the rollouts")
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--keep", required=True, help="sft.jsonl whose ids passed the contamination check")
    parser.add_argument("--out", required=True)
    parser.add_argument("--source", default="mbpp_train")
    parser.add_argument("--fractions", default="0.25,0.5,0.75")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--per-question", type=int, default=2)
    parser.add_argument("--answer-tokens", type=int, default=2048)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    from transformers import AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    open_id = tokenizer.convert_tokens_to_ids("<think>")
    allowed = {r["id"] for r in read_jsonl(Path(args.keep))}
    by_question = defaultdict(list)
    for r in read_jsonl(Path(args.rollouts)):
        if (r["source"] == args.source and r["id"] in allowed and r["correct"] and r["closed_thinking"]
                and not r["truncated"]):
            by_question[r["id"]].append(r)

    jobs = []
    fractions = [float(f) for f in args.fractions.split(",")]
    for group in by_question.values():
        for rollout in group[:args.per_question]:
            thinking = [t for t in rollout["thinking_ids"] if t != open_id]
            for fraction in fractions:
                text = tokenizer.decode(thinking[:int(fraction * len(thinking))])
                jobs.append((rollout, fraction, tokenizer.encode(thinking_text([text]), add_special_tokens=False)))
    print(f"{len(by_question)} questions, {len(jobs)} cuts to check", flush=True)
    inputs, results = out / "cuts.jsonl", out / "verdicts.jsonl"
    write_jsonl(inputs, [{"key": i, "prompt_ids": r["prompt_ids"], "thinking_ids": t, "max_tokens": args.answer_tokens,
                          "row": {f: r[f] for f in GRADE_FIELDS if f in r}} for i, (r, _, t) in enumerate(jobs)])
    command = [sys.executable, "check_vllm.py", "--model", args.model, "--inputs", str(inputs), "--out", str(results),
               "--samples", str(args.samples), "--dtype", args.dtype]
    if args.adapter:
        command += ["--adapter", args.adapter]
    subprocess.run(command, check=True)
    verdicts = {v["key"]: v for v in read_jsonl(results)}

    latest = {}  # rollout -> its latest cut with a wrong answer
    failed = Counter()
    for i, (rollout, fraction, thinking) in enumerate(jobs):
        v = verdicts[i]
        if v["wrong_ids"]:
            failed[fraction] += 1
            key = (rollout["id"], rollout["sample"])
            if key not in latest or fraction > latest[key][0]:
                latest[key] = (fraction, rollout, thinking + v["wrong_ids"])
    pairs = [{"id": r["id"], "source": r["source"], "kind": "hard", "prompt_ids": r["prompt_ids"],
              "chosen_ids": r["output_ids"], "rejected_ids": rejected} for _, r, rejected in latest.values()]
    write_jsonl(out / "pairs.jsonl", pairs)
    total = Counter(f for _, f, _ in jobs)
    text = "\n".join([f"cut at {f:.0%} of the thinking: {failed[f]} of {total[f]} cuts give a wrong answer"
                      for f in fractions] + [f"{len(pairs)} hard pairs (full right reply over its latest wrong cut)"])
    (out / "summary.txt").write_text(text + "\n")
    print(text, flush=True)


if __name__ == "__main__":
    main()
