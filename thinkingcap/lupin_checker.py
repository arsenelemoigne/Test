"""LUPIN0.x checker practice: "is this solution right? Yes/No" examples from the model's own graded answers (user, 27 Oct).

VeriThinker (2505.17941) trained a reasoning model only to judge solutions (Yes/No, labels from the final answer) and its
thinking got ~44% shorter at equal accuracy, fewer "Wait" re-checks after right steps (7B; it failed at 1.5B). Here the
same practice is a SMALL extra loss inside LUPIN0's DPO run (dpo.py --checker): the labels come free from step 1's
grading (the answer for math, the tests for code, the checkers for tool calls and instructions).

Each example: the question's own messages (a tool list stays in its system message), then the user turn gets
"Here is a proposed solution: ... Is this solution correct? Answer Yes or No."; the solution is the answer's thinking and
reply text. The target is "Yes" or "No" with thinking switched off (Qwen3's empty think block). Yes and No are balanced
per evaluation part; answers longer than --max-reply tokens are skipped (the question must fit with them).

    python lupin_checker.py --data data/pool-dp1.jsonl --rollouts runs/bon/probe-ans-dp1/rollouts.jsonl \\
        --out runs/bon/checker-dp1.jsonl
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from build_pool import part_of

ASK = "\n\nHere is a proposed solution:\n<solution>\n{solution}\n</solution>\n\nIs this solution correct? Answer Yes or No."


def read(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", required=True, help="the questions file the rollouts were sampled from")
    parser.add_argument("--rollouts", nargs="+", required=True)
    parser.add_argument("--max-reply", type=int, default=1500, help="longest answer used, in tokens")
    parser.add_argument("--per-question", type=int, default=2, help="answers judged per question (one right, one wrong first)")
    parser.add_argument("--max-examples", type=int, default=4000)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # probe_curve.py names a question f"{source}-{index of its row in --data}"
    messages = {f"{row['source']}-{i}": row["prompt"] for i, row in enumerate(read(args.data))}
    groups = defaultdict(list)
    for path in args.rollouts:
        for r in read(path):
            if r["id"] in messages and r["output_tokens"] <= args.max_reply:
                groups[r["id"]].append(r)
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    by_part = defaultdict(lambda: {True: [], False: []})
    for qid, answers in groups.items():
        rng.shuffle(answers)
        ok = lambda r: bool(r["correct"]) and r["closed_thinking"] and not r["truncated"]
        picked = [next((r for r in answers if ok(r)), None), next((r for r in answers if not ok(r)), None)]
        picked = [r for r in picked if r is not None] + [r for r in answers if r not in picked]
        for r in picked[:args.per_question]:
            text = tokenizer.decode(r["output_ids"], skip_special_tokens=False)
            for tag in ("<think>", "</think>", "<|im_end|>", "<|endoftext|>"):
                text = text.replace(tag, "")
            msgs = [dict(m) for m in messages[qid]]
            msgs[-1]["content"] = msgs[-1]["content"] + ASK.format(solution=text.strip())
            prompt = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False)
            target = tokenizer.encode("Yes" if ok(r) else "No", add_special_tokens=False) + [end]
            by_part[part_of(r)][ok(r)].append({"id": qid, "source": r["source"], "part": part_of(r), "label": ok(r),
                                               "prompt_ids": prompt, "target_ids": target})
    examples, counts = [], Counter()
    parts = [p for p in by_part if p]
    per_part = args.max_examples // max(1, len(parts))
    for part in parts:
        n = min(len(by_part[part][True]), len(by_part[part][False]), per_part // 2)  # balanced Yes / No
        for label in (True, False):
            rng.shuffle(by_part[part][label])
            examples += by_part[part][label][:n]
            counts[(part, "Yes" if label else "No")] = n
    rng.shuffle(examples)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(e) + "\n" for e in examples))
    text = "\n".join([f"{len(examples)} checker examples -> {out}"] + [f"  {a} / {b}: {n}" for (a, b), n in sorted(counts.items())])
    print(text, flush=True)


if __name__ == "__main__":
    main()
