"""Training questions of every kind the evaluation scores: math, code, tool calls, instructions.

Writes data/types.jsonl in the layout probe_curve.py reads ({"prompt": messages, "task",
"answer", "source"}) plus each grader's fields for typed.py. Run after
utils/train/prepare_data.py and utils/eval/benchmarks.py:

    python prepare_types.py --per-source 400

| source        | task          | from                                       | scored like  |
|---------------|---------------|--------------------------------------------|--------------|
| gsm8k         | math          | data/train.jsonl                           | SVAMP        |
| mbpp_train    | code          | MBPP (all splits) minus every MBPP+ task   | MBPP+        |
| hermes_tools  | tool          | Hermes function calling, one call per reply, plus "no tool fits" pairs | BFCL |
| ifeval_like   | instructions  | argilla/ifeval-like-data (filtered)        | IFEval       |

Prompts copy the evaluation's own wording (MBPP+ instruction and name reminder, BFCL's system
prompt), so the model is trained in the format it is tested in. No evaluation question is used.
"""

import argparse
import ast
import json
import random
import re
from pathlib import Path

from typed import bfcl_function, bfcl_reference, entry_point


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--questions", default="questions")
    parser.add_argument("--out", default="data/types.jsonl")
    parser.add_argument("--per-source", type=int, default=400, help="questions kept per source")
    parser.add_argument("--irrelevant-share", type=float, default=0.25,
                        help="share of the tool questions paired with tools that cannot answer them")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    rows = []
    for source, build in (("gsm8k", gsm8k), ("mbpp_train", mbpp_train), ("hermes_tools", hermes_tools),
                          ("ifeval_like", ifeval_like)):
        built = list(build(args, rng))
        rng.shuffle(built)
        rows += built[:args.per_source]
        print(f"{source}: {len(built)} usable, kept {min(len(built), args.per_source)}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {len(rows)} questions to {args.out}")


def gsm8k(args, rng):
    for line in Path("data/train.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["source"] == "gsm8k":
            yield row


def mbpp_train(args, rng):
    from datasets import load_dataset
    from utils.eval.benchmarks import MBPP_INSTRUCTION, MBPP_NAME_REMINDER, paths

    evaluated = {int(re.sub(r"\D", "", json.loads(l)["task_id"]))
                 for l in paths(args.questions)["mbpp"].read_text().splitlines() if l.strip()}
    for split in ("train", "validation", "prompt", "test"):
        for task in load_dataset("google-research-datasets/mbpp", "full", split=split):
            if task["task_id"] in evaluated:
                continue
            name = entry_point(task["test_list"][0])
            if not name:
                continue
            prompt = f'"""\n{task["text"]}\n{task["test_list"][0]}\n"""\n'
            content = f"{MBPP_INSTRUCTION}\n\n{prompt}\n{MBPP_NAME_REMINDER.format(name=name)}"
            yield {"source": "mbpp_train", "task": "code", "answer": "", "grader": "code",
                   "entry_point": name, "tests": task["test_list"], "setup": task["test_setup_code"],
                   "prompt": [{"role": "user", "content": content}]}


TOOLS = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.DOTALL)
CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def hermes_tools(args, rng):
    from datasets import load_dataset
    from utils.eval.benchmarks import bfcl_system_prompt, paths

    root = paths(args.questions)["bfcl"]
    parsed = []
    for row in load_dataset("NousResearch/hermes-function-calling-v1", "func_calling_singleturn", split="train"):
        turns = {t["from"]: t["value"] for t in row["conversations"]}
        try:
            tools = json.loads(TOOLS.search(turns["system"]).group(1))
            calls = [json.loads(c) for c in CALL.findall(turns["gpt"])]
            functions = [bfcl_function(t) for t in tools]
        except (AttributeError, KeyError, ValueError, TypeError):
            continue
        if len(calls) != 1 or not isinstance(calls[0].get("arguments"), dict):
            continue
        called = [f for f in functions if f["name"] == calls[0]["name"]]
        if not called or not all(re.fullmatch(r"[A-Za-z_]\w*", f["name"]) for f in functions):
            continue
        parsed.append((turns["human"], functions, called[0], calls[0]["arguments"], row["category"]))
    irrelevant = int(len(parsed) * args.irrelevant_share)
    for index, (question, functions, called, arguments, category) in enumerate(parsed):
        if index < irrelevant:  # pair the question with another domain's tools: no call is right
            others = [p for p in parsed if p[4] != category]
            functions = rng.choice(others)[1]
            kind, reference = "irrelevance", None
        else:
            kind = "multiple" if len(functions) > 1 else "simple"
            reference = bfcl_reference(called, arguments)
        yield {"source": "hermes_tools", "task": "tool", "answer": "", "grader": "bfcl",
               "category": kind, "function": functions, "reference": reference,
               "prompt": [{"role": "system", "content": bfcl_system_prompt(root, functions)},
                          {"role": "user", "content": question}]}


def ifeval_like(args, rng):
    from datasets import load_dataset
    from utils.eval.benchmarks import paths

    evaluated = {json.loads(l)["prompt"] for l in
                 (paths(args.questions)["ifeval"] / "data/input_data.jsonl").read_text().splitlines() if l.strip()}
    for index, row in enumerate(load_dataset("argilla/ifeval-like-data", "filtered", split="train")):
        if row["prompt"] in evaluated:
            continue
        try:
            ids = ast.literal_eval(row["instruction_id_list"]) if isinstance(row["instruction_id_list"], str) \
                else row["instruction_id_list"]
            kwargs = json.loads(row["kwargs"]) if isinstance(row["kwargs"], str) else row["kwargs"]
        except (ValueError, SyntaxError):
            continue
        if len(ids) != len(kwargs):
            continue
        yield {"source": "ifeval_like", "task": "instructions", "answer": "", "grader": "ifeval",
               "id": f"ifeval_like-{row['key']}", "instruction_id_list": ids, "kwargs": kwargs,
               "original_prompt": row["prompt"], "prompt": [{"role": "user", "content": row["prompt"]}]}


if __name__ == "__main__":
    main()
