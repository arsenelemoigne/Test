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
    parser.add_argument("--sources", default="gsm8k,mbpp_train,hermes_tools,ifeval_like")
    parser.add_argument("--kod-max-task", type=int, default=45, help="kodcode: longest task, Qwen3 tokens")
    parser.add_argument("--kod-max-solution", type=int, default=90, help="kodcode: longest reference solution")
    parser.add_argument("--tool-mix", default="0.48,0.24,0.28",
                        help="toolace: shares of one-function, several-function and no-fit questions (BFCL's)")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    rows = []
    builders = {"gsm8k": gsm8k, "gsm8k_all": gsm8k_all, "gsm8k_easy": gsm8k_easy, "mbpp_train": mbpp_train, "hermes_tools": hermes_tools,
                "toolace": toolace, "ifeval_like": ifeval_like, "mc": multiple_choice, "kodcode": kodcode,
                "gsm_plus": gsm_plus, "gsm_ic": gsm_ic}
    for source in args.sources.split(","):
        build = builders[source]
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


def multiple_choice(args, rng):
    """ARC-Easy and CommonsenseQA from data/train.jsonl: not in the evaluation, but brevity there spills over."""
    for line in Path("data/train.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["source"] in ("arc_easy", "commonsense_qa"):
            yield row


def gsm8k_all(args, rng):
    """GSM8K train and test splits (neither is an evaluation benchmark), in prepare_data.py's layout.

    Meant to be filtered afterwards to the questions the raw model solves briefly, which match
    SVAMP's difficulty; ASDiv and MAWPS, which SVAMP was built from, are deliberately not used.
    """
    from datasets import load_dataset
    from utils.train.prepare_data import MATH_INSTRUCTION, row

    for split in ("train", "test"):
        for r in load_dataset("openai/gsm8k", "main", split=split):
            answer = r["answer"].split("####")[-1].strip().replace(",", "")
            yield row(r["question"], MATH_INSTRUCTION, answer, "math", "gsm8k_all")


def gsm8k_easy(args, rng):
    """GSM8K TRAIN problems whose reference solution makes 1 or 2 calculations (counted by its <<...>> annotations): the
    difficulty SVAMP's paper describes (one- or two-operation grade-school word problems), from a README-suggested training
    set. The test split is left out (GSM-Plus and GSM-IC are built on it); ASDiv and MAWPS (SVAMP's sources) are not used."""
    from datasets import load_dataset
    from utils.train.prepare_data import MATH_INSTRUCTION, row

    for r in load_dataset("openai/gsm8k", "main", split="train"):
        if 1 <= r["answer"].count("<<") <= 2:
            answer = r["answer"].split("####")[-1].strip().replace(",", "")
            yield row(r["question"], MATH_INSTRUCTION, answer, "math", "gsm8k_easy")


GSM_PLUS_KINDS = ("distraction insertion", "problem understanding", "reversing operation", "adding operation",
                  "numerical substitution")


def gsm_plus(args, rng):
    """GSM-Plus (Li et al. 2024): GSM8K test problems perturbed, a check against shortcut solving.

    The SVAMP-like kinds only (irrelevant information, rephrased questions, reversed or added operations,
    changed numbers); "critical thinking" has no numeric answer, digit expansion and fraction conversion
    test arithmetic size rather than careful reading. Neither GSM8K nor GSM-Plus is an evaluation
    benchmark, and neither comes from ASDiv/MAWPS (SVAMP's sources).
    """
    from datasets import load_dataset
    from utils.train.prepare_data import MATH_INSTRUCTION, row

    for r in load_dataset("qintongli/GSM-Plus", split="test"):
        answer = str(r["answer"]).strip().replace(",", "")
        if r["perturbation_type"] not in GSM_PLUS_KINDS or not re.fullmatch(r"-?\d+(\.\d+)?", answer):
            continue
        yield row(r["question"], MATH_INSTRUCTION, answer, "math", "gsm_plus")


def gsm_ic(args, rng):
    """GSM-IC (Shi et al. 2023): GSM8K problems with one irrelevant sentence inserted; in-topic ones (harder)."""
    import urllib.request
    from utils.train.prepare_data import MATH_INSTRUCTION, row

    url = "https://raw.githubusercontent.com/google-research-datasets/GSM-IC/main/GSM-IC_2step.json"
    for r in json.loads(urllib.request.urlopen(url, timeout=60).read()):
        answer = str(r["answer"]).strip().replace(",", "")
        if r.get("sentence_label") != "in_topic" or not re.fullmatch(r"-?\d+(\.\d+)?", answer):
            continue
        yield row(r["new_question"], MATH_INSTRUCTION, answer, "math", "gsm_ic")


def toolace(args, rng):
    """ToolACE single-turn tool calls, already in BFCL's function format and call syntax.

    The first user turn and the first assistant reply of each conversation; the reply is a
    Python-style call list, or prose when no function fits (BFCL's irrelevance). Questions are
    drawn to --tool-mix, BFCL's shares of one-function, several-function and no-fit questions.
    """
    from datasets import load_dataset
    from utils.eval.benchmarks import bfcl_system_prompt, paths
    from utils.eval.graders import decode_calls

    root = paths(args.questions)["bfcl"]
    pools = {"simple": [], "multiple": [], "irrelevance": []}
    for r in load_dataset("Team-ACE/ToolACE", split="train"):
        system, turns = r["system"], r["conversations"]
        start = system.find("[{")
        if start < 0 or len(turns) < 2 or turns[0]["from"] != "user" or turns[1]["from"] != "assistant":
            continue
        try:
            functions = json.JSONDecoder().raw_decode(system[start:])[0]
        except ValueError:
            continue
        if not functions or not all(f.get("name") for f in functions):
            continue
        # BFCL calls are Python: "Market Trends API" becomes Market_Trends_API, in the functions and the reply
        names = {f["name"]: identifier(f["name"]) for f in functions}
        if len(set(names.values())) < len(names):
            continue
        reply = turns[1]["value"].strip()
        for old, new in sorted(names.items(), key=lambda kv: -len(kv[0])):
            reply = reply.replace(old + "(", new + "(")
        for f in functions:
            f["name"] = names[f["name"]]
            f.setdefault("parameters", {"type": "dict", "properties": {}, "required": []})
            f["parameters"].setdefault("required", [])
            f["parameters"]["type"] = "dict"
        try:
            calls = decode_calls(reply)
        except Exception:
            calls = None
        if calls is None:
            kind, reference = "irrelevance", None
        elif len(calls) == 1:
            name, arguments = next(iter(calls[0].items()))
            called = [f for f in functions if f["name"] == name]
            if not called:
                continue
            kind = "simple" if len(functions) == 1 else "multiple"
            reference = bfcl_reference(called[0], arguments)
        else:
            continue  # parallel calls: not in the evaluation
        def example(category, offered):
            return {"source": "toolace", "task": "tool", "answer": "", "grader": "bfcl", "category": category,
                    "function": offered, "reference": reference,
                    "prompt": [{"role": "system", "content": bfcl_system_prompt(root, offered)},
                               {"role": "user", "content": turns[0]["value"]}]}

        pools[kind].append(example(kind, functions))
        if kind == "multiple":  # ToolACE rarely offers one function: the called one alone makes BFCL's simple case
            pools["simple"].append(example("simple", called))
    shares = [float(x) for x in args.tool_mix.split(",")]
    for (kind, pool), share in zip(pools.items(), shares):
        rng.shuffle(pool)
        print(f"toolace {kind}: {len(pool)} usable", flush=True)
        yield from pool[:int(round(share * args.per_source))]


def identifier(name):
    """A function name as a Python identifier: other characters become underscores."""
    name = re.sub(r"[^\w.]", "_", name.strip())
    return name if re.match(r"[A-Za-z_]", name) else "f_" + name


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


TEST_FUNCTION = re.compile(r"^def (test_\w*)\(\s*\):", re.MULTILINE)


def kodcode(args, rng):
    """Clean code practice in MBPP's format: the MBPP-like slice of KodCode-Light-RL-10K.

    KodCode (synthetic tasks verified by unit tests) is a different dataset from MBPP/MBPP+. Loader from the
    mvt-pertype session: one-function easy/medium tasks whose tests are plain argument-free test_ functions,
    turned into asserts run as MBPP's are, the first assert shown as MBPP's example. Kept only where the task
    is at most 45 tokens and the reference solution at most 90 (Qwen3 tokenizer): measured on CPU, that slice
    (~900 tasks; median 28 / 63 tokens) is close to MBPP+ (17 / 29), all of KodCode is not (99 / 114).
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from utils.eval.benchmarks import MBPP_INSTRUCTION, MBPP_NAME_REMINDER

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    size = lambda text: len(tokenizer.encode(text or "", add_special_tokens=False))
    for task in load_dataset("KodCode/KodCode-Light-RL-10K", split="train"):
        if task["style"] != "instruct" or task["gpt_difficulty"] not in ("easy", "medium"):
            continue
        if len(task["test_info"] or []) != 1:
            continue
        name, test = task["test_info"][0]["function_name"], task["test"]
        if not name or "pytest" in test or "@" in test or "solution" not in test.split("\n", 1)[0]:
            continue
        body = "\n".join(l for l in test.splitlines() if not re.match(r"\s*(from solution import|import solution)", l))
        functions = TEST_FUNCTION.findall(body)
        example = next((l.strip() for l in body.splitlines() if l.strip().startswith("assert") and f"{name}(" in l), None)
        if not functions or not example or len(functions) != body.count("def test_"):
            continue
        if size(task["question"]) > args.kod_max_task or size(task["solution"]) > args.kod_max_solution:
            continue
        prompt = f'"""\n{task["question"].strip()}\n{example}\n"""\n'
        content = f"{MBPP_INSTRUCTION}\n\n{prompt}\n{MBPP_NAME_REMINDER.format(name=name)}"
        yield {"source": "kodcode", "task": "code", "answer": "", "grader": "code", "entry_point": name,
               "tests": [body, *[f"{f}()" for f in functions]], "setup": "",
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
            # the prose mentions "<tools> </tools>" before the real block: take the last non-empty one
            tools = json.loads([b for b in TOOLS.findall(turns["system"]) if b.strip()][-1])
            calls = [json.loads(c) for c in CALL.findall(turns["gpt"])]
            functions = [bfcl_function(t) for t in tools]
        except (IndexError, KeyError, ValueError, TypeError):
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
