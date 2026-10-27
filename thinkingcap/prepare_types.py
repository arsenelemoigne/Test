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
    parser.add_argument("--tool-clean", action="store_true",
                        help="toolace v2 (dp2): BFCL type names, single-turn only, references grounded in the question, "
                             "no-fit questions made by offering another conversation's function")
    parser.add_argument("--tool-near", action="store_true",
                        help="toolace (dp4): also no-fit questions offered only the functions the reply did not call")
    parser.add_argument("--ifeval-match", type=int, default=0,
                        help="ifeval_like v2 (dp2): this many questions drawn to IFEval's instruction mix (0: old filter)")
    parser.add_argument("--kod-dataset", default="KodCode/KodCode-Light-RL-10K",
                        help="kodcode: KodCode/KodCode-V1 for a big pool (the 10K slice runs short after contamination)")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    rows = []
    builders = {"gsm8k": gsm8k, "gsm8k_all": gsm8k_all, "gsm8k_easy": gsm8k_easy, "gsm8k_easy_test": gsm8k_easy_test, "metamath_easy": metamath_easy, "ifeval_built": ifeval_built, "gsm8k_matched": gsm8k_matched, "metamath_sv": metamath_sv, "mbpp_train": mbpp_train, "hermes_tools": hermes_tools,
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
            yield svamp_row(r["question"], answer, "gsm8k_easy")


SVAMP_LAST_LINE = "The last line of your reply must start with 'Final answer:' followed by the number alone."


def svamp_row(question, answer, source):
    """A math question in the evaluation's own SVAMP wording (as gsm8k_matched and metamath_sv). prepare_data.py's
    MATH_INSTRUCTION ("... like: Final answer: 42") made 25% of dp3's gsm8k_easy answers unreadable and 11% loop to the
    cap (8,580 answers: 2,187 no final answer, 941 capped, 292 really wrong), so those questions looked hard and long."""
    from utils.train.prepare_data import row
    out = row(question.strip() + "\n\n" + SVAMP_LAST_LINE, "", answer, "math", source)
    out["prompt"][0]["content"] = out["prompt"][0]["content"].rstrip()
    return out


def gsm8k_easy_test(args, rng):
    """GSM8K TEST problems with 1 or 2 calculations (dp4: dp3's math pool lacked easy, short-thinking questions; GSM8K
    is not an evaluation benchmark, GSM-Plus / GSM-IC built on its test split are not used in dp4)."""
    from datasets import load_dataset
    from utils.train.prepare_data import MATH_INSTRUCTION, row

    for r in load_dataset("openai/gsm8k", "main", split="test"):
        if 1 <= r["answer"].count("<<") <= 2:
            answer = r["answer"].split("####")[-1].strip().replace(",", "")
            yield svamp_row(r["question"], answer, "gsm8k_easy_test")


def metamath_easy(args, rng):
    """MetaMathQA's GSM_Rephrased rows whose original is a 1-2 calculation GSM8K problem (dp4): a rewording keeps the
    final answer, so a row is linked to the easy GSM8K question with the same answer that shares most of its words
    (Jaccard >= 0.4); unlinked rows are dropped. Same prompt wording as gsm8k_easy."""
    from collections import defaultdict
    from datasets import load_dataset
    from utils.train.prepare_data import MATH_INSTRUCTION, row

    words = lambda t: set(re.findall(r"[a-z]+", t.lower()))
    easy = defaultdict(list)
    for split in ("train", "test"):
        for r in load_dataset("openai/gsm8k", "main", split=split):
            if 1 <= r["answer"].count("<<") <= 2:
                easy[r["answer"].split("####")[-1].strip().replace(",", "")].append(words(r["question"]))
    for r in load_dataset("meta-math/MetaMathQA", split="train"):
        if r["type"] != "GSM_Rephrased":
            continue
        answer = r["response"].rsplit("The answer is:", 1)[-1].strip().replace(",", "")
        w = words(r["query"])
        if not w or not any(len(w & o) / len(w | o) >= 0.4 for o in easy.get(answer, ())):
            continue
        yield svamp_row(r["query"], answer, "metamath_easy")


def gsm8k_matched(args, rng):
    """The training part of track.py mathmatch's GSM8K corpus matched to SVAMP's operation count x kind and length (the
    held-out math test part, data/gsm8k_matched-test.jsonl, is never drawn here)."""
    for line in Path("data/gsm8k_matched-train.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            yield {k: v for k, v in r.items() if k not in ("id", "words", "ops", "kind")}


def metamath_sv(args, rng):
    """MetaMathQA's GSM_SV and GSM_FOBAR rows: GSM8K TRAIN problems rewritten with an unknown x (solve for x) or asked
    backwards from a given answer, the twisted wordings closest to SVAMP's variations (michal-bida's math mix). MetaMathQA
    is built from GSM8K and MATH training sets only; plain-number answers only; same prompt wording as gsm8k_matched."""
    from datasets import load_dataset
    from utils.train.prepare_data import row

    numeric = "\n\nThe last line of your reply must start with 'Final answer:' followed by the number alone."
    for r in load_dataset("meta-math/MetaMathQA", split="train"):
        if r["type"] not in ("GSM_SV", "GSM_FOBAR"):
            continue
        answer = r["response"].rsplit("The answer is:", 1)[-1].strip().replace(",", "")
        try:
            float(answer)
        except ValueError:
            continue
        out = row(r["query"] + numeric, "", answer, "math", "metamath_sv")
        out["prompt"][0]["content"] = out["prompt"][0]["content"].rstrip()  # row() appends an empty instruction
        yield out


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
        if args.tool_clean and (MULTI_TURN.search(turns[0]["value"]) or not bfcl_types(functions)):
            continue  # BFCL simple/multiple/irrelevance are single-turn; untyped parameters fail BFCL's checker
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
            if args.tool_clean:  # an undecodable reply is often a malformed call, not a refusal: no-fit is made below
                continue
            kind, reference = "irrelevance", None
        elif len(calls) == 1:
            name, arguments = next(iter(calls[0].items()))
            called = [f for f in functions if f["name"] == name]
            if not called:
                continue
            if args.tool_clean and not grounded(arguments, turns[0]["value"]):
                continue  # one GPT-written answer with values the question never states (dates, ids): ungradable
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
            if args.tool_near:  # dp4: the same question offered only the functions the reply did NOT call (close
                others = [f for f in functions if f["name"] != called[0]["name"]]  # domain, harder to refuse)
                # a function sharing a name word with the called one may answer it too ("trackAdImpression" for "ad
                # tracking data"): such a question is not no-fit, and its "wrong" answers would pass for difficulty
                if not any(name_words(f["name"]) & name_words(called[0]["name"]) for f in others):
                    pools["irrelevance"].append(example("irrelevance", others) | {"reference": None, "origin": "near"})
    if args.tool_clean:  # BFCL's irrelevance: a question offered a function from an unrelated conversation
        donors = pools["simple"]
        for row in list(pools["simple"]):
            words = set(re.findall(r"[a-z]{5,}", row["prompt"][1]["content"].lower())) - COMMON
            made = 0
            for _ in range(40):  # two different unrelated functions per question: dp1 ran short of no-fit questions
                donor = rng.choice(donors)["function"]
                text = json.dumps(donor).lower()
                if not any(w in text for w in words):
                    pools["irrelevance"].append(row | {"category": "irrelevance", "function": donor, "reference": None,
                                                       "origin": "far",
                                                       "prompt": [{"role": "system", "content": bfcl_system_prompt(root, donor)},
                                                                  row["prompt"][1]]})
                    made += 1
                    if made == 2:
                        break
    shares = [float(x) for x in args.tool_mix.split(",")]
    for (kind, pool), share in zip(pools.items(), shares):
        rng.shuffle(pool)
        print(f"toolace {kind}: {len(pool)} usable", flush=True)
        yield from pool[:int(round(share * args.per_source))]


COMMON = {"about", "could", "would", "there", "their", "which", "these", "those", "please", "using", "given", "provide",
          "information", "details", "should", "where", "while", "other", "after", "before", "within", "first", "number"}
MULTI_TURN = re.compile(r"Inquirer:|historical dialog|Response assistant", re.IGNORECASE)
BFCL_TYPES = {"string", "integer", "float", "boolean", "array", "dict", "tuple", "any"}
TYPE_NAMES = {"int": "integer", "number": "float", "double": "float", "long": "integer", "str": "string",
              "bool": "boolean", "object": "dict", "list": "array"}


def bfcl_types(functions):
    """ToolACE's parameter types -> BFCL's names, in place ("int" fails BFCL's checker on every right call: ~10%
    of dp1's ToolACE simple questions). False when a type is still unknown to BFCL."""
    def fix(spec):
        if not isinstance(spec, dict):
            return True
        if "type" in spec:
            spec["type"] = TYPE_NAMES.get(str(spec["type"]).lower(), str(spec["type"]).lower())
            if spec["type"] not in BFCL_TYPES:
                return False
        return all(fix(v) for v in spec.get("properties", {}).values()) and fix(spec.get("items"))
    return all(all(fix(p) for p in f["parameters"].get("properties", {}).values()) for f in functions)


def grounded(arguments, question):
    """Every given argument value is stated in the question (BFCL's references are; ToolACE's often are not)."""
    text = re.sub(r"[^a-z0-9.]+", " ", question.lower())
    numbers = {float(n) for n in re.findall(r"\d+(?:\.\d+)?", question.replace(",", ""))}
    def stated(value):
        if isinstance(value, bool) or value in ("", None):
            return True
        if isinstance(value, (int, float)):
            return abs(value) in numbers
        if isinstance(value, str):
            v = re.sub(r"[^a-z0-9.]+", " ", value.lower()).strip()
            return not v or v in text
        if isinstance(value, list):
            return all(stated(v) for v in value)
        if isinstance(value, dict):
            return all(stated(v) for v in value.values())
        return False
    return all(stated(v) for v in arguments.values())


def name_words(name):
    """Words of a function name ("Get_Ad_Tracking", "getAdTracking" -> {"tracking"}), 4+ letters, minus verbs like get."""
    parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])", name.replace(".", "_"))
    return {w.lower() for w in parts if len(w) >= 4} - {"list", "fetch", "search", "find", "create", "update", "info",
                                                        "data", "details", "check", "calculate", "retrieve"}


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
    for task in load_dataset(args.kod_dataset, split="train"):
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


def ifeval_buildable(registry, ids, kwargs, prompt):
    """Does IFEval's checker accept these instruction arguments? The grader calls build_description(**kwargs) on every
    answer, so a row it rejects would crash the whole answering run; tried once here, at candidate time."""
    import random as stdrandom
    state = stdrandom.getstate()
    try:
        for i, kw in zip(ids, kwargs):
            instruction = registry.INSTRUCTION_DICT[i](i)
            instruction.build_description(**{k: v for k, v in (kw or {}).items() if v is not None})
            wants = instruction.get_instruction_args_keys() if hasattr(instruction, "get_instruction_args_keys") else []
            if "prompt" in wants:
                instruction.build_description(prompt=prompt)
        return True
    except Exception:
        return False
    finally:
        stdrandom.setstate(state)


def ifeval_built(args, rng):
    """dp5: IFEval prompts the model thinks LONGER on. dp3's held-out check: IFEval-like thinking 246 vs 316 tokens on the
    evaluation (real argilla rows 232, Dolly-based built ones 258), replies 354 vs 488: the base tasks were too short
    ("Why do you love Biriyani?"). Here the base task is a human-written request from HuggingFaceH4/no_robots (train;
    Generation, Brainstorm, Open QA, Rewrite, Summarize: letters, stories, posts, rewrites of a given text, like IFEval's
    own "write a 300+ word ..." tasks), 8-150 words, followed by IFEval's own instruction sentences (ifeval_fill: the
    1/2/3-instruction split and every instruction id's share follow IFEval). --ifeval-match prompts (default 12,000)."""
    from collections import Counter
    from datasets import load_dataset
    from utils.eval.benchmarks import paths

    lines = [json.loads(l) for l in (paths(args.questions)["ifeval"] / "data/input_data.jsonl").read_text().splitlines()
             if l.strip()]
    n = args.ifeval_match or 12000
    want_n = {k: n * c / len(lines) for k, c in Counter(len(r["instruction_id_list"]) for r in lines).items()}
    want_id = {k: n * c / len(lines) for k, c in Counter(i for r in lines for i in r["instruction_id_list"]).items()}
    bases = [r["prompt"].strip() for r in load_dataset("HuggingFaceH4/no_robots", split="train")
             if r["category"] in ("Generation", "Brainstorm", "Open QA", "Rewrite", "Summarize")
             and 8 <= len(r["prompt"].split()) <= 150]
    bases = bases * (1 + n // max(1, len(bases)))  # ~6,000 requests: each may carry up to two different instruction sets
    built = ifeval_fill(args, rng, want_n, want_id, Counter(), Counter(), n, bases=bases)
    print(f"ifeval_built: {len(built)} prompts from {len(set(bases))} no_robots requests", flush=True)
    for r in built:
        yield r | {"origin": "built-long", "id": r["id"].replace("ifeval_built", "ifeval_long")}


def ifeval_fill(args, rng, want_n, want_id, got_n, got_id, room, bases=None):
    """Prompts for the instruction ids IFEval-like data cannot supply: argilla's rows with 1-3 instructions hold almost no
    no-comma, quotation, repeat-prompt, all-capitals or two-responses instructions (60 no-comma among ~40,000 such rows
    in 3 of its 10 shards). Each prompt is a Dolly writing / brainstorming / open question (no context; not an evaluation
    set) followed by IFEval's own instruction sentences, written by IFEval's instruction library with its own random
    arguments (build_description), so the official checker grades them exactly as it grades IFEval. The rarest missing
    id leads each prompt; its partners come from the other missing ids, never two that IFEval's registry marks as
    conflicting. Updates got_n / got_id in place."""
    import random as stdrandom
    from datasets import load_dataset
    from utils.eval.benchmarks import paths
    from utils.eval.graders import importable

    official = paths(args.questions)["ifeval"]
    with importable(Path(official).parent):
        from instruction_following_eval import instructions_registry as registry
    conflicts = registry.conflict_make(dict(registry.INSTRUCTION_CONFLICTS))
    if bases is None:
        bases = [r["instruction"].strip() for r in load_dataset("databricks/databricks-dolly-15k", split="train")
                 if r["category"] in ("creative_writing", "brainstorming", "open_qa", "general_qa") and not r["context"]
                 and 4 <= len(r["instruction"].split()) <= 60]
    rng.shuffle(bases)
    built = []
    for base in bases:
        missing = {i: want_id[i] - got_id[i] for i in want_id if want_id[i] - got_id[i] >= 1}
        if not missing or len(built) >= room:
            break
        lead = max(missing, key=missing.get)
        sizes = {n: max(0.0, want_n[n] - got_n[n]) for n in want_n}
        n = rng.choices(list(sizes), weights=[w or 1e-3 for w in sizes.values()])[0]
        ids = [lead]
        while len(ids) < n:
            options = [i for i in missing if i not in ids and not any(i in conflicts.get(j, ()) or j in conflicts.get(i, ())
                                                                      for j in ids)]
            if not options:
                break
            ids.append(rng.choices(options, weights=[missing[i] for i in options])[0])
        stdrandom.seed(rng.random())  # build_description draws its arguments from the random module
        # the repeat-prompt instruction goes last and its request is everything before it, as in IFEval
        ids.sort(key=lambda i: i == "combination:repeat_prompt")
        prompt, kwargs = base, []
        for i in ids:
            instruction = registry.INSTRUCTION_DICT[i](i)
            if i == "combination:repeat_prompt":
                text = instruction.build_description(prompt_to_repeat=prompt)
            else:
                text = instruction.build_description()
            kwargs.append({k: v for k, v in (instruction.get_instruction_args() or {}).items() if v is not None})
            prompt += " " + text
        if not ifeval_buildable(registry, ids, kwargs, prompt):
            continue
        got_n[len(ids)] += 1
        got_id.update(ids)
        built.append({"source": "ifeval_like", "origin": "built", "task": "instructions", "answer": "",
                      "grader": "ifeval", "id": f"ifeval_built-{len(built)}", "instruction_id_list": ids,
                      "kwargs": kwargs, "original_prompt": prompt, "prompt": [{"role": "user", "content": prompt}]})
    return built


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
    """argilla/ifeval-like-data prompts with IFEval's own instruction ids and kwargs (graded by IFEval's checker).

    Old mode: the "filtered" config as is. dp1 showed it is far easier than IFEval (original Qwen 89% vs 49%, thinking
    -43%): it is mostly "at least N sentences", "include keywords", "N highlighted sections" (91-93% passed), and it has
    no repeat-prompt, response-language or two-responses rows and almost no no-comma / quotation / all-capitals ones.
    --ifeval-match N (dp2): N prompts drawn so that both the number of instructions per prompt (1 to 3) and each
    instruction id's share follow IFEval's own 541 prompts; the "default" config (550k rows) supplies the hard ids.
    """
    from datasets import load_dataset
    from utils.eval.benchmarks import paths

    lines = [json.loads(l) for l in (paths(args.questions)["ifeval"] / "data/input_data.jsonl").read_text().splitlines()
             if l.strip()]
    evaluated = {r["prompt"] for r in lines}
    rows = load_dataset("argilla/ifeval-like-data", "filtered", split="train")
    if args.ifeval_match:
        from collections import Counter
        from itertools import chain
        rows = chain(rows, load_dataset("argilla/ifeval-like-data", "default", split="train"))
        n_share = Counter(len(r["instruction_id_list"]) for r in lines)
        id_share = Counter(i for r in lines for i in r["instruction_id_list"])
        want_n = {n: args.ifeval_match * c / len(lines) for n, c in n_share.items()}
        want_id = {i: args.ifeval_match * c / len(lines) for i, c in id_share.items()}
        got_n, got_id, kept, seen = Counter(), Counter(), [], set()
        from utils.eval.graders import importable
        with importable(Path(paths(args.questions)["ifeval"]).parent):
            from instruction_following_eval import instructions_registry as registry
        keys = {}  # each instruction id's argument names, as IFEval's own prompts give them
        for r in lines:
            for i, kw in zip(r["instruction_id_list"], r["kwargs"]):
                keys.setdefault(i, set()).update(k for k, v in kw.items() if v is not None)
    for index, row in enumerate(rows):
        if "prompt" not in row:  # the default config: "instruction", no key, one kwargs dict for all instructions
            row = {"prompt": row["instruction"], "key": f"d{index}", "instruction_id_list": row["instruction_id_list"],
                   "kwargs": row["kwargs"]}
        if row["prompt"] in evaluated or "[response_options]" in row["prompt"]:
            continue
        try:
            ids = ast.literal_eval(row["instruction_id_list"]) if isinstance(row["instruction_id_list"], str) \
                else row["instruction_id_list"]
            kwargs = json.loads(row["kwargs"]) if isinstance(row["kwargs"], str) else row["kwargs"]
        except (ValueError, SyntaxError, TypeError):
            continue
        if not isinstance(ids, list) or not isinstance(kwargs, (list, dict)):
            continue  # some default-config rows carry no kwargs at all ("null")
        if isinstance(kwargs, dict):
            if not args.ifeval_match or any(i not in keys for i in ids):
                continue
            kwargs = [{k: kwargs[k] for k in keys[i] if kwargs.get(k) is not None} for i in ids]
            if any(len(kw) < len(keys[i]) for kw, i in zip(kwargs, ids)):
                continue  # an argument the instruction needs is missing
        if len(ids) != len(kwargs):
            continue
        out = {"source": "ifeval_like", "task": "instructions", "answer": "", "grader": "ifeval",
               "id": f"ifeval_like-{row['key']}", "instruction_id_list": ids, "kwargs": kwargs,
               "original_prompt": row["prompt"], "prompt": [{"role": "user", "content": row["prompt"]}]}
        if not args.ifeval_match:
            yield out
            continue
        if row["prompt"] in seen or len(ids) not in want_n or any(i not in want_id for i in ids):
            continue
        if not ifeval_buildable(registry, ids, out["kwargs"], row["prompt"]):
            continue  # e.g. language "zh": IFEval's checker has no such code and crashed dp2's answering (KeyError)
        seen.add(row["prompt"])
        kept.append(out)
    if args.ifeval_match:
        # quota draw, rarest instruction first: a prompt is kept while its instruction count and every one of its ids are
        # under IFEval's share (filling common ids first blocked the rare ones: 4,132 prompts, no-comma 8 of 1,464)
        eligible, kept = kept, []
        rng.shuffle(eligible)
        supply = Counter(i for r in eligible for i in r["instruction_id_list"])
        taken = set()
        for target in sorted(want_id, key=lambda i: supply[i] / want_id[i]):
            for k, r in enumerate(eligible):
                if got_id[target] >= want_id[target] or len(kept) >= args.ifeval_match:
                    break
                ids = r["instruction_id_list"]
                if k in taken or target not in ids or got_n[len(ids)] + 1 > want_n[len(ids)] + 1 \
                        or any(got_id[i] + 1 > want_id[i] + 1 for i in ids):
                    continue
                taken.add(k)
                kept.append(r)
                got_n[len(ids)] += 1
                got_id.update(ids)
        real = len(kept)
        kept += ifeval_fill(args, rng, want_n, want_id, got_n, got_id, args.ifeval_match - len(kept))
        short = {i: round(w - got_id[i]) for i, w in want_id.items() if got_id[i] < w - 1}
        print(f"ifeval_like matched: {real} real + {len(kept) - real} built prompts ({len(eligible)} real eligible); by instruction count "
              f"{dict(sorted(got_n.items()))}" + (f"; ids short of IFEval's share: {short}" if short else ""), flush=True)
        yield from kept


if __name__ == "__main__":
    main()
