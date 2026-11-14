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
    parser.add_argument("--ifeval-bases", default="no_robots",
                        help="ifeval_built (12 Nov): request sources, from no_robots,dolly,oasst1,oasst2,wildchat (all human-written)")
    parser.add_argument("--ifeval-match", type=int, default=0,
                        help="ifeval_like v2 (dp2): this many questions drawn to IFEval's instruction mix (0: old filter)")
    parser.add_argument("--kod-difficulty", default="easy,medium",
                        help="kodcode: GPT difficulty labels kept (dp7 adds hard: dp5's code pool lacked the long-thinking tail)")
    parser.add_argument("--kod-dataset", default="KodCode/KodCode-Light-RL-10K",
                        help="kodcode: KodCode/KodCode-V1 for a big pool (the 10K slice runs short after contamination)")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    rows = []
    builders = {"gsm8k": gsm8k, "gsm8k_all": gsm8k_all, "gsm8k_easy": gsm8k_easy, "gsm8k_easy_test": gsm8k_easy_test, "metamath_easy": metamath_easy, "ifeval_built": ifeval_built, "toolace_near_verified": toolace_near_verified, "glaive_nofit_verified": glaive_nofit_verified, "glaive_nofit_hard": glaive_nofit_hard, "xlam_irrelevance": xlam_irrelevance, "gsm8k_std": gsm8k_std, "glaive_tools": glaive_tools, "gsm8k_matched": gsm8k_matched, "metamath_sv": metamath_sv, "mbpp_train": mbpp_train, "hermes_tools": hermes_tools,
                "toolace": toolace, "ifeval_like": ifeval_like, "ifeval_bida": ifeval_bida, "glaive_nofit_hard2": glaive_nofit_hard2, "kodcode_full": kodcode_full, "mc": multiple_choice, "kodcode": kodcode,
                "gsm_plus": gsm_plus, "gsm_ic": gsm_ic,
                "tr_math500": tr_math500, "tr_skywork": tr_skywork, "tr_arc_c": tr_arc_c, "tr_csqa": tr_csqa,
                "tr_mmlu": tr_mmlu, "tr_humaneval": tr_humaneval, "tr_aime25": tr_aime25,
                "tr_bfcl_live_simple": tr_bfcl_live_simple, "tr_bfcl_live_multiple": tr_bfcl_live_multiple,
                "tr_bfcl_live_irrelevance": tr_bfcl_live_irrelevance, "tr_bbh": tr_bbh, "tr_boolq": tr_boolq,
                "tr_ifbench": tr_ifbench}
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


def gsm8k_std(args, rng):
    """1 Nov: standard GSM8K train + test (michal-bida's main math source; harder for Qwen3-0.6B than our SVAMP-matched
    math, so more questions it sometimes gets wrong -> more "right > wrong" pairs), in SVAMP's wording. Labelled
    gsm8k_matched so build_pool / select_pool treat it as math; questions already in the pool are dropped by newtop."""
    from datasets import load_dataset

    for split in ("train", "test"):
        for r in load_dataset("openai/gsm8k", "main", split=split):
            answer = r["answer"].split("####")[-1].strip().replace(",", "")
            yield svamp_row(r["question"], answer, "gsm8k_matched")


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
        if task["style"] != "instruct" or task["gpt_difficulty"] not in args.kod_difficulty.split(","):
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


# michal-bida's instruction sources (2 Nov, user: "an instructions top-up built from his two sources"), after his
# prepare_domains.py: allenai/IF_multi_constraints_upto5 (1-3 IFEval-checkable constraints, no math/code/safety/table/
# science base request) and allenai/RLVR-IFeval (one constraint, mapped to the IFEval checker with the same meaning).
# Rows keep our ifeval_like format (source ifeval_like, origin if_multi / rlvr) so the pool treats them as instructions.
IF_EXCLUDED_KEYS = ("math", "jailbreak", "wildguard", "table_gpt", "science", "codealpaca", "code")
RLVR_TO_IFEVAL = {
    "validate_lowercase": ("change_case:english_lowercase", {}),
    "validate_uppercase": ("change_case:english_capital", {}),
    "validate_frequency_capital_words": ("change_case:capital_word_frequency",
                                         {"N": "capital_frequency", "quantifier": "capital_relation"}),
    "validate_end": ("startend:end_checker", {"end_phrase": "end_phrase"}),
    "validate_quotation": ("startend:quotation", {}),
    "validate_sections": ("detectable_format:multiple_sections", {"N": "num_sections", "section_splitter": "section_spliter"}),
    "validate_word_constraint": ("length_constraints:number_words", {"N": "num_words", "quantifier": "relation"}),
    "verify_sentence_constraint": ("length_constraints:number_sentences", {"N": "num_sentences", "quantifier": "relation"}),
    "validate_placeholders": ("detectable_content:number_placeholders", {"N": "num_placeholders"}),
    "validate_two_responses": ("combination:two_responses", {}),
    "validate_highlighted_sections": ("detectable_format:number_highlighted_sections", {"N": "num_highlights"}),
    "validate_json_format": ("detectable_format:json_format", {}),
    "validate_paragraphs": ("length_constraints:nth_paragraph_first_word",
                            {"N": "num_paragraphs", "i": "nth_paragraph", "first_word": "first_word"}),
    "verify_paragraph_count": ("length_constraints:number_paragraphs", {"N": "num_paragraphs"}),
    "verify_postscript": ("detectable_content:postscript", {"postscript_marker": "postscript_marker"}),
    "validate_no_commas": ("punctuation:no_comma", {}),
    "validate_forbidden_words": ("keywords:forbidden_words", {"forbidden_words": "forbidden_words"}),
    "verify_keywords": ("keywords:existence", {"keyword_list": "keywords"}),
    "validate_title": ("detectable_format:title", {}),
    "verify_bullet_points": ("detectable_format:number_bullet_lists", {"N": "num_bullets"}),
}


def _if_base_ok(base):
    letters = [c for c in base if c.isalpha()]
    if not letters or sum(c.isascii() for c in letters) / len(letters) <= 0.97 or "<unk>" in base:
        return False
    if re.search(r"(?i)\b(python|javascript|sql|java|c\+\+) (function|code|program|script)|write a function", base):
        return False
    return not re.search(r"\\\(|\\\[|\$[^$]+\$|\\frac|\^\d", base)


def ifeval_bida(args, rng):
    """michal-bida's two instruction sources, as IFEval-graded rows (see the comment above)."""
    from collections import defaultdict
    from datasets import load_dataset
    from utils.eval.benchmarks import paths
    from utils.eval.graders import importable

    lines = [json.loads(l) for l in (paths(args.questions)["ifeval"] / "data/input_data.jsonl").read_text().splitlines()
             if l.strip()]
    evaluated = {r["prompt"] for r in lines}
    limits = defaultdict(int)  # largest count IFEval itself asks for (e.g. num_paragraphs 5)
    for r in lines:
        for kw in r["kwargs"]:
            for k, v in (kw or {}).items():
                if isinstance(v, int) and not isinstance(v, bool):
                    limits[k] = max(limits[k], v)
    with importable(Path(paths(args.questions)["ifeval"]).parent):
        from instruction_following_eval import instructions_registry as registry
    allowed = set(registry.INSTRUCTION_DICT)

    def row(content, ids, kwargs, key, origin):
        if content in evaluated or not ifeval_buildable(registry, ids, kwargs, content):
            return None
        return {"source": "ifeval_like", "origin": origin, "task": "instructions", "answer": "", "grader": "ifeval",
                "id": f"ifeval_bida-{origin}-{key}", "instruction_id_list": ids, "kwargs": kwargs,
                "original_prompt": content, "prompt": [{"role": "user", "content": content}]}

    for k, r in enumerate(load_dataset("allenai/IF_multi_constraints_upto5", split="train")):
        truth = r["ground_truth"]
        truth = (ast.literal_eval(truth) if isinstance(truth, str) else truth)[0]
        ids = truth["instruction_id"]
        if not set(ids) <= allowed or len(ids) > 3 or any(w in r["key"] for w in IF_EXCLUDED_KEYS):
            continue
        messages = r["messages"]
        content = (ast.literal_eval(messages) if isinstance(messages, str) else messages)[0]["content"]
        base = content
        for constraint in r["constraint"].split("\t"):
            base = base.replace(constraint.strip(), " ")
        if not _if_base_ok(base):
            continue
        kwargs = [{a: b for a, b in (kw or {}).items() if b is not None} for kw in truth["kwargs"]]
        out = row(content, ids, kwargs, f"m{k}", "if_multi")
        if out:
            yield out
    for k, r in enumerate(load_dataset("allenai/RLVR-IFeval", split="train")):
        truth = json.loads(r["ground_truth"])
        if truth["func_name"] not in RLVR_TO_IFEVAL:
            continue
        instruction, names = RLVR_TO_IFEVAL[truth["func_name"]]
        kwargs = {ours: truth[theirs] for theirs, ours in names.items()}
        for key in ("capital_relation", "relation"):
            if key in kwargs and kwargs is not None:
                if kwargs[key] == "around":
                    kwargs = None
                    break
                if kwargs[key] == "at most":
                    count = {"capital_relation": "capital_frequency",
                             "relation": "num_words" if "num_words" in kwargs else "num_sentences"}[key]
                    kwargs[count] += 1
                    kwargs[key] = "less than"
        if kwargs is None or any(isinstance(v, int) and v > limits.get(a, 0) for a, v in kwargs.items()):
            continue
        messages = r["messages"]
        content = (ast.literal_eval(messages) if isinstance(messages, str) else messages)[0]["content"]
        if not _if_base_ok(content.replace(r["constraint"].strip(), " ")):
            continue
        out = row(content, [instruction], [kwargs], f"r{k}", "rlvr")
        if out:
            yield out


KOD_IMPERATIVE = re.compile(r"^(?:Please )?(?:Write|Create|Implement|Design|Develop|Define|Construct|Generate|Build|Devise|"
                            r"Given|You are given)\b")


def kodcode_full(args, rng):
    """michal-bida's code selection (2 Nov): all of KodCode-V1's Filter / Prefill / Leetcode / Evol tasks, easy and
    medium, questions up to 700 characters phrased as an instruction to write code; NO length cap to look like MBPP+
    (our kodcode keeps tasks <= 45 tokens). Same MBPP-format conversion and test handling as kodcode()."""
    import copy
    from datasets import load_dataset
    from utils.eval.benchmarks import MBPP_INSTRUCTION, MBPP_NAME_REMINDER

    for task in load_dataset("KodCode/KodCode-V1", split="train"):
        if task["subset"] not in ("Filter", "Prefill", "Leetcode", "Evol") or task["gpt_difficulty"] == "hard":
            continue
        question = task["question"].strip()
        if len(question) > 700 or "```" in question or not KOD_IMPERATIVE.match(question):
            continue
        info = task["test_info"] or []
        info = ast.literal_eval(info) if isinstance(info, str) else info
        if len(info) != 1 or not info[0].get("function_name"):
            continue
        name, test = info[0]["function_name"], task["test"]
        if "pytest" in test or "@" in test or "solution" not in test.split("\n", 1)[0]:
            continue
        body = "\n".join(l for l in test.splitlines() if not re.match(r"\s*(from solution import|import solution)", l))
        functions = TEST_FUNCTION.findall(body)
        example = next((l.strip() for l in body.splitlines() if l.strip().startswith("assert") and f"{name}(" in l), None)
        if not functions or not example or len(functions) != body.count("def test_"):
            continue
        prompt = f'"""\n{question}\n{example}\n"""\n'
        content = f"{MBPP_INSTRUCTION}\n\n{prompt}\n{MBPP_NAME_REMINDER.format(name=name)}"
        yield {"source": "kodcode", "origin": f"kodcode_v1/{task['subset']}", "task": "code", "answer": "",
               "grader": "code", "entry_point": name, "tests": [body, *[f"{f}()" for f in functions]], "setup": "",
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


GLAIVE_CALL = re.compile(r"<functioncall>\s*(\{.*?\})\s*<\|endoftext\|>", re.DOTALL)


def glaive_tools(args, rng):
    """dp7: one-call tool questions from glaiveai/glaive-function-calling-v2 (113k single-turn chats, Apache 2.0), many of
    them compute-style ("calculate the monthly loan payment ...") like BFCL simple, where ToolACE's are mostly API lookups
    that the model answers with less thinking (dp5 check: BFCL simple thinking 120 vs 143). Same clean-up as ToolACE /
    Hermes: BFCL type names, every argument value stated in the question. One function offered -> simple, several -> multiple."""
    from datasets import load_dataset
    from utils.eval.benchmarks import bfcl_system_prompt, paths

    root = paths(args.questions)["bfcl"]
    decoder = json.JSONDecoder()
    for r in load_dataset("glaiveai/glaive-function-calling-v2", split="train"):
        system, chat = r["system"], r["chat"]
        start = system.find("{")
        if start < 0 or not chat.startswith("USER:"):
            continue
        tools, text = [], system[start:]
        try:
            while text.strip():
                obj, end = decoder.raw_decode(text.strip())
                tools.append(obj)
                text = text.strip()[end:]
        except ValueError:
            continue
        question = chat[len("USER:"):].split("ASSISTANT:", 1)[0].strip()
        reply = chat.split("ASSISTANT:", 1)[1] if "ASSISTANT:" in chat else ""
        m = GLAIVE_CALL.search(reply.split("USER:", 1)[0])
        if not m or not question:
            continue
        try:
            call = json.loads(m.group(1).replace("'{", "{").replace("}'", "}"))
            name, arguments = call["name"], call.get("arguments") or {}
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            functions = [bfcl_function(t) for t in tools]
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
        called = [f for f in functions if f["name"] == name]
        if not called or not isinstance(arguments, dict) or not all(re.fullmatch(r"[A-Za-z_]\w*", f["name"]) for f in functions):
            continue
        if not bfcl_types(functions) or not grounded(arguments, question):
            continue
        yield {"source": "glaive_tools", "task": "tool", "answer": "", "grader": "bfcl",
               "category": "simple" if len(functions) == 1 else "multiple", "function": functions,
               "reference": bfcl_reference(called[0], arguments),
               "prompt": [{"role": "system", "content": bfcl_system_prompt(root, functions)},
                          {"role": "user", "content": question}]}


def toolace_near_verified(args, rng):
    """dp6: harder no-fit tool questions (nofit_verify.py, run on the Mac with the user's OpenRouter key): a ToolACE
    question offered the 1-3 functions from other conversations closest to it in meaning (sentence embeddings), kept
    only when Haiku answered NO to two differently worded "could one of these functions fulfil this request?" checks.
    Read from the repo clone (the Colab stage copies only *.py and plan files to the working directory)."""
    from utils.eval.benchmarks import bfcl_system_prompt, paths

    root = paths(args.questions)["bfcl"]
    path = next((p for p in (Path("/content/T/thinkingcap/nofit_near_verified.jsonl"),
                             Path(__file__).with_name("nofit_near_verified.jsonl")) if p.exists()), None)
    assert path, "nofit_near_verified.jsonl not found"
    for k, r in enumerate(json.loads(l) for l in path.read_text().splitlines() if l.strip()):
        yield {"source": "toolace", "origin": "near-verified", "task": "tool", "answer": "", "grader": "bfcl",
               "category": "irrelevance", "function": r["functions"], "reference": None, "id": f"toolace_nv-{k}",
               "prompt": [{"role": "system", "content": bfcl_system_prompt(root, r["functions"])},
                          {"role": "user", "content": r["question"]}]}


def glaive_nofit_verified(args, rng):
    """dp9: no-fit questions in BFCL's calculation style (nofit_verify.py --hard --rank 15,60 on Glaive, run on the Mac):
    a Glaive question offered ONE Glaive function ranked 15th-60th closest in meaning to it (past the near-synonyms of the
    function it really needs: a BMI request offered calculate_speed), kept only on Haiku's NO + NO. ToolACE look-alikes
    stayed too easy (dp8: 87% vs BFCL 79%)."""
    from utils.eval.benchmarks import bfcl_system_prompt, paths

    root = paths(args.questions)["bfcl"]
    path = next((p for p in (Path("/content/T/thinkingcap/nofit_glaive_verified.jsonl"),
                             Path(__file__).with_name("nofit_glaive_verified.jsonl")) if p.exists()), None)
    assert path, "nofit_glaive_verified.jsonl not found"
    for k, r in enumerate(json.loads(l) for l in path.read_text().splitlines() if l.strip()):
        yield {"source": "glaive_tools", "origin": "near-verified-glaive", "task": "tool", "answer": "", "grader": "bfcl",
               "category": "irrelevance", "function": r["functions"], "reference": None, "id": f"glaive_nv-{k}",
               "prompt": [{"role": "system", "content": bfcl_system_prompt(root, r["functions"])},
                          {"role": "user", "content": r["question"]}]}


def xlam_irrelevance(args, rng):
    """1 Nov: michal-bida's no-fit source (MadeAgents/xlam-irrelevance-7.5k: a question with xLAM tools of which none
    fits; the right answer calls nothing). Our Glaive no-fit questions were 98% solved by the original Qwen (almost no
    "right > wrong" pairs); his tool domain was ~45% xLAM irrelevance and his no-fit accuracy rose 6 pts. xLAM parameter
    specs {name: {description, type, default}} -> BFCL dicts; a parameter without a default is required."""
    from datasets import load_dataset
    from utils.eval.benchmarks import bfcl_system_prompt, paths

    root = paths(args.questions)["bfcl"]
    for k, row in enumerate(load_dataset("MadeAgents/xlam-irrelevance-7.5k", split="train")):
        try:
            tools = json.loads(row["tools"])
            functions = []
            for tool in tools:
                props, required = {}, []
                for pname, spec in (tool.get("parameters") or {}).items():
                    kind = str(spec.get("type", "str")).split(",")[0].strip().lower()
                    kind = "array" if kind.startswith(("list", "array")) else "dict" if kind.startswith(("dict", "object")) else kind
                    props[pname] = {"type": kind, "description": spec.get("description", "")}
                    if "optional" not in str(spec.get("type", "")).lower() and "default" not in spec:
                        required.append(pname)
                functions.append({"name": tool["name"], "description": tool.get("description", ""),
                                  "parameters": {"type": "dict", "properties": props, "required": required}})
        except (ValueError, TypeError, KeyError, AttributeError):
            continue
        if not functions or json.loads(row["answers"] or "[]") or not bfcl_types(functions) \
                or not all(re.fullmatch(r"[A-Za-z_]\w*", f["name"]) for f in functions):
            continue
        yield {"source": "glaive_tools", "origin": "xlam-irrelevance", "task": "tool", "answer": "", "grader": "bfcl",
               "category": "irrelevance", "function": functions, "reference": None, "id": f"xlam_irr-{k}",
               "prompt": [{"role": "system", "content": bfcl_system_prompt(root, functions)},
                          {"role": "user", "content": row["query"]}]}


def glaive_nofit_hard(args, rng):
    """dp13 (30 Oct): HARDER no-fit questions: a Glaive question offered ONE Glaive function ranked 4th-15th closest in
    meaning (past the exact synonyms, but same family: a tip request offered calculate_tax, Fahrenheit-to-Celsius offered
    get_temperature), kept only on Haiku's NO + NO (nofit_verify.py --rank 4,15). The first batch (ranks 15-60) stayed
    too easy: only 189 of its pool candidates were ever answered wrong."""
    from utils.eval.benchmarks import bfcl_system_prompt, paths

    root = paths(args.questions)["bfcl"]
    path = next((p for p in (Path("/content/T/thinkingcap/nofit_glaive_hard.jsonl"),
                             Path(__file__).with_name("nofit_glaive_hard.jsonl")) if p.exists()), None)
    assert path, "nofit_glaive_hard.jsonl not found"
    for k, r in enumerate(json.loads(l) for l in path.read_text().splitlines() if l.strip()):
        yield {"source": "glaive_tools", "origin": "near-hard-glaive", "task": "tool", "answer": "", "grader": "bfcl",
               "category": "irrelevance", "function": r["functions"], "reference": None, "id": f"glaive_nh-{k}",
               "prompt": [{"role": "system", "content": bfcl_system_prompt(root, r["functions"])},
                          {"role": "user", "content": r["question"]}]}


def glaive_nofit_hard2(args, rng):
    """(12 Nov, user: enough hard no-fit questions to match the test in all 3 steps) the HARDEST look-alikes: a Glaive
    question offered ONE Glaive function ranked 1st-4th closest in meaning (never its own), kept only on Haiku's NO + NO
    (nofit_verify.py --hard --rank 1,4 --singles 1, pairs already judged in earlier batches skipped)."""
    from utils.eval.benchmarks import bfcl_system_prompt, paths

    root = paths(args.questions)["bfcl"]
    path = next((p for p in (Path("/content/T/thinkingcap/nofit_glaive_hard2.jsonl"),
                             Path(__file__).with_name("nofit_glaive_hard2.jsonl")) if p.exists()), None)
    assert path, "nofit_glaive_hard2.jsonl not found"
    for k, r in enumerate(json.loads(l) for l in path.read_text().splitlines() if l.strip()):
        yield {"source": "glaive_tools", "origin": "nearest-hard-glaive", "task": "tool", "answer": "", "grader": "bfcl",
               "category": "irrelevance", "function": r["functions"], "reference": None, "id": f"glaive_nh2-{k}",
               "prompt": [{"role": "system", "content": bfcl_system_prompt(root, r["functions"])},
                          {"role": "user", "content": r["question"]}]}


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
    which = [x for x in getattr(args, "ifeval_bases", "no_robots").split(",") if x]
    bases = []
    if "no_robots" in which:
        bases += [r["prompt"].strip() for r in load_dataset("HuggingFaceH4/no_robots", split="train")
                  if r["category"] in ("Generation", "Brainstorm", "Open QA", "Rewrite", "Summarize")
                  and 8 <= len(r["prompt"].split()) <= 150]
    if "dolly" in which:  # (12 Nov) human-written requests (Databricks staff), long enough to be a real task (dp3: short ones
        # like "Why do you love Biriyani?" made the model think too little), no given context
        bases += [r["instruction"].strip() for r in load_dataset("databricks/databricks-dolly-15k", split="train")
                  if r["category"] in ("creative_writing", "brainstorming", "open_qa", "general_qa") and not r["context"]
                  and 12 <= len(r["instruction"].split()) <= 150]
    if "oasst1" in which:  # (12 Nov) human-written first turns of OpenAssistant conversations (English, no code)
        bases += [r["text"].strip() for r in load_dataset("OpenAssistant/oasst1", split="train")
                  if r["role"] == "prompter" and r["parent_id"] is None and r["lang"] == "en" and "```" not in r["text"]
                  and 12 <= len(r["text"].split()) <= 150]
    if "haiku" in which:  # (14 Nov, user: "for any missing data, you can generate it with haiku") gen_ifeval_requests.py
        path = next((p_ for p_ in (Path("/content/T/thinkingcap/ifeval_requests_haiku.jsonl"),
                                   Path(__file__).with_name("ifeval_requests_haiku.jsonl")) if p_.exists()), None)
        assert path, "ifeval_requests_haiku.jsonl not found"
        bases += [json.loads(l)["text"] for l in path.read_text().splitlines() if l.strip()]
    if "oasst2" in which:  # (13 Nov, user: step 2 short of IFEval) OpenAssistant's 2nd release: more human-written first turns
        try:
            bases += [r["text"].strip() for r in load_dataset("OpenAssistant/oasst2", split="train")
                      if r["role"] == "prompter" and r["parent_id"] is None and r["lang"] == "en" and "```" not in r["text"]
                      and 12 <= len(r["text"].split()) <= 150]
        except Exception as e:  # a source that will not load must not stop the others
            print(f"ifeval_built: oasst2 not loaded ({type(e).__name__}: {e})", flush=True)
    if "wildchat" in which:  # (13 Nov) real users' first requests (WildChat-1M): English, not flagged toxic, streamed
        try:
            got = 0
            for r in load_dataset("allenai/WildChat-1M", split="train", streaming=True):
                if got >= 40000:
                    break
                conv = r.get("conversation") or []
                if r.get("language") != "English" or r.get("toxic") or not conv or conv[0].get("role") != "user":
                    continue
                t = (conv[0].get("content") or "").strip()
                if "```" in t or not 12 <= len(t.split()) <= 150:
                    continue
                bases.append(t)
                got += 1
        except Exception as e:
            print(f"ifeval_built: wildchat not loaded ({type(e).__name__}: {e})", flush=True)
    coding = re.compile(r"\b(python|code|coding|script|function|program|programming|sql|javascript|java|c\+\+|html|css|"
                        r"regex|bash|api|json|excel formula)\b", re.I)
    bases = list(dict.fromkeys(b for b in bases if not coding.search(b)))  # writing tasks like IFEval's; each request once
    bases = bases * (1 + n // max(1, len(bases)))  # ~6,000 requests: each may carry up to two different instruction sets
    built = ifeval_fill(args, rng, want_n, want_id, Counter(), Counter(), n, bases=bases)
    print(f"ifeval_built: {len(built)} prompts from {len(set(bases))} requests ({which})", flush=True)
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
        built.append({"source": "ifeval_like", "origin": "built", "base": base, "task": "instructions", "answer": "",
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
        if args.tool_clean and (not bfcl_types(functions) or not grounded(calls[0]["arguments"], turns["human"])):
            continue  # dp6: the same clean-up as ToolACE (BFCL type names, values stated in the question)
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


# ---------------------------------------------------------------------------------------------------------------------------
# (11 Nov, user: "how much does this transfer ... or is it benchmark overfitting?") TRANSFER sets: benchmarks never trained on
# and never imitated by our training data. Same answer formats as the training helpers (Final answer: lines; MBPP-style code).
MATH_INSTRUCTION_TR = "Put the final answer on the last line, like: Final answer: 42"
CHOICE_INSTRUCTION_TR = "Answer with the label of the correct option. Put it on the last line, like: Final answer: B"


def _tr_row(question, instruction, answer, task, source, qid):
    return {"prompt": [{"role": "user", "content": f"{question}\n\n{instruction}"}], "answer": answer, "task": task,
            "source": source, "pool_id": qid}


def tr_math500(args, rng):
    """MATH-500 (competition math, harder than GSM8K / SVAMP), the problems with a whole-number answer."""
    from datasets import load_dataset
    for r in load_dataset("HuggingFaceH4/MATH-500", split="test"):
        a = str(r["answer"]).strip()
        if re.fullmatch(r"-?\d+", a):
            yield _tr_row(r["problem"], MATH_INSTRUCTION_TR, a, "math", "tr_math500", f"tr-math500-{r['unique_id']}") | {"level": r["level"]}


def tr_aime25(args, rng):
    """(12 Nov, user) AIME 2025 (I + II, 30 problems, integer answers), the benchmark of 'The Art of Efficient Reasoning'
    (arXiv 2602.20945, Table 2: Qwen3-0.6B 14.9k -> 8.9k tokens, Mean@8 13.3 -> 24.6 at a 32k budget)."""
    from datasets import load_dataset
    rows = None
    for repo, kw in (("math-ai/aime25", {"split": "test"}), ("yentinglin/aime_2025", {"split": "train"}),
                     ("opencompass/AIME2025", None)):
        try:
            if kw is None:
                rows = [r for cfg in ("AIME2025-I", "AIME2025-II") for r in load_dataset(repo, cfg, split="test")]
            else:
                rows = list(load_dataset(repo, **kw))
            if rows:
                print(f"tr_aime25: {len(rows)} problems from {repo}", flush=True)
                break
        except Exception as e:  # try the next public copy
            print(f"tr_aime25: {repo} failed: {e}", flush=True)
    for i, r in enumerate(rows or []):
        q = r.get("problem") or r.get("question")
        a = str(r.get("answer")).strip()
        if q and re.fullmatch(r"-?\d+", a):
            yield _tr_row(q, MATH_INSTRUCTION_TR, str(int(a)), "math", "tr_aime25", f"tr-aime25-{r.get('id', i)}")


def tr_skywork(args, rng):
    """Skywork-OR1 math (the README's suggested 'maths with headroom'), the easy band with whole-number answers; never used."""
    from datasets import load_dataset
    for k, r in enumerate(load_dataset("Skywork/Skywork-OR1-RL-Data", split="math")):
        gold = json.loads(r["reward_model"]["ground_truth"])
        if len(gold) == 1 and re.fullmatch(r"-?\d+", gold[0].strip()) \
                and r["extra_info"]["model_difficulty"]["DeepSeek-R1-Distill-Qwen-1.5B"] <= 8:
            yield _tr_row(r["prompt"][-1]["content"], MATH_INSTRUCTION_TR, gold[0].strip(), "math", "tr_skywork", f"tr-sky-{k}")


def _options(question, labels, texts):
    return question + "\n\n" + "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))


def tr_arc_c(args, rng):
    """ARC-Challenge (science questions; our training never had multiple choice), test split."""
    from datasets import load_dataset
    for r in load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test"):
        yield _tr_row(_options(r["question"], r["choices"]["label"], r["choices"]["text"]), CHOICE_INSTRUCTION_TR,
                      r["answerKey"], "multiple_choice", "tr_arc_c", f"tr-arcc-{r['id']}")


def tr_csqa(args, rng):
    """CommonsenseQA, validation split."""
    from datasets import load_dataset
    for r in load_dataset("tau/commonsense_qa", split="validation"):
        yield _tr_row(_options(r["question"], r["choices"]["label"], r["choices"]["text"]), CHOICE_INSTRUCTION_TR,
                      r["answerKey"], "multiple_choice", "tr_csqa", f"tr-csqa-{r['id']}")


def tr_mmlu(args, rng):
    """MMLU (57 subjects of knowledge questions), test split."""
    from datasets import load_dataset
    for k, r in enumerate(load_dataset("cais/mmlu", "all", split="test")):
        yield _tr_row(_options(f"({r['subject'].replace('_', ' ')}) {r['question']}", "ABCD", r["choices"]),
                      CHOICE_INSTRUCTION_TR, "ABCD"[r["answer"]], "multiple_choice", "tr_mmlu", f"tr-mmlu-{k}")


def tr_humaneval(args, rng):
    """HumanEval (code from a different source than MBPP / KodCode), graded by its own check() in MBPP's harness."""
    from datasets import load_dataset
    from utils.eval.benchmarks import MBPP_INSTRUCTION, MBPP_NAME_REMINDER
    for r in load_dataset("openai/openai_humaneval", split="test"):
        name = r["entry_point"]
        content = f"{MBPP_INSTRUCTION}\n\n{r['prompt']}\n{MBPP_NAME_REMINDER.format(name=name)}"
        yield {"source": "tr_humaneval", "task": "code", "answer": "", "grader": "code", "entry_point": name,
               "tests": [r["test"], f"check({name})"], "setup": r["prompt"].split("def " + name)[0],
               "prompt": [{"role": "user", "content": content}], "pool_id": f"tr-he-{r['task_id']}"}


# ---------------------------------------------------------------------------------------------------------------------------
# (12 Nov) TRANSFER v2 (track.py TRANSFER_V2_SIZES): the types the first transfer set lacks, graded by the same checkers the
# test uses. Tool calls: BFCL's LIVE splits (user-contributed, never in the test's non-live BFCL) at the harness's own gorilla
# commit, graded by its AST checker. Multiple choice: BIG-Bench Hard (option tasks) and BoolQ in tr_arc_c's format. Instructions:
# IFBench (58 instruction types new to IFEval), graded by its own verifiers (typed.py grader "ifbench").

def reference_reply(reference):
    """A BFCL possible answer -> the reply that states it: the first allowed value of each argument, optional ones left out."""
    (name, arguments), = reference[0].items()
    given = ", ".join(f"{k}={values[0]!r}" for k, values in arguments.items() if values and values[0] != "")
    return f"[{name}({given})]"


# The test's BFCL prompts reach 6,123 characters; live ones 40k (dozens of functions), which with a 32,768-token answer cap would
# overflow Qwen3-0.6B's 40,960-token context in one vLLM engine. 12,000 characters (about 3-4k tokens) keeps 93% / 97% of the
# gradable multiple / no-fit questions and all simple ones.
BFCL_LIVE_MAX_CHARS = 12000


def _bfcl_live(args, rng, category):
    """BFCL v3 live_<category> (named BFCL_v4_live_* at the pinned commit; same questions), python functions. Kept: one user
    message (no extra system turn), a prompt of at most BFCL_LIVE_MAX_CHARS, and for simple / multiple one gold call that
    BFCL's own checker accepts when stated as a reply (reference_reply), so every kept question is gradable."""
    from utils.eval.benchmarks import REVISIONS, bfcl_system_prompt, collect, paths, read_jsonl
    from utils.eval.graders import grade_call, load_bfcl

    root = paths(args.questions)["bfcl"]
    files = [f"bfcl_eval/data/BFCL_v4_live_{category}.json"]
    if category != "irrelevance":
        files.append(f"bfcl_eval/data/possible_answer/BFCL_v4_live_{category}.json")
    collect("https://raw.githubusercontent.com/ShishirPatil/gorilla/"
            f"{REVISIONS['bfcl']}/berkeley-function-call-leaderboard/", root, files)
    data = root / "bfcl_eval/data"
    references = {} if category == "irrelevance" else {
        r["id"]: r["ground_truth"] for r in read_jsonl(data / f"possible_answer/BFCL_v4_live_{category}.json")}
    checker, language = load_bfcl(root)
    for task in read_jsonl(data / f"BFCL_v4_live_{category}.json"):
        turns = task["question"]
        if len(turns) != 1 or [m["role"] for m in turns[0]] != ["user"]:
            continue
        row = {"source": f"tr_bfcl_live_{category}", "task": "tool", "answer": "", "grader": "bfcl", "category": category,
               "function": task["function"], "reference": references.get(task["id"]),
               "prompt": [{"role": "system", "content": bfcl_system_prompt(root, task["function"])}, turns[0][0]],
               "pool_id": f"tr-{task['id']}"}
        if sum(len(m["content"]) for m in row["prompt"]) > BFCL_LIVE_MAX_CHARS:
            continue
        if category != "irrelevance":
            if len(row["reference"]) != 1:
                continue
            try:
                if not grade_call(row, reference_reply(row["reference"]), checker, language)[0]:
                    continue
            except Exception:  # a schema the checker cannot type
                continue
        yield row


def tr_bfcl_live_simple(args, rng):
    yield from _bfcl_live(args, rng, "simple")


def tr_bfcl_live_multiple(args, rng):
    yield from _bfcl_live(args, rng, "multiple")


def tr_bfcl_live_irrelevance(args, rng):
    yield from _bfcl_live(args, rng, "irrelevance")


BBH_OPTION = re.compile(r"^\(([A-Z])\)\s*(.*)$")


def tr_bbh(args, rng, per_task=10):
    """BIG-Bench Hard (lukaemon/bbh, test), the 17 tasks whose answer is one of the listed options, re-labelled "A. text" as
    tr_arc_c; per_task questions per task drawn with rng (questions whose target is not an option label are dropped first)."""
    from datasets import get_dataset_config_names, load_dataset
    for task in sorted(get_dataset_config_names("lukaemon/bbh")):
        usable = []
        for k, r in enumerate(load_dataset("lukaemon/bbh", task, split="test")):
            if "\nOptions:\n" not in r["input"]:
                continue
            question, options = r["input"].rsplit("\nOptions:\n", 1)
            found = [BBH_OPTION.match(line) for line in options.strip().split("\n")]
            target = re.fullmatch(r"\(([A-Z])\)", r["target"].strip())
            labels = [m.group(1) for m in found if m]
            if not all(found) or not target or labels != [chr(65 + i) for i in range(len(labels))] \
                    or target.group(1) not in labels:
                continue
            usable.append(_tr_row(_options(question.strip(), labels, [m.group(2).strip() for m in found]),
                                  CHOICE_INSTRUCTION_TR, target.group(1), "multiple_choice", "tr_bbh", f"tr-bbh-{task}-{k}")
                          | {"bbh_task": task})
        if len(usable) >= per_task:  # tasks without options (yes/no, numbers, words) have none
            yield from rng.sample(usable, per_task)


def tr_boolq(args, rng):
    """BoolQ (google/boolq, validation): a passage and a yes/no question, as two options A. True / B. False."""
    from datasets import load_dataset
    for k, r in enumerate(load_dataset("google/boolq", split="validation")):
        q = r["question"].strip()
        question = f"{r['passage'].strip()}\n\nQuestion: {q[:1].upper() + q[1:]}?"
        yield _tr_row(_options(question, "AB", ["True", "False"]), CHOICE_INSTRUCTION_TR, "A" if r["answer"] else "B",
                      "multiple_choice", "tr_boolq", f"tr-boolq-{k}")


def tr_ifbench(args, rng):
    """IFBench (allenai/IFBench_test, 300 prompts, 1-2 of 58 instruction types), graded by IFBench's own verifiers: none of
    its instruction ids is in IFEval's registry, so typed.py's grader "ifbench" (pinned IFBench commit) checks them."""
    from datasets import load_dataset
    for r in load_dataset("allenai/IFBench_test", split="train"):
        kwargs = [{k: v for k, v in (kw or {}).items() if v is not None} for kw in r["kwargs"]]
        yield {"source": "tr_ifbench", "task": "instructions", "answer": "", "grader": "ifbench",
               "id": f"tr-ifbench-{r['key']}", "pool_id": f"tr-ifbench-{r['key']}",
               "instruction_id_list": r["instruction_id_list"], "kwargs": kwargs, "original_prompt": r["prompt"],
               "prompt": [{"role": "user", "content": r["prompt"]}]}


if __name__ == "__main__":
    main()

