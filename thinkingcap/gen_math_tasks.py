"""(18 Nov, user: slices 20-24 must resemble the eval set and rounds 1-19 on every metric) NEW SVAMP-style math word problems
written by Gemini 4 Argon (user's Google key): the easy SVAMP-like questions of our math pool are used up at slice 24
(the math left solves 85% vs the test's 89%).

Each problem: a short elementary story with whole numbers, one or two operations (+ - * /), a question; the answer is a whole
number. Argon also gives the arithmetic expression; a problem is kept only if that expression (numbers and + - * / only,
evaluated here with Python's ast, nothing executed) gives exactly the stated whole-number answer, every number in the
expression appears in the question, and the statement has SVAMP's length. Deduplicated by statement.
Key from the environment (never written): GOOGLE_API_KEY.

    python gen_math_tasks.py --calls 60 --seed 1 --out tools_tgt_math_argon_m1.jsonl
"""
import argparse
import ast
import json
import operator
import random
import re
from concurrent.futures import ThreadPoolExecutor

from gen_code_tasks import COST, argon, parse

SETTINGS = """a school trip, a bakery, a library, a garden, a farm, a toy shop, a football season, a bus route, a bird watch,
a fruit stand, a birthday party, a swimming pool, a bookshop, a zoo, a craft fair, a train journey, a classroom, a fishing
trip, a camping weekend, a pet shop, a parking lot, a music band, a marathon, a recycling drive, a science club""".replace(
    "\n", " ").split(", ")

PROMPT = """Write 10 elementary-school math word problems in the style of the SVAMP benchmark. Setting: {setting}.
Each problem is 2 or 3 short sentences (15 to 45 words in all) with whole numbers, and ends with one question. It needs ONE or
TWO arithmetic operations (+, -, *, /) and its answer is a whole number. Vary the structure like SVAMP does: some ask for a
total, some for a difference ("how many more"), some for what was there before ("how many did he have at first"), some
include one number that is not needed. Do not reuse well-known textbook problems; invent your own names and numbers.
For each problem give: "question" (the problem text), "expression" (the arithmetic that solves it, using only the numbers
of the question, + - * / and parentheses), "answer" (the whole-number result).
Return ONLY a JSON array of 10 objects: {{"question": "...", "expression": "...", "answer": 0}}"""

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def evaluate(expr):
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            return OPS[type(n.op)](ev(n.left), ev(n.right))
        raise ValueError("not plain arithmetic")
    return ev(ast.parse(expr, mode="eval"))


def ok(it):
    q, e, a = it.get("question"), it.get("expression"), it.get("answer")
    if not (isinstance(q, str) and isinstance(e, str)) or not 12 <= len(q.split()) <= 55:
        return False
    try:
        v, a = evaluate(e), float(a)
    except Exception:
        return False
    if a != int(a) or a < 0 or abs(v - a) > 1e-9:
        return False
    nums_q = set(re.findall(r"\d+(?:\.\d+)?", q.replace(",", "")))
    nums_e = re.findall(r"\d+(?:\.\d+)?", e)
    ops = len(re.findall(r"[-+*/]", e))
    return 1 <= ops <= 2 and all(n in nums_q for n in nums_e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    settings = [rng.choice(SETTINGS) for _ in range(a.calls)]
    with ThreadPoolExecutor(6) as ex:
        items = [dict(x, setting=s) for s, b in zip(settings, ex.map(lambda s: parse(argon(PROMPT.format(setting=s))), settings))
                 for x in b]
    seen, kept = set(), []
    for it in items:
        if not isinstance(it, dict):
            continue
        key = re.sub(r"\W+", " ", str(it.get("question", "")).lower()).strip()
        if key in seen or not ok(it):
            continue
        seen.add(key)
        kept.append({"question": it["question"].strip(), "expression": it["expression"], "answer": str(int(float(it["answer"]))),
                     "setting": it["setting"]})
    with open(a.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print(f"argon math: {len(items)} generated, {len(kept)} kept (whole-number answer checked from the expression); "
          f"cost ${COST['usd']:.3f}", flush=True)


if __name__ == "__main__":
    main()
