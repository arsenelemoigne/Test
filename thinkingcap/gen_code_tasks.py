"""(16 Nov, user: "(b) Have Argon write coding tasks in the same style as the test, each with a function signature, tests and
a reference solution that must pass its own tests before it's used") NEW MBPP-style code tasks for od5c / od5r's slices
20-24, written by Gemini 4 Argon (user's Google key). The hard KodCode tasks are used up.

What makes a code task hard for the original Qwen3-0.6B (its stored MBPP-style answers): short statements whose tests probe
exact behaviour (edge cases: empty input, ties, negatives, boundaries, duplicates, ordering, types), where its first idea is
almost right. Each task: a one-sentence MBPP-style statement, the function name, a reference solution, and 6-8 asserts.

Validation (on this Mac, before anything is used): the reference solution and ALL its asserts run in a fresh subprocess
(python -I, empty temporary folder, 10 s timeout); a task is kept only if every assert passes, the statement is short (MBPP
length) and the solution is short. Deduplicated by statement and by function name.
Key from the environment (never written): GOOGLE_API_KEY.

    python gen_code_tasks.py --calls 6 --seed 1 --out tools_tgt_code_argon_pilot.jsonl
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TOPICS = """string manipulation, list processing, dictionaries and counting, number theory, sorting with custom keys,
intervals and ranges, matrices and grids, bit manipulation, date-free calendar arithmetic, sequences and recurrences,
searching, sets and uniqueness, tuples and records, prefix sums, two pointers, greedy choices, parsing simple formats,
geometry with integers, combinatorics with small numbers, text statistics""".replace("\n", " ").split(", ")

PROMPT = """Create 10 Python programming tasks in the style of the MBPP benchmark. Topic: {topic}.
Each task is a ONE-sentence statement starting with "Write a function to" or "Write a python function to" (at most 30
words), solvable by a short function (at most 12 lines). Make them HARD for a small model in this specific way: the
statement looks simple, but the tests check exact behaviour that a quick solution gets wrong: edge cases (empty input,
single element, ties, negative numbers, zero, duplicates, boundaries, ordering of the result, return type such as list vs
tuple, int vs float). Do not make them long or algorithmically deep. Every behaviour the tests check must follow from the
statement itself (no hidden conventions), but you may state it briefly ("return them in ascending order", "return None
if ..."). English only.
{novel}Each task: "question" (the statement), "function_name" (snake_case), "solution" (the full reference function, no
imports unless needed, no printing), "tests" (6 to 8 assert lines calling the function, at least 3 of them edge cases).
Return ONLY a JSON array of 10 objects: {{"question": "...", "function_name": "...", "solution": "...", "tests": ["assert ...", ...]}}"""

PROMPT_MEDIUM = """Create 10 Python programming tasks in the style of the MBPP benchmark. Topic: {topic}.
Each task is a ONE-sentence statement starting with "Write a function to" or "Write a python function to" (at most 30
words), solvable by a short function (3 to 10 lines). Make them ORDINARY, MEDIUM-DIFFICULTY tasks, like most of MBPP: a
clear everyday task with one natural solution, no traps. The tests are like MBPP's: 5 or 6 asserts on normal inputs, at most
one simple edge case. A careful small model should get them right most of the time, but not always (a slip in an index, an
off-by-one, a wrong return type). Every behaviour the tests check must follow from the statement itself. English only.
{novel}Each task: "question" (the statement), "function_name" (snake_case), "solution" (the full reference function, no
imports unless needed, no printing), "tests" (5 or 6 assert lines calling the function).
Return ONLY a JSON array of 10 objects: {{"question": "...", "function_name": "...", "solution": "...", "tests": ["assert ...", ...]}}"""

NOVEL_MEDIUM = """INVENT NEW TASKS: do NOT reproduce or paraphrase known benchmark problems (MBPP, HumanEval, APPS, LeetCode or
Codeforces classics, textbook exercises such as Fibonacci, factorial, primes, palindromes, anagrams). Give each task a small
concrete setting of its own (shelf labels, ticket codes, sensor readings, playlist lengths, seat maps, ...) and word it in
your own way, not with stock phrases.
"""

NOVEL = """INVENT NEW TASKS: do NOT reproduce or paraphrase known benchmark problems (MBPP, HumanEval, APPS, LeetCode or
Codeforces classics, textbook exercises such as Fibonacci, factorial, primes, palindromes, anagrams, binomial
coefficients, derangements or Stirling numbers). Give each task a small concrete setting of its own (shelf labels, ticket
codes, sensor readings, playlist lengths, seat maps, ...) and word it in your own way, not with stock phrases.
Mix the difficulty: about half the tasks should be ones a careful small model gets right about half of the time (one
tricky edge case), the other half harder (two or three edge cases at once).
"""

COST = {"usd": 0.0}


def argon_openrouter(content, max_tokens=20000):
    """(20 Nov) The same Gemini 4 Argon through OpenRouter (key OPENROUTER_API_KEY from the environment, never written)."""
    body = {"model": "google/gemini-4-argon", "max_tokens": max_tokens, "temperature": 1.0,
            "reasoning": {"effort": "low"}, "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"],
                                          "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                d = json.loads(r.read())
            u = d.get("usage", {})
            COST["usd"] += float(u.get("cost") or (u.get("prompt_tokens", 0) * 2e-6 + u.get("completion_tokens", 0) * 10e-6))
            return ((d.get("choices") or [{}])[0].get("message", {}).get("content") or "").strip()
        except Exception:
            if attempt == 3:
                return ""
            time.sleep(10 * (attempt + 1))


def argon(content, max_tokens=20000):
    if not os.environ.get("GOOGLE_API_KEY") and os.environ.get("OPENROUTER_API_KEY"):  # (20 Nov) Google key out of credit
        return argon_openrouter(content, max_tokens)
    body = {"contents": [{"role": "user", "parts": [{"text": content}]}],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 1.0, "thinkingConfig": {"thinkingLevel": "low"}}}
    req = urllib.request.Request("https://generativelanguage.googleapis.com/v1beta/models/gemini-4-argon:generateContent",
                                 data=json.dumps(body).encode(),
                                 headers={"x-goog-api-key": os.environ["GOOGLE_API_KEY"], "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                d = json.loads(r.read())
            m = d.get("usageMetadata", {})
            COST["usd"] += ((m.get("promptTokenCount", 0) - m.get("cachedContentTokenCount", 0)) * 2e-6
                            + (m.get("candidatesTokenCount", 0) + m.get("thoughtsTokenCount", 0)) * 10e-6)
            parts = (d.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
        except Exception:
            if attempt == 3:
                return ""
            time.sleep(10 * (attempt + 1))


def parse(text):
    try:
        out = json.loads(text[text.find("["): text.rfind("]") + 1])
        return [x for x in out if isinstance(x, dict)]
    except ValueError:
        return []


def passes(item):
    """Run the reference solution and every assert in a fresh, isolated interpreter (10 s)."""
    code = item["solution"].rstrip() + "\n\n" + "\n".join(item["tests"]) + "\nprint('ALL_OK')\n"
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "task.py"
        f.write_text(code)
        try:
            r = subprocess.run([sys.executable, "-I", str(f)], cwd=d, capture_output=True, text=True, timeout=10,
                               env={"PATH": "/usr/bin:/bin"})
        except subprocess.TimeoutExpired:
            return False
    return r.returncode == 0 and "ALL_OK" in r.stdout


def well_formed(it):
    q, name, sol, tests = it.get("question"), it.get("function_name"), it.get("solution"), it.get("tests")
    if not (isinstance(q, str) and isinstance(name, str) and isinstance(sol, str) and isinstance(tests, list)):
        return False
    if not re.fullmatch(r"[a-z_][a-z0-9_]*", name) or f"def {name}(" not in sol:
        return False
    if len(q.split()) > 40 or len(sol.strip().splitlines()) > 16 or not 5 <= len(tests) <= 10:
        return False
    if any(not (isinstance(t, str) and t.strip().startswith("assert") and f"{name}(" in t) for t in tests):
        return False
    return not re.search(r"\b(input|open|exec|eval|__import__|os\.|sys\.|subprocess|socket)\b", sol + " ".join(tests))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--medium", action="store_true", help="(18 Nov, user) ordinary medium-difficulty MBPP-like tasks, test-like "
                    "style, for continuity between the slices (the edge-case batches made slice 20+ code harder than the test)")
    ap.add_argument("--novel", action="store_true", help="(16 Nov) new tasks only, mixed difficulty (most of the first batch "
                    "shared rare 8-word sequences with the test's code questions and was dropped by the contamination check)")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    topics = [rng.choice(TOPICS) for _ in range(a.calls)]
    with ThreadPoolExecutor(6) as ex:
        items = [dict(x, topic=t) for t, b in zip(topics, ex.map(lambda t: parse(argon((PROMPT_MEDIUM if a.medium else PROMPT).format(topic=t, novel=(NOVEL_MEDIUM if a.medium else NOVEL) if a.novel else ""))), topics)) for x in b]
    seen_q, seen_n, cands = set(), set(), []
    for it in items:
        if not well_formed(it):
            continue
        k = it["question"].strip().lower()
        if k in seen_q or it["function_name"] in seen_n:
            continue
        seen_q.add(k)
        seen_n.add(it["function_name"])
        cands.append(it)
    with ThreadPoolExecutor(8) as ex:
        ok = list(ex.map(passes, cands))
    kept = [it for it, g in zip(cands, ok) if g]
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in kept))
    print(f"argon code: {len(items)} generated, {len(cands)} well-formed and unique, {len(kept)} pass their own tests; "
          f"cost ${COST['usd']:.3f}", flush=True)


if __name__ == "__main__":
    main()
