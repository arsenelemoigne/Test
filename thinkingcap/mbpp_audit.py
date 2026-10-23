"""Audit the MBPP rows the gentle best-of-8 SFT trained on (runs/bon/select-r1/sft.jsonl) against MBPP+.

MBPP+ is built from MBPP, and MBPP holds some of its problems twice under other ids, so beyond the word-overlap check
every training task is compared with every MBPP+ task on three signals: share of word 8-grams found in one MBPP+ task,
any exact 13-word run, the same function name, and the wording similarity of the task description (difflib ratio on
lower-cased words, after the shared instruction lines are removed). Prints the counts and the closest pairs side by side.

    python mbpp_audit.py --sft runs/bon/select-r1/sft.jsonl --rollouts runs/bon/probe/rollouts.jsonl
"""

import argparse
import difflib
import json
import re
from collections import Counter
from pathlib import Path


def words(text):
    return re.findall(r"[a-z0-9_]+", text.lower())


def grams(ws, n):
    return {" ".join(ws[i:i + n]) for i in range(max(0, len(ws) - n + 1))}


def function_names(text):
    return set(re.findall(r"\b([a-z_][a-z0-9_]*)\s*\(", text.lower())) - {"print", "assert", "len", "range", "int", "str"}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sft", default="runs/bon/select-r1/sft.jsonl")
    parser.add_argument("--rollouts", default="runs/bon/probe/rollouts.jsonl")
    parser.add_argument("--questions", default="questions/questions.jsonl")
    parser.add_argument("--source", default="mbpp_train")
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    tests = [json.loads(l) for l in Path(args.questions).read_text().splitlines() if l.strip()]
    tests = [q for q in tests if q["dataset"] == "mbpp"]
    text_of = lambda q: [m["content"] for m in q["messages"] if m["role"] == "user"][-1]
    test_texts = {q["id"]: text_of(q) for q in tests}
    lines = Counter(l.strip() for t in test_texts.values() for l in set(t.splitlines()) if l.strip())
    boiler = {l for l, n in lines.items() if n >= 50}
    strip = lambda t: "\n".join(l for l in t.splitlines() if l.strip() not in boiler)
    test_words = {k: words(strip(t)) for k, t in test_texts.items()}
    test_names = {k: function_names(t) for k, t in test_texts.items()}

    ids = {r["id"] for r in map(json.loads, Path(args.sft).read_text().splitlines()) if r.get("source") == args.source}
    question = {}
    for line in Path(args.rollouts).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["id"] in ids:
                question.setdefault(r["id"], r["question"])
    print(f"{len(ids)} {args.source} rows in {args.sft}; {len(question)} with their question text; "
          f"{len(tests)} MBPP+ evaluation tasks", flush=True)

    results = []
    for qid, text in question.items():
        ws = words(strip(text))
        mine8, mine13, names = grams(ws, 8), grams(ws, 13), function_names(text)
        best = None
        for tid, tw in test_words.items():
            share = len(mine8 & grams(tw, 8)) / max(1, len(mine8))
            exact = bool(mine13 & grams(tw, 13))
            same_name = bool(names & test_names[tid])
            ratio = difflib.SequenceMatcher(None, ws, tw).ratio()
            score = (exact, same_name and ratio >= 0.5, ratio, share)
            if best is None or score > best[0]:
                best = (score, tid, share, exact, same_name, ratio)
        results.append((qid, text) + best[1:])

    count = lambda f: sum(1 for r in results if f(r))
    print(f"8-gram share >= 50%: {count(lambda r: r[3] >= 0.5)}; >= 30%: {count(lambda r: r[3] >= 0.3)}", flush=True)
    print(f"exact 13-word run: {count(lambda r: r[4])}", flush=True)
    print(f"same function name as some MBPP+ task: {count(lambda r: r[5])}", flush=True)
    print(f"wording similarity >= 0.8: {count(lambda r: r[6] >= 0.8)}; >= 0.6: {count(lambda r: r[6] >= 0.6)}; "
          f">= 0.5 with the same function name: {count(lambda r: r[5] and r[6] >= 0.5)}", flush=True)
    print(f"\nclosest {args.top} (by: exact run, same name + similar wording, wording similarity):", flush=True)
    for qid, text, tid, share, exact, same_name, ratio in sorted(results, key=lambda r: (r[4], r[5] and r[6] >= 0.5, r[6]),
                                                                 reverse=True)[:args.top]:
        mine = " ".join(words(strip(text)))[:110]
        theirs = " ".join(test_words[tid])[:110]
        print(f"- {qid} ~ {tid}: sim {ratio:.2f}, 8-gram {share:.0%}, 13-run {exact}, same name {same_name}\n"
              f"    train: {mine}\n    test : {theirs}", flush=True)


if __name__ == "__main__":
    main()
