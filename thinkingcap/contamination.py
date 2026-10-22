"""Check training questions against the evaluation's questions before training on them (rule 2).

A training question is flagged when at least --share of its word 8-grams appear in one single
evaluation question: a near copy, not just shared boilerplate (BFCL's system prompt and IFEval's
constraint wordings are shared by design, so only the last user message is compared). Flagged
rows are dropped from the SFT file and listed.

    python contamination.py --data runs/bon/select/sft.jsonl --rollouts runs/bon/probe/rollouts.jsonl
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def grams(text, n=8):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", required=True, help="sft.jsonl to clean in place")
    parser.add_argument("--rollouts", required=True, help="rollouts.jsonl holding each id's question text")
    parser.add_argument("--questions", default="questions/questions.jsonl")
    parser.add_argument("--share", type=float, default=0.5)
    args = parser.parse_args()

    index = defaultdict(set)
    long_runs = set()  # every 13-word run of the evaluation questions: sharing one exactly also counts
    tests = {}
    rows_q = [json.loads(l) for l in Path(args.questions).read_text().splitlines() if l.strip()]
    texts = {q["id"]: [m["content"] for m in q["messages"] if m["role"] == "user"][-1] for q in rows_q}
    # boilerplate shared by many evaluation questions (MBPP+'s instruction, also in our code prompts) is not overlap
    line_count = Counter(l.strip() for t in texts.values() for l in set(t.splitlines()) if l.strip())
    boiler = {l for l, n in line_count.items() if n >= 50}
    strip = lambda t: "\n".join(l for l in t.splitlines() if l.strip() not in boiler)
    for q in rows_q:
        text = strip(texts[q["id"]])
        tests[q["id"]] = (q["dataset"], text)
        for g in grams(text):
            index[g].add(q["id"])
        long_runs.update(grams(text, 13))
    question_of = {}
    for line in Path(args.rollouts).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            question_of.setdefault(r["id"], r["question"])

    rows = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]
    kept, flagged = [], []
    for row in rows:
        mine = grams(strip(question_of.get(row["id"], "")))
        hits = Counter(t for g in mine for t in index.get(g, ()))
        best = hits.most_common(1)
        share = best[0][1] / len(mine) if best and mine else 0.0
        if share >= args.share or grams(strip(question_of.get(row["id"], "")), 13) & long_runs:
            flagged.append((row["id"], row["source"], best[0][0], round(share, 2)))
        else:
            kept.append(row)
    Path(args.data).write_text("".join(json.dumps(r) + "\n" for r in kept))
    print(f"contamination: {len(rows)} rows checked against {len(tests)} evaluation questions, "
          f"{len(flagged)} flagged and dropped", flush=True)
    for qid, source, test_id, share in flagged[:20]:
        print(f"   {qid} ({source}) ~ {test_id} ({tests[test_id][0]}): {share:.0%} of its 8-grams", flush=True)


if __name__ == "__main__":
    main()
