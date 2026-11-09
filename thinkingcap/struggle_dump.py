"""Which full-test questions did a candidate lose, and why? (user, 9 Nov: "get a lot of data on the questions it struggled
and why, we will try to repair")

From the full evaluations (records.jsonl: per sample the full text, the answer after </think>, output and thinking
tokens, truncated, boundary_error, parse_error, correct) of the original Qwen and a candidate, plus the questions file:

  1. every wrong candidate answer gets ONE reason, checked in this order:
       cut off        hit the token limit (truncated)
       broken format  unbalanced <think> tags (boundary_error)
       unreadable     the grader could not read an answer (parse_error)
       no thinking    fewer than 20 thinking tokens
       stopped early  thinking < 1/3 of the original's median thinking on its RIGHT answers to the same question
       wrong answer   anything else: finished, readable, thought for a normal length, and still wrong
     and the same for the original's wrong answers, so each reason shows as before -> after, per part;
  2. regressed questions = the candidate solves at least 3 of 8 fewer than the original; per part their count, the
     reasons of their wrong answers, the thinking ratio (candidate wrong / original right, median), and for math the
     candidate's final number vs the gold one (off by a little, off by a lot, no number);
  3. regressed.jsonl: per regressed question the question text, gold answer, both solve rates and mean tokens, one
     right answer of the original and one wrong answer of the candidate (full text), with their reasons.

    python struggle_dump.py --base runs/mvtf/full-base --cand runs/mvtf/full-rc1-soup-30 --questions questions \\
        --out runs/mvtf/st-rc1-soup-30
"""

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

PARTS = ("svamp", "mbpp", "ifeval", "bfcl_simple_python", "bfcl_multiple", "bfcl_irrelevance")
REASONS = ("cut off", "broken format", "unreadable", "no thinking", "stopped early", "wrong answer")


def read(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def by_question(run):
    out = defaultdict(list)
    for r in read(Path(run) / "records.jsonl"):
        out[(r["dataset"], r["id"])].append(r)
    return out


def right(r):
    return bool(r.get("correct")) and not r.get("truncated")


def reason(r, right_thinking):
    """Why a wrong answer is wrong (the first that applies)."""
    if r.get("truncated"):
        return "cut off"
    if r.get("boundary_error"):
        return "broken format"
    if r.get("parse_error"):
        return "unreadable"
    thinking = r.get("reasoning_tokens") or 0
    if thinking < 20:
        return "no thinking"
    if right_thinking and thinking < right_thinking / 3:
        return "stopped early"
    return "wrong answer"


def question_text(row):
    for key in ("messages", "prompt"):
        if isinstance(row.get(key), list):
            users = [m["content"] for m in row[key] if m.get("role") == "user"]
            if users:
                return users[-1] if isinstance(users[-1], str) else json.dumps(users[-1])
    for key in ("question", "text", "prompt", "Body"):
        if isinstance(row.get(key), str):
            return row[key]
    return json.dumps({k: v for k, v in row.items() if k not in ("dataset", "id")})[:2000]


def last_number(text):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text or "")
    try:
        return float(nums[-1].replace(",", "")) if nums else None
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", required=True)
    parser.add_argument("--cand", required=True)
    parser.add_argument("--questions", default="questions")
    parser.add_argument("--drop", type=int, default=3, help="regressed = solved at least this many of 8 fewer")
    parser.add_argument("--show", type=int, default=3, help="examples printed per part (all are written to the file)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    base, cand = by_question(args.base), by_question(args.cand)
    rows = {(r["dataset"], r["id"]): r for r in read(Path(args.questions) / "questions.jsonl")}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary, examples = {}, []
    for part in PARTS:
        keys = sorted(k for k in base if k[0] == part and k in cand)
        if not keys:
            continue
        reasons = {"original": Counter(), "candidate": Counter()}
        regressed, reg_reasons, ratios, math_gap = [], Counter(), [], Counter()
        for k in keys:
            b, c = base[k], cand[k]
            b_right_think = [r.get("reasoning_tokens") or 0 for r in b if right(r)]
            median_right = statistics.median(b_right_think) if b_right_think else None
            for who, answers in (("original", b), ("candidate", c)):
                for r in answers:
                    if not right(r):
                        reasons[who][reason(r, median_right)] += 1
            drop = sum(map(right, b)) - sum(map(right, c))
            if drop < args.drop:
                continue
            wrong = [r for r in c if not right(r)]
            regressed.append(k)
            for r in wrong:
                reg_reasons[reason(r, median_right)] += 1
                if median_right:
                    ratios.append((r.get("reasoning_tokens") or 0) / median_right)
                if part == "svamp":
                    got, gold = last_number(r.get("response") or r.get("text")), last_number(str(rows.get(k, {}).get("answer", "")))
                    math_gap["no number" if got is None or gold is None else "off by <= 10%" if
                             abs(got - gold) <= 0.1 * max(1.0, abs(gold)) else "off by more"] += 1
            show_right = next((r for r in b if right(r)), None)
            show_wrong = min(wrong, key=lambda r: r.get("reasoning_tokens") or 0) if wrong else None
            examples.append({
                "part": part, "id": k[1], "question": question_text(rows.get(k, {})),
                "gold": rows.get(k, {}).get("answer"), "original_solved": f"{sum(map(right, b))}/{len(b)}",
                "candidate_solved": f"{sum(map(right, c))}/{len(c)}",
                "original_tokens": round(statistics.fmean(r["output_tokens"] for r in b)),
                "candidate_tokens": round(statistics.fmean(r["output_tokens"] for r in c)),
                "candidate_reasons": dict(Counter(reason(r, median_right) for r in wrong)),
                "original_right_text": show_right and show_right.get("text"),
                "candidate_wrong_text": show_wrong and show_wrong.get("text"),
                "candidate_wrong_reason": show_wrong and reason(show_wrong, median_right),
                "candidate_wrong_thinking": show_wrong and show_wrong.get("reasoning_tokens")})
        n_b = sum(len(base[k]) for k in keys)
        summary[part] = {
            "questions": len(keys), "regressed": len(regressed),
            "wrong_reasons_original": {x: reasons["original"][x] for x in REASONS},
            "wrong_reasons_candidate": {x: reasons["candidate"][x] for x in REASONS},
            "regressed_reasons": dict(reg_reasons),
            "regressed_thinking_ratio_median": round(statistics.median(ratios), 2) if ratios else None,
            "math_answer_gap": dict(math_gap) if part == "svamp" else None}
        print(f"\n== {part}: {len(keys)} questions, {len(regressed)} regressed (>= {args.drop}/8 fewer solved)")
        print("  wrong answers by reason (share of ALL answers), original -> candidate:")
        for x in REASONS:
            o, cnd = reasons["original"][x], reasons["candidate"][x]
            print(f"    {x:<14} {100 * o / n_b:5.1f}% -> {100 * cnd / n_b:5.1f}%  ({cnd - o:+d} answers)")
        if regressed:
            total = sum(reg_reasons.values())
            print("  on regressed questions, the candidate's wrong answers: " +
                  ", ".join(f"{x} {100 * reg_reasons[x] / total:.0f}%" for x in REASONS if reg_reasons[x]))
            if ratios:
                print(f"  their thinking vs the original's right answers (median ratio): {statistics.median(ratios):.2f}")
            if math_gap:
                print("  math, final number vs gold: " + ", ".join(f"{x} {n}" for x, n in math_gap.items()))
        for e in [e for e in examples if e["part"] == part][:args.show]:
            print(f"  -- {e['id']}: original {e['original_solved']}, candidate {e['candidate_solved']}, "
                  f"tokens {e['original_tokens']} -> {e['candidate_tokens']}, gold {str(e['gold'])[:60]!r}")
            print("     Q: " + " ".join(e["question"].split())[:300])
            print(f"     candidate wrong ({e['candidate_wrong_reason']}, {e['candidate_wrong_thinking']} thinking tokens), "
                  "end of its answer: " + " ".join((e["candidate_wrong_text"] or "").split())[-400:])
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "regressed.jsonl").write_text("".join(json.dumps(e) + "\n" for e in examples))
    print(f"\nwrote {out / 'summary.json'} and {out / 'regressed.jsonl'} ({len(examples)} regressed questions)")


if __name__ == "__main__":
    main()
