"""Which IFEval rules does a shortened model break? (user, 9 Nov: "why is IFEval not performing well?")

The records hold one verdict per answer (all of a question's rules must hold), so each rule TYPE is scored on the questions
that contain it: per rule type, the original's and each candidate's accuracy on those questions (all samples), the change,
and the median thinking / response tokens of the candidate's answers there. Length rules (length_constraints:*) also get
the response's word / sentence / paragraph count against the asked number, for the candidate's WRONG answers: below or
above the bound. Last, wrong answers split by thinking length: under half the original's median right-answer thinking on
the question, or not.

    python ifeval_rules.py --questions questions --runs runs/mvtf/full-base runs/mvtf/full-rc1-soup-30 runs/mvtf/full-rpE-80
    (the first run is the original Qwen)
"""
import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


def read(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def answers(run):
    out = defaultdict(list)
    for r in read(Path(run) / "records.jsonl"):
        if r["dataset"] == "ifeval":
            out[r["id"]].append(r)
    return out


def right(r):
    return bool(r.get("correct")) and not r.get("truncated")


def thinking(r):
    return r.get("reasoning_tokens") or 0


def response_tokens(r):
    return max(0, (r.get("output_tokens") or 0) - thinking(r))


def counts(text):
    text = text or ""
    return {"words": len(text.split()), "sentences": len([s for s in re.split(r"[.!?]+", text) if s.strip()]),
            "paragraphs": len([p for p in re.split(r"\n\s*\n|\*\*\*", text) if p.strip()])}


def length_check(rule, kwargs, response):
    """('below' | 'above' | 'ok' | None) for a length rule's asked number."""
    c = counts(response)
    unit = "words" if "words" in rule else "sentences" if "sentences" in rule else "paragraphs" if "paragraphs" in rule else None
    if not unit:
        return None
    n = kwargs.get("num_words") or kwargs.get("num_sentences") or kwargs.get("num_paragraphs")
    relation = kwargs.get("relation", "")
    if n is None:
        return None
    if relation == "less than":
        return "above" if c[unit] >= n else "ok"
    if relation == "at least" or "paragraph" in rule:
        return "below" if c[unit] < n else "ok"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--questions", default="questions")
    ap.add_argument("--runs", nargs="+", required=True, help="the original first, then the candidates")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    rows = {r["id"]: r for r in read(Path(args.questions) / "questions.jsonl") if r.get("dataset") == "ifeval"}
    runs = {Path(r).name: answers(r) for r in args.runs}
    base_name = Path(args.runs[0]).name
    base = runs[base_name]
    report = {}
    for name, cand in list(runs.items())[1:]:
        keys = [k for k in cand if k in base and k in rows]
        by_rule, by_family = defaultdict(list), defaultdict(list)
        for k in keys:
            for rule in rows[k]["instruction_id_list"]:
                by_rule[rule].append(k)
                by_family[rule.split(":")[0]].append(k)
        acc = lambda run, ks: 100 * statistics.fmean(right(a) for k in ks for a in run[k])  # noqa: E731
        print(f"\n== {name} vs {base_name}: {len(keys)} IFEval questions; all: {acc(base, keys):.1f} -> {acc(cand, keys):.1f}")
        fam_rows = []
        for fam, ks in sorted(by_family.items(), key=lambda x: -len(x[1])):
            ks = sorted(set(ks))
            d = acc(cand, ks) - acc(base, ks)
            th = statistics.median(thinking(a) for k in ks for a in cand[k])
            th0 = statistics.median(thinking(a) for k in ks for a in base[k])
            rs = statistics.median(response_tokens(a) for k in ks for a in cand[k])
            rs0 = statistics.median(response_tokens(a) for k in ks for a in base[k])
            fam_rows.append({"family": fam, "questions": len(ks), "base": round(acc(base, ks), 1), "cand": round(acc(cand, ks), 1),
                             "change": round(d, 1), "thinking": [th0, th], "response": [rs0, rs]})
        print("  by rule family (questions containing it):")
        for r in fam_rows:
            print(f"    {r['family']:<22} n={r['questions']:<4} {r['base']:5.1f} -> {r['cand']:5.1f} ({r['change']:+5.1f})  "
                  f"thinking {r['thinking'][0]:>5.0f} -> {r['thinking'][1]:>4.0f}  response {r['response'][0]:>4.0f} -> {r['response'][1]:>4.0f}")
        rule_rows = []
        for rule, ks in by_rule.items():
            if len(ks) >= 8:
                rule_rows.append((acc(cand, ks) - acc(base, ks), rule, len(ks), acc(base, ks), acc(cand, ks)))
        rule_rows.sort()
        print("  rules that lost most (>= 8 questions):")
        for d, rule, n, a0, a1 in rule_rows[:10]:
            print(f"    {rule:<45} n={n:<4} {a0:5.1f} -> {a1:5.1f} ({d:+5.1f})")
        print("  rules that gained most:")
        for d, rule, n, a0, a1 in rule_rows[-4:][::-1]:
            print(f"    {rule:<45} n={n:<4} {a0:5.1f} -> {a1:5.1f} ({d:+5.1f})")
        length = defaultdict(lambda: defaultdict(int))
        for k in keys:
            q = rows[k]
            for rule, kw in zip(q["instruction_id_list"], q.get("kwargs") or [{}] * len(q["instruction_id_list"])):
                if not rule.startswith("length_constraints"):
                    continue
                for who, run in (("original", base), (name, cand)):
                    for a in run[k]:
                        if not right(a):
                            v = length_check(rule, kw or {}, a.get("response"))
                            if v:
                                length[who][v] += 1
        if length:
            print("  wrong answers on length rules, response vs the asked number: " +
                  "; ".join(f"{who}: " + ", ".join(f"{v} {n}" for v, n in sorted(d.items())) for who, d in length.items()))
        short_think = {"original": [0, 0], name: [0, 0]}
        for k in keys:
            rt = [thinking(a) for a in base[k] if right(a)]
            if not rt:
                continue
            med = statistics.median(rt)
            for who, run in (("original", base), (name, cand)):
                for a in run[k]:
                    if not right(a):
                        short_think[who][0] += thinking(a) < med / 2
                        short_think[who][1] += 1
        print("  wrong answers thinking < half the original's right-answer median: " +
              "; ".join(f"{who} {s}/{n}" for who, (s, n) in short_think.items()))
        report[name] = {"families": fam_rows, "rules_lost": [r[1:] for r in rule_rows[:10]],
                        "length_wrong": {w: dict(d) for w, d in length.items()}, "short_thinking_wrong": short_think}
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "ifeval_rules.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
