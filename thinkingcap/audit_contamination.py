"""Audit: every training question ever prepared, against every evaluation question (rule 2).

For each training question (the last user message of each row of the given data files), the share of
its word 8-grams found in one single evaluation question, and whether any 13-gram matches exactly.
Writes a report: per file and source, how many questions reach 30% / 50% overlap or share a 13-gram,
and the closest pairs side by side, so a person can judge them.

    python audit_contamination.py --data data/*.jsonl --out runs/audit
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def grams(ws, n):
    return {" ".join(ws[i:i + n]) for i in range(max(0, len(ws) - n + 1))}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", nargs="+", required=True, help="training data files (prepare_types.py rows)")
    parser.add_argument("--questions", default="questions/questions.jsonl")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    index8, index13, tests = defaultdict(set), defaultdict(set), {}
    rows = [json.loads(l) for l in Path(args.questions).read_text().splitlines() if l.strip()]
    texts = {q["id"]: [m["content"] for m in q["messages"] if m["role"] == "user"][-1] for q in rows}
    # boilerplate (e.g. MBPP+'s instruction, shared by every code question and by our code prompts) is not overlap
    line_count = Counter(l.strip() for t in texts.values() for l in set(t.splitlines()) if l.strip())
    boiler = {l for l, c in line_count.items() if c >= 50}
    strip = lambda t: "\n".join(l for l in t.splitlines() if l.strip() not in boiler)
    for q in rows:
        text = strip(texts[q["id"]])
        tests[q["id"]] = (q["dataset"], text)
        ws = words(text)
        for g in grams(ws, 8):
            index8[g].add(q["id"])
        for g in grams(ws, 13):
            index13[g].add(q["id"])

    lines, closest = [], []
    flagged = Counter()
    for path in args.data:
        counts = defaultdict(Counter)
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            text = strip(row["prompt"][-1]["content"])
            ws = words(text)
            mine = grams(ws, 8)
            hits = Counter(t for g in mine for t in index8.get(g, ()))
            best, share = (hits.most_common(1)[0][0], hits.most_common(1)[0][1] / len(mine)) if hits and mine else (None, 0.0)
            exact = any(index13.get(g) for g in grams(ws, 13))
            c = counts[row.get("source", "?")]
            c["rows"] += 1
            c[">=30%"] += share >= 0.3
            c[">=50%"] += share >= 0.5
            c["13-gram"] += exact
            if best:
                closest.append((share, exact, Path(path).name, row.get("source"), text, best))
        for source, c in sorted(counts.items()):
            lines.append(f"{Path(path).name:<28}{source:<16}{c['rows']:>7}{c['>=30%']:>8}{c['>=50%']:>8}{c['13-gram']:>9}")
            flagged[">=50%"] += c[">=50%"]
    header = [f"{'file':<28}{'source':<16}{'rows':>7}{'>=30%':>8}{'>=50%':>8}{'13-gram':>9}"]
    report = ["Training questions vs the 2,759 evaluation questions (share of a training question's word 8-grams",
              "found in ONE evaluation question; 13-gram = an exact 13-word run shared with some evaluation question).",
              "Training rows used by the rounds are those with < 50% (the contamination check drops the rest).", ""]
    report += header + lines + ["", "The 15 closest pairs (training question | closest evaluation question):"]
    for share, exact, name, source, text, test in sorted(closest, reverse=True)[:15]:
        report.append(f"- {share:.0%}{' (13-gram)' if exact else ''} {name}/{source}: {text[:160]!r}")
        report.append(f"    ~ {test} ({tests[test][0]}): {tests[test][1][:160]!r}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "contamination_report.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
