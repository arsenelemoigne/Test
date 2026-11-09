"""Where does a candidate lose accuracy and save tokens? (user, 9 Nov: "what type of math questions does it solve poorly,
in the length / difficulty distribution?")

From two full evaluations (2,759 questions x 8: the original Qwen and a candidate), per test part, questions are binned
  1. by difficulty = the ORIGINAL's solve rate over its 8 samples (never, 1-3/8, 4-6/8, 7/8, always), and
  2. by length = the original's mean output tokens on the question, in quartiles within the part (Q1 shortest),
and each bin reports: questions, the original's and the candidate's accuracy, the change, the bin's share of the part's
whole accuracy change, mean tokens of both, and tokens saved (1 - candidate sum / original sum). "Broken" = questions the
original solves >= 7/8 that the candidate solves <= 4/8.

    python difficulty_breakdown.py --base runs/mvtf/full-base --cand runs/mvtf/full-rc1-soup-30 --out runs/mvtf/bd-rc1-soup-30
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

PARTS = ("svamp", "mbpp", "ifeval", "bfcl_simple_python", "bfcl_multiple", "bfcl_irrelevance")
DIFFICULTY = (("never", 0, 0), ("1-3/8", 1, 3), ("4-6/8", 4, 6), ("7/8", 7, 7), ("always", 8, 8))


def per_question(run):
    """(dataset, id) -> list of (correct, output tokens), one per sample."""
    out = defaultdict(list)
    for line in (Path(run) / "records.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[(r["dataset"], r["id"])].append((float(bool(r["correct"])), r["output_tokens"]))
    return out


def bin_row(label, keys, base, cand, part_change):
    acc = lambda run: 100 * statistics.fmean(c for k in keys for c, _ in run[k])
    tokens = lambda run: sum(t for k in keys for _, t in run[k]) / sum(len(run[k]) for k in keys)
    a0, a1 = acc(base), acc(cand)
    t0, t1 = tokens(base), tokens(cand)
    return {"bin": label, "questions": len(keys), "base_acc": round(a0, 1), "cand_acc": round(a1, 1),
            "change": round(a1 - a0, 1), "share_of_change": round((a1 - a0) * len(keys) / part_change, 2) if part_change else None,
            "base_tokens": round(t0), "cand_tokens": round(t1), "saved": round(100 * (1 - t1 / t0), 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", required=True)
    parser.add_argument("--cand", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    base, cand = per_question(args.base), per_question(args.cand)
    report = {}
    for part in PARTS:
        keys = sorted(k for k in base if k[0] == part and k in cand)
        if not keys:
            continue
        part_change = sum(100 * (statistics.fmean(c for c, _ in cand[k]) - statistics.fmean(c for c, _ in base[k]))
                          for k in keys)  # summed over questions, so the bins' shares add up to 1
        rows = []
        for label, low, high in DIFFICULTY:
            ks = [k for k in keys if low <= round(8 * statistics.fmean(c for c, _ in base[k])) <= high]
            if ks:
                rows.append(bin_row(label, ks, base, cand, part_change))
        by_len = sorted(keys, key=lambda k: statistics.fmean(t for _, t in base[k]))
        quarter = len(by_len) / 4
        length_rows = []
        for q in range(4):
            ks = by_len[round(q * quarter):round((q + 1) * quarter)]
            lo = round(statistics.fmean(t for _, t in base[ks[0]]))
            hi = round(statistics.fmean(t for _, t in base[ks[-1]]))
            length_rows.append(bin_row(f"Q{q + 1} ({lo}-{hi} tok)", ks, base, cand, part_change))
        broken = [k for k in keys if statistics.fmean(c for c, _ in base[k]) >= 7 / 8
                  and statistics.fmean(c for c, _ in cand[k]) <= 4 / 8]
        report[part] = {"difficulty": rows, "length": length_rows, "broken": len(broken),
                        "broken_ids": [k[1] for k in broken][:40]}
        print(f"\n== {part} ({len(keys)} questions; change {part_change / len(keys):+.1f} pts; "
              f"broken {len(broken)}: solved >= 7/8 before, <= 4/8 after)")
        for title, table in (("by the original's solve rate", rows), ("by the original's length (quartiles)", length_rows)):
            print(f"  {title}:")
            for r in table:
                print(f"    {r['bin']:<20} n={r['questions']:<4} acc {r['base_acc']:5.1f} -> {r['cand_acc']:5.1f} "
                      f"({r['change']:+5.1f}; share of loss {r['share_of_change'] if r['share_of_change'] is not None else '-'})"
                      f"  tokens {r['base_tokens']:>5} -> {r['cand_tokens']:>5} (saved {r['saved']:+.0f}%)")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "breakdown.json").write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out / 'breakdown.json'}")


if __name__ == "__main__":
    main()
