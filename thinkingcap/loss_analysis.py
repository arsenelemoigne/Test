"""Where does a model's accuracy loss come from? (13 Nov, user: "take a new look at od2, od3-r3, od3-r4 ... different
types of accuracy loss, so we can target them in the online DPO").

Question by question against the ORIGINAL's full-test answers (same questions, x8 each):
  net change = sum over questions of (model's solve rate - original's), per type;
  LOST questions = the model solves them less often than the original; the lost answers there are sorted by cause:
    cut off at the cap / thinking never closed / repetition loop / unparseable or wrong format (tool calls) /
    stopped thinking early (< 0.5x the original's median right thinking on that question) / somewhat short (0.5-0.8x) /
    IFEval response too short (< 0.7x the original's median response: length rules) / full length but wrong;
  each lost question is weighted by its loss (p0 - p1), so the causes add up to the type's lost accuracy;
  also split by how reliably the original solved the question (8/8, 6-7/8, 3-5/8, 1-2/8).
Usage: python loss_analysis.py --base BASE_records.jsonl --cand NAME=records.jsonl [NAME=...] [--out report.txt]
"""
import argparse
import json
import re
import statistics as st
from collections import Counter, defaultdict

TRUE = (True, "True", "true", 1, "1")


def load(path):
    d = defaultdict(list)
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            d[(r["dataset"], str(r["id"]))].append(r)
    return d


def loop(text):
    w = re.findall(r"\w+", (text or "").lower())
    g = Counter(tuple(w[i:i + 8]) for i in range(max(0, len(w) - 7)))
    return bool(g) and max(g.values()) >= 4


def cause(r, b_think, b_resp, ds):
    t = int(r["reasoning_tokens"])
    resp = int(r["output_tokens"]) - t
    if r.get("truncated") in TRUE:
        return "cut off at the cap"
    if "</think>" not in (r.get("text") or "</think>"):
        return "thinking never closed"
    if loop(r.get("text") or r.get("response")):
        return "repetition loop"
    if r.get("parse_error") not in (None, "None", "", "null"):
        return "unparseable / wrong format"
    if b_think and t < 0.5 * b_think:
        return "stopped thinking early (<0.5x)"
    if ds == "ifeval" and b_resp and resp < 0.7 * b_resp:
        return "IFEval response too short (<0.7x)"
    if b_think and t < 0.8 * b_think:
        return "somewhat short thinking (0.5-0.8x)"
    return "full length but wrong"


def rel_bin(p0):
    return "orig 8/8" if p0 == 1 else "orig 6-7/8" if p0 >= 0.75 else "orig 3-5/8" if p0 >= 0.375 else "orig 1-2/8" if p0 > 0 else "orig 0/8"


def analyse(name, base, cand):
    out = [f"\n===== {name} vs the original (full test, same questions) ====="]
    by_ds = defaultdict(list)
    for key, rs in cand.items():
        if key in base:
            by_ds[key[0]].append(key)
    for ds in sorted(by_ds):
        keys = by_ds[ds]
        n = len(keys)
        net, lost_w, gain_w = 0.0, 0.0, 0.0
        causes, bins, n_lost_q = Counter(), Counter(), 0
        think_ratio_right, think_ratio_all = [], []
        for key in keys:
            bs, rs = base[key], cand[key]
            p0 = st.fmean(r["correct"] in TRUE for r in bs)
            p1 = st.fmean(r["correct"] in TRUE for r in rs)
            net += p1 - p0
            b_right = [int(r["reasoning_tokens"]) for r in bs if r["correct"] in TRUE]
            b_think = st.median(b_right) if b_right else None
            b_resp = st.median(int(r["output_tokens"]) - int(r["reasoning_tokens"]) for r in bs)
            if b_think:
                think_ratio_all += [int(r["reasoning_tokens"]) / b_think for r in rs]
            if p1 < p0:
                n_lost_q += 1
                lost_w += p0 - p1
                bins[rel_bin(p0)] += p0 - p1
                wrong = [r for r in rs if r["correct"] not in TRUE]
                for r in wrong:  # the question's loss shared over its wrong answers' causes
                    causes[cause(r, b_think, b_resp, ds)] += (p0 - p1) / len(wrong)
                if b_think:
                    think_ratio_right += [int(r["reasoning_tokens"]) / b_think for r in rs if r["correct"] in TRUE]
            elif p1 > p0:
                gain_w += p1 - p0
        out.append(f"{ds}: {n} q | net {100 * net / n:+.2f} pts = lost {-100 * lost_w / n:.2f} + gained {100 * gain_w / n:+.2f} "
                   f"| {n_lost_q} questions lost | thinking vs original's right median: median "
                   f"{st.median(think_ratio_all) if think_ratio_all else float('nan'):.2f}x")
        if lost_w:
            out.append("   lost accuracy by cause (pts of the type's accuracy):  " + "; ".join(
                f"{k} {100 * v / n:.2f} ({100 * v / lost_w:.0f}%)" for k, v in causes.most_common()))
            out.append("   lost accuracy by how reliably the original solved it:  " + "; ".join(
                f"{k} {100 * v / n:.2f} ({100 * v / lost_w:.0f}%)" for k, v in sorted(bins.items())))
            if think_ratio_right:
                out.append(f"   on the lost questions, the model's RIGHT answers think {st.median(think_ratio_right):.2f}x "
                           f"the original's right median")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--cand", nargs="+", required=True, help="NAME=records.jsonl")
    ap.add_argument("--out", default="")
    ap.add_argument("--noise", action="store_true", help="also the original vs itself (sampling noise)")
    a = ap.parse_args()
    base = load(a.base)
    lines = []
    if a.noise:  # the original against ITSELF: answers 0-3 vs 4-7 (what pure sampling noise looks like, at x4 each)
        half0 = {k: [r for r in v if int(r.get("sample_index", 0)) < 4] for k, v in base.items()}
        half1 = {k: [r for r in v if int(r.get("sample_index", 0)) >= 4] for k, v in base.items()}
        lines += analyse("NOISE REFERENCE: original answers 4-7 vs its answers 0-3 (x4 each)",
                         {k: v for k, v in half0.items() if v}, {k: v for k, v in half1.items() if v})
    for spec in a.cand:
        name, path = spec.split("=", 1)
        lines += analyse(name, base, load(path))
    text = "\n".join(lines) + "\n"
    print(text, flush=True)
    if a.out:
        open(a.out, "w").write(text)


if __name__ == "__main__":
    main()
