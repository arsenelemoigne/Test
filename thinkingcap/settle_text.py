"""Thinking written after the answer is already found (text version; user 18 Nov). CPU, light, all answers.

For each answer whose final response can be located inside its own thinking, the "settle point" is the first position in
the thinking where that final answer appears; the post-settlement share is the part of the thinking written after it.
  - math (svamp): the final number ("Final answer: N"); first appearance as a standalone number in the thinking; a stricter
    version requires an answer context just before it (=, is, total, answer, so, therefore) to skip early mentions of the
    question's own numbers;
  - tool calls (bfcl_simple_python / bfcl_multiple): the function name of the final call, and the point where the name AND
    every argument value of the final call have all appeared;
  - code (mbpp): the first "def <name>(" of the final function inside the thinking (start of the first draft).
Tokens are measured by characters x (thinking tokens / thinking characters) per answer. Reported for right answers and for
all answers: detection rate, mean thinking tokens, tokens after the settle point, post-settlement share (over detected,
and over all answers with 0 for undetected), and how much of the thinking cut (original -> trained) is post-settlement.

python settle_text.py --base ~/Downloads/tc_full/base.jsonl --new ~/Downloads/tc_full/od5c16.jsonl --out runs/settle
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

NUMRE = r"(?<![\w.])-?{}(?:\.0+)?(?![\w]|\.\d)"
CTX = re.compile(r"(?:=|\bis\b|\btotal\b|\banswer\b|\bso\b|\btherefore\b|\bthus\b|\bequals\b)[^.\n]{0,25}$", re.I)


def split(text):
    i = (text or "").find("</think>")
    th = text[text.find("<think>") + 7: i] if "<think>" in text and i >= 0 else ""
    return th, (text[i + 8:] if i >= 0 else "")


def math_settle(th, resp, strict):
    m = re.search(r"Final answer:\s*\$?(-?[\d,]*\.?\d+)", resp) or re.search(r"(-?\d[\d,]*\.?\d*)\s*$", resp.strip())
    if not m:
        return None
    num = m.group(1).replace(",", "").rstrip(".")
    if num.endswith(".0"):
        num = num[:-2]
    pat = re.compile(NUMRE.format(re.escape(num)))
    for mm in pat.finditer(th):
        if not strict or CTX.search(th[max(0, mm.start() - 40): mm.start()]):
            return mm.end()
    return None


def call_parts(resp):
    m = re.search(r"\[?\s*([A-Za-z_][\w.]*)\s*\((.*)\)\s*\]?", resp, re.S)
    if not m:
        return None, []
    vals = []
    for a in re.finditer(r"\w+\s*=\s*(\"[^\"]*\"|'[^']*'|\[[^\]]*\]|[^,)]+)", m.group(2)):
        v = a.group(1).strip().strip("\"'")
        if v and v.lower() not in ("true", "false", "none") and len(v) <= 60:
            vals.append(v)
    return m.group(1), vals


def tool_settle(th, resp, full):
    name, vals = call_parts(resp)
    if not name:
        return None
    short = name.split(".")[-1]
    i = th.find(short)
    if i < 0:
        return None
    end = i + len(short)
    if not full:
        return end
    for v in vals:
        j = th.find(v)
        if j < 0:
            return None
        end = max(end, j + len(v))
    return end


def code_settle(th, resp):
    m = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", resp)
    if not m:
        return None
    j = re.search(r"def\s+" + re.escape(m.group(1)) + r"\s*\(", th)
    return j.start() if j else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", default="runs/settle")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    measures = {"svamp": [("math: final number", lambda th, r: math_settle(th, r, False)),
                          ("math: final number in an answer context", lambda th, r: math_settle(th, r, True))],
                "bfcl_simple_python": [("one function: function named", lambda th, r: tool_settle(th, r, False)),
                                       ("one function: name + all argument values", lambda th, r: tool_settle(th, r, True))],
                "bfcl_multiple": [("several functions: function named", lambda th, r: tool_settle(th, r, False)),
                                  ("several functions: name + all argument values", lambda th, r: tool_settle(th, r, True))],
                "mbpp": [("code: first draft of the final function starts", code_settle)]}
    res = {}
    for who, path in (("original", args.base), ("trained", args.new)):
        acc = defaultdict(lambda: defaultdict(float))
        for line in open(path):
            r = json.loads(line)
            if r["dataset"] not in measures or r.get("truncated"):
                continue
            th, resp = split(r["text"])
            T = float(r.get("reasoning_tokens") or 0)
            if not th or not T:
                continue
            tpc = T / len(th)
            for label, f in measures[r["dataset"]]:
                pos = f(th, resp)
                for subset in ("all", "right") if r.get("correct") else ("all",):
                    g = acc[(label, subset)]
                    g["n"] += 1
                    g["think"] += T
                    if pos is not None:
                        after = max(0.0, (len(th) - pos) * tpc)
                        g["det"] += 1
                        g["after"] += after
                        g["think_det"] += T
        for (label, subset), g in acc.items():
            res.setdefault(label, {}).setdefault(subset, {})[who] = {
                "answers": int(g["n"]), "detected": g["det"] / g["n"], "thinking": g["think"] / g["n"],
                "after_tokens": g["after"] / g["n"], "share_detected": g["after"] / g["think_det"] if g["think_det"] else None,
                "share_all": g["after"] / g["think"]}
    lines = ["== thinking written after the final answer first appears (not truncated answers)"]
    for label, sub in res.items():
        for subset in ("right", "all"):
            if subset not in sub:
                continue
            o, t = sub[subset]["original"], sub[subset]["trained"]
            cut = o["thinking"] - t["thinking"]
            post_cut = o["after_tokens"] - t["after_tokens"]
            sub[subset]["post_share_of_cut"] = post_cut / cut if cut else None
            lines.append(f"-- {label} [{subset} answers]: detected {100 * o['detected']:.1f}% -> {100 * t['detected']:.1f}%; "
                         f"thinking {o['thinking']:.0f} -> {t['thinking']:.0f} tokens; after the answer {o['after_tokens']:.0f} -> "
                         f"{t['after_tokens']:.0f} tokens; share after (detected) {100 * (o['share_detected'] or 0):.1f}% -> "
                         f"{100 * (t['share_detected'] or 0):.1f}%, (all) {100 * o['share_all']:.1f}% -> {100 * t['share_all']:.1f}%; "
                         f"post-settlement part of the cut {100 * (sub[subset]['post_share_of_cut'] or 0):.0f}%")
    (out / "settle.json").write_text(json.dumps(res, indent=1))
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
