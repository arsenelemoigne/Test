"""Labelled cuts at every paragraph end, of right AND wrong replies: the training data for a learned stop.

For each kept question (contamination-checked ids in --keep): the longest right reply and, with --include-wrong, the
longest finished wrong reply. Each reply's thinking is cut after its paragraph ends (at most --max-cuts of them, evenly
spaced, always including the first) and once in full (its natural stop); "</think>" is closed at each cut and the
model answers --samples times from there (check_vllm.py). A cut's label is its share of right answers. Wrong replies
show the two cases a stopper can gain on: questions it never gets right (stopping early is free) and replies that
were right earlier and thought themselves wrong (stopping early gains accuracy).

Outputs (as cal_cuts.py, so the probes read it): cuts.jsonl (+ reply, frac, para, n_para), verdicts.jsonl,
confidence.jsonl (teacher-forced probability of the full reply's own answer), replies.jsonl, summary.txt.

    python label_cuts.py --model models/rc1-rl-40 --rollouts runs/bon/probe-plotA/rollouts.jsonl \\
        --keep runs/bon/plotA-clean.jsonl --out runs/bon/pairs-plotA/cal --include-wrong
"""

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from probe_curve import read_jsonl, write_jsonl
from prune_traces import GRADE_FIELDS, split_steps, thinking_text


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--rollouts", nargs="+", required=True)
    parser.add_argument("--keep", nargs="+", required=True, help="files whose rows' ids are the clean questions")
    parser.add_argument("--out", required=True)
    parser.add_argument("--include-wrong", action="store_true")
    parser.add_argument("--max-cuts", type=int, default=10, help="paragraph-end cuts per reply, plus the full thinking")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--answer-tokens", type=int, default=2048)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    open_id, close_id = tok.convert_tokens_to_ids("<think>"), tok.convert_tokens_to_ids("</think>")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    allowed = {r["id"] for k in args.keep for r in read_jsonl(Path(k))}
    by_q = defaultdict(list)
    for path in args.rollouts:
        for r in read_jsonl(Path(path)):
            if r["id"] in allowed and r["closed_thinking"] and not r["truncated"]:
                by_q[r["id"]].append(r)
    replies = []
    for group in by_q.values():
        right = sorted([r for r in group if r["correct"]], key=lambda r: -r["output_tokens"])[:1]
        wrong = sorted([r for r in group if not r["correct"]], key=lambda r: -r["output_tokens"])[:1] if args.include_wrong else []
        replies += right + wrong

    jobs = []  # (reply index, share of thinking tokens kept, paragraph index, paragraphs, cut thinking ids)
    for i, r in enumerate(replies):
        text = tok.decode([t for t in r["thinking_ids"] if t != open_id and t != close_id]).strip()
        steps = split_steps(text)
        n = len(steps)
        ends = list(range(1, n))  # cut after paragraph k (1..n-1); n = the full thinking, added below
        if len(ends) > args.max_cuts:
            ends = sorted({ends[round(j * (len(ends) - 1) / (args.max_cuts - 1))] for j in range(args.max_cuts)})
        full = len(tok.encode(text, add_special_tokens=False)) or 1
        for k in ends + [n]:
            ids = tok.encode(thinking_text(steps[:k]), add_special_tokens=False)
            kept = len(tok.encode("".join(steps[:k]).strip(), add_special_tokens=False))
            jobs.append((i, min(1.0, kept / full), k, n, ids))
    cut_path, verdict_path = out / "cuts.jsonl", out / "verdicts.jsonl"
    if not verdict_path.exists():
        write_jsonl(cut_path, [{"key": k, "reply": i, "frac": f, "para": p, "n_para": n,
                                "prompt_ids": replies[i]["prompt_ids"], "thinking_ids": t, "max_tokens": args.answer_tokens,
                                "row": {g: replies[i][g] for g in GRADE_FIELDS if g in replies[i]}}
                               for k, (i, f, p, n, t) in enumerate(jobs)])
        write_jsonl(out / "replies.jsonl", [{"reply": i, "id": r["id"], "source": r["source"], "correct": bool(r["correct"]),
                                             "output_tokens": r["output_tokens"]} for i, r in enumerate(replies)])
        print(f"{len(by_q)} questions, {len(replies)} replies ({sum(r['correct'] for r in replies)} right), "
              f"{len(jobs)} cuts to check x {args.samples}", flush=True)
        subprocess.run([sys.executable, "check_vllm.py", "--model", args.model, "--inputs", str(cut_path), "--out",
                        str(verdict_path), "--samples", str(args.samples), "--dtype", args.dtype], check=True)
    verdicts = {v["key"]: v for v in read_jsonl(verdict_path)}
    conf_path = out / "confidence.jsonl"
    if not conf_path.exists():
        from cal_cuts import confidences
        items = []
        for i, f, p, n, t in jobs:
            ids = replies[i]["output_ids"]
            answer = tok.decode(ids[ids.index(close_id) + 1:], skip_special_tokens=True).lstrip("\n")
            items.append((replies[i]["prompt_ids"], t, tok.encode(answer, add_special_tokens=False) or [close_id]))
        write_jsonl(conf_path, [{"key": k, "confidence": c} for k, c in enumerate(confidences(args.model, None, items, args.dtype))])

    # summary: per data type, right vs wrong replies, success along the thinking, and what a perfect stop would give
    lines = [f"{'source':<16}{'replies':>8}{'right':>7}{'cuts':>7}  success by share of thinking kept "
             "(0-20/20-40/40-60/60-80/80-<100/full)   perfect stop: saved / accuracy gained"]
    per = defaultdict(list)
    for k, (i, f, p, n, t) in enumerate(jobs):
        per[i].append((f, p == n, verdicts[k]["right"] / max(1, verdicts[k]["of"]), len(t)))
    for source in sorted({r["source"] for r in replies}):
        idx = [i for i, r in enumerate(replies) if r["source"] == source]
        bins = defaultdict(list)
        saved, gained = [], []
        for i in idx:
            for f, is_full, s, size in per[i]:
                b = "full" if is_full else min(4, int(f * 5))
                bins[b].append(s)
            full_s = next(s for f, is_full, s, size in per[i] if is_full)
            full_size = next(size for f, is_full, s, size in per[i] if is_full)
            # perfect stop: the earliest cut at least as good as the full thinking (ties -> earliest)
            best = min(((size, -s) for f, is_full, s, size in per[i] if s >= full_s), default=(full_size, -full_s))
            saved.append(1 - best[0] / max(1, full_size))
            gained.append(max(s for _, _, s, _ in per[i]) - full_s)
        cells = " / ".join(f"{sum(bins[b]) / len(bins[b]):.2f}" if bins[b] else "-" for b in [0, 1, 2, 3, 4, "full"])
        lines.append(f"{source:<16}{len(idx):>8}{sum(replies[i]['correct'] for i in idx):>7}"
                     f"{sum(len(per[i]) for i in idx):>7}  {cells}   {sum(saved) / len(saved):.0%} / "
                     f"+{100 * sum(gained) / len(gained):.1f} pts (best cut vs natural stop)")
    text = "\n".join(lines)
    print(text, flush=True)
    (out / "summary.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
