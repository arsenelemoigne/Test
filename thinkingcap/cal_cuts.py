"""Calibrated cuts: where can the model's own correct replies stop thinking and still be right, per data type?

For each correct, finished reply (the model's own best-of-8 rollouts), the thinking is cut at --fractions of its
tokens (snapped back to the last paragraph break), "</think>" is closed there, and the model answers --samples
times from the cut (check_vllm.py, the model that wrote the rollouts). Each cut's success rate (right / samples) is
its measured confidence.

Calibration, per source: for each threshold tau in --taus, every reply stops at its earliest cut whose success rate
is >= tau (or keeps its full thinking when none is). The expected accuracy of a model that learned to stop there is
the mean success rate at the chosen cuts; its loss is 1 minus that, in points. The lowest tau whose loss stays within
the source's allowance (--allow, points; math is strict) is kept: the most saving at the accuracy we ask for.

Pairs, for DPO (dpo.py kinds):
  shorten   chosen = thinking cut at the calibrated point + "</think>" + a right answer written from there
            rejected = the full reply it was cut from (right but longer); only when the cut saves >= --min-save
  hard      chosen = the same calibrated cut + right answer; rejected = an earlier cut of the same reply + its
            wrong answer: where stopping becomes too early (teaches the line from both sides)

    python cal_cuts.py --rollouts runs/bon/probe-x/rollouts.jsonl --keep runs/bon/select-x/sft.jsonl \\
        --model models/ca1-soup-10 --out runs/bon/cal-x
"""

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from probe_curve import read_jsonl, write_jsonl
from prune_traces import GRADE_FIELDS, thinking_text


def cut_text(text, fraction):
    """The thinking's first `fraction`, ending at the last paragraph break before that point (or the point itself)."""
    point = int(len(text) * fraction)
    head = text[:point]
    brk = head.rfind("\n\n")
    return head[:brk] if brk > len(text) * fraction * 0.5 else head


def calibrate(entries, taus, allow):
    """entries: [(full_len, [(cut_len, rate), ...] by fraction)]. Returns (tau, saved, loss) for the lowest tau
    within `allow` points, or (None, 0, 0) when no tau is."""
    best = (None, 0.0, 0.0)
    for tau in sorted(taus, reverse=True):
        kept, rates = [], []
        for full, cuts in entries:
            hit = next(((n, r) for n, r in cuts if r >= tau), None)
            kept.append(1 - hit[0] / full if hit else 0.0)
            rates.append(hit[1] if hit else 1.0)
        loss = 100 * (1 - sum(rates) / len(rates))
        if loss <= allow:
            best = (tau, sum(kept) / len(kept), loss)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter", help="the model that wrote the rollouts, as a LoRA")
    parser.add_argument("--rollouts", nargs="+", required=True)
    parser.add_argument("--keep", nargs="+", required=True, help="sft.jsonl files whose ids passed the contamination check")
    parser.add_argument("--out", required=True)
    parser.add_argument("--fractions", default="0.2,0.35,0.5,0.65,0.8")
    parser.add_argument("--taus", default="1.0,0.75,0.5")
    parser.add_argument("--allow", default="gsm_plus=0.5,gsm_ic=0.5,kodcode=1,default=2",
                        help="source=points of expected accuracy loss allowed by the calibration")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--per-question", type=int, default=1, help="correct replies cut per question (longest first)")
    parser.add_argument("--min-save", type=float, default=0.25)
    parser.add_argument("--answer-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=0, help="smoke test: at most this many replies")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    from transformers import AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    open_id = tokenizer.convert_tokens_to_ids("<think>")
    allowed = {r["id"] for k in args.keep for r in read_jsonl(Path(k))}
    by_question = defaultdict(list)
    for path in args.rollouts:
        for r in read_jsonl(Path(path)):
            if r["id"] in allowed and r["correct"] and r["closed_thinking"] and not r["truncated"]:
                by_question[r["id"]].append(r)
    fractions = [float(f) for f in args.fractions.split(",")]
    replies = [r for group in by_question.values()
               for r in sorted(group, key=lambda r: -r["output_tokens"])[:args.per_question]]
    if args.limit:
        replies = replies[:args.limit]

    cut_path, verdict_path = out / "cuts.jsonl", out / "verdicts.jsonl"
    jobs = []  # (reply index, fraction, cut thinking ids)
    for i, r in enumerate(replies):
        text = tokenizer.decode([t for t in r["thinking_ids"] if t != open_id]).strip()
        for f in fractions:
            jobs.append((i, f, tokenizer.encode(thinking_text([cut_text(text, f)]), add_special_tokens=False)))
    if not verdict_path.exists():
        write_jsonl(cut_path, [{"key": k, "prompt_ids": replies[i]["prompt_ids"], "thinking_ids": t,
                                "max_tokens": args.answer_tokens,
                                "row": {f: replies[i][f] for f in GRADE_FIELDS if f in replies[i]}}
                               for k, (i, _, t) in enumerate(jobs)])
        print(f"{len(by_question)} questions, {len(replies)} replies, {len(jobs)} cuts to check", flush=True)
        command = [sys.executable, "check_vllm.py", "--model", args.model, "--inputs", str(cut_path),
                   "--out", str(verdict_path), "--samples", str(args.samples), "--dtype", args.dtype]
        if args.adapter:
            command += ["--adapter", args.adapter]
        subprocess.run(command, check=True)
    verdicts = {v["key"]: v for v in read_jsonl(verdict_path)}

    per_reply = defaultdict(list)  # reply index -> [(fraction, cut ids, verdict)] in fraction order
    for k, (i, f, t) in enumerate(jobs):
        per_reply[i].append((f, t, verdicts[k]))
    allow = dict(spec.split("=") for spec in args.allow.split(","))
    taus = [float(t) for t in args.taus.split(",")]
    by_source = defaultdict(list)
    for i, cuts in per_reply.items():
        by_source[replies[i]["source"]].append(i)

    pairs, lines = [], [f"{'source':<16}{'replies':>8}{'tau':>6}{'saved':>8}{'exp. loss':>11}{'shorten':>9}{'hard':>6}"]
    for source, indices in sorted(by_source.items()):
        entries = [(replies[i]["output_tokens"],
                    [(len(t) + (len(v["right_ids"] or v["wrong_ids"] or [])), v["right"] / max(1, v["of"]))
                     for _, t, v in per_reply[i]]) for i in indices]
        tau, saved, loss = calibrate(entries, taus, float(allow.get(source, allow["default"])))
        n_short = n_hard = 0
        if tau is not None:
            for i in indices:
                r, cuts = replies[i], per_reply[i]
                hit = next((j for j, (_, _, v) in enumerate(cuts) if v["right"] / max(1, v["of"]) >= tau and v["right_ids"]), None)
                if hit is None:
                    continue
                _, thinking, verdict = cuts[hit]
                chosen = thinking + verdict["right_ids"]
                base = {"id": r["id"], "source": source, "prompt_ids": r["prompt_ids"], "chosen_ids": chosen}
                if len(chosen) <= (1 - args.min_save) * r["output_tokens"]:
                    pairs.append(base | {"kind": "shorten", "rejected_ids": r["output_ids"]})
                    n_short += 1
                early = [(t, v) for _, t, v in cuts[:hit] if v["wrong_ids"]]
                if early:
                    t, v = early[-1]  # the latest cut that still fails: the most tempting place to stop too early
                    pairs.append(base | {"kind": "hard", "rejected_ids": t + v["wrong_ids"]})
                    n_hard += 1
        lines.append(f"{source:<16}{len(indices):>8}{tau if tau is not None else '-':>6}{saved:>8.0%}{loss:>11.1f}"
                     f"{n_short:>9}{n_hard:>6}")
    write_jsonl(out / "pairs.jsonl", pairs)
    lines.append(f"total {len(pairs)} pairs; saved = mean share of the reply cut away at tau; exp. loss = points, "
                 "if the model learned to stop at every calibrated cut")
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text + "\n")
    print(text, flush=True)


if __name__ == "__main__":
    main()
