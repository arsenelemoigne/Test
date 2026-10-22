"""Calibrated cuts: where can the model's own correct replies stop thinking and still be right, per data type?

Confidence vs outcome: at every cut we also read the model's own confidence, the probability it gives (per token,
geometric mean) to the answer it finally wrote, right after "</think>" is closed at the cut (teacher forcing; works
for every type, no gold string needed). The summary compares that confidence with the real outcome (the share of
the --samples re-answers that are right), bin by bin (calibration table, ECE), and picks, per source, the confidence
threshold c: each reply stops at its earliest cut where confidence >= c; the lowest c whose real accuracy stays
within the allowance is kept. Pairs are built at the confidence-chosen cut (the model's own signal), validated by
the real re-answers; the outcome-only rule (tau) is reported beside it for comparison.

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


def confidences(model_path, adapter, items, dtype, batch_tokens=12000):
    """exp(mean log p(answer token | prompt + cut thinking)) for each (prompt, cut, answer) item."""
    import torch
    from transformers import AutoModelForCausalLM
    from probe_curve import score
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=getattr(torch, dtype) if device == "cuda" else torch.float32).to(device)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    rows = [(k, 0, prompt + cut, answer[:256]) for k, (prompt, cut, answer) in enumerate(items)]
    scores = score(model, rows, batch_tokens, device)
    del model
    torch.cuda.empty_cache()
    import math
    return [math.exp(scores[(k, 0)] / max(1, len(items[k][2][:256]))) for k in range(len(items))]


def calibration_table(points, bins=(0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.01)):
    """points: [(confidence, real success rate)] -> (rows of (lo, hi, n, mean confidence, real rate), ECE)."""
    rows, ece = [], 0.0
    for lo, hi in zip(bins, bins[1:]):
        inside = [(c, r) for c, r in points if lo <= c < hi]
        if inside:
            conf = sum(c for c, _ in inside) / len(inside)
            real = sum(r for _, r in inside) / len(inside)
            rows.append((lo, min(hi, 1.0), len(inside), conf, real))
            ece += len(inside) / len(points) * abs(conf - real)
    return rows, ece


def by_confidence(entries, thresholds, allow):
    """entries: [(full_len, [(cut_len, rate, confidence), ...])]. The lowest confidence threshold whose real
    accuracy loss stays within `allow` points: (c, saved, loss), or (None, 0, 0)."""
    best = (None, 0.0, 0.0)
    for c in sorted(thresholds, reverse=True):
        kept, rates = [], []
        for full, cuts in entries:
            hit = next(((n, r) for n, r, conf in cuts if conf >= c), None)
            kept.append(1 - hit[0] / full if hit else 0.0)
            rates.append(hit[1] if hit else 1.0)
        loss = 100 * (1 - sum(rates) / len(rates))
        if loss <= allow:
            best = (c, sum(kept) / len(kept), loss)
    return best


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
    parser.add_argument("--fractions", default="0.01,0.05,0.1,0.2,0.35,0.5,0.65,0.8",
                        help="cut points, as shares of the reply's thinking (0.01: almost no thinking)")
    parser.add_argument("--taus", default="1.0,0.75,0.5")
    parser.add_argument("--allow", default="gsm_plus=0.5,gsm_ic=0.5,kodcode=1,default=2",
                        help="source=points of expected accuracy loss allowed by the calibration")
    parser.add_argument("--samples", type=int, default=20, help="re-answers per cut: the real outcome in 5% steps")
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
    conf_path = out / "confidence.jsonl"
    if not conf_path.exists():
        close_id = tokenizer.convert_tokens_to_ids("</think>")
        items = []
        for i, f, t in jobs:
            r = replies[i]
            ids = r["output_ids"]
            answer_text = tokenizer.decode(ids[ids.index(close_id) + 1:], skip_special_tokens=True).lstrip("\n")
            items.append((r["prompt_ids"], t, tokenizer.encode(answer_text, add_special_tokens=False) or [close_id]))
        values = confidences(args.model, args.adapter, items, args.dtype)
        write_jsonl(conf_path, [{"key": k, "confidence": c} for k, c in enumerate(values)])
    conf = {c["key"]: c["confidence"] for c in read_jsonl(conf_path)}

    per_reply = defaultdict(list)  # reply index -> [(fraction, cut ids, verdict)] in fraction order
    for k, (i, f, t) in enumerate(jobs):
        per_reply[i].append((f, t, verdicts[k] | {"confidence": conf[k]}))
    allow = dict(spec.split("=") for spec in args.allow.split(","))
    taus = [float(t) for t in args.taus.split(",")]
    by_source = defaultdict(list)
    for i, cuts in per_reply.items():
        by_source[replies[i]["source"]].append(i)

    thresholds = [x / 100 for x in range(5, 100, 5)]
    calib_lines = ["", "Confidence vs real outcome (bins of the model's confidence at a cut -> how often its re-answers "
                   "were really right):"]
    pairs, lines = [], [f"{'source':<16}{'replies':>8}{'conf c':>8}{'saved':>8}{'loss':>7}{'| tau':>7}{'saved':>8}"
                        f"{'loss':>7}{'shorten':>9}{'hard':>6}{'ECE':>7}"]
    for source, indices in sorted(by_source.items()):
        size = lambda t, v: len(t) + len(v["right_ids"] or v["wrong_ids"] or [])
        entries = [(replies[i]["output_tokens"], [(size(t, v), v["right"] / max(1, v["of"])) for _, t, v in per_reply[i]])
                   for i in indices]
        centries = [(replies[i]["output_tokens"],
                     [(size(t, v), v["right"] / max(1, v["of"]), v["confidence"]) for _, t, v in per_reply[i]])
                    for i in indices]
        budget = float(allow.get(source, allow["default"]))
        tau, saved, loss = calibrate(entries, taus, budget)
        c, csaved, closs = by_confidence(centries, thresholds, budget)
        table, ece = calibration_table([(v["confidence"], v["right"] / max(1, v["of"])) for i in indices
                                        for _, _, v in per_reply[i]])
        calib_lines.append(f"  {source}: " + "; ".join(f"{lo:.1f}-{hi:.1f}: n {n}, conf {cf:.2f} -> real {rl:.2f}"
                                                      for lo, hi, n, cf, rl in table) + f" (ECE {ece:.2f})")
        n_short = n_hard = 0
        if c is not None:
            for i in indices:
                r, cuts = replies[i], per_reply[i]
                hit = next((j for j, (_, _, v) in enumerate(cuts) if v["confidence"] >= c), None)
                if hit is None or not cuts[hit][2]["right_ids"]:
                    continue
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
        lines.append(f"{source:<16}{len(indices):>8}{c if c is not None else '-':>8}{csaved:>8.0%}{closs:>7.1f}"
                     f"{tau if tau is not None else '-':>7}{saved:>8.0%}{loss:>7.1f}{n_short:>9}{n_hard:>6}{ece:>7.2f}")
    write_jsonl(out / "pairs.jsonl", pairs)
    lines.append(f"total {len(pairs)} pairs at the confidence threshold c; saved = mean share of the reply cut away; "
                 "loss = real accuracy points lost if the model learned to stop at every chosen cut; tau = the "
                 "outcome-only rule, for comparison; ECE = gap between confidence and real accuracy (0 = calibrated)")
    text = "\n".join(lines + calib_lines)
    (out / "summary.txt").write_text(text + "\n")
    print(text, flush=True)


if __name__ == "__main__":
    main()
