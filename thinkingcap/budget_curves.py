"""Forced early-stop curves (consult30 priority 1, user 18 Nov): measurement only, no training.

Each natural answer of a model (--rollouts from probe_curve.py, Dev-512) is cut at fixed points of its own thinking and at
fixed absolute thinking budgets; "</think>" is closed there and the SAME model writes the final answer once (check_vllm.py,
the evaluation's sampling). An answer whose natural thinking already fits a budget keeps its natural outcome.

Rungs: fractions f of the answer's own thinking (--fracs, cut at the last sentence end within f x length) and absolute
budgets B tokens (--budgets). One rung per call with --max-new-rounds 1, so the caller can save to Drive between rungs;
CURVES_DONE marks the end. Outputs in --out: rung-<name>-inputs/verdicts.jsonl, curves.json, summary.txt:
  - per type and rung: accuracy (%) and mean output tokens (cut thinking + answer, or the natural answer when not cut);
  - settle point: for each naturally right answer, the smallest fraction f from which every larger cut is also right;
    post-settlement share = 1 - f (the thinking written after the answer was already decided).

    python budget_curves.py --rollouts runs/bon/dev512-base/rollouts.jsonl --out runs/eff/curves-base --max-new-rounds 1
"""
import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from budget_ladder import Cutter, finished, right
from build_pool import part_of
from probe_curve import read_jsonl, write_jsonl
from prune_traces import GRADE_FIELDS, thinking_text

PARTS = ("svamp", "mbpp", "ifeval", "bfcl_simple_python", "bfcl_multiple", "bfcl_irrelevance")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B", help="the model that wrote the answers AND finishes the cuts")
    ap.add_argument("--samples", default="0,1", help="which of each question's answers are cut")
    ap.add_argument("--fracs", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--budgets", default="0,25,50,100,150,200,300,500,800,1200")
    ap.add_argument("--answer-tokens", type=int, default=2048)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-new-rounds", type=int, default=99)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    open_id, close_id = tok.convert_tokens_to_ids("<think>"), tok.convert_tokens_to_ids("</think>")
    cutter = Cutter(tok, open_id, close_id)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keep_s = {int(x) for x in args.samples.split(",")}
    groups = defaultdict(list)
    for r in read_jsonl(Path(args.rollouts)):
        groups[r["id"]].append(r)
    answers = {}
    for q, g in groups.items():
        if part_of(g[0]) not in PARTS:
            continue
        for s, r in enumerate(g):
            if s in keep_s:
                answers[(q, s)] = r
    nat_len = {k: len(cutter.think_ids(r)) for k, r in answers.items()}
    rungs = [("f", float(x)) for x in args.fracs.split(",")] + [("b", int(x)) for x in args.budgets.split(",")]

    def budget(kind, v, k):
        return int(v * nat_len[k]) if kind == "f" else v

    def rname(kind, v):
        return f"f{int(round(v * 100)):02d}" if kind == "f" else f"b{v}"

    new_rounds = 0
    for kind, v in rungs:
        name = rname(kind, v)
        vf = out / f"rung-{name}-verdicts.jsonl"
        if vf.exists():
            continue
        if new_rounds >= args.max_new_rounds:
            print(f"curves: next rung {name}", flush=True)
            return
        jobs = []
        for k, r in answers.items():
            b = budget(kind, v, k)
            if nat_len[k] > b and r.get("prompt_ids"):
                text = cutter.cut_text(k, r, b)
                jobs.append({"key": f"{k[0]}|{k[1]}", "prompt_ids": r["prompt_ids"],
                             "thinking_ids": tok.encode(thinking_text([text]), add_special_tokens=False),
                             "max_tokens": args.answer_tokens, "row": {g: r[g] for g in GRADE_FIELDS if g in r}})
        write_jsonl(out / f"rung-{name}-inputs.jsonl", jobs)
        print(f"curves: rung {name}: {len(jobs)} cuts of {len(answers)} answers", flush=True)
        if jobs:
            subprocess.run([sys.executable, "check_vllm.py", "--model", args.model, "--inputs",
                            str(out / f"rung-{name}-inputs.jsonl"), "--out", str(out / f"rung-{name}-tmp.jsonl"),
                            "--samples", "1", "--dtype", args.dtype], check=True)
            (out / f"rung-{name}-tmp.jsonl").rename(vf)
        else:
            vf.touch()
        new_rounds += 1
    (out / "CURVES_DONE").write_text("done\n")
    summarize(out, answers, nat_len, rungs, rname, budget, args)


def summarize(out, answers, nat_len, rungs, rname, budget, args):
    res = {}  # (k, rung name) -> (right, output tokens)
    for kind, v in rungs:
        name = rname(kind, v)
        inputs = {x["key"]: x for x in read_jsonl(out / f"rung-{name}-inputs.jsonl")}
        for x in read_jsonl(out / f"rung-{name}-verdicts.jsonl"):
            q, s = x["key"].rsplit("|", 1)
            ans = x.get("right_ids") or x.get("wrong_ids")
            cut = len(inputs[x["key"]]["thinking_ids"])
            res[((q, int(s)), name)] = (x["right"] > 0, cut + 1 + (len(ans) if ans else args.answer_tokens))
    by_part = defaultdict(list)
    for k, r in answers.items():
        by_part[part_of(r)].append(k)
    curves, settle = {}, {}
    for p, ks in by_part.items():
        curves[p] = {"natural": {"acc": 100 * sum(right(answers[k]) for k in ks) / len(ks),
                                 "tokens": sum(answers[k]["output_tokens"] for k in ks) / len(ks),
                                 "thinking": sum(nat_len[k] for k in ks) / len(ks), "n": len(ks)}}
        for kind, v in rungs:
            name = rname(kind, v)
            ok, tk = [], []
            for k in ks:
                if (k, name) in res:
                    a, t = res[(k, name)]
                else:  # not cut at this rung: its natural outcome
                    a, t = right(answers[k]), answers[k]["output_tokens"]
                ok.append(a)
                tk.append(t)
            curves[p][name] = {"acc": 100 * sum(ok) / len(ok), "tokens": sum(tk) / len(tk), "kind": kind, "value": v}
        fr = [(v, rname("f", v)) for kind, v in rungs if kind == "f"]
        shares = []
        for k in ks:
            if not right(answers[k]) or nat_len[k] == 0:
                continue
            f_settle = 1.0
            for v, name in sorted(fr, reverse=True):  # walk down from the longest cut
                a = res[(k, name)][0] if (k, name) in res else True
                if not a:
                    break
                f_settle = v
            shares.append(1 - f_settle)
        settle[p] = {"post_settlement_share": sum(shares) / len(shares) if shares else None, "answers": len(shares)}
    (out / "curves.json").write_text(json.dumps({"curves": curves, "settle": settle}, indent=1))
    lines = [f"forced early-stop curves of {args.model} on {args.rollouts} (samples {args.samples})"]
    for p in PARTS:
        if p not in curves:
            continue
        c = curves[p]
        lines.append(f"== {p}: natural {c['natural']['acc']:.1f}% at {c['natural']['tokens']:.0f} tokens "
                     f"({c['natural']['thinking']:.0f} thinking); post-settlement share "
                     f"{100 * (settle[p]['post_settlement_share'] or 0):.1f}% of the thinking of {settle[p]['answers']} right answers")
        lines.append("   " + "  ".join(f"{n}: {c[n]['acc']:.1f}%/{c[n]['tokens']:.0f}t" for n in c if n != "natural"))
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
