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
    ap.add_argument("--samples", default="0,1", help="answers of each question cut at the fractions (settle points)")
    ap.add_argument("--budget-samples", default="0,1,2,3", help="answers of each question cut at the absolute budgets (curves)")
    ap.add_argument("--fracs", default=",".join(f"{x / 100:.2f}" for x in range(0, 100, 5)))
    ap.add_argument("--budget-levels", default="", help="(18 Nov, user: density at EQUAL caps) a budget_levels.json of another "
                    "run: use ITS per-type budget levels b0..b9 (e.g. the trained model cut at the original's caps)")
    ap.add_argument("--no-fracs", action="store_true", help="budget rungs only (no fraction cuts)")
    ap.add_argument("--budgets", default="auto", help="'auto' = per type: 0 and the 10th..90th percentiles of that type's "
                    "natural thinking lengths (this model); or a comma list applied to every type")
    ap.add_argument("--match-rollouts", default="", help="(18 Nov, user) another model's answers to the same questions: adds "
                    "two rungs on --budget-samples, mt = cut at that model's mean thinking length for the TYPE, mq = for the "
                    "same QUESTION (the original cut where the trained model stops)")
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
    keep_f = {int(x) for x in args.samples.split(",")}
    keep_b = {int(x) for x in args.budget_samples.split(",")}
    keep_s = keep_f | keep_b
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
    part_k = {k: part_of(r) for k, r in answers.items()}
    if args.budget_levels:
        lv = json.loads(Path(args.budget_levels).read_text())["levels"]
        levels = {p: lv[p] for p in lv}
        rungs = [("f", float(x)) for x in args.fracs.split(",")] + [("b", j) for j in range(10)]
    elif args.budgets == "auto":  # per type: 0 and deciles of its natural thinking lengths, as budget INDEX rungs b0..b9
        per = defaultdict(list)
        for k in answers:
            per[part_k[k]].append(nat_len[k])
        levels = {p: [0] + [sorted(v)[min(len(v) - 1, int(q / 10 * len(v)))] for q in range(1, 10)] for p, v in per.items()}
        rungs = [("f", float(x)) for x in args.fracs.split(",")] + [("b", j) for j in range(10)]
    else:
        fixed = [int(x) for x in args.budgets.split(",")]
        levels = defaultdict(lambda: fixed)
        rungs = [("f", float(x)) for x in args.fracs.split(",")] + [("b", j) for j in range(len(fixed))]
    if args.no_fracs:
        rungs = [r for r in rungs if r[0] != "f"]
    match_q, match_t = {}, {}
    if args.match_rollouts:
        mlen = defaultdict(list)
        for r in read_jsonl(Path(args.match_rollouts)):
            mlen[r["id"]].append((part_of(r), len(cutter.think_ids(r))))
        tl = defaultdict(list)
        for q, xs in mlen.items():
            match_q[q] = int(sum(x[1] for x in xs) / len(xs))
            tl[xs[0][0]] += [x[1] for x in xs]
        match_t = {p: int(sum(v) / len(v)) for p, v in tl.items()}
        rungs += [("m", "t"), ("m", "q")]
    (out / "budget_levels.json").write_text(json.dumps({"levels": {p: levels[p] for p in set(part_k.values())},
                                                        "match_type": match_t}))

    def budget(kind, v, k):
        if kind == "m":
            return match_t.get(part_k[k], 10 ** 9) if v == "t" else match_q.get(k[0], 10 ** 9)
        return int(v * nat_len[k]) if kind == "f" else levels[part_k[k]][v]

    def in_rung(kind, k):
        return k[1] in (keep_f if kind == "f" else keep_b)

    def rname(kind, v):
        if kind == "m":
            return f"m{v}"
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
            if not in_rung(kind, k):
                continue
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
    summarize(out, answers, nat_len, rungs, rname, budget, args, in_rung, levels)


def summarize(out, answers, nat_len, rungs, rname, budget, args, in_rung, levels):
    import random
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
            ok, tk, nat_ok, nat_tk, qs = [], [], [], [], []
            for k in ks:
                if not in_rung(kind, k):
                    continue
                if (k, name) in res:
                    a, t = res[(k, name)]
                else:  # not cut at this rung: its natural outcome
                    a, t = right(answers[k]), answers[k]["output_tokens"]
                ok.append(a)
                tk.append(t)
                nat_ok.append(right(answers[k]))
                nat_tk.append(answers[k]["output_tokens"])
                qs.append(k[0])
            if not ok:
                continue
            # paired change vs the same answers' natural outcome, 95% bootstrap over questions
            byq = defaultdict(list)
            for q_, a, n_ in zip(qs, ok, nat_ok):
                byq[q_].append(a - n_)
            keys = list(byq)
            rnd = random.Random(0)
            boots = []
            for _ in range(400):
                smp = [byq[rnd.choice(keys)] for _ in keys]
                tot = sum(len(x) for x in smp)
                boots.append(100 * sum(sum(x) for x in smp) / tot)
            boots.sort()
            curves[p][name] = {"acc": 100 * sum(ok) / len(ok), "tokens": sum(tk) / len(tk), "kind": kind, "value": v,
                               "budget": levels[p][v] if kind == "b" else None, "n": len(ok),
                               "nat_acc": 100 * sum(nat_ok) / len(nat_ok), "nat_tokens": sum(nat_tk) / len(nat_tk),
                               "dacc": 100 * (sum(ok) - sum(nat_ok)) / len(ok), "dacc_ci": [boots[10], boots[389]]}
        fr = [(v, rname("f", v)) for kind, v in rungs if kind == "f"]
        shares = []
        for k in ks:
            if not in_rung("f", k) or not right(answers[k]) or nat_len[k] == 0:
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
        for n in c:
            if n == "natural":
                continue
            x = c[n]
            lab = n if x["kind"] in ("f", "m") else f"{n}={x['budget']}t"
            lines.append(f"   {lab:10s} acc {x['acc']:5.1f}% ({x['dacc']:+5.1f} [{x['dacc_ci'][0]:+.1f}, {x['dacc_ci'][1]:+.1f}] vs natural) "
                         f"tokens {x['tokens']:6.0f} (natural {x['nat_tokens']:.0f}), n={x['n']}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
