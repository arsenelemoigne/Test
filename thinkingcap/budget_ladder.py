"""Budget ladder, phase 0 (BUDGET_LADDER_PLAN.md, user 10 Nov): measurement only, no training.

For every question, the original model's 8 natural answers (--rollouts, from probe_curve.py) give its rate k0/8. Each data
type gets its own ladder of thinking budgets: 0, then the 5/10/20/30/40/50/65/80/90th percentiles of that type's natural
RIGHT-answer thinking lengths. At a rung B, each natural answer whose thinking is longer than B is cut at the last sentence
end within B tokens (a hard token cut if no sentence fits), "</think>" is closed there and the model writes the final answer
once (check_vllm.py, the evaluation's sampling); an answer that already stopped within B keeps its natural outcome.
A question's NEEDED BUDGET b* is the lowest rung where it is right at least k0 - --tolerance times out of 8 (as right as the
original, one answer of slack). Questions climb only until they qualify (k0 = 0: until some cut is right = "rescued").
One rung per call with --max-new-rounds 1, so the caller can save to Drive between rungs; LADDER_DONE marks the end.

Outputs in --out: rungs.json, round-<j>-inputs.jsonl / -verdicts.jsonl (every cut with its answer, the pair material),
ladder.jsonl (one line per question), summary.txt (per type: b* distribution, the ORACLE curve = tokens and accuracy if
every question used exactly b*, shorter-is-more-right and rescued counts, whether b* is predictable (rank correlation with the question's
length and the original's rate), and the official-weighting overall with the go / no-go rule: >= 60% saved at accuracy >= -0.5).

    python budget_ladder.py --rollouts runs/bon/ld-L1/rollouts.jsonl --out runs/bon/ld-L1/ladder --max-new-rounds 1
"""
import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from build_pool import part_of
from probe_curve import read_jsonl, write_jsonl
from prune_traces import GRADE_FIELDS, SENTENCE, thinking_text

PCTS = (5, 10, 20, 30, 40, 50, 65, 80, 90)
PARTS = ("svamp", "mbpp", "ifeval", "bfcl_simple_python", "bfcl_multiple", "bfcl_irrelevance")
BFCL_W = {"bfcl_simple_python": 400, "bfcl_multiple": 200, "bfcl_irrelevance": 240}


def finished(r):
    return bool(r["closed_thinking"]) and not r["truncated"]


def right(r):
    return bool(r["correct"]) and finished(r)


def quantile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0


def rank_corr(xs, ys):
    """Spearman's rho with average ranks (ties), no scipy needed."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        out, i = [0.0] * len(v), 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2
            i = j + 1
        return out
    if len(xs) < 3:
        return float("nan")
    a, b = ranks(xs), ranks(ys)
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else float("nan")


class Cutter:
    """An answer's thinking as sentences with cumulative token counts: the longest sentence prefix within B tokens."""
    def __init__(self, tok, open_id, close_id):
        self.tok, self.open_id, self.close_id, self.cache = tok, open_id, close_id, {}

    def think_ids(self, r):
        return [t for t in (r.get("thinking_ids") or []) if t not in (self.open_id, self.close_id)]

    def cut_text(self, key, r, budget):
        if budget <= 0:
            return ""
        if key not in self.cache:
            text = self.tok.decode(self.think_ids(r)).strip()
            pieces = [p for p in SENTENCE.findall(text) if p]
            cum, total = [], 0
            for p in pieces:
                total += len(self.tok.encode(p, add_special_tokens=False))
                cum.append(total)
            self.cache[key] = (pieces, cum)
        pieces, cum = self.cache[key]
        k = 0
        while k < len(cum) and cum[k] <= budget:
            k += 1
        if k:
            return "".join(pieces[:k])
        return self.tok.decode(self.think_ids(r)[:budget])  # no whole sentence fits: a hard cut


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--tolerance", type=int, default=1, help="answers (of the 8) a rung may lose vs the natural answers")
    ap.add_argument("--answer-tokens", type=int, default=2048)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-new-rounds", type=int, default=99, help="rungs to run in this call (the caller saves between)")
    ap.add_argument("--checker", default="check_vllm.py", help="test hook: the script that answers from the cuts")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    open_id, close_id = tok.convert_tokens_to_ids("<think>"), tok.convert_tokens_to_ids("</think>")
    cutter = Cutter(tok, open_id, close_id)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    for r in read_jsonl(Path(args.rollouts)):
        groups[r["id"]].append(r)
    part_of_q = {q: part_of(g[0]) for q, g in groups.items()}
    groups = {q: g for q, g in groups.items() if part_of_q[q] in PARTS}
    n_of = {q: len(g) for q, g in groups.items()}
    k0 = {q: sum(right(r) for r in g) for q, g in groups.items()}
    nat_len = {(q, s): len(cutter.think_ids(r)) for q, g in groups.items() for s, r in enumerate(g)}

    rungs_f = out / "rungs.json"
    if rungs_f.exists():
        rungs = json.loads(rungs_f.read_text())
    else:  # per type: 0 + the percentiles of its natural RIGHT answers' thinking (fixed once, reused on resume)
        rungs = {}
        for p in PARTS:
            lens = [nat_len[(q, s)] for q, g in groups.items() if part_of_q[q] == p for s, r in enumerate(g) if right(r)]
            rungs[p] = sorted({0} | {max(1, quantile(lens, c / 100)) for c in PCTS}) if lens else [0]
        rungs_f.write_text(json.dumps(rungs, indent=1))
    n_rounds = max(len(v) for v in rungs.values())

    # outcome[(q, s, j)] = (right, tokens, cut_tokens); natural answers that stop within the rung keep their own outcome
    outcome = {}
    for j in range(n_rounds):
        vf = out / f"round-{j}-verdicts.jsonl"
        if vf.exists():
            inputs = {x["key"]: x for x in read_jsonl(out / f"round-{j}-inputs.jsonl")}
            for v in read_jsonl(vf):
                q, s, _ = v["key"].rsplit("|", 2)
                s = int(s)
                ans = v.get("right_ids") or v.get("wrong_ids")
                cut = len(inputs[v["key"]]["thinking_ids"])
                outcome[(q, s, j)] = (v["right"] > 0, cut + (len(ans) if ans else args.answer_tokens), cut)

    def status():
        """b*[q] = rung index, 'nat' (never qualifies on the evaluated rungs), or None (still climbing)."""
        best, rescued = {}, set()
        for q in groups:
            p = part_of_q[q]
            for j, budget in enumerate(rungs[p]):
                res = []
                for s, r in enumerate(groups[q]):
                    if nat_len[(q, s)] <= budget:
                        res.append(right(r))
                    elif (q, s, j) in outcome:
                        res.append(outcome[(q, s, j)][0])
                    else:
                        res = None
                        break
                if res is None:
                    break
                k = sum(res)
                if k0[q] == 0:
                    if k > 0:
                        rescued.add(q)
                        best[q] = j
                        break
                elif k >= k0[q] - args.tolerance:
                    best[q] = j
                    break
            else:
                best[q] = "nat"
        return best, rescued

    new_rounds = 0
    while True:
        best, _ = status()
        climbing = [q for q in groups if q not in best]
        if not climbing:
            break
        # the lowest rung any climbing question still needs
        j = min(next(j for j in range(len(rungs[part_of_q[q]]))
                     if any((q, s, j) not in outcome and nat_len[(q, s)] > rungs[part_of_q[q]][j] for s in range(n_of[q])))
                for q in climbing)
        if new_rounds >= args.max_new_rounds:
            print(f"ladder: {len(climbing)} questions still climbing; next rung index {j}", flush=True)
            return
        jobs = []
        for q in climbing:
            p = part_of_q[q]
            if j >= len(rungs[p]):
                continue
            budget = rungs[p][j]
            for s, r in enumerate(groups[q]):
                if nat_len[(q, s)] > budget and (q, s, j) not in outcome:
                    text = cutter.cut_text((q, s), r, budget)
                    jobs.append({"key": f"{q}|{s}|{j}", "prompt_ids": r["prompt_ids"],
                                 "thinking_ids": tok.encode(thinking_text([text]), add_special_tokens=False),
                                 "max_tokens": args.answer_tokens, "row": {g: r[g] for g in GRADE_FIELDS if g in r}})
        inf, vf = out / f"round-{j}-inputs.jsonl", out / f"round-{j}-verdicts.jsonl"
        prior = read_jsonl(inf) if inf.exists() else []
        write_jsonl(inf, prior + jobs)
        print(f"ladder: rung index {j}: {len(jobs)} cuts of {len(climbing)} climbing questions", flush=True)
        if jobs:
            tmp = out / f"round-{j}-new.jsonl"
            write_jsonl(tmp, jobs)
            subprocess.run([sys.executable, args.checker, "--model", args.model, "--inputs", str(tmp), "--out",
                            str(out / f"round-{j}-new-verdicts.jsonl"), "--samples", "1", "--dtype", args.dtype], check=True)
            got = read_jsonl(out / f"round-{j}-new-verdicts.jsonl")
            prev = read_jsonl(vf) if vf.exists() else []
            write_jsonl(vf, prev + got)
            inputs = {x["key"]: x for x in read_jsonl(inf)}
            for v in read_jsonl(vf):
                q, s, _ = v["key"].rsplit("|", 2)
                ans = v.get("right_ids") or v.get("wrong_ids")
                cut = len(inputs[v["key"]]["thinking_ids"])
                outcome[(q, int(s), j)] = (v["right"] > 0, cut + (len(ans) if ans else args.answer_tokens), cut)
        else:  # nothing to cut at this rung for anyone still climbing: mark the rung evaluated with no jobs
            vf.touch()
        new_rounds += 1

    best, rescued = status()
    (out / "LADDER_DONE").write_text("done\n")
    summarize(args, out, groups, part_of_q, n_of, k0, nat_len, rungs, outcome, best, rescued)


def summarize(args, out, groups, part_of_q, n_of, k0, nat_len, rungs, outcome, best, rescued):
    def at(q, j):
        """(rights, mean tokens) of question q at rung index j ('nat' = natural)."""
        toks, k = [], 0
        for s, r in enumerate(groups[q]):
            if j == "nat" or nat_len[(q, s)] <= rungs[part_of_q[q]][j]:
                k += right(r)
                toks.append(r["output_tokens"])
            else:
                ok, t, _ = outcome[(q, s, j)]
                k += ok
                toks.append(t)
        return k, sum(toks) / len(toks)

    rows, lines = [], []
    per = {}
    lines.append(f"tolerance: {args.tolerance} answer(s) of {max(n_of.values())}; never-solved questions climb until a cut is right")
    for p in PARTS:
        qs = [q for q in groups if part_of_q[q] == p]
        if not qs:
            continue
        nat_tok = sum(at(q, "nat")[1] for q in qs)
        orc_tok = sum(at(q, best[q])[1] for q in qs)
        nat_acc = sum(k0[q] / n_of[q] for q in qs) / len(qs)
        orc_acc = sum(at(q, best[q])[0] / n_of[q] for q in qs) / len(qs)
        dist = defaultdict(int)
        for q in qs:
            dist[best[q]] += 1
        more_right = sum(any(j != "nat" and all((q, s, j) in outcome or nat_len[(q, s)] <= rungs[p][j] for s in range(n_of[q]))
                             and at(q, j)[0] > k0[q] for j in range(len(rungs[p]))) for q in qs if k0[q] > 0)
        solved = [q for q in qs if k0[q] > 0]
        b_idx = [len(rungs[p]) if best[q] == "nat" else best[q] for q in solved]
        qlen = [len(groups[q][0]["prompt_ids"]) for q in solved]
        rho_len = rank_corr(b_idx, qlen)
        rho_k0 = rank_corr(b_idx, [k0[q] for q in solved])
        per[p] = (nat_tok / len(qs), orc_tok / len(qs), nat_acc, orc_acc)
        lines.append(f"\n== {p}: {len(qs)} questions; rungs {rungs[p]}")
        lines.append(f"   original right {100 * nat_acc:.1f}%; never solved {sum(k0[q] == 0 for q in qs)}; always solved "
                     f"{sum(k0[q] == n_of[q] for q in qs)}")
        lines.append("   needed budget b* (questions): " + ", ".join(
            f"{('natural' if j == 'nat' else rungs[p][j])}: {dist[j]}" for j in list(range(len(rungs[p]))) + ["nat"] if dist[j]))
        lines.append(f"   ORACLE (every question at b*): tokens {nat_tok / len(qs):.0f} -> {orc_tok / len(qs):.0f} per answer "
                     f"(saved {100 * (1 - orc_tok / nat_tok):.1f}%), accuracy {100 * nat_acc:.1f} -> {100 * orc_acc:.1f} "
                     f"({100 * (orc_acc - nat_acc):+.1f} pts)")
        lines.append(f"   a shorter rung MORE right than natural: {more_right} of {len(solved)} solved questions; "
                     f"never-solved rescued by a cut: {sum(q in rescued for q in qs)} of {sum(k0[q] == 0 for q in qs)}")
        lines.append(f"   predictable? rank corr of b* with question length {rho_len:+.2f}, with the original's rate {rho_k0:+.2f}")
        for q in qs:
            rows.append({"id": q, "part": p, "k0": k0[q], "n": n_of[q], "rungs": rungs[p],
                         "b_star": best[q], "b_star_tokens": None if best[q] == "nat" else rungs[p][best[q]],
                         "rescued": q in rescued, "nat_tokens": round(at(q, "nat")[1], 1),
                         "oracle_tokens": round(at(q, best[q])[1], 1), "oracle_right": at(q, best[q])[0]})
    write_jsonl(out / "ladder.jsonl", rows)
    # official weighting: 4 benchmarks equal (BFCL pooled 400/200/240); tokens saved = 1 - geometric mean of ratios
    bench = {}
    for p in ("svamp", "mbpp", "ifeval"):
        if p in per:
            nt, ot, na, oa = per[p]
            bench[p] = (ot / nt, oa - na)
    bf = [p for p in BFCL_W if p in per]
    if bf:
        w = sum(BFCL_W[p] for p in bf)
        nt = sum(BFCL_W[p] * per[p][0] for p in bf) / w
        ot = sum(BFCL_W[p] * per[p][1] for p in bf) / w
        bench["bfcl"] = (ot / nt, sum(BFCL_W[p] * (per[p][3] - per[p][2]) for p in bf) / w)
    if bench:
        saved = 100 * (1 - math.exp(sum(math.log(r) for r, _ in bench.values()) / len(bench)))
        acc = 100 * sum(a for _, a in bench.values()) / len(bench)
        verdict = "GO" if saved >= 60 and acc >= -0.5 else "NO-GO (below 60% saved at >= -0.5)"
        lines.append(f"\n== OVERALL ORACLE (official weighting, training questions): saved {saved:.1f}%, accuracy {acc:+.2f} pts "
                     f"-> {verdict}")
    text = "\n".join(lines)
    print(text, flush=True)
    (out / "summary.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
