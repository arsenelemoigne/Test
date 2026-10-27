"""Does the training pool behave like the evaluation? The original model on a pool sample vs on the choosing set.

The original model answers --per-part pool questions per evaluation part (probe_curve.py --generate-only, the
evaluation's sampling settings and caps); its answers on the CHOOSING set come from the evaluation run v4-base
(records.jsonl). Per part, side by side: accuracy, median and mean reply tokens, median thinking tokens, share capped,
with a 95% bootstrap interval on the choosing side (only 40 questions a part, so it is wide). Verdict per measure:
OK inside the tolerance (accuracy +-5 pts, median tokens +-15%, capped +-3 pts) or inside the choosing set's own interval,
else CHECK. The fresh set is never read.

    python pool_check.py --rollouts runs/bon/probe-poolcheck-dp1/rollouts.jsonl --reference runs/mvtf/v4-base \\
        --out runs/bon/poolcheck-dp1
"""

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from build_pool import part_of

PARTS = ("svamp", "mbpp", "bfcl_simple_python", "bfcl_multiple", "bfcl_irrelevance", "ifeval")


def read(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def stats(groups):
    """groups: {question: [(correct, tokens, thinking, capped)]} -> the measures over all answers."""
    answers = [a for g in groups.values() for a in g]
    if not answers:
        return None
    return {"questions": len(groups), "answers": len(answers),
            "accuracy": 100 * sum(a[0] for a in answers) / len(answers),
            "median_tokens": statistics.median(a[1] for a in answers),
            "mean_tokens": statistics.fmean(a[1] for a in answers),
            "median_thinking": statistics.median(a[2] for a in answers),
            "capped": 100 * sum(a[3] for a in answers) / len(answers)}


def interval(groups, key, rounds=1000, seed=0):
    """95% bootstrap interval of a measure, resampling whole questions."""
    rng = random.Random(seed)
    keys = list(groups)
    values = []
    for _ in range(rounds):
        sample = {i: groups[k] for i, k in enumerate(rng.choice(keys) for _ in keys)}
        values.append(stats(sample)[key])
    values.sort()
    return values[int(0.025 * rounds)], values[int(0.975 * rounds) - 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rollouts", required=True, help="the original model's answers on the pool sample")
    parser.add_argument("--reference", nargs="+", required=True,
                        help="the original model's evaluation runs on non-fresh sets (choosing set v4-base, questions_40 eval-base)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--pool", help="the pool file: IFEval-like questions split into real rows and IFEval-library-built ones")
    parser.add_argument("--acc-tol", type=float, default=5.0)
    parser.add_argument("--len-tol", type=float, default=0.15)
    parser.add_argument("--cap-tol", type=float, default=3.0)
    args = parser.parse_args()
    pool, ref = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    by_source = defaultdict(lambda: defaultdict(list))
    for r in read(args.rollouts):
        part = part_of(r)
        if part is None:
            continue
        thinking = len(r.get("thinking_ids") or []) or r["output_tokens"]
        answer = (bool(r["correct"]), r["output_tokens"], thinking, bool(r["truncated"]))
        pool[part][r["id"]].append(answer)
        by_source[(part, r["source"])][r["id"]].append(answer)
    for k, run in enumerate(args.reference):
        for r in read(Path(run) / "records.jsonl"):
            ref[r["dataset"]][(k, r["id"])].append((bool(r["correct"]), r["output_tokens"],
                                               r.get("reasoning_tokens") or r["output_tokens"], bool(r["truncated"])))
    report, lines = {}, ["part: pool vs choosing set [choosing 95% interval] -> verdict"]
    verdicts = []
    for part in PARTS:
        p, c = stats(pool[part]), stats(ref[part])
        if not p or not c:
            lines.append(f"  {part}: missing ({'pool' if not p else 'choosing set'})")
            continue
        rows = {}
        for key, tol, relative in (("accuracy", args.acc_tol, False), ("median_tokens", args.len_tol, True),
                                   ("median_thinking", args.len_tol, True), ("capped", args.cap_tol, False)):
            lo, hi = interval(ref[part], key)
            gap = (p[key] / c[key] - 1) if relative and c[key] else p[key] - c[key]
            ok = abs(gap) <= tol or lo <= p[key] <= hi
            rows[key] = {"pool": p[key], "choosing": c[key], "interval": [lo, hi], "gap": gap, "ok": ok}
            verdicts.append(ok)
        report[part] = {"pool": p, "choosing": c, "measures": rows,
                        "sources": {s: stats(g) for (pp, s), g in by_source.items() if pp == part}}
        cells = []
        for key, r in rows.items():
            unit = "%" if key in ("accuracy", "capped") else ""
            gap = f"{r['gap']:+.0%}" if key in ("median_tokens", "median_thinking") else f"{r['gap']:+.1f}"
            cells.append(f"{key} {r['pool']:.0f}{unit} vs {r['choosing']:.0f}{unit} [{r['interval'][0]:.0f}, "
                         f"{r['interval'][1]:.0f}] {gap} {'OK' if r['ok'] else 'CHECK'}")
        lines.append(f"  {part} ({p['questions']} pool q / {c['questions']} choosing q):\n      " + "\n      ".join(cells))
        for (pp, s), g in sorted(by_source.items()):
            if pp == part and len({s2 for (p2, s2) in by_source if p2 == part}) > 1:
                st = stats(g)
                lines.append(f"      - {s}: accuracy {st['accuracy']:.0f}%, median tokens {st['median_tokens']:.0f}")
    if args.pool:  # dp2: 61% of IFEval-like is built from IFEval's library; a pass carried by built prompts is weaker
        origin = {r["pool_id"]: r.get("origin", "real") for r in read(args.pool) if r.get("pool_id")}
        split = defaultdict(lambda: defaultdict(list))
        for r in read(args.rollouts):
            if part_of(r) == "ifeval":
                split[origin.get(r["id"], "real")][r["id"]].append(
                    (bool(r["correct"]), r["output_tokens"], len(r.get("thinking_ids") or []) or r["output_tokens"],
                     bool(r["truncated"])))
        c = stats(ref["ifeval"])
        for kind, groups in sorted(split.items()):
            st = stats(groups)
            lines.append(f"  ifeval {kind} prompts ({st['questions']} q): accuracy {st['accuracy']:.0f}% vs {c['accuracy']:.0f}%, "
                         f"median thinking {st['median_thinking']:.0f} vs {c['median_thinking']:.0f}, "
                         f"median tokens {st['median_tokens']:.0f} vs {c['median_tokens']:.0f}")
            report.setdefault("ifeval_by_origin", {})[kind] = st
    instr = defaultdict(list)  # IFEval-like: pass rate per instruction id (the choosing run does not record them)
    for r in read(args.rollouts):
        if part_of(r) == "ifeval":
            for i in r.get("instruction_id_list") or []:
                instr[i].append(bool(r["correct"]))
    if instr:
        lines.append("  ifeval pool, pass rate per instruction id (answers):")
        lines += [f"      {i}: {100 * sum(v) / len(v):.0f}% of {len(v)}" for i, v in sorted(instr.items(), key=lambda kv: -len(kv[1]))]
    lines.append(f"verdict: {'ALL OK' if verdicts and all(verdicts) else 'CHECK the lines marked CHECK'} "
                 f"({sum(verdicts)}/{len(verdicts)} measures OK)")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
