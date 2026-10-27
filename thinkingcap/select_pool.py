"""Pick the training pool by how the original model BEHAVES on each question, matched to the choosing set (dp2, user).

dp1 matched prompt lengths only, and the original Qwen did not behave on the pool as on the evaluation (IFEval-like 89%
right vs 49%, math thinking +40%, ...). Here every candidate (an oversampled build_pool.py output) has been answered
--samples times by the original model; per evaluation part, each question gets two numbers, its solve rate and its median
thinking tokens, and so does each choosing-set question (the original model's own run on it, v4-base). The two are
binned the same way (solve rate: never / under half / half or more but not always / always; thinking: the choosing
set's terciles) and the pool is drawn so each cell holds the choosing set's share of the part (+0.5 smoothing per cell:
only 40 choosing questions per part). A cell short of candidates borrows from its nearest cells (thinking first, then
solve rate); every shortfall is reported.

Selecting on the same answers that are reported would flatter the match (questions picked for a lucky draw regress
afterwards): the check is pool_check.py on NEW answers (probe_curve.py --seed) to a sample of the selected pool.
The fresh set is never read.

    python select_pool.py --pool data/poolcand-dp2.jsonl --rollouts runs/bon/probe-poolcand-dp2/rollouts.jsonl \\
        --reference runs/mvtf/v4-base --per-benchmark 3000 --out data/pool-dp2.jsonl
"""

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from build_pool import BFCL_SIZES

PARTS = ("svamp", "mbpp", "bfcl_simple_python", "bfcl_multiple", "bfcl_irrelevance", "ifeval")


def read(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def solve_bin(rate):
    return 0 if rate == 0 else 1 if rate < 0.5 else 2 if rate < 1 else 3


def think_bin(tokens, edges):
    return sum(tokens > e for e in edges)


def features(answers):
    """[(correct, thinking tokens)] -> (solve rate, median thinking)."""
    return sum(c for c, _ in answers) / len(answers), statistics.median(t for _, t in answers)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pool", required=True, help="oversampled build_pool.py output (rows carry 'part')")
    parser.add_argument("--rollouts", required=True, help="the original model's answers to every --pool question")
    parser.add_argument("--reference", nargs="+", required=True,
                        help="the original model's evaluation runs on non-fresh sets (choosing set v4-base, questions_40 eval-base)")
    parser.add_argument("--per-benchmark", type=int, default=3000, help="questions per benchmark (BFCL's 3 parts share one)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    rows = {}
    for index, r in enumerate(read(args.pool)):  # probe_curve.py names a question by its pool_id, else "<source>-<line>"
        key = r.get("pool_id") or f"{r['source']}-{index}"
        rows.setdefault(r["part"], {})[key] = r | {"_key": key}
    answers = defaultdict(list)
    for r in read(args.rollouts):
        thinking = len(r.get("thinking_ids") or []) or r["output_tokens"]
        answers[r["id"]].append((bool(r["correct"]), thinking))
    ref = defaultdict(lambda: defaultdict(list))
    for k, run in enumerate(args.reference):
        for r in read(Path(run) / "records.jsonl"):
            ref[r["dataset"]][(k, r["id"])].append((bool(r["correct"]), r.get("reasoning_tokens") or r["output_tokens"]))

    targets = {"svamp": args.per_benchmark, "mbpp": args.per_benchmark, "ifeval": args.per_benchmark}
    total = sum(BFCL_SIZES.values())
    targets |= {p: round(args.per_benchmark * n / total) for p, n in BFCL_SIZES.items()}
    pool, report, lines = [], {}, ["part: kept/wanted; cells (solve bin x thinking bin) choosing share vs kept share"]
    for part in PARTS:
        want = targets[part]
        refs = [features(a) for a in ref[part].values()]
        assert refs, f"no choosing-set answers for {part}"
        medians = sorted(t for _, t in refs)
        edges = [medians[len(medians) // 3], medians[2 * len(medians) // 3]]
        cell_ref = Counter((solve_bin(s), think_bin(t, edges)) for s, t in refs)
        cells = [(a, b) for a in range(4) for b in range(3)]
        share = {c: (cell_ref[c] + 0.5) / (len(refs) + 0.5 * len(cells)) for c in cells}
        bins, unanswered = defaultdict(list), 0
        for key, row in rows.get(part, {}).items():
            if key not in answers:
                unanswered += 1
                continue
            s, t = features(answers[key])
            bins[(solve_bin(s), think_bin(t, edges))].append(row)
        for b in bins.values():
            rng.shuffle(b)
        need = {c: round(want * share[c]) for c in cells}
        picked, borrowed = Counter(), Counter()
        chosen = []
        for c in sorted(cells, key=lambda c: -need[c]):
            got = bins[c][:need[c]]
            del bins[c][:len(got)]
            picked[c] += len(got)
            # nearest cells: same solve bin first (thinking +-1, +-2), then solve +-1
            order = sorted((d for d in cells if d != c), key=lambda d: (abs(d[0] - c[0]) * 3 + abs(d[1] - c[1])))
            for d in order:
                if len(got) >= need[c]:
                    break
                extra = bins[d][:need[c] - len(got)]
                del bins[d][:len(extra)]
                borrowed[(c, d)] += len(extra)
                got += extra
            chosen += got
        kept = Counter()
        for row in chosen:
            s, t = features(answers[row["_key"]])
            kept[(solve_bin(s), think_bin(t, edges))] += 1
        acc = lambda rs: 100 * statistics.fmean(features(answers[r["_key"]])[0] for r in rs) if rs else 0
        med = lambda rs: statistics.median(features(answers[r["_key"]])[1] for r in rs) if rs else 0
        report[part] = {"wanted": want, "kept": len(chosen), "unanswered": unanswered, "thinking_edges": edges,
                        "choosing_cells": {str(c): cell_ref[c] for c in cells},
                        "kept_cells": {str(c): kept[c] for c in cells},
                        "borrowed": {f"{c}<-{d}": n for (c, d), n in borrowed.items()},
                        "choosing_accuracy": 100 * statistics.fmean(s for s, _ in refs),
                        "choosing_median_thinking": statistics.median(medians),
                        "kept_accuracy_selection_answers": acc(chosen),
                        "kept_median_thinking_selection_answers": med(chosen),
                        "sources": dict(Counter(r["source"] for r in chosen))}
        r = report[part]
        lines.append(f"  {part}: {len(chosen)}/{want}" + (f" (SHORT {want - len(chosen)})" if len(chosen) < want else "")
                     + f"; accuracy {r['kept_accuracy_selection_answers']:.0f}% vs choosing {r['choosing_accuracy']:.0f}%"
                     + f"; median thinking {r['kept_median_thinking_selection_answers']:.0f} vs "
                     f"{r['choosing_median_thinking']:.0f}; borrowed {sum(borrowed.values())}; {r['sources']}")
        pool += [{k: v for k, v in row.items() if k != "_key"} | {"part": part} for row in chosen]
    rng.shuffle(pool)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in pool))
    out.with_suffix(".report.json").write_text(json.dumps(report, indent=1))
    lines.insert(0, f"{len(pool)} pool questions -> {out} (selection answers: the held-out check is pool_check.py)")
    out.with_suffix(".report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
