"""LUPIN spec 2.0: the training PAIRS must match the evaluation, like the pool (user, 27 Oct night: "as important").

A recipe's raw pairs (lupin_pairs.py) skew the pool: always-solved easy questions give many shorten pairs, hard ones few
or none, so training would be dominated by easy questions. Here the pairs are drawn per evaluation part (each benchmark
an equal share, BFCL split as its 400 / 200 / 240) and, inside a part, per cell of solve rate x thinking length, each
cell holding the share the ORIGINAL model's own evaluation runs have there (choosing set v4-base + questions_40
eval-base; same bins as select_pool.py, thinking in --think-bins quantiles). A question's solve rate and median thinking
come from ALL its answers (>= 10 per pool question). Inside a cell, questions are visited round-robin (one pair per
question before any question gives a second). A cell short of pairs is reported, never filled from another cell.

    python lupin_select.py --pairs runs/bon/pairs-L1/pairs.jsonl --rollouts all.jsonl --reference runs/mvtf/v4-base \\
        runs/mvtf/eval-base --total 3200 --out runs/bon/pairs-L1-matched
"""

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from build_pool import BFCL_SIZES
from select_pool import PARTS, features, solve_bin, think_bin


def read(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--rollouts", nargs="+", required=True, help="every answer to the pool questions")
    parser.add_argument("--reference", nargs="+", required=True)
    parser.add_argument("--total", type=int, default=3200, help="pairs wanted in all (200 steps x 16)")
    parser.add_argument("--think-bins", type=int, default=4)
    parser.add_argument("--part-share", default="",
                        help="P1 (29 Oct): benchmark shares instead of 25%% each, e.g. svamp=0.15,mbpp=0.25,ifeval=0.35,bfcl=0.25 "
                             "(bfcl split 400/200/240 inside)")
    parser.add_argument("--solve-tilt", default="",
                        help="P3: multiply the reference share of solve bins, e.g. 2=2,3=0.5 (more medium-difficulty, "
                             "fewer always-solved questions)")
    parser.add_argument("--think-tilt-bfcl", type=float, default=1.0,
                        help="P4: multiply the share of the longest-thinking bin in the BFCL parts by this")
    parser.add_argument("--shorten-ratio-part", default="",
                        help="per-part pressure: max shorten pairs per be-right pair, e.g. svamp=1,mbpp=2,ifeval=3,"
                             "bfcl_simple_python=2,bfcl_multiple=2,bfcl_irrelevance=2 (ours, 29 Oct: from each type's measured "
                             "accuracy cost); overrides --max-shorten-ratio for the parts it names")
    parser.add_argument("--max-shorten-ratio", type=float, default=0.0,
                        help="after matching, per part at most this many 'shorten' pairs per 'be right' pair (guard + right "
                             "+ hard); michal-bida's 1:1 balance (29 Oct: matching had pushed it to 2-3:1). 0: off")
    parser.add_argument("--kind-share-part", default="",
                        help="Run A (30 Oct build spec): per part, the share of each pair kind, e.g. "
                             "'svamp=guard:0.4,right:0.6;mbpp=shorten:0.7,guard:0.15,right:0.15'; a kind not named gets "
                             "none; parts not named keep the recipe's own mix")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    answers = defaultdict(list)
    for path in args.rollouts:
        with open(path) as handle:
            for line in handle:
                if line.strip():
                    r = json.loads(line)
                    finished = r.get("closed_thinking", True) and not r.get("truncated", False)
                    answers[r["id"]].append((bool(r["correct"]) and finished, len(r.get("thinking_ids") or []) or r["output_tokens"]))
    ref = defaultdict(lambda: defaultdict(list))
    for k, run in enumerate(args.reference):
        for r in read(Path(run) / "records.jsonl"):
            ref[r["dataset"]][(k, r["id"])].append((bool(r["correct"]), r.get("reasoning_tokens") or r["output_tokens"]))
    pairs = read(args.pairs)
    by_part = defaultdict(list)
    for p in pairs:
        by_part[p["part"]].append(p)

    bfcl = sum(BFCL_SIZES.values())
    shares = {"svamp": 0.25, "mbpp": 0.25, "ifeval": 0.25, "bfcl": 0.25}
    shares |= dict((k, float(v)) for k, v in (x.split("=") for x in args.part_share.split(",") if x))
    total_share = sum(shares.values())
    share = {k: shares[k] / total_share for k in ("svamp", "mbpp", "ifeval")}
    share |= {p: shares["bfcl"] / total_share * n / bfcl for p, n in BFCL_SIZES.items()}
    kind_share = {}
    for item in (x for x in args.kind_share_part.split(";") if x):
        part, spec = item.split("=", 1)
        kind_share[part] = {k: float(v) for k, v in (kv.split(":") for kv in spec.split(","))}
    solve_tilt = dict((int(k), float(v)) for k, v in (x.split("=") for x in args.solve_tilt.split(",") if x))
    chosen, report, lines = [], {}, [f"pairs matched to the evaluation: wanted {args.total}"]
    for part in PARTS:
        want = round(args.total * share[part])
        refs = [features(a) for a in ref[part].values()]
        medians = sorted(t for _, t in refs)
        edges = [medians[k * len(medians) // args.think_bins] for k in range(1, args.think_bins)]
        cells = [(a, b) for a in range(4) for b in range(args.think_bins)]
        cell_ref = Counter((solve_bin(s), think_bin(t, edges)) for s, t in refs)
        cell_share = {c: (cell_ref[c] + 0.5) / (len(refs) + 0.5 * len(cells)) for c in cells}
        tilt = {c: solve_tilt.get(c[0], 1.0) * (args.think_tilt_bfcl if part.startswith("bfcl") and c[1] == args.think_bins - 1
                                                 else 1.0) for c in cells}
        if any(v != 1.0 for v in tilt.values()):
            cell_share = {c: cell_share[c] * tilt[c] for c in cells}
            norm = sum(cell_share.values())
            cell_share = {c: v / norm for c, v in cell_share.items()}
        no_answers = sum(p["id"] not in answers for p in by_part.get(part, []))
        # Run A (30 Oct build spec): a part's pairs split by kind at fixed shares, each kind cell-matched on its own
        mix = kind_share.get(part) or {None: 1.0}
        got, short, kinds = Counter(), Counter(), Counter()
        empty, excluded = [], 0.0
        for kind, k_share in mix.items():
            per_cell = defaultdict(lambda: defaultdict(list))  # cell -> question -> its pairs
            for p in by_part.get(part, []):
                if p["id"] in answers and (kind is None or p["kind"] == kind):
                    s, t = features(answers[p["id"]])
                    per_cell[(solve_bin(s), think_bin(t, edges))][p["id"]].append(p)
            # cells where this recipe gives NO pair at all (mostly never-solved questions: nothing right to prefer)
            # cannot be trained on; their reference share is spread over the cells that have pairs (29 Oct: keeping
            # their quota and trimming every part to the weakest part's fill left 1,284 of 3,200 pairs). Reported.
            empty += [f"{kind or ''}{c}" for c in cells if not per_cell[c]]
            live = sum(cell_share[c] for c in cells if per_cell[c]) or 1
            excluded = max(excluded, 1 - live)
            share_k = {c: (cell_share[c] / live if per_cell[c] else 0.0) for c in cells}
            for c in cells:
                need = round(want * k_share * share_k[c])
                questions = list(per_cell[c].values())
                rng.shuffle(questions)
                for q in questions:
                    rng.shuffle(q)
                taken, depth = [], 0
                while len(taken) < need and any(len(q) > depth for q in questions):
                    for q in questions:
                        if len(q) > depth and len(taken) < need:
                            taken.append(q[depth])
                    depth += 1
                got[c] += len(taken)
                short[c] += need - len(taken)
                kinds.update(p["kind"] for p in taken)
                chosen += taken
        n = sum(got.values())
        report[part] = {"wanted": want, "kept": n, "short_cells": {str(c): v for c, v in short.items() if v > 0},
                        "cells_without_pairs": [str(c) for c in empty], "reference_share_without_pairs": excluded,
                        "kinds": dict(kinds), "pairs_without_answers": no_answers,
                        "reference_cells": {str(c): cell_ref[c] for c in cells}, "kept_cells": {str(c): got[c] for c in cells}}
        lines.append(f"  {part}: {n}/{want} pairs; kinds {dict(kinds)}; {excluded:.0%} of the reference in cells with no pair "
                     f"(spread over the rest); short {sum(short.values())} in "
                     f"{sum(v > 0 for v in short.values())} cells " +
                     ", ".join(f"solve{c[0]}/think{c[1]} -{v}" for c, v in short.items() if v > 0))
    ratios = dict((k, float(v)) for k, v in (x.split("=") for x in args.shorten_ratio_part.split(",") if x))
    if args.max_shorten_ratio > 0 or ratios:  # pressure kept AFTER matching, per part
        balanced = []
        for part in PARTS:
            ratio = ratios.get(part, args.max_shorten_ratio)
            if ratio <= 0:
                balanced += [p for p in chosen if p["part"] == part]
                continue
            mine = [p for p in chosen if p["part"] == part]
            short = [p for p in mine if p["kind"] == "shorten"]
            rest = [p for p in mine if p["kind"] != "shorten"]
            rng.shuffle(short)
            limit = round(ratio * len(rest))
            balanced += rest + short[:limit]
            report[part]["kept"] = len(rest) + min(len(short), limit)
            report[part]["shorten_dropped_for_balance"] = max(0, len(short) - limit)
        lines.append(f"pressure: shorten per be-right pair capped per part {ratios or args.max_shorten_ratio}; "
                     f"{len(chosen) - len(balanced)} shorten pairs dropped")
        chosen = balanced
    # part shares stay the evaluation's: a part that filled less of its quota (hard cells give few pairs) sets the
    # fill ratio, the other parts are trimmed to it at random (which keeps their cell mix)
    fill = min(report[part]["kept"] / max(1, report[part]["wanted"]) for part in PARTS)
    trimmed = []
    for part in PARTS:
        mine = [p for p in chosen if p["part"] == part]
        rng.shuffle(mine)
        keep = round(report[part]["wanted"] * fill)
        report[part]["after_share_trim"] = min(keep, len(mine))
        trimmed += mine[:keep]
    lines.append(f"fill ratio {fill:.2f}: every part trimmed to {fill:.0%} of its quota, {len(chosen) - len(trimmed)} pairs dropped")
    chosen = trimmed
    kept = Counter(p["part"] for p in chosen)
    lines.append("part shares (kept vs target): " + ", ".join(
        f"{p} {kept[p] / max(1, len(chosen)):.1%} vs {share[p]:.1%}" for p in PARTS))
    rng.shuffle(chosen)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pairs.jsonl").write_text("".join(json.dumps(p) + "\n" for p in chosen))
    (out / "match.json").write_text(json.dumps(report, indent=1))
    lines.insert(1, f"kept {len(chosen)} pairs, kinds {dict(Counter(p['kind'] for p in chosen))}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
