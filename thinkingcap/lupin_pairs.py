"""LUPIN step 2: preference pairs from the model's own graded answers, one recipe per variant (user, 27 Oct).

Input: rollouts.jsonl from probe_curve.py --generate-only (several graded answers per question). Answers are grouped
by question; a question's solve rate p = right answers / all answers (capped or unclosed ones count as wrong, as in the
evaluation). Pair kinds keep dpo.py's names, so --mix can weigh them:

    shorten  a right answer beats a LONGER right answer           ("be shorter")
    guard    the shortest right answer beats a wrong / capped one ("be right, and short")
    hard     a LONGER wrong answer beats a shorter wrong one        (DAST: on questions it fails, keep thinking)
    right    a right answer beats a wrong one of similar length    ("be right", says nothing about length)

Recipes (--recipe), each from the literature pass of 27 Oct (EXPERIMENT_LOG):
    long2short  Kimi k1.5 / michal-bida: shortest right > longest wrong or capped (guard), and > longest right when that
                one is >= --long-ratio x longer (shorten); per part, shorten pairs capped at the guard pairs.
    dast        DAST (2503.04472): shortest right > longest right (shorten) + longest wrong > shortest wrong (hard),
                the widest pair of each kind per question, the narrowest --margin-drop share of each kind dropped.
    adar1       Ada-R1 (2504.21659): shortest right > each of the 2 longest answers of any outcome.
    rightwrong  only similar-length right > wrong pairs (within --right-band).
Modifiers, any recipe:
    --min-solve P   difficulty rule (user, 27 Oct; ALP 2506.05256 / DAST budget / Difficulty-Aware DLER): on questions
                    solved >= P of the time the recipe runs as usual ("be shorter"); below P the question gets NO length
                    push at all: no shorten pair, and its guard / Ada-R1 pairs become one similar-length right > wrong pair,
                    so a hard question may keep thinking as long as it needs.
    --drop-similar J  drop pairs whose two answers share more than J of their token 4-grams (near-identical pairs push
                    probability onto unintended outputs: likelihood displacement, 2410.08847).
    --add-right     add the similar-length right > wrong pairs to any recipe.
Never-solved questions give no pair except hard (DAST); always-solved ones no guard / right pair.

    python lupin_pairs.py --rollouts runs/bon/probe-ans-dp1/rollouts.jsonl --recipe long2short --out runs/bon/pairs-L0
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from build_pool import part_of


def read(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def finished(r):
    return r["closed_thinking"] and not r["truncated"]


def right(r):
    return bool(r["correct"]) and finished(r)


def grams(ids, n=4):
    return {tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)}


def similar(a, b):
    ga, gb = grams(a), grams(b)
    return len(ga & gb) / max(1, len(ga | gb))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rollouts", nargs="+", required=True)
    parser.add_argument("--recipe", required=True, choices=("long2short", "dast", "adar1", "rightwrong"))
    parser.add_argument("--long-ratio", type=float, default=1.5)
    parser.add_argument("--margin-drop", type=float, default=0.15)
    parser.add_argument("--right-band", type=float, default=0.25)
    parser.add_argument("--min-solve", type=float, default=0.0)
    parser.add_argument("--min-solve-part", default="",
                        help="per-part protection override, e.g. svamp=0.8 (29 Oct: math lost most; Lightning McQwen3 / "
                             "michal-bida both lost on SVAMP)")
    parser.add_argument("--right-parts", default="",
                        help="parts that also get a similar-length right > wrong pair per question (Lightning McQwen3's "
                             "math-only correctness DPO), e.g. svamp")
    parser.add_argument("--easy-ratio", type=float, default=0.0,
                        help="long2short: on ALWAYS-solved questions a longer right answer counts from this ratio (29 Oct: 1.2; "
                             "their right answers are close in length, so 1.5x gave few pairs)")
    parser.add_argument("--shorten-cap", type=float, default=1.0,
                        help="long2short: shorten pairs per part at most this many times the guard pairs (always-solved "
                             "questions give no guard pair)")
    parser.add_argument("--drop-similar", type=float, default=0.0)
    parser.add_argument("--add-right", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    groups = defaultdict(list)
    for path in args.rollouts:
        for r in read(path):
            groups[r["id"]].append(r)

    pairs, counts = [], Counter()
    overrides = dict((k, float(v)) for k, v in (x.split("=") for x in args.min_solve_part.split(",") if x))
    right_parts = set(x for x in args.right_parts.split(",") if x)

    def add(kind, chosen, rejected, q):
        pairs.append({"id": q[0]["id"], "source": q[0]["source"], "part": part_of(q[0]), "kind": kind,
                      "prompt_ids": chosen["prompt_ids"], "chosen_ids": chosen["output_ids"],
                      "rejected_ids": rejected["output_ids"],
                      "margin": abs(len(rejected["output_ids"]) - len(chosen["output_ids"])) /
                                max(len(chosen["output_ids"]), len(rejected["output_ids"]))})

    for q in groups.values():
        rights = sorted((r for r in q if right(r)), key=lambda r: r["output_tokens"])
        wrongs = [r for r in q if not right(r)]  # wrong, capped or unclosed: all score as wrong
        p = len(rights) / len(q)
        part = part_of(q[0])
        may_shorten = p >= overrides.get(part, args.min_solve)
        if not may_shorten:  # a hard question: only "be right", never "be shorter" (nor "short right > long wrong")
            if args.recipe in ("long2short", "adar1"):
                close = [(abs(r["output_tokens"] - w["output_tokens"]), r, w) for r in rights for w in wrongs if finished(w)
                         and abs(r["output_tokens"] - w["output_tokens"]) <= args.right_band * max(r["output_tokens"], w["output_tokens"])]
                if close:
                    _, r, w = min(close, key=lambda t: t[0])
                    add("right", r, w, q)
            if args.recipe != "dast":
                continue
        if args.recipe == "long2short":
            if rights and wrongs:
                add("guard", rights[0], max(wrongs, key=lambda r: r["output_tokens"]), q)
            ratio = args.easy_ratio if args.easy_ratio and len(rights) == len(q) else args.long_ratio
            if may_shorten and len(rights) > 1 and rights[-1]["output_tokens"] >= ratio * rights[0]["output_tokens"]:
                add("shorten", rights[0], rights[-1], q)
        elif args.recipe == "dast":
            if may_shorten and len(rights) > 1 and rights[-1]["output_tokens"] > rights[0]["output_tokens"]:
                add("shorten", rights[0], rights[-1], q)
            done = sorted((r for r in wrongs if finished(r)), key=lambda r: r["output_tokens"])
            if len(done) > 1 and done[-1]["output_tokens"] > done[0]["output_tokens"]:
                add("hard", done[-1], done[0], q)
        elif args.recipe == "adar1":
            if rights:
                others = sorted((r for r in q if r is not rights[0]), key=lambda r: -r["output_tokens"])[:2]
                for o in others:
                    if right(o) and (not may_shorten or o["output_tokens"] <= rights[0]["output_tokens"]):
                        continue
                    add("shorten" if right(o) else "guard", rights[0], o, q)
        if args.recipe == "rightwrong" or args.add_right or part in right_parts:
            close = [(abs(r["output_tokens"] - w["output_tokens"]), r, w) for r in rights for w in wrongs if finished(w)
                     and abs(r["output_tokens"] - w["output_tokens"]) <= args.right_band * max(r["output_tokens"], w["output_tokens"])]
            if close:
                _, r, w = min(close, key=lambda t: t[0])
                add("right", r, w, q)

    if args.recipe == "long2short":  # michal-bida's balance: per part, no more shorten than guard pairs
        kept = []
        for part in {p["part"] for p in pairs}:
            mine = [p for p in pairs if p["part"] == part]
            short = [p for p in mine if p["kind"] == "shorten"]
            rng.shuffle(short)
            limit = round(args.shorten_cap * sum(p["kind"] == "guard" for p in mine))
            counts[(part, "shorten capped from")] = len(short)
            kept += [p for p in mine if p["kind"] != "shorten"] + short[:limit]
        pairs = kept
    if args.recipe == "dast" and args.margin_drop > 0:  # DAST: the narrowest pairs of each kind carry little signal
        kept = []
        for kind in {p["kind"] for p in pairs}:
            mine = sorted((p for p in pairs if p["kind"] == kind), key=lambda p: p["margin"])
            kept += mine[int(len(mine) * args.margin_drop):]
        pairs = kept
    if args.drop_similar > 0:
        before = len(pairs)
        pairs = [p for p in pairs if similar(p["chosen_ids"], p["rejected_ids"]) <= args.drop_similar]
        counts[("all", "dropped as near-identical")] = before - len(pairs)
    for p in pairs:
        counts[(p["part"], p["kind"])] += 1
        p.pop("margin")
    rng.shuffle(pairs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pairs.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pairs))
    solve = [sum(right(r) for r in q) / len(q) for q in groups.values()]
    text = "\n".join([f"{len(pairs)} pairs, recipe {args.recipe}" + (f", min solve {args.min_solve}" if args.min_solve else "")
                      + f" ({len(groups)} questions; never solved {sum(s == 0 for s in solve)}, always solved "
                      f"{sum(s == 1 for s in solve)})"] + [f"  {a} / {b}: {n}" for (a, b), n in sorted(counts.items())])
    (out / "summary.txt").write_text(text + "\n")
    print(text, flush=True)


if __name__ == "__main__":
    main()
