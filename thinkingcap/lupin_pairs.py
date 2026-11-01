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
    --band PART:LO:HI  (1 Nov, math hole) on PART's questions whose answers think LO-HI tokens on average (mean thinking
                    length of the answers given), no shorten pair: the trained models over-cut exactly there (SVAMP
                    470-905: -13.5 pts) while the cut is free above it. Guard pairs stay.
    --overcut FILE  ("don't over-cut") a trained model's answers to banded questions (track.py onpolicy): per question its
                    shortest finished WRONG answer that thinks < --overcut-ratio x the winner loses to the original's
                    median-length right answer (kind guard, tagged overcut; added after the shorten cap). The original's
                    own answers give the same pair where one of its wrong answers is that short ("natural" cases).
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
    parser.add_argument("--median-shorten-part", default="",
                        help="Run B (30 Oct build spec): per part, PART=P: the shorten pair becomes the MEDIAN-length right "
                             "answer > the longest right one (>= --long-ratio x), only on questions solved >= P "
                             "(teaches 'drop the runaway tail', not 'skip the re-read'), e.g. svamp=0.8")
    parser.add_argument("--band", default="", help="PART:LO:HI, no shorten pair on PART's questions thinking LO-HI")
    parser.add_argument("--overcut", nargs="*", default=[], help="trained models' rollouts on banded questions (losers)")
    parser.add_argument("--overcut-ratio", type=float, default=0.75)
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
    band_part, band_lo, band_hi = (args.band.split(":") + ["", "0", "0"])[:3] if args.band else ("", 0, 0)
    band_lo, band_hi = float(band_lo), float(band_hi)

    def think(r):
        return len(r.get("thinking_ids") or r["output_ids"])

    def in_band(q):
        return part_of(q[0]) == band_part and band_lo <= sum(think(r) for r in q) / len(q) < band_hi
    overrides = dict((k, float(v)) for k, v in (x.split("=") for x in args.min_solve_part.split(",") if x))
    right_parts = set(x for x in args.right_parts.split(",") if x)
    median_parts = dict((k, float(v)) for k, v in (x.split("=") for x in args.median_shorten_part.split(",") if x))

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
        banded = in_band(q)
        if banded:
            counts[(part, "in band")] += 1
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
            if part in median_parts:  # Run B: median right > longest right, on questions solved >= the part's P
                mid = rights[(len(rights) - 1) // 2] if rights else None
                if p >= median_parts[part] and len(rights) > 2 and rights[-1]["output_tokens"] >= args.long_ratio * mid["output_tokens"]:
                    add("shorten", mid, rights[-1], q)
            elif may_shorten and len(rights) > 1 and rights[-1]["output_tokens"] >= ratio * rights[0]["output_tokens"]:
                if banded:
                    counts[(part, "shorten dropped (band)")] += 1
                else:
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
    if args.band:  # "don't over-cut": a too-short wrong answer loses to a normal right one
        trained = defaultdict(list)
        for path in args.overcut:
            for r in read(path):
                trained[r["id"]].append(r)
        for q in groups.values():
            if not in_band(q):
                continue
            rights = sorted((r for r in q if right(r)), key=lambda r: r["output_tokens"])
            if not rights:
                continue
            win = rights[(len(rights) - 1) // 2]
            for who, answers in (("trained", trained.get(q[0]["id"], [])), ("natural", q)):
                short = sorted((r for r in answers if finished(r) and not right(r)
                                and think(r) < args.overcut_ratio * think(win)), key=think)
                if short:
                    pairs.append({"id": q[0]["id"], "source": q[0]["source"], "part": part_of(q[0]), "kind": "guard",
                                  "overcut": who, "prompt_ids": win["prompt_ids"], "chosen_ids": win["output_ids"],
                                  "rejected_ids": short[0]["output_ids"], "margin": 0.0})
                    counts[(part_of(q[0]), f"overcut ({who})")] += 1
        counts[("all", "overcut questions with trained answers")] = len(trained)
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
        p.pop("overcut", None)
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
