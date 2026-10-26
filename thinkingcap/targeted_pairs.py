"""Targeted DPO pairs from a label_cuts.py lot: shorten code and instructions, let math think longer and be right.

From the lot's paragraph-end cuts (20 re-answers each) and the model's 8 replies per question:
  code / instructions (--shorten-sources)   kind "shorten": chosen = the EARLIEST cut whose re-answers are right at least
      as often as the full reply (and >= --safe) + a right answer from there; rejected = the full reply (right or wrong).
      Only when the cut is >= --min-save shorter. Thinking barely changes these types' outcome, so the cut keeps the answer.
  math (--long-sources)   kind "hard": chosen = a full right reply; rejected = its latest cut that still fails
      (<= --fail of re-answers right) + the wrong answer written there: stopping too early is what goes wrong.
      kind "guard": chosen = the question's longest right reply; rejected = its shortest finished wrong reply, when the
      right one is longer: more thinking, more right.
  --right-sources (math)   kind "right": the question's own right reply beats its own wrong one of similar length
      (within --right-band), so the pair says "be right" and nothing about length (Lightning McQwen3's correctness
      DPO; dpo.py's NLL anchor keeps the right reply likely). One pair per question.
  a type in both lists (code in stopB6) gets both kinds: shorten where a cut is safe, hard/guard where stopping fails.
  --guard-mode shorter (stopV3): the guard flips to the balanced-DPO direction: the question's SHORTEST right reply beats
      its LONGEST wrong or unfinished one, when the wrong one is longer, so it protects accuracy and shortens at once
      ("longer" says "more thinking, more right", which is wrong for code and instructions); --guard-sources picks the types
      (default: the --long-sources). --hard-ratio R caps "hard" pairs at R x the type's shorten pairs for the
      --hard-cap-sources (types whose answer barely depends on thinking), so brakes cannot outweigh the stops.
  every other type: no pairs (left as they are).
--rule question (shorten only; questions with >= --min-samples full replies, else the strict rule above): the bar is the
  question's own rate q over all its full replies (unfinished ones count as wrong, as in the evaluation), the quantity the
  benchmark scores, not >= 95%. A cut is safe when it beats q + --margin on the first half of its re-answers and still
  reaches q on the second half (no luck from picking the earliest cut); lots labelled before the per-answer outcomes were
  kept use the cut's whole rate + margin and require the next cut to reach q as well. Questions never solved (q = 0) still
  need a right answer from the cut; teaching "stop with the same wrong answer" is left for a separate, later test.

    python targeted_pairs.py --cal runs/bon/pairs-plotA/cal --rollouts runs/bon/probe-plotA/rollouts.jsonl \\
        runs/bon/kprobe-plotA/rollouts.jsonl --keep runs/bon/plotA-clean.jsonl --out runs/bon/pairs-tg1
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def read(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cal", required=True)
    parser.add_argument("--rollouts", nargs="+", required=True)
    parser.add_argument("--keep", required=True, help="clean question ids (label_cuts' --keep)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--shorten-sources", default="kodcode,ifeval_like")
    parser.add_argument("--long-sources", default="gsm_ic,gsm_plus")
    parser.add_argument("--right-sources", default="none")
    parser.add_argument("--guard-mode", default="longer", choices=("longer", "shorter"),
                        help="guard pairs: longest right over shortest wrong (longer) or shortest right over longest wrong")
    parser.add_argument("--guard-sources", default="", help="types given guard pairs (default: --long-sources)")
    parser.add_argument("--hard-ratio", type=float, default=0.0, help="cap hard pairs at this x shorten pairs (0: no cap)")
    parser.add_argument("--hard-cap-sources", default="none", help="types the --hard-ratio cap applies to")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--right-band", type=float, default=0.25, help="right pairs: lengths within this share")
    parser.add_argument("--safe", type=float, default=0.95)
    parser.add_argument("--fail", type=float, default=0.25)
    parser.add_argument("--min-save", type=float, default=0.25)
    parser.add_argument("--rule", default="strict", choices=("strict", "question"))
    parser.add_argument("--margin", type=float, default=0.10, help="question rule: share above q a cut must reach")
    parser.add_argument("--min-samples", type=int, default=20, help="question rule: full replies needed for a reliable q")
    args = parser.parse_args()
    cal, out = Path(args.cal), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    short_src, long_src = set(args.shorten_sources.split(",")), set(args.long_sources.split(","))
    right_src = set(args.right_sources.split(","))
    guard_src = set(args.guard_sources.split(",")) if args.guard_sources else long_src
    allowed = {r["id"] for r in read(args.keep)}
    rollouts, tries, unfinished = defaultdict(list), defaultdict(list), defaultdict(list)
    for p in args.rollouts:
        if Path(p).exists():
            for r in read(p):
                if r["id"] in allowed:
                    tries[r["id"]].append(bool(r["correct"]) and r["closed_thinking"] and not r["truncated"])
                if r["id"] in allowed and r["closed_thinking"] and not r["truncated"]:
                    rollouts[r["id"]].append(r)
                elif r["id"] in allowed:
                    unfinished[r["id"]].append(r)
    replies = {r["reply"]: r for r in read(cal / "replies.jsonl")}
    verdict = {v["key"]: v for v in read(cal / "verdicts.jsonl")}
    cuts = defaultdict(list)
    for c in read(cal / "cuts.jsonl"):
        cuts[c["reply"]].append(c)

    def full_reply(rep):
        """The rollout a label_cuts reply was made from (same question, outcome and length)."""
        return next((r for r in rollouts[rep["id"]] if bool(r["correct"]) == rep["correct"]
                     and r["output_tokens"] == rep["output_tokens"]), None)

    pairs, counts = [], defaultdict(int)
    for i, rep in replies.items():
        src = rep["source"]
        if src not in short_src | long_src:
            continue
        cs = sorted(cuts[i], key=lambda c: c["para"])
        full = full_reply(rep)
        if full is None or not cs:
            continue
        rate = lambda c: verdict[c["key"]]["right"] / max(1, verdict[c["key"]]["of"])
        full_rate = rate(cs[-1])  # the last cut is the full thinking (the natural stop)
        base = {"id": rep["id"], "source": src, "prompt_ids": full["prompt_ids"]}
        if src in short_src:
            hit = next((c for c in cs[:-1] if rate(c) >= max(args.safe, full_rate) and verdict[c["key"]]["right_ids"]), None)
            if args.rule == "question" and len(tries[rep["id"]]) >= args.min_samples:
                q = sum(tries[rep["id"]]) / len(tries[rep["id"]])

                def safe(k):
                    v = verdict[cs[k]["key"]]
                    if not v["right_ids"]:
                        return False
                    if len(v.get("outcomes") or []) >= 4:  # choose on one half, check on the other
                        half = len(v["outcomes"]) // 2
                        first, second = v["outcomes"][:half], v["outcomes"][half:]
                        return sum(first) / len(first) >= min(1.0, q + args.margin) and sum(second) / len(second) >= q
                    return rate(cs[k]) >= min(1.0, q + args.margin) and rate(cs[k + 1]) >= q

                hit = next((cs[k] for k in range(len(cs) - 1) if safe(k)), None)
                if hit:
                    counts[(src, "question rule")] += 1
            if hit:
                chosen = hit["thinking_ids"] + verdict[hit["key"]]["right_ids"]
                if len(chosen) <= (1 - args.min_save) * len(full["output_ids"]):
                    pairs.append(base | {"kind": "shorten", "chosen_ids": chosen, "rejected_ids": full["output_ids"]})
                    counts[(src, "shorten", "right" if rep["correct"] else "wrong")] += 1
        if src in long_src and rep["correct"]:  # a type may be in both lists: shorten where safe, brake where not
            early = [c for c in cs[:-1] if rate(c) <= args.fail and verdict[c["key"]]["wrong_ids"]]
            if early:
                c = early[-1]
                pairs.append(base | {"kind": "hard", "chosen_ids": full["output_ids"],
                                     "rejected_ids": c["thinking_ids"] + verdict[c["key"]]["wrong_ids"]})
                counts[(src, "hard")] += 1
    for qid, group in rollouts.items():  # guard: longest right over shortest wrong, or (shorter) the reverse
        src = group[0]["source"]
        if src not in guard_src:
            continue
        right = [r for r in group if r["correct"]]
        wrong = [r for r in group if not r["correct"]]
        if args.guard_mode == "shorter":
            wrong += unfinished[qid]  # a capped or unclosed reply is wrong, as in the evaluation
        if right and wrong:
            if args.guard_mode == "longer":
                r, w = max(right, key=lambda r: r["output_tokens"]), min(wrong, key=lambda r: r["output_tokens"])
                keep = r["output_tokens"] > w["output_tokens"]
            else:
                r, w = min(right, key=lambda r: r["output_tokens"]), max(wrong, key=lambda r: r["output_tokens"])
                keep = r["output_tokens"] < w["output_tokens"]
            if keep:
                pairs.append({"id": qid, "source": src, "kind": "guard", "prompt_ids": r["prompt_ids"],
                              "chosen_ids": r["output_ids"], "rejected_ids": w["output_ids"]})
                counts[(src, "guard")] += 1
    for qid, group in rollouts.items():  # "be right": own right vs own wrong of similar length
        src = group[0]["source"]
        if src not in right_src:
            continue
        right = [r for r in group if r["correct"]]
        wrong = [r for r in group if not r["correct"]]
        close = [(abs(r["output_tokens"] - w["output_tokens"]), r, w) for r in right for w in wrong
                 if abs(r["output_tokens"] - w["output_tokens"]) <= args.right_band * max(r["output_tokens"], w["output_tokens"])]
        if close:
            _, r, w = min(close, key=lambda t: t[0])
            pairs.append({"id": qid, "source": src, "kind": "right", "prompt_ids": r["prompt_ids"],
                          "chosen_ids": r["output_ids"], "rejected_ids": w["output_ids"]})
            counts[(src, "right")] += 1
    capped = set(args.hard_cap_sources.split(",")) if args.hard_ratio > 0 else set()
    if capped:  # brakes may not outweigh the stops on types whose answer barely depends on thinking
        rng = random.Random(args.seed)
        kept = []
        for src in sorted({p["source"] for p in pairs}):
            mine = [p for p in pairs if p["source"] == src]
            hard = [p for p in mine if p["kind"] == "hard"]
            if src in capped:
                rng.shuffle(hard)
                limit = int(args.hard_ratio * sum(p["kind"] == "shorten" for p in mine))
                counts[(src, "hard capped from")] = len(hard)
                hard = hard[:limit]
                counts[(src, "hard")] = len(hard)
            kept += [p for p in mine if p["kind"] != "hard"] + hard
        pairs = kept
        rng.shuffle(pairs)
    (out / "pairs.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pairs))
    text = "\n".join([f"{len(pairs)} targeted pairs"] + [f"  {' / '.join(k)}: {n}" for k, n in sorted(counts.items())])
    (out / "summary.txt").write_text(text + "\n")
    print(text, flush=True)


if __name__ == "__main__":
    main()
