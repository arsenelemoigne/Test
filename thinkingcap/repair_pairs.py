"""Track A pairs (GAME_PLAN.md v3, 5 Nov): "be right" pairs from a TRAINED model's own answers, no shortening pressure.

Per question (the model's 4 answers), two kinds, each a right answer over a FINISHED wrong one (no cut rejections):
  guard  the shortest right answer > the shortest finished wrong answer
  right  a right answer > a finished wrong answer of similar length (max(Lc/Lr, Lr/Lc) <= --ratio), exactly half of each
         part's right pairs with the right answer strictly LONGER (a correctness signal without a length signal)
Quotas per part (--quota part=n), half guard / half right; <= 3 pairs per question, <= 2 right pairs per question; no
sample pair used twice. --kinds guard (9 Nov): guard pairs only, the whole quota (the user's own 21 Oct rule, bon_pairs'
guard; the similar-length "right" kind matches Lightning McQwen3's correctness DPO and is left out). A question gives at
most one guard pair, so a part needs at least its quota of mixed-outcome questions. Mixed-outcome questions are the only source (a pair needs a right and a wrong answer).
--think-only PARTS (9 Nov, user, from the rc1-soup-30 full-test diagnosis): for these parts (ifeval) "shortest" is the
shortest THINKING, never the shortest whole answer, so a pair never rewards a shorter final response (length constraints:
"at least 600 words"). --prefer-reread (same day): the chosen right answer is the shortest one that RE-CHECKS THE QUESTION
(what exactly is asked, the example / assert, the constraints, whether a tool fits; regex markers on its thinking) when one
exists within 1.5x the shortest right answer's thinking, else the shortest: the shortened model's lost questions were
right calculations of the wrong question, the re-check it had dropped.
--count-rules (10 Nov, user): IFEval questions with a COUNTING rule (word / sentence / paragraph counts, keyword frequency,
number of sections, bullets or placeholders) fail with responses that are too short (the full-test diagnostic): there the
chosen right answer must have a RESPONSE at least as long as the rejected one's and THINKING no longer than 1.2x the
rejected one's (the pair then differs in response length and correctness only: "write what is asked", not "think more");
when no right answer fits, the question gives no pair.
The output is ordered so that every --block consecutive pairs hold the intended part x kind mix (training reads it in
order with dpo.py --keep-order: a checkpoint at step 5 has seen a representative slice, not one part).

    python repair_pairs.py --rollouts rp/rollouts.jsonl --out pairs-rp --quota svamp=640,mbpp=400,bfcl_irrelevance=240,...
"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from build_pool import part_of


REREAD = re.compile(r"\b(the (question|problem|task|user|prompt) (is )?(asks?|asking|says|wants|wanted|states|requires|mentions)|"
                    r"(re-?read|reading|read) (the|it) (question|problem|again|carefully)|let me re-?read|"
                    r"(they('re| are)|it('s| is)|we('re| are)) (being )?asked|asked (for|to)|"
                    r"(only|actually) (asks?|wants?)|wait,? (the|it) (question|problem|asks)|"
                    r"(check|compare|verify|test)\w* (it )?(against|with)? ?the (example|assert|test|sample|constraints?)|"
                    r"(none of the|no) (available )?(functions?|tools?) (fit|match|appl|is relevant|are relevant|can)|"
                    r"(does|do) (the|this|these|any) (functions?|tools?) (fit|match|apply|help)|"
                    r"the (constraints?|requirements?|instructions?) (are|is|say|require))", re.I)


COUNT_RULES = ("length_constraints:", "keywords:frequency", "keywords:letter_frequency", "detectable_format:number_",
               "detectable_format:multiple_sections", "detectable_content:number_placeholders",
               "change_case:capital_word_frequency")


def counting(r):
    return any(rule.startswith(COUNT_RULES) for rule in (r.get("instruction_id_list") or []))


def response_len(r):
    return len(r["output_ids"]) - len(r.get("thinking_ids") or [])


def thinking_len(r):  # (10 Nov, reviewer) an EMPTY thinking is 0 tokens, not the whole answer; only a missing field falls back
    ids = r.get("thinking_ids")
    return len(ids) if ids is not None else r["output_tokens"]


def finished(r):
    return r["closed_thinking"] and not r["truncated"]


def right(r):
    return bool(r["correct"]) and finished(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rollouts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quota", required=True, help="part=n,... (pairs per part, half guard / half right)")
    ap.add_argument("--ratio", type=float, default=1.25)
    ap.add_argument("--block", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--think-only", default="", help="parts (comma) whose shortness is the thinking only (ifeval)")
    ap.add_argument("--count-rules", action="store_true",
                    help="ifeval counting-rule questions: chosen response >= rejected's, chosen thinking <= 1.2x rejected's")
    ap.add_argument("--prefer-reread", action="store_true",
                    help="chosen = the shortest right answer that re-checks the question, within 1.5x the shortest's thinking")
    ap.add_argument("--shorten-min-right", type=float, default=0.75,
                    help="guard,shorten: a shorten pair only where this share of the answers is right")
    ap.add_argument("--shorten-ratio", type=float, default=1.5,
                    help="guard,shorten: the longest right answer must be >= this x the shortest")
    ap.add_argument("--kinds", default="guard,right", choices=("guard,right", "guard", "guard,shorten"),
                    help="guard,right: half / half (Track A); guard: only shortest right > shortest finished wrong")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    quota = {k: int(v) for k, v in (x.split("=") for x in args.quota.split(","))}
    groups = defaultdict(list)
    for path in args.rollouts:
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                groups[r["id"]].append(r)
    think_only = set(filter(None, args.think_only.split(",")))
    decode = None
    if args.prefer_reread:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        decode = lambda r: tok.decode(r.get("thinking_ids") or r["output_ids"], skip_special_tokens=True)  # noqa: E731
    rereads = Counter()
    cand = defaultdict(lambda: {"guard": [], "right_long": [], "right_short": [], "shorten": [], "shorten_soft": []})
    for qid, q in groups.items():
        part = part_of(q[0])
        if part not in quota:
            continue
        size = thinking_len if part in think_only else (lambda r: r["output_tokens"])
        rights = sorted((r for r in q if right(r)), key=size)
        wrongs = sorted((r for r in q if finished(r) and not right(r)), key=size)
        if (args.kinds == "guard,shorten" and len(rights) >= 2 and len(rights) / len(q) >= args.shorten_min_right
                and not (part == "ifeval" and counting(q[0]))):  # (10 Nov, odpo) a counting rule needs its full response
            spread = size(rights[-1]) / max(1, size(rights[0]))
            if spread >= args.shorten_ratio:
                cand[part]["shorten"].append((qid, rights[0], rights[-1]))
            elif spread >= 1.25:  # backup supply: used only if the clear (>= 1.5x) pairs run short
                cand[part]["shorten_soft"].append((qid, rights[0], rights[-1]))
        if not rights or not wrongs:
            continue
        chosen = rights[0]
        if args.count_rules and part == "ifeval" and counting(q[0]):
            rej = wrongs[0]
            fit = [r for r in rights if response_len(r) >= response_len(rej) and thinking_len(r) <= 1.2 * thinking_len(rej)]
            rereads[("ifeval", "counting-rule questions")] += 1
            if not fit:
                rereads[("ifeval", "counting-rule questions without a fitting right answer (no pair)")] += 1
                continue
            chosen = min(fit, key=thinking_len)
            rereads[("ifeval", "counting-rule pairs")] += 1
            cand[part]["guard"].append((qid, chosen, rej))
            continue
        if decode:
            checked = [r for r in rights if thinking_len(r) <= 1.5 * thinking_len(rights[0]) and REREAD.search(decode(r))]
            chosen = checked[0] if checked else rights[0]
            rereads[(part, "chosen re-checks")] += bool(checked)
            rereads[(part, "rejected re-checks")] += bool(REREAD.search(decode(wrongs[0])))
            rereads[(part, "guard questions")] += 1
        cand[part]["guard"].append((qid, chosen, wrongs[0]))
        close = [(r, w) for r in rights for w in wrongs
                 if max(r["output_tokens"], w["output_tokens"]) <= args.ratio * min(r["output_tokens"], w["output_tokens"])]
        rng.shuffle(close)
        for r, w in close:
            cand[part]["right_long" if r["output_tokens"] > w["output_tokens"] else "right_short"].append((qid, r, w))
    pairs, counts = [], Counter()
    per_q, right_q, used = Counter(), Counter(), set()

    def take(part, kind, pool, n):
        rng.shuffle(pool)
        got = []
        for qid, c, w in pool:
            if len(got) >= n:
                break
            key = (id(c), id(w))
            if key in used or per_q[qid] >= 3 or (kind != "guard" and right_q[qid] >= 2):
                continue
            used.add(key)
            per_q[qid] += 1
            right_q[qid] += kind != "guard"
            got.append({"id": qid, "source": c["source"], "part": part, "kind": kind, "prompt_ids": c["prompt_ids"],
                        "chosen_ids": c["output_ids"], "rejected_ids": w["output_ids"],
                        "chosen_tokens": c["output_tokens"], "rejected_tokens": w["output_tokens"]})
        return got
    by_part = {}
    for part, n in quota.items():
        if args.kinds == "guard,shorten":  # (10 Nov, odpo) half "be right", half "be shorter when sure"; no backfill
            g = take(part, "guard", cand[part]["guard"], n // 2)
            sh = take(part, "shorten", cand[part]["shorten"], n - n // 2)
            soft = take(part, "shorten", cand[part]["shorten_soft"], n - n // 2 - len(sh))
            counts[(part, "shorten from the 1.25-1.5x backup")] += len(soft)
            counts[(part, "shorten candidates (>= 1.5x / 1.25-1.5x)")] = f"{len(cand[part]['shorten'])} / {len(cand[part]['shorten_soft'])}"
            sh += soft
            by_part[part] = g + sh
            counts[(part, "guard")] += len(g)
            counts[(part, "shorten")] += len(sh)
            counts[(part, "SHORT of quota")] = max(0, n - len(by_part[part]))
            continue
        if args.kinds == "guard":
            by_part[part] = take(part, "guard", cand[part]["guard"], n)
            for p in by_part[part]:
                counts[(part, "guard")] += 1
            counts[(part, "SHORT of quota")] = max(0, n - len(by_part[part]))
            continue
        g = take(part, "guard", cand[part]["guard"], n // 2)
        r_long = take(part, "right", cand[part]["right_long"], (n - n // 2) // 2)
        r_short = take(part, "right", cand[part]["right_short"], n - n // 2 - len(r_long))
        if len(g) < n // 2:  # a short guard supply is filled with right pairs, never with another kind
            r_short += take(part, "right", cand[part]["right_short"], n // 2 - len(g))
        by_part[part] = g + r_long + r_short
        for p in by_part[part]:
            counts[(part, p["kind"] + ("" if p["kind"] == "guard" else
                                       " (right longer)" if p["chosen_tokens"] > p["rejected_tokens"] else " (right shorter)"))] += 1
        counts[(part, "SHORT of quota")] = max(0, n - len(by_part[part]))
    # stratified order: interleave the parts by their shares, kinds alternating inside each part
    queues = {part: sorted(ps, key=lambda p: (p["kind"], rng.random())) for part, ps in by_part.items()}
    for part in queues:
        g = [p for p in queues[part] if p["kind"] == "guard"]
        r = [p for p in queues[part] if p["kind"] != "guard"]
        queues[part] = [x for pair in zip(g, r) for x in pair] + g[len(r):] + r[len(g):]
    total = sum(len(q) for q in queues.values())
    share = {part: len(q) / max(1, total) for part, q in queues.items()}
    credit = Counter()
    while any(queues.values()):  # each round adds every part's share; a part emits a pair per whole unit of credit
        for part, q in queues.items():
            credit[part] += share[part]
            while credit[part] >= 1 and q:
                pairs.append(q.pop(0))
                credit[part] -= 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pairs.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pairs))
    lines = [f"{len(pairs)} pairs ({len(per_q)} questions)"]
    for (part, what), n in sorted(counts.items()):
        lines.append(f"  {part} / {what}: {n}")
    for part in quota:
        ps = by_part[part]
        if ps:
            med = lambda xs: sorted(xs)[len(xs) // 2]  # noqa: E731
            lines.append(f"  {part}: median chosen / rejected tokens {med([p['chosen_tokens'] for p in ps])} / "
                         f"{med([p['rejected_tokens'] for p in ps])}; chosen longer in "
                         f"{sum(p['chosen_tokens'] > p['rejected_tokens'] for p in ps) / len(ps):.0%}")
    for (part, what), n in sorted(rereads.items()):
        if what.startswith("counting-rule"):
            lines.append(f"  {part}: {what}: {n}")
    ifeval_q = [q for q in groups.values() if part_of(q[0]) == "ifeval"]
    if ifeval_q:
        lines.append(f"  ifeval questions answered: {len(ifeval_q)}, with a counting rule: {sum(counting(q[0]) for q in ifeval_q)}")
    for part in quota:
        if rereads[(part, "guard questions")]:
            n = rereads[(part, "guard questions")]
            lines.append(f"  {part}: re-check of the question in the chosen answer {rereads[(part, 'chosen re-checks')] / n:.0%}, "
                         f"in the rejected {rereads[(part, 'rejected re-checks')] / n:.0%} ({n} guard questions)")
    first = Counter(p["part"] for p in pairs[:args.block])
    lines.append(f"  first {args.block} pairs by part: {dict(first)}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
