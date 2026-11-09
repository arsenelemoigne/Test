"""Track A pairs (GAME_PLAN.md v3, 5 Nov): "be right" pairs from a TRAINED model's own answers, no shortening pressure.

Per question (the model's 4 answers), two kinds, each a right answer over a FINISHED wrong one (no cut rejections):
  guard  the shortest right answer > the shortest finished wrong answer
  right  a right answer > a finished wrong answer of similar length (max(Lc/Lr, Lr/Lc) <= --ratio), exactly half of each
         part's right pairs with the right answer strictly LONGER (a correctness signal without a length signal)
Quotas per part (--quota part=n), half guard / half right; <= 3 pairs per question, <= 2 right pairs per question; no
sample pair used twice. --kinds guard (9 Nov): guard pairs only, the whole quota (the user's own 21 Oct rule, bon_pairs'
guard; the similar-length "right" kind matches Lightning McQwen3's correctness DPO and is left out). A question gives at
most one guard pair, so a part needs at least its quota of mixed-outcome questions. Mixed-outcome questions are the only source (a pair needs a right and a wrong answer).
The output is ordered so that every --block consecutive pairs hold the intended part x kind mix (training reads it in
order with dpo.py --keep-order: a checkpoint at step 5 has seen a representative slice, not one part).

    python repair_pairs.py --rollouts rp/rollouts.jsonl --out pairs-rp --quota svamp=640,mbpp=400,bfcl_irrelevance=240,...
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from build_pool import part_of


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
    ap.add_argument("--kinds", default="guard,right", choices=("guard,right", "guard"),
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
    cand = defaultdict(lambda: {"guard": [], "right_long": [], "right_short": []})
    for qid, q in groups.items():
        part = part_of(q[0])
        if part not in quota:
            continue
        rights = sorted((r for r in q if right(r)), key=lambda r: r["output_tokens"])
        wrongs = sorted((r for r in q if finished(r) and not right(r)), key=lambda r: r["output_tokens"])
        if not rights or not wrongs:
            continue
        cand[part]["guard"].append((qid, rights[0], wrongs[0]))
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
            if key in used or per_q[qid] >= 3 or (kind == "right" and right_q[qid] >= 2):
                continue
            used.add(key)
            per_q[qid] += 1
            right_q[qid] += kind == "right"
            got.append({"id": qid, "source": c["source"], "part": part, "kind": kind, "prompt_ids": c["prompt_ids"],
                        "chosen_ids": c["output_ids"], "rejected_ids": w["output_ids"],
                        "chosen_tokens": c["output_tokens"], "rejected_tokens": w["output_tokens"]})
        return got
    by_part = {}
    for part, n in quota.items():
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
        r = [p for p in queues[part] if p["kind"] == "right"]
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
    first = Counter(p["part"] for p in pairs[:args.block])
    lines.append(f"  first {args.block} pairs by part: {dict(first)}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
