"""The consultant pack on the Mac (4 Nov, user: everything on the CPU): the ORIGINAL Qwen's answers to the training
questions + the test's profile in numbers, nothing of our work, for an independent strategy analysis by Argon.

Reads the files line by line (little memory). Writes to --out:
    test_profile.md        the test (astra_test_card.md without our data map) + the original model on the FULL test
                           (full-base: 2,759 x 8), per part: accuracy, thinking quantiles, cut share, solve-rate spread
    raw_meta.jsonl         one line of numbers per training answer (id, type, source, sample, right, cut, thinking_tokens,
                           tokens, question_tokens)
    raw_text_sample.jsonl  ~3,000 full answers (question + answer), stratified by type x outcome
    train_profile.md       the same statistics for the training questions, side by side with the test

    python consult_pack.py --answers lupin-all-...-a4.jsonl --test-records full-base-records.jsonl --out pack/
"""
import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from build_pool import part_of

COLS = ["questions", "answers", "accuracy_pct", "always_right_pct", "sometimes_pct", "never_right_pct", "think_mean",
        "think_p10", "think_p25", "think_p50", "think_p75", "think_p90", "think_p99", "cut_pct", "think_right_p50",
        "think_wrong_p50"]


def quant(xs, f):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(f * len(xs)))] if xs else 0


def profile(groups):
    """groups: question -> [(right, thinking_tokens, cut), ...]"""
    ans = [a for g in groups.values() for a in g]
    rates = [sum(r for r, _, _ in g) / len(g) for g in groups.values() if g]
    th = [t for _, t, _ in ans]
    pct = lambda n: round(100 * n / max(1, len(rates)))  # noqa: E731
    return {"questions": len(groups), "answers": len(ans),
            "accuracy_pct": round(100 * sum(r for r, _, _ in ans) / max(1, len(ans)), 1),
            "always_right_pct": pct(sum(x == 1 for x in rates)), "sometimes_pct": pct(sum(0 < x < 1 for x in rates)),
            "never_right_pct": pct(sum(x == 0 for x in rates)),
            "think_mean": round(statistics.fmean(th)) if th else 0, "think_p10": quant(th, .1), "think_p25": quant(th, .25),
            "think_p50": quant(th, .5), "think_p75": quant(th, .75), "think_p90": quant(th, .9), "think_p99": quant(th, .99),
            "cut_pct": round(100 * sum(c for _, _, c in ans) / max(1, len(ans)), 1),
            "think_right_p50": quant([t for r, t, _ in ans if r], .5), "think_wrong_p50": quant([t for r, t, _ in ans if not r], .5)}


def table(stats, title):
    lines = [title, "", "| part | " + " | ".join(COLS) + " |", "|" + "---|" * (len(COLS) + 1)]
    lines += [f"| {k} | " + " | ".join(str(v[c]) for c in COLS) + " |" for k, v in sorted(stats.items())]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--answers", required=True)
    ap.add_argument("--test-records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=3000)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    close_id = tok.convert_tokens_to_ids("</think>")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tg = defaultdict(lambda: defaultdict(list))
    for line in open(args.test_records):
        r = json.loads(line)
        tg[r["dataset"]][r["id"]].append((str(r["correct"]) == "True", int(r.get("reasoning_tokens") or 0),
                                          str(r["truncated"]) == "True"))
    test = {ds: profile(g) for ds, g in tg.items()}
    trg = defaultdict(lambda: defaultdict(list))
    strata = defaultdict(list)
    seen_n = defaultdict(int)
    rng = random.Random(0)
    skipped = 0
    with open(out / "raw_meta.jsonl", "w") as meta:
        for line in open(args.answers):
            r = json.loads(line)
            part = part_of(r) if "source" in r else None
            if not part:
                skipped += 1
                continue
            ids_ = r["output_ids"]
            th = ids_.index(close_id) if close_id in ids_ else len(ids_)
            right = bool(r.get("correct")) and not r.get("truncated")
            cut = bool(r.get("truncated"))
            trg[part][r["id"]].append((right, th, cut))
            meta.write(json.dumps({"id": r["id"], "type": part, "source": r.get("source"), "sample": len(trg[part][r["id"]]) - 1,
                                   "right": right, "cut": cut, "thinking_tokens": th, "tokens": len(ids_),
                                   "question_tokens": len(r.get("prompt_ids", []))}) + "\n")
            key = (part, "right" if right else "cut" if cut else "wrong")
            bucket = strata[key]  # reservoir sampling (uniform over the stratum): memory stays small
            seen_n[key] += 1
            if len(bucket) < 400:
                bucket.append(r)
            else:
                j = rng.randrange(seen_n[key])
                if j < 400:
                    bucket[j] = r
    per = max(1, args.sample // max(1, len(strata)))
    with open(out / "raw_text_sample.jsonl", "w") as f:
        n = 0
        for (part, outcome), items in sorted(strata.items()):
            for r in items[:per]:
                f.write(json.dumps({"id": r["id"], "type": part, "outcome": outcome,
                                    "text": tok.decode(r.get("prompt_ids", []) + r["output_ids"], skip_special_tokens=False)}) + "\n")
                n += 1
    train = {p: profile(g) for p, g in trg.items()}
    card = (Path(__file__).parent / "astra_test_card.md").read_text().split("## What the practice")[0]
    (out / "test_profile.md").write_text(card + "\n" + table(test, "## The ORIGINAL model on the full test (2,759 x 8), numbers only"))
    (out / "train_profile.md").write_text(table(train, "## The ORIGINAL model on the training questions") + "\n" +
                                          table(test, "## ... and on the test, for comparison"))
    (out / "thinking_off.md").write_text((Path(__file__).parent / "astra_thinking_off.md").read_text())
    sizes = {p.name: round(p.stat().st_size / 1e6, 1) for p in out.iterdir()}
    print(f"pack: {sum(len(g) for g in trg.values())} training questions, {n} sample texts, {skipped} rows skipped; MB {sizes}")


if __name__ == "__main__":
    main()
