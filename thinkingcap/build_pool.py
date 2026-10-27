"""A training pool that looks like the evaluation, part by part: format, prompt length and sub-type mix (user, 27 Oct).

Every evaluation part (SVAMP, MBPP+, BFCL simple / multiple / irrelevance, IFEval) gets training questions from the
sources that imitate it (prepare_types.py writes them in the part's own prompt format):

    svamp               gsm8k_matched + metamath_sv (--math-mix, michal-bida's 70 / 30 by default)
    mbpp                kodcode
    bfcl_simple_python  toolace one-function questions      (BFCL's own sizes split the tool budget: 400 / 200 / 240)
    bfcl_multiple       toolace several-function questions
    bfcl_irrelevance    toolace no-function-fits questions
    ifeval              ifeval_like

Prompt length (chat template applied, tool list included) is matched to the CHOOSING set's (questions_val; never the
fresh set): the choosing questions of a part give --bins quantile bins and the pool draws the same number of questions in
each bin, within the choosing questions' shortest-to-longest range. A bin short of candidates is filled from its
neighbours, a source short of questions by the part's other sources; both are reported.

Contamination: a candidate is dropped when it shares an 8-word sequence with any evaluation question (all of them, not
only the choosing set), ignoring sequences that occur in 3 or more evaluation questions: those are shared templates (BFCL's
system prompt, IFEval's constraint wording), which made the older 13-word check drop 997 IFEval-like questions.

    python build_pool.py --candidates data/cand-dp1.jsonl data/kodcode3.jsonl --choosing questions_val20261023 \\
        --all-questions questions --per-benchmark 3000 --out data/pool-dp1.jsonl
"""

import argparse
import json
import random
import re
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path

# BFCL's question counts in the evaluation: the tool budget is split in these proportions
BFCL_SIZES = {"bfcl_simple_python": 400, "bfcl_multiple": 200, "bfcl_irrelevance": 240}
TOOL_CATEGORY = {"simple": "bfcl_simple_python", "multiple": "bfcl_multiple", "irrelevance": "bfcl_irrelevance"}


def part_of(row):
    """The evaluation part a training row imitates (None: a source this pool does not use)."""
    source = row["source"]
    if source in ("gsm8k_matched", "gsm8k_easy", "metamath_sv"):
        return "svamp"
    if source == "kodcode":
        return "mbpp"
    if source == "ifeval_like":
        return "ifeval"
    if source in ("toolace", "hermes_tools"):
        return TOOL_CATEGORY.get(row.get("category"))
    return None


def read(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def words(messages):
    text = " ".join(m["content"] for m in messages if m["role"] != "assistant")
    return re.findall(r"[a-z0-9]+", text.lower())


def ngrams(tokens, n=8):
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--candidates", nargs="+", required=True, help="prepare_types.py outputs")
    parser.add_argument("--choosing", required=True, help="the choosing-set questions directory (questions_val...)")
    parser.add_argument("--all-questions", default="questions", help="every evaluation question, for contamination")
    parser.add_argument("--per-benchmark", type=int, default=3000, help="questions per benchmark (BFCL's 3 parts share one)")
    parser.add_argument("--math-mix", default="gsm8k_matched=0.7,metamath_sv=0.3")
    parser.add_argument("--tool-sources", default="toolace", help="sources for the BFCL parts")
    parser.add_argument("--bins", type=int, default=5, help="prompt-length quantile bins per part (40 questions -> 5)")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="tokenizer for prompt lengths")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def length(messages):
        return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=True))

    # the evaluation side: choosing-set prompt lengths per part, and every question's 8-word sequences
    choosing = defaultdict(list)
    for q in read(Path(args.choosing) / "questions.jsonl"):
        choosing[q["dataset"]].append(length(q["messages"]))
    seen = Counter()
    for q in read(Path(args.all_questions) / "questions.jsonl"):
        seen.update(ngrams(words(q["messages"])))
    eval_grams = {g for g, n in seen.items() if n < 3}  # shared templates (>= 3 questions) are not evidence of copying

    tool_sources = set(args.tool_sources.split(","))
    math_mix = {k: float(v) for k, v in (x.split("=") for x in args.math_mix.split(","))}
    by_part = defaultdict(lambda: defaultdict(list))
    dropped, texts = Counter(), set()
    for path in args.candidates:
        for row in read(path):
            part = part_of(row)
            if part is None or (part.startswith("bfcl") and row["source"] not in tool_sources):
                continue
            if part == "svamp" and row["source"] not in math_mix:
                continue
            key = json.dumps(row["prompt"])
            if key in texts:  # the same question from two candidate files
                continue
            texts.add(key)
            if ngrams(words(row["prompt"])) & eval_grams:
                dropped[(part, row["source"])] += 1
                continue
            by_part[part][row["source"]].append(row)

    targets = {"svamp": args.per_benchmark, "mbpp": args.per_benchmark, "ifeval": args.per_benchmark}
    total = sum(BFCL_SIZES.values())
    targets |= {p: round(args.per_benchmark * n / total) for p, n in BFCL_SIZES.items()}
    pool, report = [], {}
    for part, want in targets.items():
        ref = sorted(choosing.get(part, []))
        assert ref, f"no choosing-set questions for {part}"
        edges = [ref[min(len(ref) - 1, round(len(ref) * k / args.bins))] for k in range(1, args.bins)]
        shares = math_mix if part == "svamp" else {s: 1.0 / len(by_part[part]) for s in by_part[part]}
        bins = {}
        for source in shares:
            bins[source] = defaultdict(list)
            for r in by_part[part][source]:
                r["_len"] = length(r["prompt"])
                if ref[0] <= r["_len"] <= ref[-1]:  # outside the choosing set's range: no bin would match it
                    bins[source][bisect_right(edges, r["_len"])].append(r)
            for b in bins[source].values():
                rng.shuffle(b)

        def draw(source, k, n):
            """Up to n rows of a source from bin k, then from its nearest bins."""
            got = []
            for step in range(args.bins):
                for j in ((k,) if step == 0 else (k - step, k + step)):
                    if len(got) < n and 0 <= j < args.bins:
                        extra = bins[source][j][:n - len(got)]
                        del bins[source][j][:len(extra)]
                        got += extra
            return got

        picked, short, filled = [], 0, Counter()
        for source, share in shares.items():
            need = round(want * share)
            for k in range(args.bins):
                n = need // args.bins + (1 if k < need % args.bins else 0)
                take = draw(source, k, n)
                for other in shares:  # a source short of questions is topped up by the part's other sources
                    if len(take) < n and other != source:
                        extra = draw(other, k, n - len(take))
                        filled[other] += len(extra)
                        take += extra
                short += n - len(take)
                picked += take
        lens = sorted(r["_len"] for r in picked)
        quant = lambda xs, f: xs[min(len(xs) - 1, int(f * len(xs)))] if xs else None
        report[part] = {
            "wanted": want, "kept": len(picked), "short": short, "topped_up_by": dict(filled),
            "sources": dict(Counter(r["source"] for r in picked)),
            "available": {s: len(v) for s, v in by_part[part].items()},
            "contamination_dropped": {s: n for (p, s), n in dropped.items() if p == part},
            "prompt_tokens_choosing": [quant(ref, f) for f in (0.1, 0.25, 0.5, 0.75, 0.9)],
            "prompt_tokens_pool": [quant(lens, f) for f in (0.1, 0.25, 0.5, 0.75, 0.9)]}
        for r in picked:
            r.pop("_len", None)
            pool.append(r | {"part": part})
    rng.shuffle(pool)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in pool))
    out.with_suffix(".report.json").write_text(json.dumps(report, indent=1))
    lines = [f"{len(pool)} pool questions -> {out}", "part: kept/wanted (short), sources, prompt tokens p10/p25/p50/p75/p90 "
             "pool vs choosing, contamination drops"]
    for part, r in report.items():
        lines.append(f"  {part}: {r['kept']}/{r['wanted']}" + (f" (SHORT {r['short']})" if r["short"] else "")
                     + f"; {r['sources']}; pool {r['prompt_tokens_pool']} vs choosing {r['prompt_tokens_choosing']}"
                     + (f"; dropped {r['contamination_dropped']}" if r["contamination_dropped"] else ""))
    print("\n".join(lines), flush=True)
    out.with_suffix(".report.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
