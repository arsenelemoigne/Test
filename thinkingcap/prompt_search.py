"""Which prompt makes the model think shortest without losing answers? Candidates for prompt distillation.

Lightning McQwen3 distilled a terse "ex-draft" prompt into Qwen3-0.6B (sample with the prompt, delete it, SFT on the
replies). This measures candidate prompts first, on TRAINING questions (a label lot), never on the evaluation sets:
a prompt chosen on the benchmarks would be fitting to them. Every candidate answers the same questions with the
evaluation's sampling settings and caps; each is paired against the no-prompt "base" candidate per question.

A candidate (prompt_candidates.py) has an optional "system" text (appended to an existing system message, e.g. the
tool list, else a new one) and an optional "prefill" (the start of the thinking, counted in its tokens). "leak" lists
words from the prompt; a reply leaks when its thinking uses one the question does not contain, or says "example"
when the question does not (Lightning McQwen3: leaking replies teach the model to talk about a prompt it will not see).

Reported per candidate and type: accuracy change, tokens saved (on all questions and on those base finished), leak and
cap rates; overall tokens saved = 1 - geometric mean of the per-type token ratios, as the challenge scores it. Each
candidate's replies are saved as soon as it finishes, so a lost runtime only repeats the candidate under way.

    python prompt_search.py --data data/lot-V2.jsonl --per-source 50 --samples 4 --out runs/bon/ps1
"""

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

from probe_curve import read_jsonl, rollout_rows, write_jsonl
from utils.eval.recipe import SAMPLING, SEED


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", required=True, help="typed questions (a label lot)")
    parser.add_argument("--candidates", default="", help="a JSON list of candidates (default: prompt_candidates.py)")
    parser.add_argument("--only", default="", help="comma-separated candidate names (default: all)")
    parser.add_argument("--sources", default="", help="comma-separated sources (default: all in --data)")
    parser.add_argument("--per-source", type=int, default=50)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--code-max-tokens", type=int, default=8192)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--out", required=True)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.candidates:
        candidates = json.loads(Path(args.candidates).read_text())
    else:
        from prompt_candidates import CANDIDATES
        candidates = CANDIDATES
    if args.only:
        keep = set(args.only.split(",")) | {"base"}
        candidates = [c for c in candidates if c["name"] in keep]
    assert candidates[0]["name"] == "base", "the first candidate must be the no-prompt base"
    questions = pick(args)
    (out / "candidates.json").write_text(json.dumps(candidates, indent=1))
    if not args.summarize_only:
        todo = [c for c in candidates if not (out / f"rollouts-{c['name']}.jsonl").exists()]
        if todo:
            sample(questions, todo, args, out)
    summarize(questions, candidates, out)


def pick(args):
    keep = set(args.sources.split(",")) if args.sources else None
    by_source = defaultdict(list)
    for index, row in enumerate(read_jsonl(Path(args.data))):
        if keep is None or row["source"] in keep:
            by_source[row["source"]].append({**row, "id": f"{row['source']}-{index}"})
    rng = random.Random(SEED)
    picked = []
    for source in sorted(by_source):
        rows = by_source[source]
        rng.shuffle(rows)
        picked += rows[:args.per_source]
    return picked


def with_prompt(question, candidate):
    """The question's messages with the candidate's system text added."""
    msgs = [dict(m) for m in question["prompt"]]
    text = candidate.get("system")
    if text:
        if msgs and msgs[0]["role"] == "system":
            msgs[0]["content"] = msgs[0]["content"].rstrip() + "\n\n" + text
        else:
            msgs.insert(0, {"role": "system", "content": text})
    return msgs


def sample(questions, candidates, args, out):
    import os
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    close_id = tokenizer.convert_tokens_to_ids("</think>")
    eos = tokenizer.eos_token_id
    eos_ids = eos if isinstance(eos, list) else [eos]

    def ids(question, candidate):
        prompt = tokenizer.apply_chat_template(with_prompt(question, candidate), tokenize=True,
                                               add_generation_prompt=True, enable_thinking=True)
        if candidate.get("prefill"):  # Qwen3 opens its thinking with "<think>\n": start it for the model
            prompt = prompt + tokenizer.encode("<think>\n" + candidate["prefill"], add_special_tokens=False)
        return prompt

    longest = max(len(ids(q, c)) for q in questions for c in candidates)
    engine = LLM(model=args.model, dtype=args.dtype, seed=SEED, gpu_memory_utilization=0.85,
                 max_model_len=max(args.max_tokens, args.code_max_tokens) + longest + 16, generation_config="vllm")
    params = [SamplingParams(n=args.samples, seed=SEED, stop_token_ids=eos_ids, **SAMPLING,
                             max_tokens=args.code_max_tokens if q["task"] == "code" else args.max_tokens)
              for q in questions]
    for candidate in candidates:
        prompts = [ids(q, candidate) for q in questions]
        outputs = engine.generate([{"prompt_token_ids": p} for p in prompts], params)
        extra = len(tokenizer.encode(candidate["prefill"], add_special_tokens=False)) if candidate.get("prefill") else 0
        rows = rollout_rows(questions, prompts, outputs, tokenizer, close_id, eos_ids)
        for row in rows:
            row["thinking_text"] = tokenizer.decode(row["thinking_ids"], skip_special_tokens=True)
            row["tokens"] = row["output_tokens"] + extra  # the prefill counts: the distilled model would write it
            for key in ("prompt_ids", "thinking_ids"):
                row.pop(key)
        write_jsonl(out / f"rollouts-{candidate['name']}.jsonl", rows)
        print(f"{candidate['name']}: {len(rows)} replies saved", flush=True)


def leaks(row, words):
    question = row["question"].lower()
    thinking = row["thinking_text"].lower()
    if re.search(r"\bexamples?\b", thinking) and "example" not in question:
        return True
    return any(w.lower() in thinking and w.lower() not in question for w in words)


def summarize(questions, candidates, out):
    by_cand = {}
    for c in candidates:
        path = out / f"rollouts-{c['name']}.jsonl"
        if path.exists():
            by_cand[c["name"]] = defaultdict(list)
            for r in read_jsonl(path):
                by_cand[c["name"]][r["id"]].append(r)
    base = by_cand.get("base")
    if not base:
        print("no base replies yet")
        return
    sources = sorted({q["source"] for q in questions})
    finished = {i for i, rs in base.items() if not any(r["truncated"] for r in rs)}
    table, report = [], {}
    for c in candidates:
        mine = by_cand.get(c["name"])
        if mine is None:
            continue
        per, ratios, ratios_fin, deltas = {}, [], [], []
        for s in sources:
            ids = [i for i in base if base[i][0]["source"] == s and i in mine]
            if not ids:
                continue
            mean = lambda d, sel, key: sum(r[key] for i in sel for r in d[i]) / max(1, sum(len(d[i]) for i in sel))
            acc_b = sum(r["correct"] for i in ids for r in base[i]) / sum(len(base[i]) for i in ids)
            acc_c = sum(r["correct"] for i in ids for r in mine[i]) / sum(len(mine[i]) for i in ids)
            ratio = mean(mine, ids, "tokens") / max(1e-9, mean(base, ids, "tokens"))
            fin = [i for i in ids if i in finished]
            ratio_fin = mean(mine, fin, "tokens") / max(1e-9, mean(base, fin, "tokens")) if fin else float("nan")
            rows = [r for i in ids for r in mine[i]]
            per[s] = {"questions": len(ids), "acc_base": acc_b, "acc": acc_c, "delta_pp": 100 * (acc_c - acc_b),
                      "saved": 1 - ratio, "saved_finished": 1 - ratio_fin,
                      "tokens": mean(mine, ids, "tokens"), "tokens_base": mean(base, ids, "tokens"),
                      "leak": sum(leaks(r, c.get("leak", [])) for r in rows) / len(rows),
                      "capped": sum(r["truncated"] for r in rows) / len(rows)}
            ratios.append(ratio)
            deltas.append(100 * (acc_c - acc_b))
            if fin:
                ratios_fin.append(ratio_fin)
        geo = lambda xs: 1 - math.exp(sum(math.log(max(x, 1e-9)) for x in xs) / len(xs)) if xs else float("nan")
        overall = {"saved": geo(ratios), "saved_finished": geo(ratios_fin), "delta_pp": sum(deltas) / len(deltas),
                   "leak": sum(p["leak"] for p in per.values()) / len(per)}
        report[c["name"]] = {"overall": overall, "per_source": per}
        cells = "; ".join(f"{s} {p['saved']:+.0%}/{p['delta_pp']:+.1f}" + (f" leak {p['leak']:.0%}" if p["leak"] else "")
                          for s, p in per.items())
        table.append(f"{c['name']:16s} saved {overall['saved']:+.1%} (base-finished {overall['saved_finished']:+.1%}), "
                     f"accuracy {overall['delta_pp']:+.1f} pp, leaks {overall['leak']:.0%}\n    {cells}")
    (out / "results.json").write_text(json.dumps(report, indent=1))
    text = "\n".join(table)
    (out / "summary.txt").write_text(text + "\n")
    print(text, flush=True)


if __name__ == "__main__":
    main()
