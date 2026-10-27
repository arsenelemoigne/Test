"""TokenSkip (Xia et al., EMNLP 2025, arXiv 2502.12067) on Qwen3's thinking: SFT data whose thinking keeps only its
most important words (user, 27 Oct: "try it raw, maybe we will be surprised"; also a LUPIN0.x variant).

TokenSkip scores every word of a right chain of thought with LLMLingua-2 (a BERT-size classifier of "keep or drop"),
keeps the top --ratio share in their original order, and fine-tunes on the shortened chains. Its numbers (GSM8K, about
50% fewer tokens): Qwen2.5-14B -1.7 pts, 7B -5.4, 3B -9.3; the smaller the model, the more it costs. It removes words
(connectors, articles, restatements), never steps, and was tested only on instruct-model chains of 200-500 tokens, not on
thinking models. Here:

  - only the thinking is compressed; the reply after </think> (the answer, the code, the tool call) is kept token for token;
  - one right answer per question (--pick random isolates the compression; shortest stacks it on shortest-of-N);
  - the evaluation prompt is fixed, so there is no ratio in the prompt (TokenSkip's control): one ratio per run, or per
    source with --ratio-by;
  - --verify: the original model is handed the compressed thinking and writes the reply itself (eval sampling); the
    example is kept only if that reply is right (a chain that lost the work behind its answer is dropped).

Writes sft.jsonl (sft.py's format), summary.txt (per source: kept, Qwen-token ratio actually reached, verify pass rate)
and examples.txt (before / after, to read what was cut).

    python tokenskip.py --rollouts runs/bon/probe-plotB/rollouts.jsonl --ratio 0.6 --verify --out runs/bon/tskip-ts06
"""

import argparse
import json
import os
import random
import statistics
from collections import defaultdict
from pathlib import Path

LINGUA = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"


def read(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def right(r):
    return bool(r["correct"]) and r["closed_thinking"] and not r["truncated"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rollouts", nargs="+", required=True, help="original-model rollouts (probe_curve.py)")
    parser.add_argument("--ratio", type=float, default=0.6, help="share of the thinking's words kept (1.0: no compression)")
    parser.add_argument("--ratio-by", default="", help="source=ratio overrides, e.g. gsm8k_matched=0.9,metamath_sv=0.9")
    parser.add_argument("--pick", choices=("random", "shortest"), default="random")
    parser.add_argument("--min-thinking", type=int, default=32, help="thinking shorter than this many tokens is kept as is")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--examples", type=int, default=3, help="before / after examples per source")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ratios = {s: float(v) for s, v in (x.split("=") for x in args.ratio_by.split(",") if x)}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    close_id = tokenizer.convert_tokens_to_ids("</think>")

    groups = defaultdict(list)
    for path in args.rollouts:
        for r in read(path):
            if right(r) and "prompt_ids" in r and close_id in r["output_ids"]:
                groups[r["id"]].append(r)
    chosen = [min(g, key=lambda r: r["output_tokens"]) if args.pick == "shortest" else rng.choice(g)
              for g in groups.values()]
    print(f"{len(chosen)} questions with a right answer", flush=True)

    compressor = None
    if any(ratios.get(r["source"], args.ratio) < 1 for r in chosen):
        from llmlingua import PromptCompressor
        import torch
        compressor = PromptCompressor(model_name=LINGUA, use_llmlingua2=True,
                                      device_map="cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for r in chosen:
        ids = r["output_ids"]
        cut = ids.index(close_id)
        body = tokenizer.decode(ids[:cut], skip_special_tokens=False).replace("<think>", "").strip()
        ratio = ratios.get(r["source"], args.ratio)
        before = len(tokenizer.encode(body, add_special_tokens=False))
        if compressor and ratio < 1 and before >= args.min_thinking:
            # TokenSkip's settings (its LLMLingua.py): digits always kept, repeated drops merged; line breaks kept too,
            # so Qwen's paragraphs stay paragraphs
            body = compressor.compress_prompt(body, rate=ratio, force_tokens=["\n", "=", "?"], force_reserve_digit=True,
                                              drop_consecutive=True)["compressed_prompt"].replace("\n ", "\n").strip()
        thinking = tokenizer.encode("<think>\n" + body + "\n", add_special_tokens=False)
        after = len(thinking) - len(tokenizer.encode("<think>\n", add_special_tokens=False)) - 1
        rows.append({"id": r["id"], "source": r["source"], "row": r, "ratio": ratio, "before": before, "after": max(0, after),
                     "prompt_ids": r["prompt_ids"], "thinking_ids": thinking + [close_id],
                     "completion_ids": thinking + [close_id] + ids[cut + 1:]})

    passed = {}
    if args.verify:
        passed = verify(rows, args, tokenizer)
    kept = [x for x in rows if not args.verify or passed.get(x["id"])]
    (out / "sft.jsonl").write_text("".join(json.dumps({"id": x["id"], "source": x["source"], "prompt_ids": x["prompt_ids"],
                                                        "completion_ids": x["completion_ids"]}) + "\n" for x in kept))
    lines = [f"TokenSkip data: {len(kept)} of {len(rows)} examples kept"
             + (" (verify: the original model, handed the compressed thinking, answers right)" if args.verify else ""),
             "source: kept/all, ratio asked, thinking tokens median before -> after, actual ratio (sum after / sum before)"]
    examples = []
    by_source = defaultdict(list)
    for x in rows:
        by_source[x["source"]].append(x)
    for source, xs in sorted(by_source.items()):
        ks = [x for x in xs if not args.verify or passed.get(x["id"])]
        actual = sum(x["after"] for x in xs) / max(1, sum(x["before"] for x in xs))
        lines.append(f"  {source}: {len(ks)}/{len(xs)}, ratio {xs[0]['ratio']:g}, median {statistics.median(x['before'] for x in xs):.0f}"
                     f" -> {statistics.median(x['after'] for x in xs):.0f}, actual {actual:.2f}"
                     + (f", verify pass {len(ks) / len(xs):.0%}" if args.verify else ""))
        for x in rng.sample(xs, min(args.examples, len(xs))):
            ids = x["row"]["output_ids"]
            examples += [f"=== {source} {x['id']} ({x['before']} -> {x['after']} thinking tokens"
                         + (f", verify {'pass' if passed.get(x['id']) else 'FAIL'}" if args.verify else "") + ")",
                         "--- before:", tokenizer.decode(ids[:ids.index(close_id)], skip_special_tokens=False),
                         "--- after:", tokenizer.decode(x["thinking_ids"], skip_special_tokens=False), ""]
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    (out / "examples.txt").write_text("\n".join(examples) + "\n")
    print("\n".join(lines), flush=True)


def verify(rows, args, tokenizer):
    """The original model continues after the compressed thinking (and its </think>): is its own reply right?"""
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams
    from typed import full_text, is_correct
    from utils.eval.recipe import SAMPLING, SEED

    prompts = [x["prompt_ids"] + x["thinking_ids"] for x in rows]
    engine = LLM(model=args.model, dtype=args.dtype, seed=SEED, gpu_memory_utilization=0.85,
                 max_model_len=max(len(p) for p in prompts) + 4096 + 16, generation_config="vllm")
    eos = tokenizer.eos_token_id
    eos_ids = eos if isinstance(eos, list) else [eos]
    params = [SamplingParams(n=1, seed=SEED, stop_token_ids=eos_ids, **SAMPLING,
                             max_tokens=4096 if x["row"]["task"] == "code" else 2048) for x in rows]
    outputs = engine.generate([{"prompt_token_ids": p} for p in prompts], params)
    passed = {}
    for x, o in zip(rows, outputs):
        reply = list(o.outputs[0].token_ids)
        truncated = not (reply and reply[-1] in eos_ids)
        try:
            passed[x["id"]] = is_correct(x["row"], full_text(tokenizer, x["thinking_ids"] + reply), truncated)
        except Exception as error:  # an old rollout without its grader fields: not verifiable, not kept
            passed[x["id"]] = False
            errors = getattr(verify, "errors", 0) + 1
            verify.errors = errors
            if errors <= 3:
                print(f"verify: {x['id']} not gradable ({error!r})", flush=True)
    return passed


if __name__ == "__main__":
    main()
