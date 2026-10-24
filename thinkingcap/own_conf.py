"""The model's own confidence at each cut of a cal_cuts.py run: it answers once from the cut (greedy, vLLM) and the
per-token geometric-mean probability of THAT answer is kept. Unlike cal_cuts' confidence (the probability of the answer
the full, right reply wrote, i.e. of the right answer), this needs no reference: it is what a real stopper could read.
A long answer's average is drowned by easy tokens (code: brackets, indentation), so fairer variants are kept too:
own_min (its least likely token), own_low5 (geometric mean of its 5 least likely), own_first50 (first 50 tokens, as DEER
does for code) and own_certainty (minus the mean per-token entropy, from the top-5 log-probs plus the leftover mass).

    python own_conf.py --model models/rc1-rl-40 --cal runs/bon/calcheck-rc1rl40   # writes own_confidence.jsonl there
"""

import argparse
import json
import math
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--cal", required=True, help="a cal_cuts.py output folder (cuts.jsonl)")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    out = Path(args.cal) / "own_confidence.jsonl"
    if out.exists() and "own_min" in out.read_text().split("\n", 1)[0]:
        print(f"{out} exists", flush=True)
        return
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    rows = [json.loads(l) for l in (Path(args.cal) / "cuts.jsonl").read_text().splitlines() if l.strip()]
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    longest = max(len(r["prompt_ids"]) + len(r["thinking_ids"]) + r["max_tokens"] for r in rows)
    engine = LLM(model=args.model, dtype=args.dtype, seed=0, gpu_memory_utilization=0.6, max_model_len=longest + 16,
                 enforce_eager=True)
    result = {}
    for cap in sorted({r["max_tokens"] for r in rows}):
        batch = [(k, r) for k, r in enumerate(rows) if r["max_tokens"] == cap]
        params = SamplingParams(n=1, temperature=0.0, max_tokens=cap, stop_token_ids=[end], logprobs=5)
        outputs = engine.generate([{"prompt_token_ids": r["prompt_ids"] + r["thinking_ids"]} for _, r in batch], params)
        for (k, _), o in zip(batch, outputs):
            c = o.outputs[0]
            n = max(1, len(c.token_ids))
            chosen, entropy = [], []
            for token, top in zip(c.token_ids, c.logprobs or []):
                chosen.append(top[token].logprob if token in top else min(t.logprob for t in top.values()))
                probs = [math.exp(t.logprob) for t in top.values()]
                rest = max(0.0, 1 - sum(probs))
                entropy.append(-sum(q * math.log(q) for q in probs + [rest] if q > 0))
            low = sorted(chosen)[:5] or [0.0]
            result[k] = {"key": k, "own_confidence": math.exp(c.cumulative_logprob / n), "tokens": len(c.token_ids),
                         "own_min": math.exp(min(chosen, default=0.0)), "own_low5": math.exp(sum(low) / len(low)),
                         "own_first50": math.exp(sum(chosen[:50]) / max(1, len(chosen[:50]))),
                         "own_certainty": -sum(entropy) / max(1, len(entropy)),
                         "answer": c.text[-300:]}  # its own answer (tail), for the answer-stability baseline
    out.write_text("".join(json.dumps(result[k]) + "\n" for k in sorted(result)))
    print(f"own confidence for {len(result)} cuts -> {out}", flush=True)


if __name__ == "__main__":
    main()
