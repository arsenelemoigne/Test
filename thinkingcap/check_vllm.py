"""Natural-answer check for many shortened traces at once, with vLLM.

Each input row is a prompt and a shortened thinking block; the model (optionally with a LoRA)
answers --samples times from it with the evaluation's sampling settings. A trace passes only if
every answer ends inside its token cap and is graded correct (typed.is_correct). Output: one
row per input with the shortest passing answer's token ids, or null.

    python check_vllm.py --inputs check_inputs.jsonl --out check_results.jsonl --samples 2

prune_traces.py and fade_traces.py call it with --checker vllm, in a child process, so the GPU
memory vLLM takes is released when it exits.
"""

import argparse
import json
import os
from pathlib import Path

from typed import full_text, is_correct
from utils.eval.recipe import SAMPLING, SEED


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter")
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    rows = [json.loads(line) for line in Path(args.inputs).read_text().splitlines() if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    longest = max(len(r["prompt_ids"]) + len(r["thinking_ids"]) + r["max_tokens"] for r in rows)
    # the parent keeps its scoring model on the GPU: take less memory and skip CUDA graph capture,
    # which ran out of memory there
    engine = LLM(model=args.model, dtype=args.dtype, seed=SEED, gpu_memory_utilization=0.45,
                 max_model_len=longest + 16, generation_config="vllm", enforce_eager=True,
                 enable_lora=bool(args.adapter), max_lora_rank=16)
    lora = LoRARequest("adapter", 1, args.adapter) if args.adapter else None
    # one request per distinct cap, so a short multiple-choice answer does not reserve a code-sized budget
    caps = sorted({r["max_tokens"] for r in rows})
    answers = {}
    for cap in caps:
        batch = [r for r in rows if r["max_tokens"] == cap]
        params = SamplingParams(n=args.samples, max_tokens=cap, seed=SEED, stop_token_ids=[end], **SAMPLING)
        outputs = engine.generate([{"prompt_token_ids": r["prompt_ids"] + r["thinking_ids"]} for r in batch],
                                  params, lora_request=lora)
        for row, output in zip(batch, outputs):
            passing = []
            for completion in output.outputs:
                tokens = list(completion.token_ids)
                if not tokens or tokens[-1] != end:
                    passing = None  # ran out of room: the evaluation would cut it off too
                    break
                text = full_text(tokenizer, row["thinking_ids"] + tokens)
                if not is_correct(row["row"], text):
                    passing = None
                    break
                passing.append(tokens)
            answers[row["key"]] = min(passing, key=len) if passing else None
    Path(args.out).write_text("".join(json.dumps({"key": k, "answer_ids": v}) + "\n" for k, v in answers.items()))
    passed = sum(v is not None for v in answers.values())
    print(f"checked {len(rows)} traces: {passed} pass", flush=True)


if __name__ == "__main__":
    main()
