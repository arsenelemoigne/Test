"""Score several LoRA checkpoints in one vLLM engine, with evaluate.py's questions, seeds, sampling and graders.

evaluate.py loads the engine for every checkpoint and scores it alone. Here the engine loads
once with every adapter registered, and all checkpoints' requests share its batches, so a sweep
of N checkpoints costs about one evaluation. Each checkpoint gets its own --out directory with
evaluate.py's records.jsonl and results.json, which compare.py reads as usual.

Generations are statistically equivalent to evaluate.py's, not bit-identical: vLLM's output at a
fixed seed depends on what shares the batch. Use it for quick sweeps; score finalists with
evaluate.py.

    python sweep_eval.py --questions questions_40 --samples 2 \\
        --runs runs/a=adapters/x/step-10 runs/b=adapters/x/step-20
"""

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--questions", default="questions_40")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--runs", nargs="+", required=True, help="out_dir=adapter_dir, one per checkpoint")
    options = parser.parse_args()
    runs = [spec.split("=", 1) for spec in options.runs]

    import evaluate as ev
    sys.argv = ["evaluate.py", "--questions", options.questions, "--samples", str(options.samples),
                "--out", runs[0][0], "--adapter", runs[0][1]]
    args = ev.parse_arguments()  # evaluate.py's own defaults: base revision, seed, benchmarks, caps
    root = Path(args.questions).resolve()
    suite = ev.load_suite(root)
    questions = ev.select(ev.read_jsonl(ev.paths(root)["questions"]), args.benchmarks)
    outs = [Path(out).resolve() for out, _ in runs]
    for out in outs:
        ev.prepare_output(out, False)
    grading = ev.grading_context(suite, args)

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    open_id, close_id = (tokenizer.convert_tokens_to_ids(tag) for tag in ("<think>", "</think>"))
    jobs = ev.build_jobs(questions, tokenizer, args)
    caps = ev.caps_in_use(questions, args)
    rank = max(json.loads((Path(adapter) / "adapter_config.json").read_text())["r"] for _, adapter in runs)
    model = LLM(model=args.model, revision=args.revision, tokenizer_revision=args.revision,
                max_model_len=max(caps.values()) + ev.PROMPT_BUDGET, gpu_memory_utilization=0.85,
                enable_prefix_caching=False, generation_config="vllm", seed=args.seed,
                enable_lora=True, max_lora_rank=rank, max_loras=len(runs))
    loras = [LoRARequest(f"checkpoint-{i}", i + 1, str(Path(adapter).resolve())) for i, (_, adapter) in enumerate(runs)]

    work = [(i, job) for job in jobs for i in range(len(runs))]
    records = [[] for _ in runs]
    step = ev.BATCH * len(runs)
    for start in range(0, len(work), step):
        batch = work[start:start + step]
        settings = [SamplingParams(max_tokens=ev.token_cap(job["row"]["dataset"], args), seed=job["seed"],
                                   stop_token_ids=list(eos_ids), **ev.SAMPLING) for _, job in batch]
        outputs = model.generate([{"prompt_token_ids": job["ids"]} for _, job in batch], settings,
                                 lora_request=[loras[i] for i, _ in batch])
        for (i, job), output in zip(batch, outputs):
            if output.lora_request is None or output.lora_request.lora_int_id != i + 1:
                raise ValueError("Adapter routing mismatch")
            tokens = list(output.outputs[0].token_ids)
            final, reasoning, boundary = ev.split_thinking(tokens, open_id, close_id, eos_ids)
            records[i].append({"id": job["row"]["id"], "dataset": job["row"]["dataset"],
                               "sample_index": job["sample"], "seed": job["seed"],
                               "output_tokens": len(tokens), "reasoning_tokens": reasoning,
                               "truncated": not (tokens and tokens[-1] in eos_ids), "boundary_error": boundary,
                               "text": tokenizer.decode(tokens, skip_special_tokens=True),
                               "response": tokenizer.decode(final, skip_special_tokens=True)})
        print(f"{start + len(batch)}/{len(work)}", flush=True)

    for out, (_, adapter), mine in zip(outs, runs, records):
        args.out, args.adapter = str(out), adapter
        ev.grade_all(questions, mine, grading)
        ev.save_records(out, mine)
        results, failed = ev.collect(questions, mine, args.samples)
        ev.save_results(out, args, suite, grading, results, failed, caps)
        print(f"== {adapter}\n{ev.report(results, suite, failed)}", flush=True)


if __name__ == "__main__":
    main()
