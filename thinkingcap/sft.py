"""LoRA fine-tune Qwen3-0.6B on the pruned traces from prune_traces.py.

The loss covers the completion only (thinking + answer), never the prompt. The adapter is
saved in .safetensors and loads straight into the evaluation:

    python sft.py --data runs/prune/sft.jsonl --out adapters/pruned-sft
    python evaluate.py --out runs/pruned-sft --adapter adapters/pruned-sft
    python compare.py runs/base runs/pruned-sft

Every --save-every optimizer steps it also writes a checkpoint, giving several points on the
length/accuracy curve.
"""

import argparse
import json
import math
import random
from pathlib import Path


def main():
    args = parse_arguments()
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    torch.manual_seed(args.seed)
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    rows = [r for r in rows if len(r["prompt_ids"]) + len(r["completion_ids"]) <= args.max_len]
    if not rows:
        raise SystemExit(f"No rows of {args.data} fit in --max-len {args.max_len}")
    if args.max_per_source:
        # Easy multiple-choice traces are short and plentiful; left alone they teach "stop at once".
        shuffled = random.Random(args.seed).sample(rows, len(rows))
        kept, seen = [], {}
        for row in shuffled:
            seen[row["source"]] = seen.get(row["source"], 0) + 1
            if seen[row["source"]] <= args.max_per_source:
                kept.append(row)
        rows = kept
    for spec in args.repeat:  # "gsm8k:2": show that dataset's rows twice
        source, times = spec.split(":")
        rows += [r for r in rows if r["source"] == source] * (int(times) - 1)
    print(f"training on {len(rows)} rows", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype) if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    if args.init_adapter:  # continue an earlier round's LoRA (fading worked examples)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    trainable = [p for p in model.parameters() if p.requires_grad]
    for parameter in trainable:  # fp16 gradients cannot be unscaled: keep the LoRA weights in fp32
        parameter.data = parameter.data.float()
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    total_steps = math.ceil(len(rows) * args.epochs / args.accum)
    warmup = max(1, total_steps // 20)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(
        (step + 1) / warmup, 0.5 * (1 + math.cos(math.pi * min(1.0, step / total_steps)))))
    # fp16 needs loss scaling; bf16 and fp32 do not
    scaler = torch.amp.GradScaler(enabled=dtype == torch.float16)

    rng = random.Random(args.seed)
    order = [i for _ in range(args.epochs) for i in rng.sample(range(len(rows)), len(rows))]
    step, running, tokens = 0, 0.0, 0
    model.train()
    for position, index in enumerate(order, 1):
        row = rows[index]
        ids = torch.tensor([row["prompt_ids"] + row["completion_ids"]], device=device)
        labels = ids.clone()
        labels[:, :len(row["prompt_ids"])] = -100
        with torch.autocast(device_type=device, dtype=dtype, enabled=dtype != torch.float32):
            loss = model(input_ids=ids, labels=labels).loss
        scaler.scale(loss / args.accum).backward()
        running += loss.item()
        tokens += len(row["completion_ids"])
        if position % args.accum == 0 or position == len(order):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            schedule.step()
            step += 1
            if step % args.log_every == 0 or position == len(order):
                seen = (position - 1) % args.accum + 1
                print(f"step {step}/{total_steps}  loss {running / seen:.4f}  "
                      f"lr {schedule.get_last_lr()[0]:.2e}  completion tokens {tokens}", flush=True)
            running = 0.0
            if args.save_every and step % args.save_every == 0 and position != len(order):
                model.save_pretrained(Path(args.out) / f"step-{step}")
    model.save_pretrained(args.out)
    print(f"saved {args.out}")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--data", required=True, help="sft.jsonl from prune_traces.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--accum", type=int, default=16, help="sequences per optimizer step")
    parser.add_argument("--max-len", type=int, default=4096)
    parser.add_argument("--save-every", type=int, default=0, help="optimizer steps; 0 saves only at the end")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--dtype", default="bfloat16", help="float16 on a T4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-per-source", type=int, default=0, help="cap rows per dataset; 0 keeps all")
    parser.add_argument("--repeat", nargs="*", default=[], help="dataset:times, e.g. gsm8k:2, after the cap")
    parser.add_argument("--init-adapter", help="start from this LoRA instead of a fresh one")
    return parser.parse_args()


if __name__ == "__main__":
    main()
