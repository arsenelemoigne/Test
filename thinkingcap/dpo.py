"""DPO on the model's own pairs (chosen > rejected) from prune_traces.py --pairs, continuing a LoRA.

The reference is the starting adapter itself: its log-probabilities of every pair are computed
once before the first update, so no second model is kept in memory. Loss per pair:

    -log sigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r)))  [+ nll * mean token NLL of the chosen side]

with sequence log-probabilities summed over the completion tokens. Checkpoints every --save-every
optimizer steps, because training too long collapses DPO on a model this small (stop early).

    python dpo.py --data runs/round/pairs.jsonl --init-adapter adapters/combine-codecut --out adapters/round1
"""

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path


def main():
    args = parse_arguments()
    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    torch.manual_seed(args.seed)
    pairs = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    pairs = [p for p in pairs if len(p["prompt_ids"]) + max(len(p["chosen_ids"]), len(p["rejected_ids"])) <= args.max_len]
    rng = random.Random(args.seed)
    if args.max_per_source:
        kept, seen = [], Counter()
        for p in rng.sample(pairs, len(pairs)):
            seen[p["source"]] += 1
            if seen[p["source"]] <= args.max_per_source:
                kept.append(p)
        pairs = kept
    if not pairs:
        raise SystemExit(f"no pairs in {args.data}")
    print(f"training on {len(pairs)} pairs: " + ", ".join(
        f"{s}/{k} {n}" for (s, k), n in sorted(Counter((p["source"], p["kind"]) for p in pairs).items())), flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype) if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    for parameter in trainable:
        parameter.data = parameter.data.float()

    def logprob(prompt, completion):
        """Summed log-probability of the completion tokens and their count."""
        ids = torch.tensor([prompt + completion], device=device)
        with torch.autocast(device_type=device, dtype=dtype, enabled=dtype != torch.float32):
            logits = model(input_ids=ids).logits[0, len(prompt) - 1:-1]
        targets = ids[0, len(prompt):]
        nll = F.cross_entropy(logits.float(), targets, reduction="sum")
        return -nll, len(completion)

    model.eval()  # the reference: the starting adapter, scored once
    with torch.no_grad():
        for number, p in enumerate(pairs, 1):
            p["ref_c"] = logprob(p["prompt_ids"], p["chosen_ids"])[0].item()
            p["ref_r"] = logprob(p["prompt_ids"], p["rejected_ids"])[0].item()
            if number % 200 == 0:
                print(f"reference scored {number}/{len(pairs)}", flush=True)

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    total_steps = math.ceil(len(pairs) * args.epochs / args.accum)
    warmup = max(1, total_steps // 10)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / warmup))
    order = [i for _ in range(args.epochs) for i in rng.sample(range(len(pairs)), len(pairs))]
    step, running, margins, wins = 0, 0.0, 0.0, 0
    model.train()
    for position, index in enumerate(order, 1):
        p = pairs[index]
        pi_c, n_c = logprob(p["prompt_ids"], p["chosen_ids"])
        pi_r, _ = logprob(p["prompt_ids"], p["rejected_ids"])
        margin = (pi_c - p["ref_c"]) - (pi_r - p["ref_r"])
        loss = -F.logsigmoid(args.beta * margin)
        if args.nll:
            loss = loss + args.nll * (-pi_c / n_c)
        (loss / args.accum).backward()
        running += loss.item()
        margins += margin.item()
        wins += margin.item() > 0
        if position % args.accum == 0 or position == len(order):
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            schedule.step()
            step += 1
            seen = (position - 1) % args.accum + 1
            if step % args.log_every == 0 or position == len(order):
                print(f"step {step}/{total_steps}  loss {running / seen:.4f}  margin {margins / seen:+.2f}  "
                      f"chosen ahead {wins / seen:.0%}", flush=True)
            running, margins, wins = 0.0, 0.0, 0
            if args.save_every and step % args.save_every == 0 and position != len(order):
                model.save_pretrained(Path(args.out) / f"step-{step}")
    model.save_pretrained(args.out)
    print(f"saved {args.out} after {step} steps")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--data", required=True, help="pairs.jsonl from prune_traces.py --pairs")
    parser.add_argument("--init-adapter", required=True, help="the current model's LoRA: start and reference")
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--nll", type=float, default=0.0, help="weight of the chosen side's NLL (an anchor)")
    parser.add_argument("--accum", type=int, default=16, help="pairs per optimizer step")
    parser.add_argument("--max-len", type=int, default=4096)
    parser.add_argument("--max-per-source", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
