"""DPO on (chosen, rejected) reply pairs, continuing a LoRA, with an NLL anchor and a collapse tripwire.

    python dpo.py --pairs runs/r1/pairs/pairs.jsonl --init-adapter adapters/start --out adapters/r1

loss = -log sigmoid(beta * [(pi_c - ref_c) - (pi_r - ref_r)]) + nll * (mean token NLL of chosen)

pi and ref are summed log-probabilities of the reply tokens (thinking and answer) under the
policy and under the starting adapter; the reference is scored once before training. The NLL
anchor on the chosen reply (as in RPO, Pang et al. 2024) stops DPO from widening the margin by
lowering both sides. Pair kinds are mixed to --mix (shorten, guard, hard, right).

Every --save-every optimizer steps a checkpoint is saved. The tripwire watches held-out chosen
replies: if the probability that the reply ends right after </think> rises above --tripwire, a
collapse known from on-policy truncation DPO, training stops and the last checkpoint is kept.
"""

import argparse
import json
import math
import random
from pathlib import Path


def main():
    args = parse_arguments()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in Path(args.pairs).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if len(r["prompt_ids"]) + max(len(r["chosen_ids"]), len(r["rejected_ids"])) <= args.max_len]
    rows = mix(rows, [float(x) for x in args.mix.split(",")], rng)
    held_out, rows = rows[:args.held_out], rows[args.held_out:]
    print(f"training on {len(rows)} pairs, {len(held_out)} held out", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype) if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    close = tokenizer.convert_tokens_to_ids("</think>")
    ends = {tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.convert_tokens_to_ids("<|endoftext|>")}
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    for p in trainable:
        p.data = p.data.float()

    def logps(prompt, reply, grad):
        ids = torch.tensor([prompt + reply], device=device)
        with torch.set_grad_enabled(grad), torch.autocast(device_type=device, dtype=dtype, enabled=dtype != torch.float32):
            hidden = model(input_ids=ids, output_hidden_states=False).logits[0]
        positions = range(len(prompt) - 1, len(prompt) + len(reply) - 1)
        total = 0.0
        for start in range(0, len(reply), 1024):  # log-softmax in slices: the full vocabulary is large
            idx = list(positions)[start:start + 1024]
            logp = torch.log_softmax(hidden[idx].float(), dim=-1)
            total = total + logp.gather(1, torch.tensor(reply[start:start + 1024], device=device)[:, None]).sum()
        return total

    model.eval()
    with torch.no_grad():  # the reference: the starting adapter, scored once
        for r in rows:
            r["ref_c"] = float(logps(r["prompt_ids"], r["chosen_ids"], False))
            r["ref_r"] = float(logps(r["prompt_ids"], r["rejected_ids"], False))
    print("reference scored", flush=True)

    def tripwire():
        """Mean probability of ending the reply right after </think>, over held-out chosen replies."""
        probs = []
        with torch.no_grad():
            for r in held_out:
                reply = r["chosen_ids"]
                if close not in reply:
                    continue
                cut = reply.index(close) + 1
                ids = torch.tensor([r["prompt_ids"] + reply[:cut]], device=device)  # ends with </think>
                with torch.autocast(device_type=device, dtype=dtype, enabled=dtype != torch.float32):
                    logits = model(input_ids=ids).logits[0, -1].float()
                p = torch.softmax(logits, -1)
                probs.append(float(sum(p[e] for e in ends)))
        return sum(probs) / len(probs) if probs else 0.0

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    total = math.ceil(len(rows) * args.epochs / args.batch)
    steps = min(total, args.max_steps or total)
    order = [i for _ in range(args.epochs) for i in rng.sample(range(len(rows)), len(rows))][:steps * args.batch]
    baseline = tripwire()
    print(f"tripwire baseline {baseline:.3f}; {steps} steps of {args.batch} pairs", flush=True)
    model.train()
    step, stats = 0, []
    for position, i in enumerate(order, 1):
        r = rows[i]
        pc = logps(r["prompt_ids"], r["chosen_ids"], True)
        pr = logps(r["prompt_ids"], r["rejected_ids"], True)
        margin = args.beta * ((pc - r["ref_c"]) - (pr - r["ref_r"]))
        loss = -torch.nn.functional.logsigmoid(margin) + args.nll * (-pc / len(r["chosen_ids"]))
        (loss / args.batch).backward()
        stats.append((float(loss), float(margin) > 0))
        if position % args.batch == 0 or position == len(order):  # the last, partial batch too
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0:
                loss_avg = sum(s[0] for s in stats) / len(stats)
                acc = sum(s[1] for s in stats) / len(stats)
                print(f"step {step}/{steps}  loss {loss_avg:.4f}  pairs ordered right {acc:.0%}", flush=True)
                stats = []
            if step % args.save_every == 0 or step == steps:
                model.eval()
                wire = tripwire()
                model.train()
                print(f"step {step}: tripwire {wire:.3f} (baseline {baseline:.3f})", flush=True)
                if wire > max(args.tripwire, 3 * baseline):
                    print(f"tripwire: replies start ending right after </think>; stopped at step {step}, "
                          "keeping the previous checkpoint", flush=True)
                    break
                model.save_pretrained(Path(args.out) / f"step-{step}")
                model.save_pretrained(args.out)
    print(f"saved {args.out}")


def mix(rows, shares, rng):
    """Pairs subsampled to the shares of shorten, guard, hard and right, keeping as many as the scarcest allows."""
    kinds = {k: [r for r in rows if r["kind"] == k] for k in ("shorten", "guard", "hard", "right")}
    shares = list(shares) + [0.0] * (len(kinds) - len(shares))  # a three-share --mix leaves "right" out
    total = min(len(kinds[k]) / s for k, s in zip(kinds, shares) if s > 0) if rows else 0
    picked = []
    for (k, pool), s in zip(kinds.items(), shares):
        rng.shuffle(pool)
        picked += pool[:int(total * s)]
    rng.shuffle(picked)
    return picked


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--init-adapter", required=True, help="the starting LoRA; also the DPO reference")
    parser.add_argument("--out", required=True)
    parser.add_argument("--mix", default="0.4,0.4,0.2", help="shares of shorten, guard, hard (, right) pairs")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--nll", type=float, default=1.0, help="weight of the NLL anchor on the chosen reply")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch", type=int, default=16, help="pairs per optimizer step")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0, help="stop early (0: the whole epoch)")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--held-out", type=int, default=48, help="pairs kept aside for the tripwire")
    parser.add_argument("--tripwire", type=float, default=0.2)
    parser.add_argument("--max-len", type=int, default=9000)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
