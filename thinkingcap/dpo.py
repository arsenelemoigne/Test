"""DPO on (chosen, rejected) reply pairs, continuing a LoRA, with an NLL anchor and a collapse tripwire.

    python dpo.py --pairs runs/r1/pairs/pairs.jsonl --init-adapter adapters/start --out adapters/r1

loss = -log sigmoid(beta * [(pi_c - ref_c) - (pi_r - ref_r)]) + nll * (mean token NLL of chosen)

pi and ref are summed log-probabilities of the reply tokens (thinking and answer) under the
policy and under the starting adapter; the reference is scored once before training. The NLL
anchor on the chosen reply (as in RPO, Pang et al. 2024) stops DPO from widening the margin by
lowering both sides. Pair kinds are mixed to --mix (shorten, guard, hard, right), or all kept with --mix all.

--checker (LUPIN0.x, VeriThinker-style practice): every --checker-every pairs, one "is this solution right? Yes/No"
example (lupin_checker.py) adds --checker-weight x its answer's mean token NLL to the loss. Same model, same step;
nothing changes at test time, where the model is never asked to judge.

Every --save-every optimizer steps a checkpoint is saved. The tripwire watches held-out chosen
replies: if the probability that the reply ends right after </think> rises above --tripwire, a
collapse known from on-policy truncation DPO, training stops and the last checkpoint is kept.
"""

import argparse
import json
import math
import random
import shutil
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
    if args.mix != "all":
        rows = mix(rows, [float(x) for x in args.mix.split(",")], rng)
    else:
        rng.shuffle(rows)
    checks = []
    if args.checker:
        checks = [json.loads(l) for l in Path(args.checker).read_text().splitlines() if l.strip()]
        checks = [c for c in checks if len(c["prompt_ids"]) + len(c["target_ids"]) <= args.max_len]
        random.Random(args.seed + 1).shuffle(checks)
        print(f"checker practice: {len(checks)} examples, one every {args.checker_every} pairs, weight {args.checker_weight}",
              flush=True)
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

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ref_path, latest, state_path = out / "ref.json", out / "latest", out / "resume.pt"
    model.eval()
    if ref_path.exists():  # a lost runtime: the reference scores are kept (rows are the same for the same seed)
        for r, (c, rj) in zip(rows, json.loads(ref_path.read_text())):
            r["ref_c"], r["ref_r"] = c, rj
    else:
        with torch.no_grad():  # the reference: the starting adapter, scored once
            for r in rows:
                r["ref_c"] = float(logps(r["prompt_ids"], r["chosen_ids"], False))
                r["ref_r"] = float(logps(r["prompt_ids"], r["rejected_ids"], False))
        ref_path.write_text(json.dumps([[r["ref_c"], r["ref_r"]] for r in rows]))
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
    step, stats, check_stats = 0, [], []
    if state_path.exists() and (latest / "adapter_config.json").exists():  # resume after the last saved checkpoint
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model, load_file(str(latest / "adapter_model.safetensors")))
        for p in trainable:
            p.data = p.data.float()
        state = torch.load(state_path, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        step = state["step"]
        print(f"resuming after step {step}", flush=True)
    model.train()
    for position, i in enumerate(order, 1):
        if position <= step * args.batch:
            continue
        r = rows[i]
        pc = logps(r["prompt_ids"], r["chosen_ids"], True)
        pr = logps(r["prompt_ids"], r["rejected_ids"], True)
        margin = args.beta * ((pc - r["ref_c"]) - (pr - r["ref_r"]))
        loss = -torch.nn.functional.logsigmoid(margin) + args.nll * (-pc / len(r["chosen_ids"]))
        if checks and position % args.checker_every == 0:  # a fixed example per position: a resumed run sees the same
            c = checks[(position // args.checker_every) % len(checks)]
            check_nll = -logps(c["prompt_ids"], c["target_ids"], True) / len(c["target_ids"])
            loss = loss + args.checker_weight * check_nll
            check_stats.append(float(check_nll))
        (loss / args.batch).backward()
        stats.append((float(loss), float(margin) > 0))
        if position % args.batch == 0 or position == len(order):  # the last, partial batch too
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if args.lr_schedule == "cosine":  # warm-up, then cosine down to --lr-min at the last step (30 Oct build spec)
                if step < args.warmup:
                    lr = args.lr * (step + 1) / args.warmup
                else:
                    lr = args.lr_min + 0.5 * (args.lr - args.lr_min) * (
                        1 + math.cos(math.pi * (step - args.warmup) / max(1, steps - 1 - args.warmup)))
                for group in optimizer.param_groups:
                    group["lr"] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0:
                loss_avg = sum(s[0] for s in stats) / len(stats)
                acc = sum(s[1] for s in stats) / len(stats)
                extra = f"  checker NLL {sum(check_stats) / len(check_stats):.3f}" if check_stats else ""
                print(f"step {step}/{steps}  loss {loss_avg:.4f}  pairs ordered right {acc:.0%}{extra}", flush=True)
                stats, check_stats = [], []
            if step % args.save_every == 0 or step == steps:
                model.eval()
                wire = tripwire()
                model.train()
                print(f"step {step}: tripwire {wire:.3f} (baseline {baseline:.3f})", flush=True)
                if wire > max(args.tripwire, 3 * baseline):
                    print(f"tripwire: replies start ending right after </think>; stopped at step {step}, "
                          "keeping the previous checkpoint", flush=True)
                    break
                model.save_pretrained(out / f"step-{step}")
                model.save_pretrained(latest)  # not args.out: an adapter there means "finished" to track.py
                torch.save({"step": step, "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state()},
                           state_path)
    if (latest / "adapter_config.json").exists():  # the last checkpoint (before a tripwire stop, the previous one)
        shutil.copytree(latest, out, dirs_exist_ok=True)
    else:
        model.save_pretrained(out)
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
    parser.add_argument("--mix", default="0.4,0.4,0.2", help="shares of shorten, guard, hard (, right) pairs, or 'all'")
    parser.add_argument("--checker", default="", help="LUPIN0.x: Yes/No judging examples from lupin_checker.py")
    parser.add_argument("--checker-weight", type=float, default=0.1, help="weight of the checker NLL in the loss")
    parser.add_argument("--checker-every", type=int, default=2, help="one checker example every N pairs")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--nll", type=float, default=1.0, help="weight of the NLL anchor on the chosen reply")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant")
    parser.add_argument("--warmup", type=int, default=10, help="cosine: linear warm-up steps")
    parser.add_argument("--lr-min", type=float, default=2e-6, help="cosine: learning rate at the last step")
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
