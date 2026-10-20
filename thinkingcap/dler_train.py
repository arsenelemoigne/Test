"""DLER-style RL ("Doing Length pEnalty Right", NVIDIA, arXiv 2510.15110) for Qwen3-0.6B.

Each step:
  1. Sample --group replies to each of --gen-prompts questions (balanced across --sources, every
     source cycling through its pool so small sets such as HumanEval never run out), at the
     training temperature (1.0, as in DLER).
  2. Reward: 1 if the reply is correct *and* finished within its source's cap, else 0. The cap is
     the generation limit (--cap-by-source), so a reply that runs past it scores 0: DLER's
     truncation penalty.
  3. Dynamic sampling: drop questions whose replies all score 0 or all score 1 (no signal); sample
     more rounds until --prompts informative questions are collected or --max-rounds is reached.
  4. Advantage: reward minus its group's mean (no std), then normalised over the whole batch
     (REINFORCE++ baseline, DLER sec. 3.1).
  5. PPO update over mini-batches of --minibatch questions, one epoch: per token
     max(-A r, -A clip(r, 1 - clip_low, 1 + clip_high)), dual clip c for A < 0, plus an MSE KL to
     the base model (--kl-coef), averaged over the mini-batch's tokens; gradient norm clip 1.0.

Logged every step, overall and per source: accuracy, mean length, share of replies that hit the
cap, share of questions dropped as all-0 / all-1, entropy, clip fraction, KL. Every --guard-every
steps a fixed held-out set (per source, at the evaluation's sampling) is scored; with
--stop-on-alarm the run saves and stops when the guard falls more than --guard-drop below the
raw model or entropy rises more than --entropy-jump over 5 steps, so a person can look first.

    python dler_train.py --out adapters/dler --data data/train.jsonl,data/code.jsonl \\
        --sources gsm8k,skywork,arc_easy,commonsense_qa,humaneval \\
        --cap-by-source arc_easy=512,commonsense_qa=512,gsm8k=1024,skywork=2048,humaneval=3072 \\
        --guard-data data/val.jsonl,data/code_val.jsonl --mirror /content/drive/MyDrive/runs/dler
"""

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

from rl_train import Sampler, batch_logps, held_out_questions, is_correct, read_rows, score_replies, token_batches
from utils.eval.recipe import SAMPLING


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--init-adapter")
    parser.add_argument("--data", default="data/train.jsonl")
    parser.add_argument("--sources", default="gsm8k,skywork,arc_easy,commonsense_qa")
    parser.add_argument("--out", required=True)
    parser.add_argument("--mirror", help="copy --out here after every checkpoint (e.g. a Drive folder)")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--prompts", type=int, default=64, help="informative questions per step")
    parser.add_argument("--gen-prompts", type=int, default=96, help="questions sampled per round")
    parser.add_argument("--max-rounds", type=int, default=3, help="sampling rounds per step at most")
    parser.add_argument("--group", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2048, help="cap for sources not in --cap-by-source")
    parser.add_argument("--cap-by-source", default="")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--minibatch", type=int, default=16, help="questions per optimizer step")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--clip-low", type=float, default=0.2)
    parser.add_argument("--clip-high", type=float, default=0.28)
    parser.add_argument("--dual-clip", type=float, default=10.0)
    parser.add_argument("--kl-coef", type=float, default=5e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch-tokens", type=int, default=4096)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--guard-every", type=int, default=10)
    parser.add_argument("--guard-data", default="data/val.jsonl")
    parser.add_argument("--guard-questions", type=int, default=160)
    parser.add_argument("--guard-group", type=int, default=2)
    parser.add_argument("--guard-drop", type=float, default=0.03)
    parser.add_argument("--entropy-jump", type=float, default=0.3)
    parser.add_argument("--stop-on-alarm", action="store_true")
    parser.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    parser.add_argument("--attention-backend", default=None)
    parser.add_argument("--gpu-memory", type=float, default=0.3)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


# ---------------------------------------------------------------- pure pieces (tested)

def parse_caps(text):
    return {k: int(v) for k, v in (item.split("=") for item in filter(None, text.split(",")))}


def informative(rewards):
    """A group teaches something only if its rewards differ."""
    return len(set(rewards)) > 1


def dler_advantages(groups):
    """Rewards per group -> advantages: minus the group mean, then normalised over the batch."""
    centred = [[r - sum(g) / len(g) for r in g] for g in groups]
    flat = [a for g in centred for a in g]
    mean = sum(flat) / len(flat)
    spread = math.sqrt(sum((a - mean) ** 2 for a in flat) / len(flat))
    return [[(a - mean) / (spread + 1e-6) for a in g] for g in centred]


def ppo_token_losses(new_logps, old_logps, advantage, clip_low, clip_high, dual_clip):
    """Per-token clipped policy loss (torch tensors), and the share of tokens where clipping bit."""
    import torch
    ratio = torch.exp(new_logps - old_logps)
    unclipped = -advantage * ratio
    clipped = -advantage * torch.clamp(ratio, 1 - clip_low, 1 + clip_high)
    loss = torch.maximum(unclipped, clipped)
    if advantage < 0:
        loss = torch.minimum(loss, torch.full_like(loss, -advantage * dual_clip))
    return loss, float((clipped > unclipped).float().mean())


class CyclingPools:
    """One shuffled queue per source, restarted (reshuffled) when it runs out."""

    def __init__(self, rows, sources, rng):
        self.rng = rng
        self.pools = {s: [r for r in rows if r["source"] == s] for s in sources}
        missing = [s for s, p in self.pools.items() if not p]
        if missing:
            raise SystemExit(f"no questions for sources {missing}")
        self.queues = {s: [] for s in sources}
        self.cycles = {s: 0 for s in sources}

    def take(self, count):
        """`count` questions, as equal a share per source as possible."""
        sources = list(self.pools)
        out = []
        for i in range(count):
            source = sources[i % len(sources)]
            if not self.queues[source]:
                self.queues[source] = self.rng.sample(self.pools[source], len(self.pools[source]))
                self.cycles[source] += 1
            out.append(self.queues[source].pop())
        return out


# ---------------------------------------------------------------- training

def main():
    args = parse_arguments()
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype) if device == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    if args.init_adapter:
        model = PeftModel.from_pretrained(base, args.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(
            r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    trainable = [p for p in model.parameters() if p.requires_grad]
    for parameter in trainable:
        parameter.data = parameter.data.float()
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler(enabled=dtype == torch.float16)

    caps = parse_caps(args.cap_by_source)
    sources = args.sources.split(",")
    pools = CyclingPools(read_rows(args.data), sources, rng)
    # Sampler sizes vLLM's context from --max-tokens and --max-tokens-by-source
    args.max_tokens_by_source = args.cap_by_source
    sampler = Sampler(args, tokenizer)
    off = -1 if args.backend == "vllm" else 0
    train_sampling = {"temperature": args.temperature, "top_p": 1.0, "top_k": off}
    current = out / "current"
    model.save_pretrained(current)
    log = []

    def mirror():
        if args.mirror:
            shutil.copytree(out, args.mirror, dirs_exist_ok=True)

    def write_log():
        (out / "dler_log.jsonl").write_text("".join(json.dumps(e) + "\n" for e in log))

    guard_questions, guard_prompts = held_out_questions(args, tokenizer, args.guard_data, args.guard_questions)

    def guard_check(adapter_dir, index):
        replies = sampler.sample(guard_prompts, adapter_dir, 10 ** 6, model, group=args.guard_group,
                                 lora_id=10 ** 6 + index, sampling=SAMPLING,
                                 caps=[caps.get(q["source"], args.max_tokens) for q in guard_questions])
        return score_replies(guard_questions, replies, tokenizer)

    raw = guard_check(None, 0)
    print(json.dumps({"guard_raw": raw}), flush=True)
    (out / "guard.json").write_text(json.dumps({"raw": raw, "checks": []}, indent=1))

    def prompt_ids(question):
        return tokenizer.apply_chat_template(question["prompt"], tokenize=True, add_generation_prompt=True,
                                             enable_thinking=True)

    for step in range(args.steps):
        started = time.time()
        kept, seen = [], {s: {"questions": 0, "all0": 0, "all1": 0, "correct": 0, "replies": 0,
                              "tokens": 0, "capped": 0} for s in sources}
        rounds = 0
        while len(kept) < args.prompts and rounds < args.max_rounds:
            questions = pools.take(args.gen_prompts)
            prompts = [prompt_ids(q) for q in questions]
            limits = [caps.get(q["source"], args.max_tokens) for q in questions]
            replies = sampler.sample(prompts, current, step, model, caps=limits, sampling=train_sampling)
            rounds += 1
            for question, prompt, limit, group in zip(questions, prompts, limits, replies):
                rewards = [1.0 if r and r[-1] == eos and is_correct(
                    tokenizer.decode(r, skip_special_tokens=True), question) else 0.0 for r in group]
                tally = seen[question["source"]]
                tally["questions"] += 1
                tally["correct"] += sum(rewards)
                tally["replies"] += len(group)
                tally["tokens"] += sum(map(len, group))
                tally["capped"] += sum(not r or r[-1] != eos for r in group)
                if informative(rewards):
                    kept.append((question, prompt, group, rewards))
                else:
                    tally["all1" if rewards[0] else "all0"] += 1
        kept = kept[:args.prompts]

        stats = {"step": step + 1, "rounds": rounds, "informative": len(kept), "lr": args.lr}
        if kept:
            advantages = dler_advantages([k[3] for k in kept])
            jobs = [(prompt, reply, adv, question["source"])
                    for (question, prompt, group, _), group_adv in zip(kept, advantages)
                    for reply, adv in zip(group, group_adv) if reply]
            # old (sampling-time) and base-model log-probs, once, before any update
            model.eval()
            with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype, enabled=dtype != torch.float32):
                old = {}
                ref = {}
                for batch in token_batches(jobs, args.batch_tokens):
                    for job, lp in zip(batch, batch_logps(model, [(j[0], j[1]) for j in batch], device)):
                        old[id(job)] = lp.detach()
                    if args.kl_coef:
                        with model.disable_adapter():
                            for job, lp in zip(batch, batch_logps(model, [(j[0], j[1]) for j in batch], device)):
                                ref[id(job)] = lp.detach()
            model.train()
            order = list(range(len(kept)))
            rng.shuffle(order)
            per_source = {s: {"entropy": 0.0, "tokens": 0, "clip": 0.0, "clip_n": 0, "kl": 0.0} for s in sources}
            loss_total = 0.0
            groups_of = {}
            index = 0
            for k, (question, prompt, group, _) in enumerate(kept):
                count = sum(1 for r in group if r)
                groups_of[k] = jobs[index:index + count]
                index += count
            for start in range(0, len(order), args.minibatch):
                chunk = [job for k in order[start:start + args.minibatch] for job in groups_of[k]]
                if not chunk:
                    continue
                total_tokens = sum(len(j[1]) for j in chunk)
                for batch in token_batches(chunk, args.batch_tokens):
                    entropies = []
                    with torch.autocast(device_type=device, dtype=dtype, enabled=dtype != torch.float32):
                        new = batch_logps(model, [(j[0], j[1]) for j in batch], device, entropies)
                    loss = 0.0
                    for job, lp, entropy in zip(batch, new, entropies):
                        token_loss, clip_share = ppo_token_losses(lp, old[id(job)], job[2], args.clip_low,
                                                                  args.clip_high, args.dual_clip)
                        if args.kl_coef:
                            kl = 0.5 * (lp - ref[id(job)]) ** 2
                            token_loss = token_loss + args.kl_coef * kl
                            per_source[job[3]]["kl"] += float(kl.detach().sum())
                        loss = loss + token_loss.sum() / total_tokens
                        tally = per_source[job[3]]
                        tally["entropy"] += float(entropy.detach().sum())
                        tally["tokens"] += len(job[1])
                        tally["clip"] += clip_share * len(job[1])
                    scaler.scale(loss).backward()
                    loss_total += float(loss)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            model.save_pretrained(current)
            tokens = sum(v["tokens"] for v in per_source.values())
            stats.update(loss=loss_total, entropy=sum(v["entropy"] for v in per_source.values()) / max(1, tokens),
                         clip_fraction=sum(v["clip"] for v in per_source.values()) / max(1, tokens),
                         kl=sum(v["kl"] for v in per_source.values()) / max(1, tokens))
        else:
            per_source = {s: {"entropy": 0.0, "tokens": 0, "clip": 0.0, "kl": 0.0} for s in sources}
        replies_all = sum(v["replies"] for v in seen.values())
        stats.update(accuracy=sum(v["correct"] for v in seen.values()) / max(1, replies_all),
                     mean_tokens=sum(v["tokens"] for v in seen.values()) / max(1, replies_all),
                     capped=sum(v["capped"] for v in seen.values()) / max(1, replies_all),
                     seconds=round(time.time() - started))
        stats["by_source"] = {
            s: {"accuracy": round(v["correct"] / max(1, v["replies"]), 3),
                "mean_tokens": round(v["tokens"] / max(1, v["replies"])),
                "capped": round(v["capped"] / max(1, v["replies"]), 3),
                "all0": round(v["all0"] / max(1, v["questions"]), 3),
                "all1": round(v["all1"] / max(1, v["questions"]), 3),
                "entropy": round(per_source[s]["entropy"] / max(1, per_source[s]["tokens"]), 3),
                "clip_fraction": round(per_source[s]["clip"] / max(1, per_source[s]["tokens"]), 4),
                "pool_cycles": pools.cycles[s]}
            for s, v in seen.items()}

        alarm = None
        if args.guard_every and (step + 1) % args.guard_every == 0:
            result = guard_check(current, step + 1)
            stats["guard"] = result
            drop = raw["all"]["accuracy"] - result["all"]["accuracy"]
            guard = json.loads((out / "guard.json").read_text())
            guard["checks"].append({"step": step + 1, **result})
            (out / "guard.json").write_text(json.dumps(guard, indent=1))
            if drop > args.guard_drop:
                alarm = f"guard accuracy {result['all']['accuracy']:.3f} is {drop:.1%} below the raw model"
        recent = [e["entropy"] for e in log[-4:] if "entropy" in e] + ([stats["entropy"]] if "entropy" in stats else [])
        if len(recent) == 5 and recent[-1] - recent[0] > args.entropy_jump:
            alarm = alarm or f"entropy rose {recent[-1] - recent[0]:.2f} over 5 steps"
        log.append(stats)
        print(json.dumps(stats), flush=True)
        write_log()
        if args.save_every and (step + 1) % args.save_every == 0:
            model.save_pretrained(out / f"step{step + 1}")
            mirror()
        if alarm and args.stop_on_alarm:
            model.save_pretrained(out / f"step{step + 1}")
            (out / "ALARM.txt").write_text(alarm + "\n")
            mirror()
            print(f"ALARM, stopping for review: {alarm}", flush=True)
            return
    model.save_pretrained(out)
    write_log()
    mirror()
    print(f"saved {out}")


if __name__ == "__main__":
    main()
