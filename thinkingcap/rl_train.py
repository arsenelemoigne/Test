"""RL stage: reward correct answers that are as short as the model's own shortest correct one,
and push length pressure onto the thinking tokens that carry no information.

Each step samples --group replies to each of --questions training questions with the current
LoRA, then takes one policy-gradient step (GRPO-style group advantages, on-policy):

  reward    a correct reply gets 1 - alpha * |len - L*| / L*, a wrong or cut-off one
            -alpha * |len - L*| / L*, where L* is the length of the shortest correct reply in its
            group, or the group mean when none is correct (ShorterBetter's "sample optimal
            length"): every question sets its own budget.
  advantage reward minus the group mean, over the group's standard deviation.
  per token on correct replies, thinking tokens marked redundant get an extra -lambda. Two
            markers: "confidence" (tokens the policy itself found unlikely, in the bottom
            --redundant-quantile of the reply: exploratory filler, as in TokenProbe), or
            "settle" (tokens after the point where p(correct answer) settled, read with
            probe_curve.probe_scores: the model already knew).
  TokenProbe ("On the Token Value Inequality in Efficient Reasoning", --redundant tokenprobe-rs or
            tokenprobe-kl): each reply's token log-probs are z-scored (NormLP); the K best
            non-overlapping W-token windows by summed NormLP are "core", the rest redundant.
            Redundant tokens are pressed only when the reply is over budget L = core_len * scale,
            scale annealed from --scale-start to --scale-end.
              rs: A_t += beta * (alpha_core * NormLP_t [core] - lam * max(0, T - L) / T [redundant])
              kl: core tokens (and in-budget replies) are anchored to the base model with a KL
                  term beta_kl; over-budget redundant tokens lose the anchor and get
                  -gamma_t * log pi with gamma_t = lam_decay * (T - L) / T.
            Wrong replies get the plain update (plus the KL anchor in kl mode).

Start from the SFT adapter (--init-adapter) so the model already knows how to stop.

    python rl_train.py --init-adapter adapters/pruned_v2 --out adapters/rl --steps 6
    python evaluate.py --adapter adapters/rl --out runs/rl
"""

import argparse
import json
import math
import os
import random
import statistics
from pathlib import Path

from probe_curve import read_jsonl
from utils.eval.recipe import SAMPLING
from utils.train.rewards import is_task_correct


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--init-adapter", help="LoRA to continue from (the SFT warm start)")
    parser.add_argument("--data", default="data/train.jsonl")
    parser.add_argument("--sources", default="gsm8k,arc_easy,commonsense_qa")
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--questions", type=int, default=32, help="questions per step")
    parser.add_argument("--group", type=int, default=8, help="replies per question")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--alpha", type=float, default=0.5, help="weight of the distance to L*")
    parser.add_argument("--lam", type=float, default=0.1, help="extra penalty per redundant token")
    parser.add_argument("--redundant", default="confidence",
                        choices=["confidence", "settle", "tokenprobe-rs", "tokenprobe-kl", "none"])
    parser.add_argument("--window", type=int, default=10, help="TokenProbe core window W")
    parser.add_argument("--windows", type=int, default=10, help="TokenProbe core windows K")
    parser.add_argument("--scale-start", type=float, default=3.0, help="budget = core_len * scale")
    parser.add_argument("--scale-end", type=float, default=1.5)
    parser.add_argument("--rs-beta", type=float, default=1.0)
    parser.add_argument("--rs-alpha", type=float, default=0.05)
    parser.add_argument("--rs-lam", type=float, default=0.08)
    parser.add_argument("--kl-beta", type=float, default=5e-3)
    parser.add_argument("--lam-decay", type=float, default=0.2)
    parser.add_argument("--redundant-quantile", type=float, default=0.2)
    parser.add_argument("--settle-threshold", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--rank", type=int, default=16, help="only used without --init-adapter")
    parser.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    parser.add_argument("--attention-backend", default=None, help="vLLM backend; TRITON_ATTN on a T4")
    parser.add_argument("--gpu-memory", type=float, default=0.35, help="vLLM share of the GPU")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


# ---------------------------------------------------------------- rewards (pure, tested)

def length_rewards(lengths, correct, alpha):
    """ShorterBetter-style rewards for one group.

    L* is the shortest correct reply's length, or the group's mean length when none is correct
    (ShorterBetter's neutral fallback: without it, its shortest-length variants collapsed).
    A correct reply gets 1 - alpha * |len - L*| / L*, floored at 0; a wrong one gets
    -alpha * |len - L*| / L*, so a wrong reply never beats a correct one.
    """
    lengths_correct = [n for n, ok in zip(lengths, correct) if ok]
    target = min(lengths_correct) if lengths_correct else sum(lengths) / len(lengths)
    target = max(target, 1)
    return [max(0.0, 1 - alpha * abs(n - target) / target) if ok else -alpha * abs(n - target) / target
            for n, ok in zip(lengths, correct)]


def group_advantages(rewards):
    """GRPO: reward minus the group mean, over the group's standard deviation (0 if all equal)."""
    mean = sum(rewards) / len(rewards)
    spread = statistics.pstdev(rewards)
    return [0.0 if spread < 1e-8 else (r - mean) / spread for r in rewards]


def confidence_redundant(logps, thinking_len, quantile):
    """Thinking positions whose token the policy found least likely: the bottom quantile."""
    thinking = logps[:thinking_len]
    if not thinking:
        return set()
    cut = sorted(thinking)[max(0, int(len(thinking) * quantile) - 1)]
    return {i for i, lp in enumerate(thinking) if lp <= cut}


def settle_redundant(p_gold_by_cut, thinking_len, threshold):
    """Thinking positions after the first cut from which p(gold) stays at or above threshold."""
    settle = None
    for cut, p in sorted(p_gold_by_cut.items()):
        if p >= threshold:
            settle = cut if settle is None else settle
        else:
            settle = None
    return set() if settle is None else set(range(settle, thinking_len))


def norm_lp(logps):
    """TokenProbe Eq. 1: each token's log-prob z-scored within its reply."""
    mean = sum(logps) / len(logps)
    spread = math.sqrt(sum((x - mean) ** 2 for x in logps) / len(logps))
    return [(x - mean) / (spread + 1e-6) for x in logps]


def core_positions(scores, window, windows):
    """The best non-overlapping windows by summed score, greedily, at most `windows` of them."""
    if len(scores) <= window:
        return set(range(len(scores)))
    sums = [sum(scores[i:i + window]) for i in range(len(scores) - window + 1)]
    taken = set()
    for start in sorted(range(len(sums)), key=lambda i: -sums[i]):
        if len(taken) >= windows * window:
            break
        span = set(range(start, start + window))
        if not span & taken:
            taken |= span
    return taken


def rs_advantages(advantage, scores, core, budget, beta, alpha, lam):
    """TokenProbe Eq. 6 on a correct reply: core tokens gain alpha * NormLP, redundant ones pay
    lam * (T - L) / T when the reply is over budget."""
    over = max(0, len(scores) - budget) / len(scores)
    return [advantage + beta * (alpha * z if i in core else -lam * over) for i, z in enumerate(scores)]


def kl_coefficients(length, core, budget, beta, lam_decay):
    """TokenProbe Eq. 4: (beta_t, gamma_t) per token. Over budget, redundant tokens drop the KL
    anchor and get gamma_t = lam_decay * (T - L) / T; everything else keeps beta."""
    if length <= budget:
        return [beta] * length, [0.0] * length
    gamma = lam_decay * (length - budget) / length
    return ([beta if i in core else 0.0 for i in range(length)],
            [0.0 if i in core else gamma for i in range(length)])


def token_advantages(advantage, length, redundant, lam):
    """Every token gets the reply's advantage; redundant ones get -lam more."""
    return [advantage - (lam if i in redundant else 0.0) for i in range(length)]


# ---------------------------------------------------------------- sampling

class Sampler:
    """Replies from the current adapter: vLLM on a GPU (reloads the LoRA each step), HF otherwise."""

    def __init__(self, args, tokenizer):
        self.args, self.tokenizer = args, tokenizer
        if args.backend == "vllm":
            os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
            from vllm import LLM
            extra = {"attention_config": {"backend": args.attention_backend}} if args.attention_backend else {}
            self.engine = LLM(model=args.model, dtype=args.dtype, seed=args.seed, enable_lora=True,
                              max_lora_rank=64, gpu_memory_utilization=args.gpu_memory,
                              max_model_len=args.max_tokens + 1024, generation_config="vllm", **extra)

    def sample(self, prompts, adapter_dir, step, model=None):
        eos = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if self.args.backend == "vllm":
            from vllm import SamplingParams
            from vllm.lora.request import LoRARequest
            params = SamplingParams(n=self.args.group, max_tokens=self.args.max_tokens,
                                    seed=self.args.seed + step, stop_token_ids=[eos], **SAMPLING)
            lora = LoRARequest(f"step{step}", step + 1, str(Path(adapter_dir).resolve()))
            outputs = self.engine.generate([{"prompt_token_ids": p} for p in prompts], params, lora_request=lora)
            return [[list(o.token_ids) for o in out.outputs] for out in outputs]
        import torch
        replies = []
        model.eval()
        for prompt in prompts:
            ids = torch.tensor([prompt], device=model.device)
            with torch.no_grad():
                out = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=True,
                                     num_return_sequences=self.args.group, max_new_tokens=self.args.max_tokens,
                                     eos_token_id=eos, pad_token_id=eos, **SAMPLING)
            group = []
            for row in out[:, len(prompt):].tolist():
                group.append(row[:row.index(eos) + 1] if eos in row else row)
            replies.append(group)
        model.train()
        return replies


# ---------------------------------------------------------------- training

def main():
    args = parse_arguments()
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
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
    for parameter in trainable:  # fp16 gradients cannot be unscaled: keep the LoRA weights in fp32
        parameter.data = parameter.data.float()
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler(enabled=dtype == torch.float16)
    close_id = tokenizer.convert_tokens_to_ids("</think>")
    eos = tokenizer.convert_tokens_to_ids("<|im_end|>")

    keep = set(args.sources.split(","))
    pool = [r for r in read_jsonl(Path(args.data)) if r["source"] in keep]
    random.shuffle(pool)
    current = out / "current"
    model.save_pretrained(current)
    sampler = Sampler(args, tokenizer)
    log = []

    for step in range(args.steps):
        scale = args.scale_start + (args.scale_end - args.scale_start) * step / max(1, args.steps - 1)
        questions = [pool[(step * args.questions + i) % len(pool)] for i in range(args.questions)]
        prompts = [tokenizer.apply_chat_template(q["prompt"], tokenize=True, add_generation_prompt=True,
                                                 enable_thinking=True) for q in questions]
        replies = sampler.sample(prompts, current, step, model)

        jobs, stats = [], {"correct": 0, "replies": 0, "tokens": 0, "signal": 0}
        for question, prompt, group in zip(questions, prompts, replies):
            correct = [bool(r) and r[-1] == eos and is_task_correct(
                tokenizer.decode(r, skip_special_tokens=True), question["answer"], question["task"])
                for r in group]
            rewards = length_rewards([len(r) for r in group], correct, args.alpha)
            advantages = group_advantages(rewards)
            stats["correct"] += sum(correct)
            stats["replies"] += len(group)
            stats["tokens"] += sum(map(len, group))
            if any(a != 0 for a in advantages):
                stats["signal"] += 1
            for reply, ok, advantage in zip(group, correct, advantages):
                thinking_len = reply.index(close_id) if close_id in reply else len(reply)
                jobs.append((prompt, reply, ok, advantage, thinking_len, question))

        # one on-policy step: loss = -sum_t A_t log pi(token_t), averaged over all tokens
        total_tokens = max(1, sum(len(j[1]) for j in jobs))
        model.train()
        loss_sum = 0.0
        for prompt, reply, ok, advantage, thinking_len, question in jobs:
            if not reply or (advantage == 0 and not ok):
                continue
            ids = torch.tensor([prompt + reply], device=device)
            with torch.autocast(device_type=device, dtype=dtype, enabled=dtype != torch.float32):
                logits = model(input_ids=ids).logits[0, len(prompt) - 1:-1]
            logps = torch.log_softmax(logits.float(), dim=-1).gather(
                1, torch.tensor(reply, device=device)[:, None])[:, 0]
            redundant = set()
            betas = gammas = None
            if args.redundant.startswith("tokenprobe"):
                scores = norm_lp(logps.tolist())
                core = core_positions(scores, args.window, args.windows)
                budget = len(core) * scale
                if args.redundant == "tokenprobe-rs":
                    advantages_t = (rs_advantages(advantage, scores, core, budget, args.rs_beta,
                                                  args.rs_alpha, args.rs_lam) if ok else [advantage] * len(reply))
                else:
                    advantages_t = [advantage] * len(reply)
                    betas, gammas = (kl_coefficients(len(reply), core, budget, args.kl_beta, args.lam_decay)
                                     if ok else ([args.kl_beta] * len(reply), [0.0] * len(reply)))
            else:
                if ok and args.redundant == "confidence":
                    redundant = confidence_redundant(logps.tolist(), thinking_len, args.redundant_quantile)
                elif ok and args.redundant == "settle":
                    redundant = settle_redundant(probe_p_gold(model, tokenizer, prompt, reply, thinking_len, question),
                                                 thinking_len, args.settle_threshold)
                advantages_t = token_advantages(advantage, len(reply), redundant, args.lam)
            weights = torch.tensor(advantages_t, device=device)
            per_token = -weights * logps
            if betas is not None:
                with torch.no_grad(), model.disable_adapter():
                    with torch.autocast(device_type=device, dtype=dtype, enabled=dtype != torch.float32):
                        ref_logits = model(input_ids=ids).logits[0, len(prompt) - 1:-1]
                    ref = torch.log_softmax(ref_logits.float(), dim=-1).gather(
                        1, torch.tensor(reply, device=device)[:, None])[:, 0]
                gap = ref - logps  # k3 estimator of KL(pi || ref), per token
                kl = torch.exp(gap) - gap - 1
                per_token = per_token + torch.tensor(betas, device=device) * kl \
                    + torch.tensor(gammas, device=device) * logps
            loss = per_token.sum() / total_tokens
            scaler.scale(loss).backward()
            loss_sum += loss.item()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        model.save_pretrained(current)

        entry = {"step": step + 1, "accuracy": stats["correct"] / stats["replies"],
                 "mean_tokens": stats["tokens"] / stats["replies"],
                 "questions_with_signal": stats["signal"], "loss": loss_sum}
        log.append(entry)
        print(json.dumps(entry), flush=True)

    model.save_pretrained(out)
    (out / "rl_log.jsonl").write_text("".join(json.dumps(e) + "\n" for e in log))
    print(f"saved {out}")


def probe_p_gold(model, tokenizer, prompt, reply, thinking_len, question):
    """{cut: raw p(gold answer)} every 32 thinking tokens, with probe_curve's packed scorer."""
    from probe_curve import BRIDGE, probe_scores
    eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
    gold = tokenizer.encode(" " + question["answer"].strip(), add_special_tokens=False)
    suffix = gold if question["task"] == "multiple_choice" else gold + [eos]
    closing = tokenizer.encode(BRIDGE, add_special_tokens=False)
    closings = {cut: closing for cut in range(1, thinking_len + 1, 32)}
    scores = probe_scores(model, prompt, reply[:thinking_len], closings, [suffix], model.device)
    return {cut: math.exp(scores[(cut, 0)]) for cut in closings}


if __name__ == "__main__":
    main()
