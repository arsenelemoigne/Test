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

Stability knobs (DLER: the penalty matters less than stable training): --lr-end with
--restarts gives a cosine decay that restarts --restarts times; --updates splits each step's
groups into that many optimizer steps; --entropy-coef adds a bonus on the policy's mean token
entropy, which is logged every step so a collapse (the model no longer exploring) shows.

AdaptThink (--no-think-share): that share of each group starts with an empty
"<think>\n\n</think>\n\n", the prefix Qwen3's own non-thinking mode uses, as part of the reply.
The policy is trained on those prefix tokens too, so it learns when skipping the thinking still
gives a correct answer; that reply is the group's shortest, so the length reward favours it
whenever it is right. --no-think-bonus adds AdaptThink's delta to correct no-think replies.

DRPO (--objective drpo; Li, Chen, Lin, Yang, ICLR 2026, arXiv 2510.04474): the right and wrong replies of a group
are never compared with each other. Each reply is scored s = its mean token log-prob; right replies are all pushed up,
each with weight w = softmax over the group's RIGHT replies of r_l / lambda, r_l = 1 - len / (the group's longest
reply), so shorter right replies get more and a long right reply is never pushed down; wrong replies are all pushed
down with weight softmax(s / tau) over the group's wrong replies (length plays no part). Relative length (not their
len / 8k) gives a tool call and a code task the same pressure. A trust region pens updates that drift from the
sampling policy: beta0 * max(0, KL_hat - delta)^2, KL_hat the mean per-token log-ratio to the step's start (only
non-zero with --updates > 1). --reward and --advantage are ignored.

Start from the SFT adapter (--init-adapter) so the model already knows how to stop.

    python rl_train.py --init-adapter adapters/pruned_v2 --out adapters/rl --steps 6
    python evaluate.py --adapter adapters/rl --out runs/rl
"""

import argparse
from collections import defaultdict
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
    parser.add_argument("--sources", default="gsm8k,skywork,arc_easy,commonsense_qa")
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
    parser.add_argument("--lr-end", type=float, help="cosine-decay the lr to this (default: constant)")
    parser.add_argument("--restarts", type=int, default=1, help="cosine cycles over the run (warm restarts)")
    parser.add_argument("--updates", type=int, default=1, help="optimizer steps per sampling step")
    parser.add_argument("--entropy-coef", type=float, default=0.0, help="bonus on mean token entropy")
    parser.add_argument("--no-think-share", type=float, default=0.0,
                        help="share of each group sampled with an empty thinking block (AdaptThink)")
    parser.add_argument("--no-think-bonus", type=float, default=0.0)
    parser.add_argument("--batch-tokens", type=int, default=4096,
                        help="padded tokens per training forward pass")
    parser.add_argument("--rank", type=int, default=16, help="only used without --init-adapter")
    parser.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    parser.add_argument("--attention-backend", default=None, help="vLLM backend; TRITON_ATTN on a T4")
    parser.add_argument("--gpu-memory", type=float, default=0.35, help="vLLM share of the GPU")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sb-alpha", type=float, default=2.0, help="shorterbetter: weight of the correctness term")
    parser.add_argument("--sb-beta", type=float, default=0.005,
                        help="shorterbetter: per-token weight of |len - SOL| (paper 0.001 on 5-11k-token replies; ours are"
                             " ~5-10x shorter, so 0.005 keeps a similar pressure)")
    parser.add_argument("--reward", choices=["shorter", "arora", "shorterbetter"], default="shorter",
                        help="arora: correct x (1 - alpha_q * sigmoid(z)), z = the reply's thinking length "
                             "standardized among the group's correct replies; wrong or cut off = 0")
    parser.add_argument("--adaptive", action="store_true",
                        help="arora: alpha_q = alpha x the group's solve rate (ALP): hard questions keep their thinking")
    parser.add_argument("--advantage", choices=["group", "batch"], default="group",
                        help="batch: reward minus the group mean, over the whole batch's spread (DLER), not each group's")
    parser.add_argument("--kl-coef", type=float, default=0.0, help="k3 KL to the model without the LoRA, per token")
    parser.add_argument("--dynamic-sampling", type=float, default=0.0,
                        help="DLER's dynamic sampling: draw this many times --questions per step and train only on groups "
                             "with at least one right reply (an all-wrong group carries no correctness signal), keeping "
                             "--questions of them (0: off)")
    parser.add_argument("--objective", choices=["grpo", "drpo"], default="grpo")
    parser.add_argument("--protect-sources", default="",
                        help="comma list of sources (e.g. math) given NO length pressure: right replies weighted equally "
                             "(drpo) / correctness-only reward (shorterbetter); they are still trained to stay right")
    parser.add_argument("--guard-drop", type=float, default=0.0,
                        help="stop training when a data type's batch accuracy (mean of its last 5 steps) falls more than "
                             "this below its mean over the first 5 steps (e.g. 0.06); the last checkpoint is kept")
    parser.add_argument("--drpo-lambda", type=float, default=0.2, help="drpo: length weight temperature (smaller: shorter)")
    parser.add_argument("--drpo-tau", type=float, default=10.0, help="drpo: log-sum-exp temperature over wrong replies")
    parser.add_argument("--drpo-delta", type=float, default=1e-4, help="drpo: trust region size")
    parser.add_argument("--drpo-beta", type=float, default=1e3, help="drpo: trust region penalty")
    parser.add_argument("--grader", choices=["task", "typed"], default="task",
                        help="typed: typed.is_correct (math, multiple choice, code tests, tools, instructions)")
    parser.add_argument("--offset", type=int, default=0,
                        help="steps already done in earlier blocks: question slice and seed continue from there")
    parser.add_argument("--save-every", type=int, default=0, help="also keep the LoRA every N steps (out/step-N)")
    parser.add_argument("--eval-only", action="store_true",
                        help="no training: score --init-adapter (or the base model) on the first --questions of --data")
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


def shorterbetter_rewards(lengths, correct, alpha, beta):
    """ShorterBetter (arXiv 2504.21370, Eq. 1) exactly: r = alpha * correct - beta * |len - SOL|, SOL = the shortest correct
    reply's length in the group, or the group's mean length when none is correct (its neutral fallback; the shortest-reply
    target collapsed in its ablation). Absolute token lengths, not relative ones as in length_rewards."""
    lengths_correct = [n for n, ok in zip(lengths, correct) if ok]
    sol = min(lengths_correct) if lengths_correct else sum(lengths) / len(lengths)
    return [alpha * float(ok) - beta * abs(n - sol) for n, ok in zip(lengths, correct)]


def arora_rewards(thinking, correct, alpha):
    """Arora & Zanette (2502.04463): a correct reply gets 1 - alpha * sigmoid(z), z its thinking length
    standardized among the group's correct replies (shorter than its siblings: more than 1 - alpha/2); a wrong
    or cut-off reply gets 0, so being right always beats being short."""
    lengths = [n for n, ok in zip(thinking, correct) if ok]
    if not lengths:
        return [0.0] * len(thinking)
    mean = sum(lengths) / len(lengths)
    spread = statistics.pstdev(lengths) if len(lengths) > 1 else 0.0
    sigmoid = lambda z: 1 / (1 + math.exp(-z))
    return [1 - alpha * sigmoid((n - mean) / (spread + 1e-6)) if ok else 0.0 for n, ok in zip(thinking, correct)]


def drpo_weights(lengths, correct, old_scores, lam, tau):
    """DRPO per-reply coefficients for one group: positive for right replies (they sum to 1, shorter ones more:
    softmax of r_l / lam, r_l = 1 - len / the group's longest reply), negative for wrong ones (they sum to -1:
    softmax of the replies' old mean log-prob / tau, length-blind). A right reply's coefficient is never <= 0."""
    longest = max(max(lengths), 1)
    right = [i for i, ok in enumerate(correct) if ok]
    wrong = [i for i, ok in enumerate(correct) if not ok]
    coef = [0.0] * len(lengths)
    for members, values, sign in ((right, [(1 - lengths[i] / longest) / lam for i in right], 1.0),
                                  (wrong, [old_scores[i] / tau for i in wrong], -1.0)):
        if members:
            top = max(values)
            exps = [math.exp(v - top) for v in values]
            total = sum(exps)
            for i, e in zip(members, exps):
                coef[i] = sign * e / total
    return coef


def learning_rate(step, steps, lr, lr_end, restarts):
    """Cosine from lr to lr_end within each of `restarts` equal cycles; constant without lr_end."""
    if lr_end is None:
        return lr
    cycle = max(1, math.ceil(steps / max(1, restarts)))
    position = (step % cycle) / max(1, cycle - 1)
    return lr_end + (lr - lr_end) * (1 + math.cos(math.pi * position)) / 2


def split_groups(jobs, parts, rng):
    """Jobs split into `parts` chunks of whole question groups (a group's advantages stay together)."""
    groups = {}
    for job in jobs:
        groups.setdefault(id(job[5]), []).append(job)
    order = list(groups.values())
    rng.shuffle(order)
    return [[j for g in order[i::parts] for j in g] for i in range(parts) if order[i::parts]]


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


def balanced_order(rows, rng):
    """Rows shuffled within each source, then spread so that EVERY stretch of the order (each step's batch) holds the
    sources in their pool proportions from the first step to the last: row k of a source with n rows is placed at
    (k + u) / n, u uniform in [0, 1), and all rows are sorted by that position. (The former round-robin gave every source
    the same count per step until the smaller ones ran out, after which steps held only the larger sources: sb1's steps
    22-30 were all math and instructions.)"""
    by_source = {}
    for row in rows:
        by_source.setdefault(row["source"], []).append(row)
    placed = []
    for source in sorted(by_source):
        group = by_source[source]
        rng.shuffle(group)
        placed += [((k + rng.random()) / len(group), row) for k, row in enumerate(group)]
    placed.sort(key=lambda x: x[0])
    return [row for _, row in placed]


def batch_logps(model, rows, device, entropies=None):
    """log pi(reply token | prefix) for each (prompt, reply) row, in one right-padded forward pass.

    Logits are computed only on reply positions, through the output head, so memory grows with
    the reply tokens rather than with batch x sequence x vocabulary.
    """
    import torch

    from probe_curve import decoder_and_head
    body, head = decoder_and_head(model)
    width = max(len(p) + len(r) for p, r in rows)
    ids = torch.zeros(len(rows), width, dtype=torch.long)
    mask = torch.zeros(len(rows), width, dtype=torch.long)
    for i, (prompt, reply) in enumerate(rows):
        ids[i, :len(prompt) + len(reply)] = torch.tensor(prompt + reply)
        mask[i, :len(prompt) + len(reply)] = 1
    hidden = body(input_ids=ids.to(device), attention_mask=mask.to(device)).last_hidden_state
    from torch.utils.checkpoint import checkpoint

    def piece(h, target):
        # vocabulary-wide logits for a few hundred positions only, recomputed in the backward pass: a 3,000-token
        # reply would otherwise keep 3,000 x 152k fp32 values alive (~1.8 GB) and run the GPU out of memory
        logp = torch.log_softmax(head(h).float(), dim=-1)
        return logp.gather(1, target[:, None])[:, 0], -(logp.exp() * logp).sum(-1)

    out = []
    for i, (prompt, reply) in enumerate(rows):
        positions = torch.arange(len(prompt) - 1, len(prompt) + len(reply) - 1, device=device)
        target = torch.tensor(reply, device=device)
        tokens, ents = [], []
        for start in range(0, len(reply), 256):
            h = hidden[i, positions[start:start + 256]]
            t, e = checkpoint(piece, h, target[start:start + 256], use_reentrant=False) if h.requires_grad \
                else piece(h, target[start:start + 256])
            tokens.append(t)
            ents.append(e)
        out.append(torch.cat(tokens))
        if entropies is not None:
            entropies.append(torch.cat(ents))
    return out


def token_batches(jobs, budget):
    """Jobs grouped so each group's padded size (rows x longest) stays within budget."""
    jobs = sorted(jobs, key=lambda j: len(j[0]) + len(j[1]))
    batches, current = [], []
    for job in jobs:
        longest = max([len(j[0]) + len(j[1]) for j in current] + [len(job[0]) + len(job[1])])
        if current and longest * (len(current) + 1) > budget:
            batches.append(current)
            current = []
        current.append(job)
    if current:
        batches.append(current)
    return batches


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


def grade_all(questions, replies, tokenizer, eos, grader):
    """Correct or not, per reply (an unfinished reply is wrong); typed grading runs code tests in parallel."""
    from concurrent.futures import ThreadPoolExecutor
    if grader == "typed":
        import typed  # noqa: F401  (imported once here, not raced by the grading threads)
    jobs = [(q, r) for q, group in zip(questions, replies) for r in group]

    def grade(job):
        question, reply = job
        if not reply or reply[-1] != eos:
            return False
        if grader == "typed":
            from typed import full_text, is_correct
            return bool(is_correct(question, full_text(tokenizer, reply)))
        return is_task_correct(tokenizer.decode(reply, skip_special_tokens=True), question["answer"], question["task"])
    with ThreadPoolExecutor(8) as pool:
        flat = list(pool.map(grade, jobs))
    out, i = [], 0
    for group in replies:
        out.append(flat[i:i + len(group)])
        i += len(group)
    return out


# ---------------------------------------------------------------- sampling

NO_THINK = "<think>\n\n</think>\n\n"  # what Qwen3's chat template inserts with enable_thinking=False

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

    def sample(self, prompts, adapter_dir, step, model=None, no_think=0):
        """--group replies per prompt; the last `no_think` of each group start with an empty
        thinking block, which is returned as part of the reply."""
        eos = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        prefix = self.tokenizer.encode(NO_THINK, add_special_tokens=False)
        counts = [(self.args.group - no_think, []), (no_think, prefix)]
        if self.args.backend == "vllm":
            from vllm import SamplingParams
            from vllm.lora.request import LoRARequest
            lora = LoRARequest(f"step{step}", step + 1, str(Path(adapter_dir).resolve())) if adapter_dir else None
            requests, params = [], []
            for n, start in counts:
                if n:
                    requests += [{"prompt_token_ids": p + start} for p in prompts]
                    params += [SamplingParams(n=n, max_tokens=self.args.max_tokens - len(start),
                                              seed=self.args.seed + step, stop_token_ids=[eos],
                                              **SAMPLING)] * len(prompts)
            outputs = self.engine.generate(requests, params, lora_request=lora)
            replies = [[] for _ in prompts]
            for index, out in enumerate(outputs):
                start = counts[0][1] if index < len(prompts) and counts[0][0] else prefix
                replies[index % len(prompts)] += [start + list(o.token_ids) for o in out.outputs]
            return replies
        import torch
        replies = []
        model.eval()
        for prompt in prompts:
            group = []
            for n, start in counts:
                if not n:
                    continue
                ids = torch.tensor([prompt + start], device=model.device)
                with torch.no_grad():
                    out = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=True,
                                         num_return_sequences=n, max_new_tokens=self.args.max_tokens - len(start),
                                         eos_token_id=eos, pad_token_id=eos, **SAMPLING)
                for row in out[:, len(prompt) + len(start):].tolist():
                    group.append(start + (row[:row.index(eos) + 1] if eos in row else row))
            replies.append(group)
        model.train()
        return replies


# ---------------------------------------------------------------- training

def held_out_eval(args):
    """Accuracy and length on a fixed, never-trained-on question set, per source."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
    keep = set(args.sources.split(","))
    questions = balanced_order([r for r in read_jsonl(Path(args.data)) if r["source"] in keep],
                               random.Random(0))[:args.questions]
    prompts = [tokenizer.apply_chat_template(q["prompt"], tokenize=True, add_generation_prompt=True,
                                             enable_thinking=True) for q in questions]
    replies = Sampler(args, tokenizer).sample(prompts, args.init_adapter, 0)
    close_id = tokenizer.convert_tokens_to_ids("</think>")
    keys = ("correct", "replies", "tokens", "thinking", "skipped")
    stats = {}
    for question, group in zip(questions, replies):
        entry = stats.setdefault(question["source"], dict.fromkeys(keys, 0))
        for reply in group:
            thinking = reply.index(close_id) if close_id in reply else len(reply)
            entry["correct"] += bool(reply) and reply[-1] == eos and is_task_correct(
                tokenizer.decode(reply, skip_special_tokens=True), question["answer"], question["task"])
            entry["replies"] += 1
            entry["tokens"] += len(reply)
            entry["thinking"] += thinking
            entry["skipped"] += thinking <= 4  # "<think>\n\n" only
    stats["all"] = {k: sum(e[k] for e in stats.values()) for k in keys}
    rows = {source: {"accuracy": e["correct"] / e["replies"], "mean_tokens": e["tokens"] / e["replies"],
                     "mean_thinking_tokens": e["thinking"] / e["replies"],
                     "skipped_thinking": e["skipped"] / e["replies"]}
            for source, e in sorted(stats.items())}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "held_out.json").write_text(json.dumps(rows, indent=1))
    print(json.dumps(rows), flush=True)


def main():
    args = parse_arguments()
    if args.eval_only:
        return held_out_eval(args)
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
    pool = balanced_order([r for r in read_jsonl(Path(args.data)) if args.sources == "all" or r["source"] in keep],
                          random.Random(args.seed))
    current = out / "current"
    model.save_pretrained(current)
    sampler = Sampler(args, tokenizer)
    log = []
    rng = random.Random(args.seed)

    for step in range(args.steps):
        scale = args.scale_start + (args.scale_end - args.scale_start) * step / max(1, args.steps - 1)
        drawn = math.ceil(args.questions * args.dynamic_sampling) if args.dynamic_sampling else args.questions
        questions = [pool[((args.offset + step) * drawn + i) % len(pool)] for i in range(drawn)]
        prompts = [tokenizer.apply_chat_template(q["prompt"], tokenize=True, add_generation_prompt=True,
                                                 enable_thinking=True) for q in questions]
        no_think = round(args.group * args.no_think_share)
        replies = sampler.sample(prompts, current, args.offset + step, model, no_think)

        by_type = {}
        protect = set(filter(None, args.protect_sources.split(",")))
        jobs, stats = [], {"correct": 0, "replies": 0, "tokens": 0, "signal": 0, "thinking": 0,
                           "no_think_correct": 0, "no_think_replies": 0}
        grades = grade_all(questions, replies, tokenizer, eos, args.grader)
        dropped = 0
        if args.dynamic_sampling:  # keep informative groups only, up to --questions of them
            keep = [k for k in range(len(questions)) if any(grades[k])][:args.questions]
            dropped = len(questions) - len(keep)
            questions, prompts, replies, grades = ([xs[k] for k in keep] for xs in (questions, prompts, replies, grades))
        groups = []
        for question, prompt, group, correct in zip(questions, prompts, replies, grades):
            if args.reward == "arora":
                thinking = [r.index(close_id) if close_id in r else len(r) for r in group]
                alpha = args.alpha * (sum(correct) / len(correct) if args.adaptive else 1.0)
                rewards = arora_rewards(thinking, correct, alpha)
            elif args.reward == "shorterbetter":
                rewards = shorterbetter_rewards([len(r) for r in group], correct, args.sb_alpha,
                                                0.0 if question.get("source") in protect else args.sb_beta)
            else:
                rewards = length_rewards([len(r) for r in group], correct, args.alpha)
            skipped = [i >= len(group) - no_think for i in range(len(group))]
            rewards = [r + (args.no_think_bonus if ok and skip else 0.0)
                       for r, ok, skip in zip(rewards, correct, skipped)]
            advantages = group_advantages(rewards) if args.advantage == "group" else None
            groups.append(rewards)
            stats["no_think_correct"] += sum(ok for ok, skip in zip(correct, skipped) if skip)
            stats["no_think_replies"] += sum(skipped)
            stats["thinking"] += sum(r.index(close_id) if close_id in r else len(r) for r in group)
            stats["correct"] += sum(correct)
            kind = question.get("source", "all")  # per data type: a collapse of one type shows even in a mixed batch
            by_type.setdefault(kind, [0, 0, 0])
            by_type[kind][0] += sum(correct)
            by_type[kind][1] += len(group)
            by_type[kind][2] += sum(r.index(close_id) if close_id in r else len(r) for r in group)
            stats["replies"] += len(group)
            stats["tokens"] += sum(map(len, group))
            if advantages and any(a != 0 for a in advantages):
                stats["signal"] += 1
            for index, (reply, ok) in enumerate(zip(group, correct)):
                thinking_len = reply.index(close_id) if close_id in reply else len(reply)
                jobs.append([prompt, reply, ok, advantages[index] if advantages else (len(groups) - 1, index),
                             thinking_len, question])
        old_logps = {}
        if args.objective == "drpo":  # coefficients per reply, from the sampling policy's scores (no gradient)
            model.eval()
            with torch.no_grad():
                for batch in token_batches([tuple(j) for j in jobs if j[1]], args.batch_tokens):
                    with torch.autocast(device_type=device, dtype=dtype, enabled=dtype != torch.float32):
                        lps = batch_logps(model, [(j[0], j[1]) for j in batch], device)
                    for j, lp in zip(batch, lps):
                        old_logps[id(j[1])] = lp.detach().float()
            by_group = defaultdict(list)
            for job in jobs:
                by_group[id(job[5])].append(job)
            stats["signal"] = 0
            for members in by_group.values():
                scores = [float(old_logps[id(j[1])].mean()) if j[1] else 0.0 for j in members]
                protected = members[0][5].get("source") in protect
                coef = drpo_weights([len(j[1]) for j in members], [j[2] for j in members], scores,
                                    1e9 if protected else args.drpo_lambda, args.drpo_tau)
                for j, c in zip(members, coef):
                    j[3] = c
                stats["signal"] += any(c != 0 for c in coef)
        elif args.advantage == "batch":  # DLER: center per group, scale by the whole batch's spread
            centered = [[r - sum(g) / len(g) for r in g] for g in groups]
            spread = statistics.pstdev([c for g in centered for c in g]) or 1.0
            for job in jobs:
                g, i = job[3]
                job[3] = centered[g][i] / spread
            stats["signal"] = sum(any(abs(c) > 1e-8 for c in g) for g in centered)
        jobs = [tuple(j) for j in jobs]

        # on-policy update(s): loss = -sum_t A_t log pi(token_t), averaged over the chunk's tokens
        lr = learning_rate(step, args.steps, args.lr, args.lr_end, args.restarts)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        model.train()
        loss_sum, entropy_sum, entropy_count = 0.0, 0.0, 0
        active = [j for j in jobs if j[1] and not (j[3] == 0 and not j[2])]
        for chunk in split_groups(active, args.updates, rng):
            total_tokens = max(1, sum(len(j[1]) for j in chunk))
            n_questions = max(1, len({id(j[5]) for j in chunk}))
            if args.objective == "drpo":
                kl_sum, kl_terms = 0.0, []
                for batch in token_batches(chunk, args.batch_tokens):
                    with torch.autocast(device_type=device, dtype=dtype, enabled=dtype != torch.float32):
                        batch_lp = batch_logps(model, [(j[0], j[1]) for j in batch], device)
                    loss = 0.0
                    for j, logps in zip(batch, batch_lp):
                        loss = loss - j[3] * logps.float().mean() / n_questions
                        kl_terms.append(float((old_logps[id(j[1])].to(logps.device) - logps.float()).sum()))
                    scaler.scale(loss).backward()
                    loss_sum += float(loss)
                kl_hat = sum(kl_terms) / total_tokens
                if kl_hat > args.drpo_delta:  # trust region: a second pass adds the penalty's gradient
                    scale_kl = 2 * args.drpo_beta * (kl_hat - args.drpo_delta)
                    for batch in token_batches(chunk, args.batch_tokens):
                        with torch.autocast(device_type=device, dtype=dtype, enabled=dtype != torch.float32):
                            batch_lp = batch_logps(model, [(j[0], j[1]) for j in batch], device)
                        penalty = sum((old_logps[id(j[1])].to(lp.device) - lp.float()).sum()
                                      for j, lp in zip(batch, batch_lp)) * scale_kl / total_tokens
                        scaler.scale(penalty).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                continue
            for batch in token_batches(chunk, args.batch_tokens):
                with torch.autocast(device_type=device, dtype=dtype, enabled=dtype != torch.float32):
                    entropies = []
                    batch_lp = batch_logps(model, [(j[0], j[1]) for j in batch], device, entropies)
                refs = None
                if args.redundant == "tokenprobe-kl" or args.kl_coef:
                    with torch.no_grad(), model.disable_adapter():
                        with torch.autocast(device_type=device, dtype=dtype, enabled=dtype != torch.float32):
                            refs = batch_logps(model, [(j[0], j[1]) for j in batch], device)
                loss = 0.0
                for index, (prompt, reply, ok, advantage, thinking_len, question) in enumerate(batch):
                    logps = batch_lp[index]
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
                            redundant = settle_redundant(
                                probe_p_gold(model, tokenizer, prompt, reply, thinking_len, question),
                                thinking_len, args.settle_threshold)
                        advantages_t = token_advantages(advantage, len(reply), redundant, args.lam)
                    per_token = -torch.tensor(advantages_t, device=device) * logps
                    if betas is None and args.kl_coef:
                        betas = [args.kl_coef] * len(reply)
                        gammas = [0.0] * len(reply)
                    if betas is not None:
                        gap = refs[index] - logps  # k3 estimator of KL(pi || ref), per token
                        kl = torch.exp(gap) - gap - 1
                        per_token = per_token + torch.tensor(betas, device=device) * kl \
                            + torch.tensor(gammas, device=device) * logps
                    if args.entropy_coef:
                        per_token = per_token - args.entropy_coef * entropies[index]
                    entropy_sum += float(entropies[index].detach().sum())
                    entropy_count += len(reply)
                    loss = loss + per_token.sum() / total_tokens
                scaler.scale(loss).backward()
                loss_sum += float(loss)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        model.save_pretrained(current)
        if args.save_every and (step + 1) % args.save_every == 0:
            model.save_pretrained(out / f"step-{step + 1}")

        entry = {"step": step + 1, "accuracy": stats["correct"] / stats["replies"],
                 "mean_tokens": stats["tokens"] / stats["replies"],
                 "mean_thinking_tokens": stats["thinking"] / stats["replies"],
                 "questions_with_signal": stats["signal"], "dropped_groups": dropped, "loss": loss_sum, "lr": lr,
                 "entropy": entropy_sum / max(1, entropy_count)}
        entry["by_type"] = {k: {"accuracy": round(c / max(1, n), 3), "thinking": round(t / max(1, n)), "replies": n}
                            for k, (c, n, t) in sorted(by_type.items())}
        if stats["no_think_replies"]:
            entry["no_think_accuracy"] = stats["no_think_correct"] / stats["no_think_replies"]
        log.append(entry)
        print(json.dumps(entry), flush=True)
        if args.guard_drop > 0 and len(log) >= 10:  # per-type accuracy guard (user: strict checks per data type)
            def acc(entries, kind):
                c = sum(e["by_type"][kind]["accuracy"] * e["by_type"][kind]["replies"] for e in entries if kind in e["by_type"])
                n = sum(e["by_type"][kind]["replies"] for e in entries if kind in e["by_type"])
                return c / n if n >= 40 else None
            fallen = []
            for kind in sorted({k for e in log for k in e["by_type"]}):
                first, last = acc(log[:5], kind), acc(log[-5:], kind)
                if first is not None and last is not None and last < first - args.guard_drop:
                    fallen.append(f"{kind} {first:.2f} -> {last:.2f}")
            if fallen:
                print(f"guard: stopping at step {step + 1}, accuracy fell: {'; '.join(fallen)}", flush=True)
                model.save_pretrained(out / f"step-{step + 1}")
                break

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
