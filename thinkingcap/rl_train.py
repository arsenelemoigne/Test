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

Protected sources (--protect-sources, e.g. humaneval): those questions are still sampled and
trained on, but rewarded for correctness only (1 right, 0 wrong) with no length term and no
token-level pressure, optionally anchored to the base model by --protect-kl. They keep the model
practising those tasks while other sources teach brevity, so the length pressure cannot erode them.

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


def is_correct(text, question):
    """The training grader for maths and multiple choice; code runs its unit tests (code_tasks.py)."""
    if question["task"] == "code":
        from code_tasks import is_code_correct
        return is_code_correct(text, question["answer"])
    return is_task_correct(text, question["answer"], question["task"])


def source_caps(args):
    """{source: reply token cap} from --max-tokens-by-source."""
    return {k: int(v) for k, v in (item.split("=") for item in filter(None, args.max_tokens_by_source.split(",")))}


def read_rows(paths):
    """Rows of one or several comma-separated jsonl files."""
    return [row for path in paths.split(",") for row in read_jsonl(Path(path))]


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
    parser.add_argument("--protect-sources", default="",
                        help="comma-separated sources trained for correctness only (no length pressure)")
    parser.add_argument("--max-tokens-by-source", default="",
                        help="per-source reply cap overriding --max-tokens, e.g. humaneval=4096")
    parser.add_argument("--source-alpha", default="",
                        help="per-source length weight overriding --alpha, e.g. gsm8k=0.25,skywork=0.25")
    parser.add_argument("--mirror", help="copy --out here after every saved checkpoint (e.g. a Drive folder)")
    parser.add_argument("--protect-kl", type=float, default=0.0,
                        help="KL anchor to the base model on protected sources' tokens")
    parser.add_argument("--lam-decay", type=float, default=0.2)
    parser.add_argument("--redundant-quantile", type=float, default=0.2)
    parser.add_argument("--settle-threshold", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lr-end", type=float, help="cosine-decay the lr to this (default: constant)")
    parser.add_argument("--restarts", type=int, default=1, help="cosine cycles over the run (warm restarts)")
    parser.add_argument("--updates", type=int, default=1, help="optimizer steps per sampling step")
    parser.add_argument("--guard-every", type=int, default=0,
                        help="every N steps, score a fixed held-out set against the raw model; keep the "
                             "shortest checkpoint within --guard-drop (out/best), stop after --guard-patience misses")
    parser.add_argument("--guard-data", default="data/val.jsonl")
    parser.add_argument("--guard-questions", type=int, default=128)
    parser.add_argument("--guard-group", type=int, default=2)
    parser.add_argument("--guard-drop", type=float, default=0.03, help="largest accepted accuracy drop vs raw")
    parser.add_argument("--guard-patience", type=int, default=3, help="misses in a row before stopping")
    parser.add_argument("--guard-action", choices=["soften", "stop"], default="soften",
                        help="on a miss, soften: roll back to the best checkpoint so far (or the start), "
                             "halve the lr and the length pressure, and go on; stop: only count the miss")
    parser.add_argument("--save-every", type=int, default=0, help="also keep the adapter every N steps (out/stepN)")
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
    """Rows shuffled within each source, then interleaved round-robin, so every step sees every
    source in equal measure however unequal their sizes; --seed picks a different slice."""
    by_source = {}
    for row in rows:
        by_source.setdefault(row["source"], []).append(row)
    queues = []
    for source in sorted(by_source):
        rng.shuffle(by_source[source])
        queues.append(by_source[source])
    order, index = [], 0
    while any(index < len(q) for q in queues):
        order += [q[index] for q in queues if index < len(q)]
        index += 1
    return order


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
    out = []
    for i, (prompt, reply) in enumerate(rows):
        positions = torch.arange(len(prompt) - 1, len(prompt) + len(reply) - 1, device=device)
        logp = torch.log_softmax(head(hidden[i, positions]).float(), dim=-1)
        out.append(logp.gather(1, torch.tensor(reply, device=device)[:, None])[:, 0])
        if entropies is not None:
            entropies.append(-(logp.exp() * logp).sum(-1))
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
                              max_model_len=max([args.max_tokens, *source_caps(args).values()]) + 1024,
                              generation_config="vllm", **extra)

    def sample(self, prompts, adapter_dir, step, model=None, no_think=0, group=None, lora_id=None, caps=None,
               sampling=None):
        """`group` (default --group) replies per prompt; the last `no_think` of each group start
        with an empty thinking block, which is returned as part of the reply. Without
        adapter_dir the raw model answers. `caps` gives each prompt its own reply cap."""
        caps = caps or [self.args.max_tokens] * len(prompts)
        sampling = sampling or SAMPLING
        eos = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        prefix = self.tokenizer.encode(NO_THINK, add_special_tokens=False)
        counts = [((group or self.args.group) - no_think, []), (no_think, prefix)]
        if self.args.backend == "vllm":
            from vllm import SamplingParams
            from vllm.lora.request import LoRARequest
            lora_id = lora_id or step + 1
            lora = LoRARequest(f"lora{lora_id}", lora_id, str(Path(adapter_dir).resolve())) if adapter_dir else None
            requests, params = [], []
            for n, start in counts:
                if n:
                    requests += [{"prompt_token_ids": p + start} for p in prompts]
                    params += [SamplingParams(n=n, max_tokens=cap - len(start), seed=self.args.seed + step,
                                              stop_token_ids=[eos], **sampling) for cap in caps]
            outputs = self.engine.generate(requests, params, lora_request=lora)
            replies = [[] for _ in prompts]
            for index, out in enumerate(outputs):
                start = counts[0][1] if index < len(prompts) and counts[0][0] else prefix
                replies[index % len(prompts)] += [start + list(o.token_ids) for o in out.outputs]
            return replies
        import contextlib
        model.eval()
        raw = model.disable_adapter() if adapter_dir is None and hasattr(model, "disable_adapter") \
            else contextlib.nullcontext()
        with raw:
            replies = self._hf_sample(model, prompts, counts, eos, caps, sampling)
        model.train()
        return replies

    def _hf_sample(self, model, prompts, counts, eos, caps, sampling):
        import torch
        replies = []
        for prompt, cap in zip(prompts, caps):
            group = []
            for n, start in counts:
                if not n:
                    continue
                ids = torch.tensor([prompt + start], device=model.device)
                with torch.no_grad():
                    out = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=True,
                                         num_return_sequences=n, max_new_tokens=cap - len(start),
                                         eos_token_id=eos, pad_token_id=eos, **sampling)
                for row in out[:, len(prompt) + len(start):].tolist():
                    group.append(start + (row[:row.index(eos) + 1] if eos in row else row))
            replies.append(group)
        return replies


# ---------------------------------------------------------------- training

def held_out_questions(args, tokenizer, path, count):
    """The first `count` questions of `path`, balanced across --sources, in a fixed order."""
    keep = set(args.sources.split(","))
    questions = balanced_order([r for r in read_rows(path) if r["source"] in keep],
                               random.Random(0))[:count]
    prompts = [tokenizer.apply_chat_template(q["prompt"], tokenize=True, add_generation_prompt=True,
                                             enable_thinking=True) for q in questions]
    return questions, prompts


def score_replies(questions, replies, tokenizer):
    """Accuracy, length, thinking length and share of skipped thinking, per source and in all."""
    eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
    close_id = tokenizer.convert_tokens_to_ids("</think>")
    keys = ("correct", "replies", "tokens", "thinking", "skipped")
    stats = {}
    for question, group in zip(questions, replies):
        entry = stats.setdefault(question["source"], dict.fromkeys(keys, 0))
        for reply in group:
            thinking = reply.index(close_id) if close_id in reply else len(reply)
            entry["correct"] += bool(reply) and reply[-1] == eos and is_correct(
                tokenizer.decode(reply, skip_special_tokens=True), question)
            entry["replies"] += 1
            entry["tokens"] += len(reply)
            entry["thinking"] += thinking
            entry["skipped"] += thinking <= 4  # "<think>\n\n" only
    stats["all"] = {k: sum(e[k] for e in stats.values()) for k in keys}
    return {source: {"accuracy": e["correct"] / e["replies"], "mean_tokens": e["tokens"] / e["replies"],
                     "mean_thinking_tokens": e["thinking"] / e["replies"],
                     "skipped_thinking": e["skipped"] / e["replies"]}
            for source, e in sorted(stats.items())}


def held_out_eval(args):
    """Accuracy and length on a fixed, never-trained-on question set, per source."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    questions, prompts = held_out_questions(args, tokenizer, args.data, args.questions)
    rows = score_replies(questions, Sampler(args, tokenizer).sample(prompts, args.init_adapter, 0), tokenizer)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "held_out.json").write_text(json.dumps(rows, indent=1))
    print(json.dumps(rows), flush=True)


def main():
    args = parse_arguments()
    if args.eval_only:
        return held_out_eval(args)
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file
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
    pool = balanced_order([r for r in read_rows(args.data) if r["source"] in keep], random.Random(args.seed))
    current = out / "current"
    model.save_pretrained(current)
    sampler = Sampler(args, tokenizer)
    log = []
    protect = set(filter(None, args.protect_sources.split(",")))
    source_alpha = {k: float(v) for k, v in (item.split("=") for item in filter(None, args.source_alpha.split(",")))}

    def mirror():
        if args.mirror:
            import shutil
            shutil.copytree(out, args.mirror, dirs_exist_ok=True)
    caps = source_caps(args)
    soften = 1.0  # halved each time the guard rolls back
    guard = None
    if args.guard_every:
        guard_questions, guard_prompts = held_out_questions(args, tokenizer, args.guard_data, args.guard_questions)

        def guard_check(adapter_dir, index):
            replies = sampler.sample(guard_prompts, adapter_dir, 10 ** 6, model, group=args.guard_group,
                                     lora_id=10 ** 6 + index,
                                     caps=[caps.get(q["source"], args.max_tokens) for q in guard_questions])
            return score_replies(guard_questions, replies, tokenizer)["all"]

        guard = {"raw": guard_check(None, 0), "best": None, "misses": 0, "checks": [], "softened": []}
        model.save_pretrained(out / "start")
        print(json.dumps({"guard_raw": guard["raw"]}), flush=True)
    rng = random.Random(args.seed)

    for step in range(args.steps):
        scale = args.scale_start + (args.scale_end - args.scale_start) * step / max(1, args.steps - 1)
        questions = [pool[(step * args.questions + i) % len(pool)] for i in range(args.questions)]
        prompts = [tokenizer.apply_chat_template(q["prompt"], tokenize=True, add_generation_prompt=True,
                                                 enable_thinking=True) for q in questions]
        no_think = round(args.group * args.no_think_share)
        replies = sampler.sample(prompts, current, step, model, no_think,
                                 caps=[caps.get(q["source"], args.max_tokens) for q in questions])

        by_source = {}
        jobs, stats = [], {"correct": 0, "replies": 0, "tokens": 0, "signal": 0, "thinking": 0,
                           "no_think_correct": 0, "no_think_replies": 0, "protected_correct": 0,
                           "protected_replies": 0, "protected_tokens": 0}
        for question, prompt, group in zip(questions, prompts, replies):
            correct = [bool(r) and r[-1] == eos and is_correct(
                tokenizer.decode(r, skip_special_tokens=True), question)
                for r in group]
            protected = question["source"] in protect
            rewards = ([1.0 if ok else 0.0 for ok in correct] if protected
                       else length_rewards([len(r) for r in group], correct,
                                           source_alpha.get(question["source"], args.alpha) * soften))
            if protected:
                stats["protected_correct"] += sum(correct)
                stats["protected_replies"] += len(group)
                stats["protected_tokens"] += sum(map(len, group))
            skipped = [i >= len(group) - no_think for i in range(len(group))]
            rewards = [r + (args.no_think_bonus if ok and skip else 0.0)
                       for r, ok, skip in zip(rewards, correct, skipped)]
            advantages = group_advantages(rewards)
            stats["no_think_correct"] += sum(ok for ok, skip in zip(correct, skipped) if skip)
            stats["no_think_replies"] += sum(skipped)
            stats["thinking"] += sum(r.index(close_id) if close_id in r else len(r) for r in group)
            per = by_source.setdefault(question["source"], {"correct": 0, "replies": 0, "tokens": 0,
                                                            "entropy": 0.0, "entropy_tokens": 0})
            per["correct"] += sum(correct)
            per["replies"] += len(group)
            per["tokens"] += sum(map(len, group))
            stats["correct"] += sum(correct)
            stats["replies"] += len(group)
            stats["tokens"] += sum(map(len, group))
            if any(a != 0 for a in advantages):
                stats["signal"] += 1
            for reply, ok, advantage in zip(group, correct, advantages):
                thinking_len = reply.index(close_id) if close_id in reply else len(reply)
                jobs.append((prompt, reply, ok, advantage, thinking_len, question))

        # on-policy update(s): loss = -sum_t A_t log pi(token_t), averaged over the chunk's tokens
        lr = learning_rate(step, args.steps, args.lr, args.lr_end, args.restarts) * soften
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        model.train()
        loss_sum, entropy_sum, entropy_count = 0.0, 0.0, 0
        active = [j for j in jobs if j[1] and not (j[3] == 0 and not j[2])]
        for chunk in split_groups(active, args.updates, rng):
            total_tokens = max(1, sum(len(j[1]) for j in chunk))
            for batch in token_batches(chunk, args.batch_tokens):
                with torch.autocast(device_type=device, dtype=dtype, enabled=dtype != torch.float32):
                    entropies = []
                    batch_lp = batch_logps(model, [(j[0], j[1]) for j in batch], device, entropies)
                refs = None
                anchored = args.protect_kl and any(j[5]["source"] in protect for j in batch)
                if args.redundant == "tokenprobe-kl" or anchored:
                    with torch.no_grad(), model.disable_adapter():
                        with torch.autocast(device_type=device, dtype=dtype, enabled=dtype != torch.float32):
                            refs = batch_logps(model, [(j[0], j[1]) for j in batch], device)
                loss = 0.0
                for index, (prompt, reply, ok, advantage, thinking_len, question) in enumerate(batch):
                    logps = batch_lp[index]
                    redundant = set()
                    betas = gammas = None
                    if question["source"] in protect:
                        advantages_t = [advantage] * len(reply)
                        if args.protect_kl:
                            betas, gammas = [args.protect_kl] * len(reply), [0.0] * len(reply)
                    elif args.redundant.startswith("tokenprobe"):
                        scores = norm_lp(logps.tolist())
                        core = core_positions(scores, args.window, args.windows)
                        budget = len(core) * scale
                        if args.redundant == "tokenprobe-rs":
                            advantages_t = (rs_advantages(advantage, scores, core, budget, args.rs_beta,
                                                          args.rs_alpha, args.rs_lam) if ok else [advantage] * len(reply))
                        else:
                            advantages_t = [advantage] * len(reply)
                            betas, gammas = (kl_coefficients(len(reply), core, budget, args.kl_beta,
                                                           args.lam_decay * soften)
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
                    if betas is not None:
                        gap = refs[index] - logps  # k3 estimator of KL(pi || ref), per token
                        kl = torch.exp(gap) - gap - 1
                        per_token = per_token + torch.tensor(betas, device=device) * kl \
                            + torch.tensor(gammas, device=device) * logps
                    if args.entropy_coef:
                        per_token = per_token - args.entropy_coef * entropies[index]
                    entropy_sum += float(entropies[index].detach().sum())
                    entropy_count += len(reply)
                    by_source[question["source"]]["entropy"] += float(entropies[index].detach().sum())
                    by_source[question["source"]]["entropy_tokens"] += len(reply)
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
            model.save_pretrained(out / f"step{step + 1}")
            (out / "rl_log.jsonl").write_text("".join(json.dumps(e) + "\n" for e in log))
            mirror()

        entry = {"step": step + 1, "accuracy": stats["correct"] / stats["replies"],
                 "mean_tokens": stats["tokens"] / stats["replies"],
                 "mean_thinking_tokens": stats["thinking"] / stats["replies"],
                 "questions_with_signal": stats["signal"], "loss": loss_sum, "lr": lr,
                 "entropy": entropy_sum / max(1, entropy_count),
                 "by_source": {source: {"accuracy": round(v["correct"] / v["replies"], 3),
                                        "mean_tokens": round(v["tokens"] / v["replies"]),
                                        "entropy": round(v["entropy"] / max(1, v["entropy_tokens"]), 3)}
                               for source, v in sorted(by_source.items())}}
        if stats["protected_replies"]:
            entry["protected_accuracy"] = stats["protected_correct"] / stats["protected_replies"]
            entry["protected_tokens"] = stats["protected_tokens"] / stats["protected_replies"]
        if stats["no_think_replies"]:
            entry["no_think_accuracy"] = stats["no_think_correct"] / stats["no_think_replies"]
        stop = False
        if guard and (step + 1) % args.guard_every == 0:
            result = guard_check(current, step + 1)
            drop = guard["raw"]["accuracy"] - result["accuracy"]
            entry.update(guard_accuracy=result["accuracy"], guard_tokens=result["mean_tokens"], guard_drop=drop)
            guard["checks"].append({"step": step + 1, **result})
            if drop <= args.guard_drop:
                guard["misses"] = 0
                if guard["best"] is None or result["mean_tokens"] < guard["best"]["mean_tokens"]:
                    model.save_pretrained(out / "best")
                    guard["best"] = {"step": step + 1, **result}
            else:
                guard["misses"] += 1
                stop = guard["misses"] >= args.guard_patience
                if args.guard_action == "soften" and not stop:
                    restore = out / ("best" if guard["best"] else "start")
                    set_peft_model_state_dict(model, load_file(restore / "adapter_model.safetensors"))
                    optimizer.state.clear()
                    model.save_pretrained(current)
                    soften /= 2
                    guard["softened"].append({"step": step + 1, "restored": restore.name, "soften": soften})
                    entry.update(rolled_back_to=restore.name, soften=soften)
        log.append(entry)
        print(json.dumps(entry), flush=True)
        if stop:
            print(f"guard: accuracy more than {args.guard_drop:.0%} below the raw model "
                  f"{args.guard_patience} checks in a row, stopping", flush=True)
            break

    model.save_pretrained(out)
    (out / "rl_log.jsonl").write_text("".join(json.dumps(e) + "\n" for e in log))
    if guard:
        (out / "guard.json").write_text(json.dumps(guard, indent=1))
    mirror()
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
