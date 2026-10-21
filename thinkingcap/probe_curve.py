"""How early does the model know its answer? Probe the answer distribution as thinking grows.

For each question this samples full rollouts, then cuts each rollout's thinking after k tokens
(k = 0, stride, 2*stride, ...), closes it with "\\n</think>\\n\\nFinal answer:" (the newline only
when the cut does not already end with one, as Qwen3 writes it) and reads the
model's probability of every candidate answer by teacher forcing -- no sampling. The rollout
is read once and every cut is scored as a short segment attached to it, so a cut costs only
its closing tokens. The curve p(gold | first k thinking tokens) shows how much of the thinking comes
after the answer is already settled.

Drop this file at the root of ThinkingCap-Test (next to train.py) and run:

    python utils/train/prepare_data.py                          # writes data/val.jsonl
    python probe_curve.py --out runs/probe --per-source 50      # GSM8K, Skywork, CSQA, ARC
    python probe_curve.py --out runs/probe --summarize-only     # re-read the results

Phase 1 generates with vLLM (the eval's sampling settings) in a child process, so the GPU is
free again for phase 2, which scores with Transformers. Both write to --out, and phase 1 is
skipped when its rollouts already exist.
"""

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from typed import full_text, is_correct
from utils.eval.recipe import SAMPLING, SEED
from utils.train.rewards import clean_answer, final_answer, same_number

# fields a typed question carries for its grader (prepare_types.py); copied into each rollout
GRADE_FIELDS = ("grader", "entry_point", "tests", "setup", "category", "function", "reference",
                "instruction_id_list", "kwargs", "original_prompt")

BRIDGE = "\n</think>\n\nFinal answer:"
OPTION = re.compile(r"^([A-Z])\. ", re.MULTILINE)
ROLLOUTS = "rollouts.jsonl"
PROBES = "probes.jsonl"


def main():
    args = parse_arguments()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.generate_only:
        questions = load_questions(args)
        if not args.chunk:
            write_jsonl(out / ROLLOUTS, generate(questions, args))
            return
        # Save every chunk as it finishes, and skip the questions already saved: a disconnect costs
        # one chunk, not the whole run. Chunks mix the sources so long and short replies share a batch.
        partial = out / (ROLLOUTS + ".partial")
        done = {r["id"] for r in read_jsonl(partial)} if partial.exists() else set()
        todo = [q for q in questions if q["id"] not in done]
        random.Random(SEED).shuffle(todo)
        print(f"{len(done)} questions already sampled, {len(todo)} to go", flush=True)

        def save(rows):
            with partial.open("a") as handle:
                handle.write("".join(json.dumps(r) + "\n" for r in rows))

        if todo:
            generate(todo, args, sink=save)
        partial.rename(out / ROLLOUTS)
        return
    if not args.summarize_only:
        if not (out / ROLLOUTS).exists():
            # vLLM keeps most of the GPU until its process exits, so it samples in a child process.
            subprocess.run([sys.executable, *sys.argv, "--generate-only"], check=True)
        rollouts = read_jsonl(out / ROLLOUTS)
        write_jsonl(out / PROBES, probe_all(rollouts, args))
    summarize(read_jsonl(out / PROBES), args.threshold, out)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter", help="sample from this LoRA on top of --model (on-policy rollouts)")
    parser.add_argument("--data", default="data/val.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-source", type=int, default=50, help="questions per dataset")
    parser.add_argument("--sources", help="comma-separated datasets to keep (default: all)")
    parser.add_argument("--samples", type=int, default=4, help="rollouts per question")
    parser.add_argument("--max-tokens", type=int, default=4096, help="token cap, as the evaluation's (4,096)")
    parser.add_argument("--code-max-tokens", type=int, default=8192, help="token cap for code, as MBPP+'s (8,192)")
    parser.add_argument("--stride", type=int, default=16,
                        help="thinking tokens between probes; 1 probes every marginal token")
    parser.add_argument("--batch-tokens", type=int, default=32768,
                        help="padded tokens per scoring forward pass; lower it if memory runs out")
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="p(gold) that counts as 'settled' in the summary")
    parser.add_argument("--dtype", default="bfloat16", help="float16 on a T4")
    parser.add_argument("--attention-backend", default=None,
                        help="vLLM attention backend; TRITON_ATTN on a T4, where FlashInfer prefill fails")
    parser.add_argument("--chunk", type=int, default=0,
                        help="sample this many questions at a time and append each chunk to rollouts.jsonl.partial, "
                             "resuming from it after a disconnect (0: one batch, saved at the end)")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--generate-only", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def load_questions(args):
    keep = set(args.sources.split(",")) if args.sources else None
    by_source = defaultdict(list)
    for index, row in enumerate(read_jsonl(Path(args.data))):
        if keep is None or row["source"] in keep:
            by_source[row["source"]].append({**row, "id": f"{row['source']}-{index}"})
    rng = random.Random(SEED)
    picked = []
    for rows in by_source.values():
        rng.shuffle(rows)
        picked += rows[:args.per_source]
    return picked


# ---------------------------------------------------------------- phase 1: rollouts

def generate(questions, args, sink=None):
    """Sample `--samples` full replies per question with the evaluation's sampling settings."""
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    close_id = tokenizer.convert_tokens_to_ids("</think>")
    eos = tokenizer.eos_token_id
    eos_ids = eos if isinstance(eos, list) else [eos]
    prompts = [tokenizer.apply_chat_template(q["prompt"], tokenize=True, add_generation_prompt=True,
                                             enable_thinking=True) for q in questions]
    extra = {"attention_config": {"backend": args.attention_backend}} if args.attention_backend else {}
    engine = LLM(model=args.model, dtype=args.dtype, seed=SEED, gpu_memory_utilization=0.85,
                 max_model_len=max(args.max_tokens, args.code_max_tokens) + max(len(p) for p in prompts) + 16,
                 generation_config="vllm", enable_lora=bool(args.adapter), max_lora_rank=32, **extra)
    from vllm.lora.request import LoRARequest
    lora = LoRARequest("adapter", 1, args.adapter) if args.adapter else None
    # one cap per question, as the evaluation does: 8,192 for code, 4,096 for everything else
    params = [SamplingParams(n=args.samples, seed=SEED, stop_token_ids=eos_ids, **SAMPLING,
                             max_tokens=args.code_max_tokens if q["task"] == "code" else args.max_tokens)
              for q in questions]
    size = args.chunk or len(questions)
    rollouts = []
    for start in range(0, len(questions), size):
        part = slice(start, start + size)
        outputs = engine.generate([{"prompt_token_ids": p} for p in prompts[part]], params[part], lora_request=lora)
        rows = rollout_rows(questions[part], prompts[part], outputs, tokenizer, close_id, eos_ids)
        if sink:
            sink(rows)
            print(f"sampled {min(start + size, len(questions))}/{len(questions)} questions", flush=True)
        else:
            rollouts += rows
    return rollouts


def rollout_rows(questions, prompts, outputs, tokenizer, close_id, eos_ids):
    rollouts = []
    for question, prompt, output in zip(questions, prompts, outputs):
        for sample, result in enumerate(output.outputs):
            tokens = list(result.token_ids)
            closed = close_id in tokens
            thinking = tokens[:tokens.index(close_id)] if closed else tokens
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            truncated = not (tokens and tokens[-1] in eos_ids)
            rollouts.append({**{f: question[f] for f in GRADE_FIELDS if f in question and f != "id"},
                "id": question["id"], "source": question["source"], "task": question["task"],
                "answer": question["answer"], "question": question["prompt"][-1]["content"],
                "sample": sample, "prompt_ids": prompt, "thinking_ids": thinking, "output_ids": tokens,
                "output_tokens": len(tokens), "closed_thinking": closed,
                "truncated": not (tokens and tokens[-1] in eos_ids),
                "correct": is_correct(question, full_text(tokenizer, tokens), truncated),
                "final": extract(text)})
    return rollouts


def extract(text):
    answer = final_answer(text.split("</think>")[-1].replace("**", ""))
    return None if answer is None else clean_answer(answer)


# ---------------------------------------------------------------- phase 2: probes

def candidates(rollout, siblings):
    """The answers to put a probability on, gold first.

    Multiple choice: every option label. Math: the gold answer plus every final answer any
    rollout of the question reached, so the distribution covers what the model actually says.
    Each math candidate ends with the end-of-turn token: without it " 4" would also count
    the probability of " 42".
    """
    if rollout["task"] == "multiple_choice":
        labels = OPTION.findall(rollout["question"]) or list("ABCDE")
        gold = rollout["answer"].strip().upper()
        return [gold] + [label for label in labels if label != gold], False
    seen = [rollout["answer"]]
    for other in siblings:
        # Equal numbers written differently ("42", "42.0") would split one answer's mass.
        if other["final"] and not any(same_number(other["final"], s) for s in seen):
            seen.append(other["final"])
    return seen, True


def probe_all(rollouts, args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = getattr(torch, args.dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
    # Qwen3 closes its thinking as "...\n</think>\n\n": add the newline only where the cut lacks one.
    bridges = {ends: tokenizer.encode(BRIDGE[1:] if ends else BRIDGE, add_special_tokens=False)
               for ends in (False, True)}
    newline_ending = {i for i in range(len(tokenizer))
                      if tokenizer.convert_ids_to_tokens(i).endswith(("Ċ", "\n"))}
    end_of_turn = tokenizer.convert_tokens_to_ids("<|im_end|>")
    open_id = tokenizer.convert_tokens_to_ids("<think>")

    by_question = defaultdict(list)
    for rollout in rollouts:
        by_question[rollout["id"]].append(rollout)

    results = []
    for number, rollout in enumerate(rollouts, 1):
        answers, terminated = candidates(rollout, by_question[rollout["id"]])
        suffixes = [tokenizer.encode(" " + a, add_special_tokens=False) + ([end_of_turn] if terminated else [])
                    for a in answers]
        thinking = rollout["thinking_ids"]
        # The rollout opens with <think>: the first probe keeps it, giving Qwen3's empty "<think>\n</think>".
        first = 1 if thinking and thinking[0] == open_id else 0
        cuts = sorted(set(range(first, len(thinking) + 1, args.stride)) | {first, len(thinking)})
        closings = {cut: bridges[cut > 0 and thinking[cut - 1] in newline_ending] for cut in cuts}
        scores = probe_scores(model, rollout["prompt_ids"], thinking, closings, suffixes, device)

        probes = []
        for cut in cuts:
            logps = [scores[(cut, index)] for index in range(len(answers))]
            raw = [math.exp(lp) for lp in logps]
            mass = sum(raw)
            dist = [p / mass for p in raw] if mass > 0 else [1 / len(raw)] * len(raw)
            top = max(range(len(dist)), key=dist.__getitem__)
            probes.append({"k": cut, "p_gold": gold_probability(raw, rollout["task"]), "p_gold_raw": raw[0], "mass": mass,
                           "top": answers[top], "p_top": dist[top],
                           "entropy": -sum(p * math.log(p) for p in dist if p > 0)})
        results.append({key: rollout[key] for key in
                        ("id", "source", "task", "answer", "sample", "output_tokens",
                         "closed_thinking", "truncated", "correct", "final")}
                       | {"thinking_tokens": len(thinking), "candidates": answers, "probes": probes})
        if number % 10 == 0 or number == len(rollouts):
            print(f"probed {number}/{len(rollouts)}", flush=True)
    return results


def gold_probability(raw, task):
    """p(gold) from the candidates' raw probabilities, gold first.

    Multiple choice: gold's share of the probability on the options, which cover every
    possible answer. Math: gold's raw probability. Its candidates are only the answers the
    rollouts happened to reach, so when they all agree gold is the sole candidate and its
    share would read 100% before any thinking.
    """
    if task == "multiple_choice":
        mass = sum(raw)
        return raw[0] / mass if mass > 0 else 0.0
    return raw[0]


def probe_scores(model, prompt, thinking, closings, suffixes, device, budget=6144):
    """{(cut, candidate): log p(suffix | prompt + thinking[:cut] + closing)} for every cut of one rollout.

    Every cut is a prefix of the same rollout, so the rollout is run once and each cut's
    closing + candidate is appended after it as a segment that attends only to the first
    len(prompt) + cut positions and to itself, at the positions it would have there. One forward
    pass scores up to `budget` segment tokens, and the result equals scoring each cut from scratch.
    """
    import torch

    body, head = decoder_and_head(model)
    dtype = head.weight.dtype
    blocked = torch.finfo(dtype).min
    context = prompt + thinking
    segments = [(cut, row, closing + suffix, len(closing))
                for cut, closing in closings.items() for row, suffix in enumerate(suffixes)]

    def causal(size):
        return torch.full((size, size), blocked, dtype=dtype, device=device).triu(1)

    scores, start = {}, 0
    while start < len(segments):
        group, size = [], 0
        while start < len(segments) and (not group or size + len(segments[start][2]) <= budget):
            group.append(segments[start])
            size += len(segments[start][2])
            start += 1
        total = len(context) + size
        ids, positions = list(context), list(range(len(context)))
        mask = torch.full((total, total), blocked, dtype=dtype, device=device)
        mask[:len(context), :len(context)] = causal(len(context))
        spans, offset = [], len(context)
        for cut, row, tokens, closing in group:
            length, width = len(prompt) + cut, len(tokens)
            ids += tokens
            positions += range(length, length + width)
            mask[offset:offset + width, :length] = 0
            mask[offset:offset + width, offset:offset + width] = causal(width)
            spans.append((cut, row, offset + closing, tokens[closing:]))
            offset += width
        with torch.no_grad():
            hidden = body(input_ids=torch.tensor([ids], device=device), attention_mask=mask[None, None],
                          position_ids=torch.tensor([positions], device=device)).last_hidden_state[0]
            for cut, row, first, suffix in spans:
                # Position t predicts token t + 1, so the suffix is read one step early.
                read = torch.arange(first - 1, first - 1 + len(suffix), device=device)
                logp = torch.log_softmax(head(hidden[read]).float(), dim=-1)
                target = torch.tensor(suffix, device=device)
                scores[(cut, row)] = logp.gather(1, target[:, None]).sum().item()
    return scores


def decoder_and_head(model):
    """The transformer body and the output projection, through a PEFT wrapper if there is one."""
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    return inner.base_model, inner.get_output_embeddings()


def score(model, rows, batch_tokens, device):
    """{(cut, candidate): log p(suffix | context)} for rows of (cut, candidate, context, suffix)."""
    import torch

    body, head = decoder_and_head(model)

    rows = sorted(rows, key=lambda row: len(row[2]) + len(row[3]))
    scores, start = {}, 0
    while start < len(rows):
        end = start + 1  # grow the batch while the padded size fits the budget
        while end < len(rows) and (end + 1 - start) * (len(rows[end][2]) + len(rows[end][3])) <= batch_tokens:
            end += 1
        batch = rows[start:end]
        width = max(len(context) + len(suffix) for _, _, context, suffix in batch)
        ids = torch.zeros(len(batch), width, dtype=torch.long)
        mask = torch.zeros(len(batch), width, dtype=torch.long)
        for row, (_, _, context, suffix) in enumerate(batch):
            sequence = context + suffix
            ids[row, :len(sequence)] = torch.tensor(sequence)
            mask[row, :len(sequence)] = 1
        with torch.no_grad():
            # Logits only where the suffix is read: the full vocabulary at every position would
            # take batch_tokens x 152k values.
            hidden = body(input_ids=ids.to(device), attention_mask=mask.to(device)).last_hidden_state
            for row, (cut, candidate, context, suffix) in enumerate(batch):
                # Position t predicts token t + 1, so the suffix is read one step early.
                positions = torch.arange(len(context) - 1, len(context) + len(suffix) - 1, device=device)
                logp = torch.log_softmax(head(hidden[row, positions]).float(), dim=-1)
                target = torch.tensor(suffix, device=device)
                scores[(cut, candidate)] = logp.gather(1, target[:, None]).sum().item()
        start = end
    return scores


# ---------------------------------------------------------------- summary

def settle_point(probes, threshold):
    """First cut from which p(gold) stays at or above the threshold, or None if it never does."""
    point = None
    for probe in probes:
        if probe["p_gold"] >= threshold:
            point = probe["k"] if point is None else point
        else:
            point = None
    return point


def summarize(results, threshold, out):
    lines = [f"p(gold) threshold for 'settled': {threshold}", ""]
    header = (f"{'source':<16}{'rollouts':>9}{'acc':>7}{'capped':>8}{'think':>8}"
              f"{'settled':>9}{'after':>8}{'p@0':>7}{'p@25%':>7}{'p@50%':>7}{'p@75%':>7}{'p@end':>7}")
    lines.append(header)
    by_source = defaultdict(list)
    for result in results:
        by_source[result["source"]].append(result)
    for source, group in sorted(by_source.items()) + [("ALL", results)]:
        correct = [r for r in group if r["correct"]]
        settled = [(settle_point(r["probes"], threshold), r["thinking_tokens"]) for r in correct]
        settled = [(point, total) for point, total in settled if point is not None]
        after = sum(total - point for point, total in settled)
        after_share = after / max(1, sum(r["thinking_tokens"] for r in correct))
        quantiles = [mean(at_fraction(r["probes"], f) for r in group) for f in (0, .25, .5, .75, 1)]
        lines.append(f"{source:<16}{len(group):>9}{mean(r['correct'] for r in group):>7.0%}"
                     f"{mean(r['truncated'] for r in group):>8.0%}"
                     f"{mean(r['thinking_tokens'] for r in group):>8.0f}"
                     f"{len(settled) / max(1, len(correct)):>9.0%}{after_share:>8.0%}"
                     + "".join(f"{q:>7.2f}" for q in quantiles))
    lines += ["",
              "acc      share of rollouts whose natural answer is correct",
              "capped   share that hit the token cap before finishing",
              "think    mean thinking tokens",
              "settled  share of correct rollouts whose p(gold) reaches the threshold and stays there",
              "after    share of correct rollouts' thinking tokens spent after that point: the room to cut",
              "p@x      mean p(gold) after x of each rollout's thinking (share of the options for",
              "         multiple choice, raw probability for math)"]
    text = "\n".join(lines)
    print(text)
    (out / "summary.txt").write_text(text + "\n")
    plot(by_source, out)


def at_fraction(probes, fraction):
    """p(gold) at the last probe at or before `fraction` of the thinking."""
    first, last = probes[0]["k"], probes[-1]["k"]
    return at_tokens(probes, first + fraction * (last - first))


def plot(by_source, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping the plot")
        return
    figure, (relative, absolute) = plt.subplots(1, 2, figsize=(12, 4.5))
    grid = [i / 20 for i in range(21)]
    for source, group in sorted(by_source.items()):
        correct = [r for r in group if r["correct"]] or group
        relative.plot(grid, [mean(at_fraction(r["probes"], f) for r in correct) for f in grid], label=source)
        longest = max(r["probes"][-1]["k"] for r in correct)
        steps = list(range(0, longest + 1, max(1, longest // 50)))
        absolute.plot(steps, [mean(at_tokens(r["probes"], k) for r in correct) for k in steps], label=source)
    relative.set(xlabel="share of the rollout's thinking", ylabel="mean p(gold)",
                 title="Correct rollouts: p(gold) vs. share of thinking", ylim=(0, 1))
    absolute.set(xlabel="thinking tokens", ylabel="mean p(gold)",
                 title="Correct rollouts: p(gold) vs. thinking tokens", ylim=(0, 1))
    if max(r["probes"][-1]["k"] for group in by_source.values() for r in group) > 1024:
        absolute.set_xscale("symlog", linthresh=256)  # long traces: keep the early tokens readable
    for axis in (relative, absolute):
        axis.grid(alpha=.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(out / "curve.png", dpi=130)
    print(f"wrote {out / 'curve.png'}")


def at_tokens(probes, k):
    """p(gold) after k thinking tokens; a rollout that already stopped keeps its final value."""
    return ([p for p in probes if p["k"] <= k] or probes[:1])[-1]["p_gold"]


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


if __name__ == "__main__":
    main()
