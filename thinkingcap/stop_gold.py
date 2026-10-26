"""Stop-token training with the lots' own labels as teacher: at every labelled paragraph end, p("</think>") toward safe/unsafe.

The label lots (label_cuts.py) cut each reply after its paragraph ends, closed "</think>" there and forced answers from
each cut; a cut's verdict is its share of right answers. This script trains exactly the decision those labels describe
and nothing else: one forward pass over the reply as the model wrote it (prompt + thinking), and at each labelled
paragraph end (the token ending the k-th paragraph, "\\n\\n") the binary cross-entropy between p("</think>" next) and

    target 1 (stop)  if the cut's re-answers are right at least as often as the reply's own full thinking (its natural
                     stop) and at least --min-rate: answering here is as good as finishing (the user's relative rule);
    target 0 (keep)  otherwise: stopping here loses answers.

No reply text is imitated (the DPO stop runs anchored the chosen reply with an NLL term, which imitated reply lengths),
no whole-reply preference is used, and safe and unsafe points both get a target, so the model learns WHERE stopping is
safe rather than one global "stop earlier" habit. Replies whose every cut and full thinking are 0 (never solved: "stop
circling" is a separate, later test) are left out. Per-type caps (--quota) keep one type from dominating; every
--save-every optimizer steps a checkpoint is saved so several training doses can be scored (the DPO stop runs all fell
off a cliff between 400 and 480 pairs seen).

    python stop_gold.py --lots runs/bon/pairs-plotB runs/bon/pairs-plotV2 --probes runs/bon/probe-plotB \\
        runs/bon/kprobe-plotB runs/bon/probe-plotV2 runs/bon/kprobe-plotV2 --out adapters/gold1
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def read(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def label(rate, full_rate, min_rate):
    """1 = stop here is as good as finishing, 0 = keep thinking."""
    return 1.0 if rate >= max(full_rate, min_rate) else 0.0


def build(lots, probes, decode, open_id, close_id, sources, min_rate):
    """[(source, prompt_ids, thinking token ids up to </think>, [(token position, target)])] from the lots; pure but for I/O."""
    rollouts = defaultdict(list)
    for p in probes:
        if Path(p, "rollouts.jsonl").exists():
            for r in read(Path(p, "rollouts.jsonl")):
                if r["closed_thinking"] and not r["truncated"]:
                    rollouts[r["id"]].append(r)
    items, misses = [], Counter()
    for lot in map(Path, lots):
        cal = lot / "cal"
        if not (cal / "verdicts.jsonl").exists():
            continue
        verdict = {v["key"]: v for v in read(cal / "verdicts.jsonl")}
        cuts = defaultdict(list)
        for c in read(cal / "cuts.jsonl"):
            cuts[c["reply"]].append(c)
        for rep in read(cal / "replies.jsonl"):
            if rep["source"] not in sources:
                continue
            full = next((r for r in rollouts[rep["id"]] if bool(r["correct"]) == rep["correct"]
                         and r["output_tokens"] == rep["output_tokens"]), None)
            cs = sorted(cuts[rep["reply"]], key=lambda c: c["para"])
            if full is None or len(cs) < 2:
                misses["no rollout"] += 1
                continue
            rate = lambda c: verdict[c["key"]]["right"] / max(1, verdict[c["key"]]["of"])
            full_rate = rate(cs[-1])  # the last cut is the full thinking
            if full_rate == 0 and all(rate(c) == 0 for c in cs):
                misses["never solved"] += 1
                continue
            tokens = full["output_ids"]
            stop = tokens.index(close_id)
            text_at = {}  # stripped thinking text up to each paragraph-end token -> that token's position
            for i in range(stop):
                if tokens[i] != open_id and decode([tokens[i]]).endswith("\n\n"):
                    text_at[decode([t for t in tokens[:i + 1] if t not in (open_id, close_id)]).strip()] = i
            points = []
            for c in cs[:-1]:
                key = decode([t for t in c["thinking_ids"] if t not in (open_id, close_id)]).strip()
                if key in text_at:
                    points.append((text_at[key], label(rate(c), full_rate, min_rate)))
                else:
                    misses["cut not aligned"] += 1
            if points:
                items.append((rep["source"], full["prompt_ids"], tokens[:stop], points))
    return items, misses


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--lots", nargs="+", required=True, help="runs/bon/pairs-LOT folders (cal/ inside)")
    parser.add_argument("--probes", nargs="+", required=True, help="runs/bon/probe-LOT / kprobe-LOT folders (rollouts)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--sources", default="gsm8k_matched,metamath_sv,kodcode,ifeval_like,toolace,hermes_tools",
                        help="types trained (GSM-IC/Plus left out: they settle by 20-40%%, unlike SVAMP; no MBPP)")
    parser.add_argument("--quota", default="kodcode=1800,ifeval_like=1800,toolace=1200,hermes_tools=500,"
                                           "gsm8k_matched=1300,metamath_sv=1300", help="most replies per type")
    parser.add_argument("--min-rate", type=float, default=0.25, help="never a stop target below this success rate")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--accumulate", type=int, default=8, help="replies per optimizer step")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=100, help="optimizer steps between checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="build and summarize the labels only (no GPU)")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    open_id, close_id = tok.convert_tokens_to_ids("<think>"), tok.convert_tokens_to_ids("</think>")
    decode = lambda ids: tok.decode(ids, skip_special_tokens=False)
    items, misses = build(args.lots, args.probes, decode, open_id, close_id, set(args.sources.split(",")), args.min_rate)
    rng = random.Random(args.seed)
    quota = dict((k, int(v)) for k, v in (x.split("=") for x in args.quota.split(","))) if args.quota else {}
    by_src = defaultdict(list)
    for it in items:
        by_src[it[0]].append(it)
    items = []
    for src, group in sorted(by_src.items()):
        rng.shuffle(group)
        items += group[:quota.get(src, len(group))]
    rng.shuffle(items)
    lines = [f"{len(items)} replies, {sum(len(p) for *_, p in items)} labelled paragraph ends; skipped: {dict(misses)}"]
    for src in sorted({s for s, *_ in items}):
        pts = [t for s, _, _, p in items if s == src for _, t in p]
        first = [next((i for i, (_, t) in enumerate(p) if t == 1), None) for s, _, _, p in items if s == src]
        lines.append(f"  {src}: {sum(s == src for s, *_ in items)} replies, {len(pts)} points, stop share "
                     f"{sum(pts) / max(1, len(pts)):.2f}, replies with a safe cut {sum(f is not None for f in first)}")
    (out / "labels.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    if args.dry_run:
        return

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    torch.manual_seed(args.seed)
    model = get_peft_model(AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype)).to("cuda"),
                           LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, task_type="CAUSAL_LM",
                                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                                                      "down_proj"]))
    model.gradient_checkpointing_enable()  # long replies ran teach1b out of memory on a 22 GB L4
    model.enable_input_require_grads()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    order = [i for _ in range(args.epochs) for i in rng.sample(range(len(items)), len(items))]
    steps = -(-len(order) // args.accumulate)
    print(f"{steps} optimizer steps of {args.accumulate} replies; checkpoint every {args.save_every}", flush=True)
    model.train()
    step, window = 0, defaultdict(lambda: {"stop": [], "keep": [], "loss": []})
    for position, k in enumerate(order, 1):
        src, prompt, thinking, points = items[k]
        last = max(i for i, _ in points) + 1  # nothing after the last labelled paragraph end is needed (causal)
        ids = torch.tensor([prompt + thinking[:last]], device="cuda")
        hidden = model.base_model.model.model(input_ids=ids).last_hidden_state[0]
        at = torch.tensor([len(prompt) + i for i, _ in points], device="cuda")
        logp = torch.log_softmax(model.base_model.model.lm_head(hidden[at]).float(), -1)[:, close_id]
        log1m = torch.log1p(-torch.exp(logp).clamp(max=1 - 1e-6))
        target = torch.tensor([t for _, t in points], device="cuda")
        loss = -(target * logp + (1 - target) * log1m).mean()
        (loss / args.accumulate).backward()
        with torch.no_grad():
            p = torch.exp(logp).tolist()
            w = window[src]
            w["stop"] += [pi for pi, (_, t) in zip(p, points) if t == 1]
            w["keep"] += [pi for pi, (_, t) in zip(p, points) if t == 0]
            w["loss"].append(float(loss))
        if position % args.accumulate == 0 or position == len(order):
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 25 == 0 or step == steps:
                m = lambda v: f"{sum(v) / len(v):.3f}" if v else "-"
                print(f"step {step}/{steps} | " + "; ".join(
                    f"{s}: loss {m(w['loss'])}, p(stop) where safe {m(w['stop'])}, where unsafe {m(w['keep'])}"
                    for s, w in sorted(window.items())), flush=True)
                window.clear()
            if step % args.save_every == 0 or step == steps:
                model.save_pretrained(out / f"step-{step}")
    model.save_pretrained(out)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
