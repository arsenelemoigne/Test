"""The judge as teacher: train the model's own "</think>" signal at every paragraph end to follow the stopping judge.

1. Judge: an MLP on one layer's hidden state at the end of a closed cut (place "after </think>", as probe_zoo's best),
   fitted on labelled cuts of the SAME model (label_cuts.py lots: safe = every forced answer right, unsafe <= --unsafe
   of 20; in between left out). Its AUC on 20% of the questions (held out by hash) is printed, then it is refitted on all.
2. Teacher labels: the model's own answers to fresh questions (--rollouts, one sampled answer each). At each paragraph
   end of the thinking (a token ending in a blank line; at most --max-cuts per answer, evenly spaced; never the natural
   end) the cut is closed as in label_cuts ("<think>\\n...\\n</think>\\n\\n") and scored by the judge. Target =
   sigmoid((p - threshold) / temperature): near 0 unless the judge is clearly confident, so small per-paragraph
   chances do not add up to an early stop somewhere.
3. Training: a fresh LoRA on --model; one forward pass per answer; at every scored paragraph end the loss is the binary
   cross-entropy between p("</think>" next) and its target. Nothing else is trained. One adapter per --thresholds value
   (same judge scores).

    python teach_stop.py --model Qwen/Qwen3-0.6B --cal runs/bon/pairs-plotB/cal runs/bon/pairs-plotB2/cal \\
        --rollouts runs/bon/probe-teach1/rollouts.jsonl --out adapters/teach1 --thresholds 0.8 0.9
"""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np


def read(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def sharpen(p, threshold, temperature):
    """Teacher target: ~0 below the threshold, ~1 above it."""
    return 1 / (1 + math.exp(-(p - threshold) / temperature))


def paragraph_ends(tokens, close_id, open_id, decode, max_cuts):
    """Positions i (inside the thinking, before </think>) whose token ends a paragraph; evenly thinned to max_cuts."""
    stop = tokens.index(close_id) if close_id in tokens else len(tokens)
    ends = [i for i in range(stop - 1) if tokens[i] != open_id and decode([tokens[i]]).endswith("\n\n")]
    if len(ends) > max_cuts:
        ends = sorted({ends[round(j * (len(ends) - 1) / (max_cuts - 1))] for j in range(max_cuts)})
    return ends


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--cal", nargs="+", required=True, help="label_cuts.py lots of --model (the judge's data)")
    parser.add_argument("--feats-root", default="/content/feats", help="reuse extract_feats.py points when present")
    parser.add_argument("--rollouts", required=True, help="the model's own answers to fresh questions")
    parser.add_argument("--out", required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--unsafe", type=int, default=15)
    parser.add_argument("--max-cuts", type=int, default=16)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.8, 0.9])
    parser.add_argument("--temperature", type=float, default=0.03)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--accumulate", type=int, default=8)
    parser.add_argument("--batch-tokens", type=int, default=12000)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from probe_knows import auc, fit_mlp

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    open_id, close_id = tok.convert_tokens_to_ids("<think>"), tok.convert_tokens_to_ids("</think>")
    dtype = getattr(torch, args.dtype)
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to("cuda").eval()

    def last_states(seqs):
        """Hidden state of --layer at each sequence's last token, packed by length."""
        feats = np.zeros((len(seqs), base.config.hidden_size), dtype=np.float32)
        order = sorted(range(len(seqs)), key=lambda k: len(seqs[k]))
        batch = []

        def flush():
            width = max(len(seqs[k]) for k in batch)
            ids = torch.zeros(len(batch), width, dtype=torch.long)
            mask = torch.zeros(len(batch), width, dtype=torch.long)
            for i, k in enumerate(batch):
                ids[i, :len(seqs[k])] = torch.tensor(seqs[k])
                mask[i, :len(seqs[k])] = 1
            with torch.no_grad():
                states = base.model(input_ids=ids.cuda(), attention_mask=mask.cuda(), output_hidden_states=True).hidden_states
            for i, k in enumerate(batch):
                feats[k] = states[args.layer][i, len(seqs[k]) - 1].float().cpu().numpy()
        for k in order:
            if batch and (len(batch) + 1) * len(seqs[k]) > args.batch_tokens:
                flush()
                batch = []
            batch.append(k)
        if batch:
            flush()
        return feats

    # 1. the judge
    xs, ys, qs = [], [], []
    for cal in map(Path, args.cal):
        cuts, replies = read(cal / "cuts.jsonl"), {r["reply"]: r for r in read(cal / "replies.jsonl")}
        verdict = {v["key"]: v for v in read(cal / "verdicts.jsonl")}
        name = cal.name if cal.name != "cal" else cal.parent.name
        cached = Path(args.feats_root) / name / "points.npy"
        if cached.exists():
            feats = np.asarray(np.load(cached, mmap_mode="r")[:, args.layer - 1, 1], dtype=np.float32)
        else:
            feats = last_states([c["prompt_ids"] + c["thinking_ids"] for c in cuts])
        for c, f in zip(cuts, feats):
            v = verdict[c["key"]]
            if v["right"] == v["of"] or v["right"] <= args.unsafe:
                xs.append(f)
                ys.append(float(v["right"] == v["of"]))
                qs.append(f"{name}/{replies[c['reply']]['id']}")
    x, y = np.stack(xs), np.array(ys)
    held = np.array([int(hashlib.md5(q.encode()).hexdigest(), 16) % 10 < 2 for q in qs])
    check = fit_mlp(x[~held], y[~held], seed=args.seed)
    held_auc = auc(check(x[held]), y[held])
    judge = fit_mlp(x, y, seed=args.seed)
    print(f"judge: {len(y)} labelled cuts ({int(y.sum())} safe), layer {args.layer}; held-out AUC {held_auc:.3f}", flush=True)

    # 2. teacher labels on the model's own answers
    rows = [r for r in read(args.rollouts) if r["closed_thinking"] and not r["truncated"]]
    decode = lambda ids: tok.decode(ids, skip_special_tokens=False)
    items, seqs = [], []
    for r in rows:
        tokens = r["output_ids"]
        for i in paragraph_ends(tokens, close_id, open_id, decode, args.max_cuts):
            text = tok.decode([t for t in tokens[:i + 1] if t not in (open_id, close_id)]).strip()
            seqs.append(r["prompt_ids"] + tok.encode("<think>\n" + text + "\n</think>\n\n", add_special_tokens=False))
            items.append((id(r), i))
    scores = judge(last_states(seqs))
    by_answer = {}
    for (rid, i), p in zip(items, scores):
        by_answer.setdefault(rid, []).append((i, float(p)))
    (out / "teacher.json").write_text(json.dumps({"held_out_auc": held_auc, "answers": len(by_answer),
                                                  "cuts": len(items), "mean_p": float(np.mean(scores)),
                                                  "share_above_0.9": float(np.mean(scores > 0.9))}))
    print(f"teacher: {len(by_answer)} answers, {len(items)} paragraph ends, mean p {np.mean(scores):.2f}, "
          f"{np.mean(scores > 0.8):.0%} above 0.8, {np.mean(scores > 0.9):.0%} above 0.9", flush=True)
    del base
    torch.cuda.empty_cache()

    # 3. one adapter per threshold
    from peft import LoraConfig, get_peft_model
    answers = [(r, by_answer[id(r)]) for r in rows if id(r) in by_answer]
    for threshold in args.thresholds:
        target_dir = out / f"t{threshold:g}"
        if (target_dir / "adapter_config.json").exists():
            continue
        torch.manual_seed(args.seed)
        model = get_peft_model(AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to("cuda"), LoraConfig(
            r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        order = list(range(len(answers)))
        random.Random(args.seed).shuffle(order)
        model.train()
        total, n, before, after = 0.0, 0, [], []
        for step, k in enumerate(order):
            r, cuts = answers[k]
            stop = r["output_ids"].index(close_id)
            ids = torch.tensor([r["prompt_ids"] + r["output_ids"][:stop]], device="cuda")
            hidden = model.base_model.model.model(input_ids=ids).last_hidden_state[0]
            positions = torch.tensor([len(r["prompt_ids"]) + i for i, _ in cuts], device="cuda")
            logp = torch.log_softmax(model.base_model.model.lm_head(hidden[positions]).float(), -1)[:, close_id]
            log1m = torch.log1p(-torch.exp(logp).clamp(max=1 - 1e-6))
            target = torch.tensor([sharpen(p, threshold, args.temperature) for _, p in cuts], device="cuda")
            loss = -(target * logp + (1 - target) * log1m).mean()
            (loss / args.accumulate).backward()
            if (step + 1) % args.accumulate == 0 or step == len(order) - 1:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total, n = total + float(loss), n + 1
            if step < 50:
                before.append(float(torch.exp(logp).mean()))
            if step >= len(order) - 50:
                after.append(float(torch.exp(logp).mean()))
            if (step + 1) % 500 == 0:
                print(f"t{threshold:g}: {step + 1}/{len(order)} answers, loss {total / n:.4f}", flush=True)
                total, n = 0.0, 0
        model.save_pretrained(target_dir)
        print(f"t{threshold:g}: saved {target_dir}; mean p(</think>) at paragraph ends {np.mean(before):.3f} (first 50) "
              f"-> {np.mean(after):.3f} (last 50)", flush=True)
        del model, optimizer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
