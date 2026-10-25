"""The judge as teacher: train the model's own "</think>" signal at every paragraph end to follow the stopping judge.

1. Judges: MLPs on one layer's hidden state at the end of a closed cut (place "after </think>", as probe_zoo's best),
   fitted on labelled cuts of the SAME model (label_cuts.py lots). The yes/no judge learns safe (every forced answer
   right) vs unsafe (<= --unsafe of 20; in between left out). The soft judge learns every cut's real success rate k/20
   with cross-entropy, so its output is a calibrated "right this often if it stops here". On 20% of the questions (held
   out by hash) both are graded by AUC, log loss, Brier and calibration error, then refitted on all.
2. Teacher labels: the model's own answers to fresh questions (--rollouts, one sampled answer each). At each paragraph
   end of the thinking (a token ending in a blank line; at most --max-cuts per answer, evenly spaced; never the natural
   end) the cut is closed as in label_cuts ("<think>\\n...\\n</think>\\n\\n") and scored, and so is the full thinking.
   The model's own confidence (own_conf.py: the geometric mean of its 5 least likely tokens when it answers greedily from
   that point) is read at the same points. Teacher per type (zbench2, timing measured within the same reply on the 240
   benchmark questions): own confidence for math and code (--own-sources: SVAMP 0.98, MBPP+ 0.83), the soft judge for
   tool calls (--judge-sources: BFCL 0.81), none for instructions and multiple choice (--protect-sources: target 0,
   "keep thinking", since stop training spills over across types). Targets, one model per --variants entry, always
   sharpened as sigmoid((p - bar) / temperature) so small per-paragraph chances do not add up to an early stop:
     rel:M    the teacher's value here against max(its value at the full thinking of the SAME answer - M, --min-p): stop
              where stopping is as good as finishing (the user's rule). Comparing within one answer uses the signal's
              timing, which transfers, and needs no absolute threshold, which did not (a bar tuned on practice math
              over-stopped SVAMP). --min-p keeps expected failures out ("stop circling" is a separate, later test).
     fixed:T  the yes/no judge above the fixed bar T, on tool calls only (kept for comparison).
3. Training: a fresh LoRA on --model; one forward pass per answer; at every scored paragraph end the loss is the binary
   cross-entropy between p("</think>" next) and its target. Nothing else is trained.

    python teach_stop.py --model Qwen/Qwen3-0.6B --cal runs/bon/pairs-plotB/cal runs/bon/pairs-plotB2/cal \\
        --rollouts runs/bon/probe-teach1/rollouts.jsonl --out adapters/teach1 --variants rel:0.05 rel:0.1
"""

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
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
    parser.add_argument("--drop-sources", default="mbpp_train", help="never learn from these (MBPP+ is built on MBPP)")
    parser.add_argument("--max-cuts", type=int, default=16)
    parser.add_argument("--variants", nargs="+", default=["rel:0.05", "rel:0.1"], help="fixed:T or rel:M, one model each")
    parser.add_argument("--min-p", type=float, default=0.3, help="rel: never a stop target below this predicted success")
    parser.add_argument("--own-sources", default="gsm_ic,gsm_plus,gsm8k,gsm8k_all,kodcode",
                        help="teacher = the model's own confidence (5 least likely answer tokens): on the 240 benchmark "
                             "questions it reads WHEN an answer is settled best for math (SVAMP 0.98) and code (0.83; zbench2)")
    parser.add_argument("--judge-sources", default="toolace,hermes_tools",
                        help="teacher = the hidden-state judge: it leads only on tool calls (BFCL 0.81; zbench2)")
    parser.add_argument("--protect-sources", default="ifeval_like,arc_easy,commonsense_qa",
                        help="trained toward 'keep thinking' (target 0) at every paragraph end: no signal is trusted there "
                             "(IFEval had 6 usable benchmark replies), and stop training spills over across types")
    parser.add_argument("--answer-tokens", type=int, default=2048, help="own confidence: cap of the forced answer at each cut")
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
    from probe_knows import auc, calibration, fit_mlp

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    open_id, close_id = tok.convert_tokens_to_ids("<think>"), tok.convert_tokens_to_ids("</think>")
    dtype = getattr(torch, args.dtype)
    dropped = set(args.drop_sources.split(",")) if args.drop_sources else set()

    # the model's own answers: every scored paragraph end, plus the full thinking (the natural stop)
    rows = [r for r in read(args.rollouts) if r["closed_thinking"] and not r["truncated"] and r["source"] not in dropped]
    decode = lambda ids: tok.decode(ids, skip_special_tokens=False)
    closed = lambda ids: tok.encode("<think>\n" + tok.decode([t for t in ids if t not in (open_id, close_id)]).strip()
                                    + "\n</think>\n\n", add_special_tokens=False)
    items, seqs = [], []
    for r in rows:
        tokens = r["output_ids"]
        ends = paragraph_ends(tokens, close_id, open_id, decode, args.max_cuts)
        if not ends:
            continue
        for i in ends:
            seqs.append(r["prompt_ids"] + closed(tokens[:i + 1]))
            items.append((id(r), i))
        seqs.append(r["prompt_ids"] + closed(tokens[:tokens.index(close_id)]))
        items.append((id(r), None))
    # own confidence at every one of those points (own_conf.py: one greedy forced answer each, vLLM, then the GPU is freed)
    own_dir = out / "own"
    own_dir.mkdir(exist_ok=True)
    if not (own_dir / "own_confidence.jsonl").exists():
        by_id = {id(r): r for r in rows}
        (own_dir / "cuts.jsonl").write_text("".join(json.dumps({"key": k, "prompt_ids": by_id[rid]["prompt_ids"],
                                                                 "thinking_ids": seq[len(by_id[rid]["prompt_ids"]):],
                                                                 "max_tokens": args.answer_tokens}) + "\n"
                                                     for k, ((rid, _), seq) in enumerate(zip(items, seqs))))
        subprocess.run([sys.executable, "own_conf.py", "--model", args.model, "--cal", str(own_dir), "--dtype", args.dtype],
                       check=True)
    own = {o["key"]: o["own_low5"] for o in read(own_dir / "own_confidence.jsonl")}
    own_p = np.array([own[k] for k in range(len(items))])
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

    # 1. the judges
    xs, success, qs = [], [], []
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
            if replies[c["reply"]]["source"] in dropped:
                continue
            v = verdict[c["key"]]
            xs.append(f)
            success.append(v["right"] / max(1, v["of"]))
            qs.append(f"{name}/{replies[c['reply']]['id']}")
    x, success = np.stack(xs), np.array(success)
    labelled = (success == 1) | (success <= args.unsafe / 20)
    label = (success == 1).astype(float)
    held = np.array([int(hashlib.md5(q.encode()).hexdigest(), 16) % 10 < 2 for q in qs])
    grades = {}
    for kind, rows, target in (("yes/no", labelled, label), ("soft", np.ones(len(success), bool), success)):
        check = fit_mlp(x[rows & ~held], target[rows & ~held], seed=args.seed)(x[held])
        ll, brier, ece = calibration(check, success[held])
        grades[kind] = {"auc": auc(check[labelled[held]], label[held][labelled[held]]), "log_loss": ll,
                        "brier": brier, "ece": ece}
        print(f"judge {kind}: held-out AUC {grades[kind]['auc']:.3f}, log loss {ll:.3f}, Brier {brier:.3f}, "
              f"calibration error {ece:.3f}", flush=True)
    hard = fit_mlp(x[labelled], label[labelled], seed=args.seed)
    soft = fit_mlp(x, success, seed=args.seed)
    print(f"judges: {len(success)} cuts ({int(labelled.sum())} yes/no labelled), layer {args.layer}", flush=True)

    # 2. teacher signals at every point: the judges (hidden state) and the model's own confidence
    feats = last_states(seqs)
    hard_p, soft_p = hard(feats), soft(feats)
    by_answer, at_end = {}, {}
    for (rid, i), h, sp, op in zip(items, hard_p, soft_p, own_p):
        if i is None:
            at_end[rid] = (float(sp), float(op))
        else:
            by_answer.setdefault(rid, []).append((i, float(h), float(sp), float(op)))
    cut = np.array([i is not None for _, i in items])
    report = {"judges": grades, "answers": len(by_answer), "cuts": int(cut.sum()),
              "yes_no_mean_p": float(hard_p[cut].mean()), "soft_mean_success_at_cuts": float(soft_p[cut].mean()),
              "soft_mean_success_at_end": float(soft_p[~cut].mean()),
              "own_low5_mean_at_cuts": float(own_p[cut].mean()), "own_low5_mean_at_end": float(own_p[~cut].mean())}
    (out / "teacher.json").write_text(json.dumps(report, indent=1))
    print(f"teacher: {len(by_answer)} answers, {int(cut.sum())} paragraph ends; yes/no p > 0.9 at "
          f"{np.mean(hard_p[cut] > 0.9):.0%}; predicted success {soft_p[cut].mean():.2f} at cuts vs "
          f"{soft_p[~cut].mean():.2f} at the full thinking", flush=True)
    del base
    torch.cuda.empty_cache()

    # 3. one adapter per variant
    from peft import LoraConfig, get_peft_model
    answers = [(r, by_answer[id(r)], at_end[id(r)]) for r in rows if id(r) in by_answer]
    split = lambda v: set(v.split(",")) if v else set()
    protected, own_src, judge_src = split(args.protect_sources), split(args.own_sources), split(args.judge_sources)
    teacher = lambda src: "own" if src in own_src else "judge" if src in judge_src else "protect"
    counts = {t: sum(teacher(r["source"]) == t for r, _, _ in answers) for t in ("own", "judge", "protect")}
    print(f"teachers: own confidence {counts['own']}, judge {counts['judge']}, keep thinking {counts['protect']} answers", flush=True)
    for variant in args.variants:
        kind, value = variant.split(":")
        value = float(value)
        target_dir = out / f"{kind}{value:g}"
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
            r, cuts, end_p = answers[k]
            stop = r["output_ids"].index(close_id)
            ids = torch.tensor([r["prompt_ids"] + r["output_ids"][:stop]], device="cuda")
            hidden = model.base_model.model.model(input_ids=ids).last_hidden_state[0]
            positions = torch.tensor([len(r["prompt_ids"]) + c[0] for c in cuts], device="cuda")
            logp = torch.log_softmax(model.base_model.model.lm_head(hidden[positions]).float(), -1)[:, close_id]
            log1m = torch.log1p(-torch.exp(logp).clamp(max=1 - 1e-6))
            who = teacher(r["source"])
            if who == "protect" or (kind == "fixed" and who != "judge"):
                goals = [0.0] * len(cuts)
            elif kind == "fixed":  # the yes/no judge above a fixed bar (tools only)
                goals = [sharpen(h, value, args.temperature) for _, h, _, _ in cuts]
            else:  # as right as finishing (minus the margin), never below --min-p; within the same answer
                pos = 2 if who == "judge" else 3  # soft judge's predicted success, or own confidence
                bar = max(end_p[pos - 2] - value, args.min_p)
                goals = [sharpen(c[pos], bar, args.temperature) for c in cuts]
            target = torch.tensor(goals, device="cuda")
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
                print(f"{variant}: {step + 1}/{len(order)} answers, loss {total / n:.4f}", flush=True)
                total, n = 0.0, 0
        model.save_pretrained(target_dir)
        print(f"{variant}: saved {target_dir}; mean p(</think>) at paragraph ends {np.mean(before):.3f} (first 50) "
              f"-> {np.mean(after):.3f} (last 50)", flush=True)
        del model, optimizer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
