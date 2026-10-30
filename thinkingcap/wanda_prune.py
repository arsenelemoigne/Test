"""SPARSE arm C (SPARSE_SPEC.md): Wanda one-shot pruning of Qwen3-0.6B + a cheap damage check, runnable on a laptop.

Wanda (Sun et al. 2023): each weight's score is |W_ij| x ||X_j||_2, the norm of its input feature over the calibration
tokens; inside each output row the lowest --sparsity share is set to zero. Done block by block IN ORDER: each block's
input norms are measured on the hidden states that come out of the already-pruned blocks before it (as the official
code does). Only the 7 linear layers of the 28 decoder blocks are pruned (not the tied embedding, not the norms).
Calibration = the original model's OWN reasoning answers (prompt + its reply), as SPARSE_SPEC asks (reasoning models
lose far less with self-generated calibration text).

Damage check (no generation): on held-out RIGHT answers (never used for calibration), the mean per-token negative
log-likelihood of the reply given the prompt, per type, for the original and each pruned model; plus "right > wrong":
on questions with both a right and a wrong answer, the share where the model gives the right one the higher mean
log-probability. These are exact computations but a PROXY: real accuracy needs generation (GPU eval).

    python wanda_prune.py --rollouts dp11check.jsonl --sparsity 0.1 0.2 0.3 0.4 0.5 --out runs/sparse/wanda-c1
"""

import argparse
import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch

from build_pool import part_of

PARTS = ("svamp", "mbpp", "ifeval", "bfcl_simple_python", "bfcl_multiple", "bfcl_irrelevance")
LINEARS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def read(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def get(module, dotted):
    for name in dotted.split("."):
        module = getattr(module, name)
    return module


def run_block(model, layer, hidden, device):
    """One decoder block on one sequence (batch 1), with the rotary position inputs the full model would pass."""
    positions = torch.arange(hidden.shape[1], device=device).unsqueeze(0)
    cos_sin = model.model.rotary_emb(hidden, positions)
    out = layer(hidden, attention_mask=None, position_ids=positions, position_embeddings=cos_sin)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def wanda(model, calib, sparsity, device):
    """Prune `model` in place; calib = list of token-id lists."""
    hiddens = [model.model.embed_tokens(torch.tensor([ids], device=device)).float().cpu() for ids in calib]
    for li, layer in enumerate(model.model.layers):
        sq = {name: torch.zeros(get(layer, name).in_features, device=device) for name in LINEARS}
        hooks = []
        for name in LINEARS:
            def hook(module, inputs, output, name=name):
                x = inputs[0].reshape(-1, inputs[0].shape[-1]).float()
                sq[name] += (x * x).sum(0)
            hooks.append(get(layer, name).register_forward_hook(hook))
        for h in hiddens:
            run_block(model, layer, h.to(device), device)
        for hk in hooks:
            hk.remove()
        for name in LINEARS:
            lin = get(layer, name)
            score = lin.weight.abs().float() * sq[name].sqrt().unsqueeze(0)
            k = int(score.shape[1] * sparsity)
            if k:
                cut = torch.topk(score, k, dim=1, largest=False).indices
                lin.weight.scatter_(1, cut, 0)
        hiddens = [run_block(model, layer, h.to(device), device).float().cpu() for h in hiddens]  # pruned outputs feed the next block


@torch.no_grad()
def reply_nll(model, prompt_ids, output_ids, device, max_len):
    ids = (prompt_ids + output_ids)[:max_len]
    n_prompt = min(len(prompt_ids), max_len - 1)
    x = torch.tensor([ids], device=device)
    hidden = model.model(x).last_hidden_state[0, n_prompt - 1:-1]  # positions that predict the reply tokens
    target = x[0, n_prompt:]
    total = 0.0
    for i in range(0, hidden.shape[0], 256):  # the 151k-word output layer in chunks: ~150 MB instead of ~2 GB per answer
        logits = model.lm_head(hidden[i:i + 256]).float()
        total += torch.nn.functional.cross_entropy(logits, target[i:i + 256], reduction="sum").item()
    return total / max(1, target.shape[0])


def damage(model, right, pairs, device, max_len):
    by_part = defaultdict(list)
    for r in right:
        by_part[r["_part"]].append(reply_nll(model, r["prompt_ids"], r["output_ids"], device, max_len))
    prefer = defaultdict(list)
    for good, bad in pairs:
        prefer[good["_part"]].append(reply_nll(model, good["prompt_ids"], good["output_ids"], device, max_len)
                                     < reply_nll(model, bad["prompt_ids"], bad["output_ids"], device, max_len))
    return ({p: statistics.fmean(v) for p, v in by_part.items()}, {p: statistics.fmean(v) for p, v in prefer.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--sparsity", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--calib", type=int, default=128, help="calibration answers (the paper's 128)")
    parser.add_argument("--calib-len", type=int, default=1024)
    parser.add_argument("--per-part", type=int, default=100, help="held-out right answers per part for the damage check")
    parser.add_argument("--pairs-per-part", type=int, default=40)
    parser.add_argument("--max-len", type=int, default=1536)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--mask-compare", default="", help="a saved mask (mask-30.pt): share of pruned weights in common")
    parser.add_argument("--save", action="store_true", help="also save each pruned model (for the GPU eval)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM

    rng = random.Random(args.seed)
    rows = read(args.rollouts)
    for r in rows:
        r["_part"] = part_of(r)
        r["_ok"] = bool(r["correct"]) and r["closed_thinking"] and not r["truncated"]
    by_q = defaultdict(list)
    for r in rows:
        by_q[r["id"]].append(r)
    qids = sorted(by_q)
    rng.shuffle(qids)
    calib_q = set()
    calib = []
    for q in qids:  # calibration: one right answer per question, mixed types
        ok = [r for r in by_q[q] if r["_ok"]]
        if ok and len(calib) < args.calib:
            calib.append((ok[0]["prompt_ids"] + ok[0]["output_ids"])[:args.calib_len])
            calib_q.add(q)
    right, pairs = [], []
    count, pcount = defaultdict(int), defaultdict(int)
    for q in qids:  # held out: questions never used for calibration
        if q in calib_q:
            continue
        ok = [r for r in by_q[q] if r["_ok"]]
        bad = [r for r in by_q[q] if not r["_ok"] and r["closed_thinking"]]
        part = by_q[q][0]["_part"]
        if ok and count[part] < args.per_part:
            right.append(ok[0])
            count[part] += 1
        if ok and bad and pcount[part] < args.pairs_per_part:
            pairs.append((ok[-1], bad[0]))
            pcount[part] += 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"calibration {len(calib)} answers; damage check {dict(count)} right answers, {dict(pcount)} right/wrong pairs", flush=True)

    def load():
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="sdpa").to(args.device)
        model.eval()
        return model

    model = load()
    # sanity: the block-by-block pass must reproduce the full model (same final hidden states up to float noise)
    with torch.no_grad():
        ids = torch.tensor([calib[0][:256]], device=args.device)
        full = model.model(ids).last_hidden_state
        h = model.model.embed_tokens(ids)
        for layer in model.model.layers:
            h = run_block(model, layer, h, args.device)
        h = model.model.norm(h)
        gap = (full - h).abs().max().item()
    print(f"sanity: block-by-block vs full model, max difference {gap:.2e}", flush=True)
    assert gap < 1e-3, "block-by-block pass does not match the full model: Wanda's inputs would be wrong"

    results = {}
    t = time.time()
    results["0.0"] = damage(model, right, pairs, args.device, args.max_len)
    print(f"original: nll {results['0.0'][0]}  right>wrong {results['0.0'][1]}  ({time.time() - t:.0f}s)", flush=True)
    for s in args.sparsity:
        model = load()
        t = time.time()
        wanda(model, calib, s, args.device)
        zero = sum((get(l, n).weight == 0).sum().item() for l in model.model.layers for n in LINEARS)
        total = sum(get(l, n).weight.numel() for l in model.model.layers for n in LINEARS)
        results[str(s)] = damage(model, right, pairs, args.device, args.max_len)
        print(f"sparsity {s:.0%} (measured {zero / total:.1%} of block weights zero, {time.time() - t:.0f}s): "
              f"nll {results[str(s)][0]}  right>wrong {results[str(s)][1]}", flush=True)
        mask = {f"{li}.{n}": (get(l, n).weight == 0).cpu() for li, l in enumerate(model.model.layers) for n in LINEARS}
        torch.save(mask, out / f"mask-{round(s * 100)}.pt")
        if args.mask_compare and Path(args.mask_compare).exists():
            other = torch.load(args.mask_compare)
            same = defaultdict(lambda: [0, 0])
            for key, m in mask.items():
                kind = key.split(".", 1)[1]
                same[kind][0] += (m & other[key]).sum().item()
                same[kind][1] += m.sum().item()
            text = "mask overlap with " + args.mask_compare + ": " + ", ".join(
                f"{k} {100 * a / max(1, b):.1f}%" for k, (a, b) in same.items())
            print(text, flush=True)
            (out / "mask_overlap.txt").write_text(text + "\n")
        if args.save:  # weights + tokenizer, so evaluate.py can load the folder as a model
            model.to(torch.bfloat16).save_pretrained(out / f"wanda-{round(s * 100)}")
            from transformers import AutoTokenizer
            AutoTokenizer.from_pretrained(args.model).save_pretrained(out / f"wanda-{round(s * 100)}")
        (out / "results.json").write_text(json.dumps(results, indent=1))
    base_nll, base_pref = results["0.0"]
    lines = ["Wanda damage check (proxy, no generation): reply NLL increase vs original (%), and right>wrong preference",
             "sparsity | " + " | ".join(PARTS)]
    for key, (nll, pref) in results.items():
        if key == "0.0":
            continue
        lines.append(f"{float(key):.0%} | " + " | ".join(
            f"+{100 * (nll[p] / base_nll[p] - 1):.0f}% nll, {100 * pref.get(p, 0):.0f}% vs {100 * base_pref.get(p, 0):.0f}%"
            for p in PARTS if p in nll))
    if (out / "mask_overlap.txt").exists():
        lines.append((out / "mask_overlap.txt").read_text().strip())
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
