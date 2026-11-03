"""Chirurgical edits (3 Nov, user): remove the raw Qwen's looping habit from a few weights, as in "Sparks of
Weight-Space Programming" (Takehi et al., MIT, 2 Nov 2026; logs github.com/RikiyaT/lm_understand-logs, Exp 8), but
measured on what we care about: thinking tokens at equal accuracy. No training: every edit is computed, not fitted.

Phases (each saves to --out and is skipped when its file exists, so a lost runtime resumes):
    mine   looping answers among the model's own graded answers (capped, >= --min-repeat of the thinking's 8-grams
           already seen); cut each just before its first repeated 32-token stretch: prefix = prompt + answer up to
           there, loop branch = the real next --branch tokens. The productive branch: --alternatives sampled
           continuations of the same prefix, keeping the one whose 8-grams overlap the prefix least (it must be
           mostly new). Pairs split by question: --find for the direction, the rest held out for the checks.
           Also: --normal finished right answers (all types) for the "nothing else moved" checks, and fixed dev /
           holdout question sets (questions never used for pairs).
    probe  teacher-forced reads of the pairs (no generation):
             direction per layer  u_l = mean(loop - productive) of the hidden state after layer l over the branch
                                  tokens, normalised; split-half cosine says whether it is stable;
             writers              per layer, how much its MLP output pushes along u_l (loop minus productive);
             copy heads           per head, the attention a branch token pays to "what came right after the last
                                  time this token appeared" (Exp 8 p25c's score: L15H7 0.887 loop vs 0.082).
    edit   candidates: each top copy head zeroed and halved (its o_proj columns); the direction removed from the
           top writer layers' down_proj, W' = W - a u (mask64 * u^T W) (p50c, mask = the 64 columns that write the
           most loop-specific signal), a = --doses; a random direction (orthogonal to u, same mask, same a) as control.
    mech   held-out pairs and normal answers, teacher-forced, every candidate vs the original: log-prob per token
           on loop branches (should fall) and productive branches / normal answers (should not), and the chance of
           closing the thinking mid-answer (should not rise). Candidates failing these are not generated.
    gen    paired generation, original and candidate on the same dev questions with the same seeds (vLLM, the
           evaluation's sampling), graded with typed.py: accuracy, thinking tokens, cap share, tokens on answers both
           finished, per type, paired accuracy difference +- SE, wrong->right / right->wrong.
    pick   the candidate with the most thinking saved at no worse than -1 point on dev (ties: accuracy), confirmed on
           the holdout set; its merged weights go to --out/best for the 720 screen (track.py chirurgical).

    python chirurgical.py --answers runs/bon/lupin-all-...-a4.jsonl --pool data/pool-dp13.jsonl --out runs/chir/c1
"""

import argparse
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))


# ---------- pure helpers (unit-tested in chirurgical_test.py) ----------

def repeat_share(ids, n=8):
    """Share of n-grams that already appeared earlier in the sequence."""
    seen, rep = set(), 0
    for i in range(len(ids) - n + 1):
        g = tuple(ids[i:i + n])
        rep += g in seen
        seen.add(g)
    return rep / max(1, len(ids) - n + 1)


def loop_start(ids, n=32):
    """Index of the first token of the first n-token stretch that already appeared earlier (None if none)."""
    seen = {}
    for i in range(len(ids) - n + 1):
        g = tuple(ids[i:i + n])
        if g in seen:
            return i
        seen[g] = i
    return None


def novelty(prefix, branch, n=8):
    """Share of the branch's n-grams that do NOT appear in prefix + earlier branch: 1 = all new text."""
    seen = {tuple(prefix[i:i + n]) for i in range(len(prefix) - n + 1)}
    seq = list(prefix[-(n - 1):]) + list(branch)
    new, total = 0, 0
    for i in range(len(seq) - n + 1):
        g = tuple(seq[i:i + n])
        new += g not in seen
        total += 1
        seen.add(g)
    return new / max(1, total)


def copy_score(ids, attn, positions):
    """Mean attention, over query positions, on the tokens that followed earlier occurrences of the query token
    (the "previous matching continuation"). attn: [heads, T, T] (one layer). Returns one score per head."""
    import torch
    heads = attn.shape[0]
    total = torch.zeros(heads)
    count = 0
    where = defaultdict(list)
    for j, t in enumerate(ids):
        where[t].append(j)
    for t in positions:
        targets = [j + 1 for j in where[ids[t]] if j + 1 < t]
        if not targets:
            continue
        total += attn[:, t, targets].float().sum(-1).cpu()
        count += 1
    return (total / max(1, count)).tolist()


def remove_direction(W, u, mask, alpha):
    """p50c's edit on a down_proj weight W [hidden, inter]: W - alpha * u (mask * (u^T W)). u: unit [hidden]."""
    import torch
    proj = (u @ W) * mask
    return W - alpha * torch.outer(u, proj)


def paired_stats(a, b):
    """a, b: 0/1 lists (original, candidate) on the same items. Difference in points and its SE."""
    d = [y - x for x, y in zip(a, b)]
    n = len(d)
    mean = sum(d) / n
    sd = statistics.pstdev(d) if n > 1 else 0.0
    return {"n": n, "original": sum(a) / n, "candidate": sum(b) / n, "diff_pp": 100 * mean,
            "se_pp": 100 * sd / math.sqrt(n), "wrong_to_right": sum(x == 0 and y == 1 for x, y in zip(a, b)),
            "right_to_wrong": sum(x == 1 and y == 0 for x, y in zip(a, b))}


# ---------- phases ----------

def think_end(ids, close_id):
    return ids.index(close_id) if close_id in ids else len(ids)


def mine(args, tok, part_of):
    out = Path(args.out)
    if (out / "pairs.jsonl").exists():
        return
    rng = random.Random(args.seed)
    close_id = tok.convert_tokens_to_ids("</think>")
    rows = {}
    for f in args.pool:
        for r in read_jsonl(f):
            if not re.search(r"prostring|stringname|stringName", json.dumps(r)):  # the "file" -> "string" corruption
                rows[r.get("pool_id", r.get("id"))] = r
    by_q = defaultdict(list)
    for f in args.answers:
        for r in read_jsonl(f):
            if r["id"] in rows:
                by_q[r["id"]].append(r)
    loops, normal = [], []
    for qid, answers in by_q.items():
        for r in answers:
            ids = r["output_ids"]
            thinking = ids[:think_end(ids, close_id)]
            if r.get("truncated") and repeat_share(thinking) >= args.min_repeat:
                s = loop_start(thinking)
                if s is not None and s >= args.branch and s + args.branch <= len(thinking):
                    # the last --context tokens before the loop (the start of an 8k-token answer adds read time, not signal)
                    loops.append({"id": qid, "part": part_of(rows[qid]), "prompt_ids": r["prompt_ids"],
                                  "prefix": thinking[max(0, s - args.context):s], "loop": thinking[s:s + args.branch]})
            elif r.get("correct") and not r.get("truncated") and close_id in ids:
                normal.append({"id": qid, "part": part_of(rows[qid]), "prompt_ids": r["prompt_ids"], "output_ids": ids})
    rng.shuffle(loops)
    seen, uniq = set(), []
    for x in loops:  # one loop per question: pairs from the same question are near copies
        if x["id"] not in seen:
            seen.add(x["id"])
            uniq.append(x)
    loops = uniq[:args.pairs]
    print(f"mine: {len(uniq)} looping questions, kept {len(loops)}; parts {dict(Counter(x['part'] for x in loops))}", flush=True)
    # productive branches: sampled continuations of the same prefix, the most novel one
    from vllm import LLM, SamplingParams
    from utils.eval.recipe import SAMPLING
    llm = LLM(model=args.model, dtype="bfloat16", seed=args.seed, gpu_memory_utilization=args.gpu_memory,
              max_model_len=args.max_len, generation_config="vllm")
    reqs = [{"prompt_token_ids": x["prompt_ids"] + x["prefix"]} for x in loops]
    keep = []
    for i in range(0, len(reqs), 256):
        outs = llm.generate(reqs[i:i + 256], SamplingParams(n=args.alternatives, max_tokens=args.branch,
                                                              seed=args.seed, **SAMPLING))
        for x, o in zip(loops[i:i + 256], outs):
            alts = [list(c.token_ids) for c in o.outputs if len(c.token_ids) == args.branch and close_id not in c.token_ids]
            if not alts:
                continue
            best = max(alts, key=lambda a: novelty(x["prefix"], a))
            if novelty(x["prefix"], best) >= args.min_novelty and novelty(x["prefix"], x["loop"]) <= 0.5:
                keep.append(x | {"productive": best, "novelty": novelty(x["prefix"], best)})
    del llm
    rng.shuffle(keep)
    for k, x in enumerate(keep):
        x["split"] = "find" if k < int(len(keep) * args.find) else "check"
    write_jsonl(out / "pairs.jsonl", keep)
    rng.shuffle(normal)
    by_part = defaultdict(list)
    for x in normal:
        by_part[x["part"]].append(x)
    normal = [x for p in sorted(by_part) for x in by_part[p][:max(1, args.normal // len(by_part))]]
    write_jsonl(out / "normal.jsonl", normal)
    used = {x["id"] for x in keep} | {x["id"] for x in normal}
    pool = [q for q in rows if q not in used and q in by_q]
    rng.shuffle(pool)
    sets = {"dev": pool[:args.dev], "holdout": pool[args.dev:args.dev + args.holdout]}
    (out / "sets.json").write_text(json.dumps(sets))
    print(f"mine: {len(keep)} pairs ({sum(x['split'] == 'find' for x in keep)} to find, rest to check), "
          f"{len(normal)} normal answers, dev {len(sets['dev'])}, holdout {len(sets['holdout'])}", flush=True)


DTYPE = "float32"  # --dtype; bfloat16 lets the 0.6B fit an 8 GB Mac for the CPU check


def load_hf(path, eager=False):
    import torch
    from transformers import AutoModelForCausalLM
    kw = {"attn_implementation": "eager"} if eager else {}
    return AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, DTYPE), **kw).to(
        "cuda" if torch.cuda.is_available() else "cpu").eval()


def branch_means(model, prompt, prefix, branch, mlp_cache=None):
    """Hidden state after every layer, averaged over the branch tokens; optionally MLP outputs / down_proj inputs."""
    import torch
    ids = torch.tensor([prompt + prefix + branch], device=model.device)
    lo = len(prompt) + len(prefix)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    hs = [h[0, lo:].mean(0).float().cpu() for h in out.hidden_states[1:]]
    return hs


def probe(args):
    import torch
    out = Path(args.out)
    if (out / "probe.pt").exists():
        return
    pairs = [x for x in read_jsonl(out / "pairs.jsonl") if x["split"] == "find"]
    model = load_hf(args.model)
    layers = model.model.layers
    L = len(layers)
    mlp_out, act_in = {}, {}

    def keep_out(i):
        return lambda m, inp, o: mlp_out.__setitem__(i, o.detach())

    def keep_in(i):
        return lambda m, inp: act_in.__setitem__(i, inp[0].detach())
    hooks = [layers[i].mlp.register_forward_hook(keep_out(i)) for i in range(L)]
    hooks += [layers[i].mlp.down_proj.register_forward_pre_hook(keep_in(i)) for i in range(L)]
    diffs = [[] for _ in range(L)]
    mlp_diff = [[] for _ in range(L)]   # per pair, per layer: mean MLP output on loop - productive (hidden)
    act_diff = [None] * L               # running sum of down_proj input means, loop - productive
    for x in pairs:
        side = {}
        for name in ("loop", "productive"):
            hs = branch_means(model, x["prompt_ids"], x["prefix"], x[name])
            lo = len(x["prompt_ids"]) + len(x["prefix"])
            side[name] = (hs, [mlp_out[i][0, lo:].mean(0).float().cpu() for i in range(L)],
                          [act_in[i][0, lo:].mean(0).float().cpu() for i in range(L)])
        for i in range(L):
            diffs[i].append(side["loop"][0][i] - side["productive"][0][i])
            mlp_diff[i].append(side["loop"][1][i] - side["productive"][1][i])
            a = side["loop"][2][i] - side["productive"][2][i]
            act_diff[i] = a if act_diff[i] is None else act_diff[i] + a
    for h in hooks:
        h.remove()
    n = len(pairs)
    u = [torch.stack(d).mean(0) for d in diffs]
    u = [v / v.norm().clamp_min(1e-8) for v in u]
    half = n // 2
    stab = []
    for i in range(L):
        a = torch.stack(diffs[i][:half]).mean(0)
        b = torch.stack(diffs[i][half:]).mean(0)
        stab.append(float(torch.nn.functional.cosine_similarity(a, b, dim=0)))
    write = [float(torch.stack(mlp_diff[i]).mean(0) @ u[i]) for i in range(L)]
    act = [v / n for v in act_diff]
    del model
    # copy heads: eager attention on loop and productive texts (cropped to --attn-len tokens)
    model = load_hf(args.model, eager=True)
    scores = {"loop": None, "productive": None}
    for name in scores:
        acc = None
        for x in pairs[:args.attn_pairs]:
            ids = (x["prompt_ids"] + x["prefix"] + x[name])[-args.attn_len:]
            positions = list(range(len(ids) - len(x[name]), len(ids)))
            with torch.no_grad():
                att = model(torch.tensor([ids], device=model.device), output_attentions=True).attentions
            s = torch.tensor([copy_score(ids, a[0], positions) for a in att])  # [L, heads]
            acc = s if acc is None else acc + s
        scores[name] = acc / max(1, min(len(pairs), args.attn_pairs))
    torch.save({"u": u, "stability": stab, "write": write, "act": act, "copy": scores}, out / "probe.pt")
    best = sorted(range(L), key=lambda i: -write[i])[:4]
    gap = scores["loop"] - scores["productive"]
    top = sorted(((float(gap[l, h]), l, h) for l in range(gap.shape[0]) for h in range(gap.shape[1])), reverse=True)[:5]
    print("probe: direction stability (split-half cosine) by layer: " + " ".join(f"{s:.2f}" for s in stab), flush=True)
    print("probe: top MLP writers (layer: loop-specific write) " + ", ".join(f"L{i}: {write[i]:.3f}" for i in best), flush=True)
    print("probe: top copy heads (loop - productive score) " + ", ".join(
        f"L{l}H{h} {g:.3f} ({float(scores['loop'][l, h]):.2f} vs {float(scores['productive'][l, h]):.2f})"
        for g, l, h in top), flush=True)


def candidates(args, probe_data, model):
    """[(name, {tensor_name: new_tensor})] edits on the ORIGINAL weights."""
    import torch
    cands = []
    L = len(model.model.layers)
    cfg = model.config
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    gap = probe_data["copy"]["loop"] - probe_data["copy"]["productive"]
    top = sorted(((float(gap[l, h]), l, h) for l in range(gap.shape[0]) for h in range(gap.shape[1])), reverse=True)
    for _, l, h in top[:args.heads]:
        name = f"model.layers.{l}.self_attn.o_proj.weight"
        W = model.state_dict()[name].clone()
        for scale in (0.0, 0.5):
            V = W.clone()
            V[:, h * hd:(h + 1) * hd] *= scale
            cands.append((f"head-L{l}H{h}x{scale:g}", {name: V}))
    write = probe_data["write"]
    layers = sorted(range(L), key=lambda i: -write[i])[:args.layers]
    g = torch.Generator().manual_seed(args.seed)
    edits = {}
    for i in layers:
        name = f"model.layers.{i}.mlp.down_proj.weight"
        W = model.state_dict()[name].float().cpu()
        u = probe_data["u"][i]
        score = (u @ W) * probe_data["act"][i]
        mask = torch.zeros(W.shape[1])
        mask[torch.topk(score, args.columns).indices] = 1.0
        r = torch.randn(u.shape, generator=g)
        r = r - (r @ u) * u
        r = r / r.norm()
        edits[name] = (W, u, mask, r)
    tag = "+".join(f"L{i}" for i in layers)
    for a in args.doses:
        cands.append((f"dir-{tag}-a{a:g}", {n: remove_direction(W, u, m, a) for n, (W, u, m, r) in edits.items()}))
    a = max(args.doses)
    cands.append((f"random-{tag}-a{a:g}", {n: remove_direction(W, r, m, a) for n, (W, u, m, r) in edits.items()}))
    return cands


def logprob_regions(model, prompt, before, region):
    """Mean log-prob per token of `region` after prompt + before (teacher-forced)."""
    import torch
    ids = torch.tensor([prompt + before + region], device=model.device)
    with torch.no_grad():
        logits = model(ids).logits[0, :-1].float()
    lp = torch.log_softmax(logits, -1)
    lo = len(prompt) + len(before)
    tgt = ids[0, lo:]
    return float(lp[lo - 1:lo - 1 + len(region)].gather(1, tgt[:, None]).mean())


def close_logprob(model, prompt, output_ids, close_id, share=0.5):
    """Log-prob of closing the thinking at --share of a finished answer's thinking (a premature close)."""
    import torch
    end = output_ids.index(close_id)
    cut = max(1, int(end * share))
    ids = torch.tensor([prompt + output_ids[:cut]], device=model.device)
    with torch.no_grad():
        logits = model(ids).logits[0, -1].float()
    return float(torch.log_softmax(logits, -1)[close_id])


def mech(args, tok):
    import torch
    out = Path(args.out)
    if (out / "mech.json").exists():
        return
    close_id = tok.convert_tokens_to_ids("</think>")
    pairs = [x for x in read_jsonl(out / "pairs.jsonl") if x["split"] == "check"][:args.check_pairs]
    normal = read_jsonl(out / "normal.jsonl")[:args.check_normal]
    probe_data = torch.load(out / "probe.pt")
    model = load_hf(args.model)
    original = {k: v.clone() for k, v in model.state_dict().items()}

    def measure():
        loop = statistics.fmean(logprob_regions(model, x["prompt_ids"], x["prefix"], x["loop"]) for x in pairs)
        prod = statistics.fmean(logprob_regions(model, x["prompt_ids"], x["prefix"], x["productive"]) for x in pairs)
        norm = statistics.fmean(logprob_regions(model, x["prompt_ids"], [], x["output_ids"][:args.normal_len]) for x in normal)
        close = statistics.fmean(close_logprob(model, x["prompt_ids"], x["output_ids"], close_id) for x in normal)
        return {"loop": loop, "productive": prod, "normal": norm, "close": close}
    base = measure()
    results = {"original": base}
    for name, edit in candidates(args, probe_data, model):
        sd = model.state_dict()
        for k, v in edit.items():
            sd[k].copy_(v.to(sd[k].dtype).to(sd[k].device))
        m = measure()
        d = {k: m[k] - base[k] for k in m}
        # selective: loop text clearly less likely, productive / normal text barely, no premature close
        d["pass"] = bool(d["loop"] < -0.01 and d["productive"] > -0.02 and d["normal"] > -0.02 and d["close"] < math.log(2))
        results[name] = d
        print(f"mech {name}: loop {d['loop']:+.4f}, productive {d['productive']:+.4f}, normal {d['normal']:+.4f}, "
              f"close {d['close']:+.3f} -> {'PASS' if d['pass'] else 'drop'}", flush=True)
        for k in edit:
            sd[k].copy_(original[k])
    (out / "mech.json").write_text(json.dumps(results, indent=1))


def survival(original, edited, dtype):
    """How much of an edit survives a cast to `dtype` (the evaluation loads bfloat16): share of edited entries still
    different from the cast original, and the cast edit's norm as a share of the float32 edit's norm."""
    import torch
    o = original.float()
    e = edited.float()
    changed = (e != o)
    oc, ec = o.to(dtype).float(), e.to(dtype).float()
    kept = ((ec != oc) & changed).sum().item() / max(1, changed.sum().item())
    norm = (ec - oc).norm().item() / max(1e-12, (e - o).norm().item())
    return {"entries_kept": kept, "norm_kept": norm}


def save_candidate(args, name, probe_data, path):
    """The original weights with one candidate's edit, saved for vLLM in --gen-dtype; prints how much of the edit
    survives the cast (3 Nov, user: is bfloat16 precise enough?)."""
    import torch
    if (Path(path) / "config.json").exists():
        return
    from transformers import AutoTokenizer
    model = load_hf(args.model)
    edit = dict(candidates(args, probe_data, model))[name]
    sd = model.state_dict()
    dtype = getattr(torch, args.gen_dtype)
    for k, v in edit.items():
        if dtype != torch.float32:
            sv = survival(sd[k], v, dtype)
            flag = "" if sv["entries_kept"] > 0.9 and sv["norm_kept"] > 0.9 else "  <-- mostly lost in the cast"
            print(f"cast {name} {k}: {100 * sv['entries_kept']:.1f}% of edited entries and "
                  f"{100 * sv['norm_kept']:.1f}% of the edit's size survive {args.gen_dtype}{flag}", flush=True)
        sd[k].copy_(v.to(sd[k].dtype).to(sd[k].device))
    model.to(dtype).save_pretrained(path)
    AutoTokenizer.from_pretrained(args.model).save_pretrained(path)


def generate_graded(args, model_path, qids, rows, tok, seed):
    """One answer per question, the evaluation's sampling, graded; returns per question (right, thinking, capped)."""
    from vllm import LLM, SamplingParams
    from utils.eval.recipe import SAMPLING
    import typed
    close_id = tok.convert_tokens_to_ids("</think>")
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    llm = LLM(model=str(model_path), dtype=args.gen_dtype, seed=seed, gpu_memory_utilization=args.gpu_memory,
              max_model_len=args.max_tokens + 2048, generation_config="vllm")
    prompts = [tok.apply_chat_template(rows[q]["prompt"], tokenize=True, add_generation_prompt=True) for q in qids]
    params = [SamplingParams(max_tokens=args.max_tokens, seed=seed + k, stop_token_ids=[eos], **SAMPLING)
              for k in range(len(qids))]
    outs = llm.generate([{"prompt_token_ids": p} for p in prompts], params)
    del llm
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    res = {}
    for q, o in zip(qids, outs):
        ids = list(o.outputs[0].token_ids)
        capped = o.outputs[0].finish_reason == "length"
        right = (not capped) and bool(typed.is_correct(rows[q], typed.full_text(tok, ids)))
        res[q] = (int(right), think_end(ids, close_id), int(capped))
    return res


def summarize(orig, cand, qids, part):
    parts = defaultdict(list)
    for q in qids:
        parts[part[q]].append(q)
    out = {}
    for p, qs in [("all", qids)] + sorted(parts.items()):
        s = paired_stats([orig[q][0] for q in qs], [cand[q][0] for q in qs])
        t0 = statistics.fmean(orig[q][1] for q in qs)
        t1 = statistics.fmean(cand[q][1] for q in qs)
        both = [q for q in qs if not orig[q][2] and not cand[q][2]]
        s |= {"thinking_original": t0, "thinking_candidate": t1, "saved_pct": 100 * (1 - t1 / max(1e-9, t0)),
              "cap_original": statistics.fmean(orig[q][2] for q in qs), "cap_candidate": statistics.fmean(cand[q][2] for q in qs),
              "finished_both": len(both),
              "saved_pct_finished_both": 100 * (1 - statistics.fmean(cand[q][1] for q in both) /
                                                max(1e-9, statistics.fmean(orig[q][1] for q in both))) if both else None}
        out[p] = s
    return out


def gen(args, tok, part_of):
    import torch
    out = Path(args.out)
    rows = {}
    for f in args.pool:
        for r in read_jsonl(f):
            rows[r.get("pool_id", r.get("id"))] = r
    sets = json.loads((out / "sets.json").read_text())
    part = {q: part_of(rows[q]) for q in sets["dev"] + sets["holdout"]}
    mech_res = json.loads((out / "mech.json").read_text())
    probe_data = torch.load(out / "probe.pt")
    passing = [k for k, v in mech_res.items() if k != "original" and v.get("pass")]
    controls = [k for k in mech_res if k.startswith("random-")]
    path = out / "gen.json"
    done = json.loads(path.read_text()) if path.exists() else {}
    cache = out / "answers"
    cache.mkdir(exist_ok=True)

    def answers(name, model_path, split):
        f = cache / f"{name}-{split}.json"
        if f.exists():
            return {k: tuple(v) for k, v in json.loads(f.read_text()).items()}
        res = generate_graded(args, model_path, sets[split], rows, tok, args.seed)
        f.write_text(json.dumps(res))
        return res
    orig_dev = answers("original", args.model, "dev")
    for name in passing + controls:
        if name in done:
            continue
        mp = Path(args.scratch) / name
        save_candidate(args, name, probe_data, mp)
        done[name] = summarize(orig_dev, answers(name, mp, "dev"), sets["dev"], part)
        a = done[name]["all"]
        print(f"gen {name} (dev): accuracy {100 * a['original']:.1f} -> {100 * a['candidate']:.1f} "
              f"({a['diff_pp']:+.1f} +- {a['se_pp']:.1f}), thinking saved {a['saved_pct']:+.1f}% "
              f"(on answers both finished {a['saved_pct_finished_both'] or 0:+.1f}%), capped "
              f"{100 * a['cap_original']:.1f}% -> {100 * a['cap_candidate']:.1f}%", flush=True)
        path.write_text(json.dumps(done, indent=1))
    ok = [k for k in passing if k in done and done[k]["all"]["diff_pp"] >= -1.0]
    if not ok:
        print("pick: no candidate saves thinking at no worse than -1 point on dev", flush=True)
        (out / "pick.json").write_text(json.dumps({"best": None}))
        return
    best = max(ok, key=lambda k: (done[k]["all"]["saved_pct"], done[k]["all"]["diff_pp"]))
    hold = summarize(answers("original", args.model, "holdout"), answers(best, Path(args.scratch) / best, "holdout"),
                     sets["holdout"], part)
    a = hold["all"]
    print(f"pick {best} (holdout): accuracy {a['diff_pp']:+.1f} +- {a['se_pp']:.1f}, thinking saved {a['saved_pct']:+.1f}%, "
          f"capped {100 * a['cap_original']:.1f}% -> {100 * a['cap_candidate']:.1f}%", flush=True)
    (out / "pick.json").write_text(json.dumps({"best": best, "dev": done[best], "holdout": hold}, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--answers", nargs="+", required=True)
    p.add_argument("--pool", nargs="+", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--out", required=True)
    p.add_argument("--scratch", default="/content/chir-models")
    p.add_argument("--phases", default="mine,probe,mech,gen")
    p.add_argument("--min-repeat", type=float, default=0.3)
    p.add_argument("--branch", type=int, default=48)
    p.add_argument("--context", type=int, default=1536, help="thinking tokens kept before the loop")
    p.add_argument("--alternatives", type=int, default=8)
    p.add_argument("--min-novelty", type=float, default=0.8)
    p.add_argument("--pairs", type=int, default=600)
    p.add_argument("--find", type=float, default=0.5)
    p.add_argument("--normal", type=int, default=300)
    p.add_argument("--dev", type=int, default=256)
    p.add_argument("--holdout", type=int, default=512)
    p.add_argument("--attn-pairs", type=int, default=64)
    p.add_argument("--attn-len", type=int, default=768)
    p.add_argument("--heads", type=int, default=3)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--columns", type=int, default=64)
    p.add_argument("--doses", type=float, nargs="+", default=[1, 2, 4, 6])
    p.add_argument("--check-pairs", type=int, default=150)
    p.add_argument("--check-normal", type=int, default=150)
    p.add_argument("--normal-len", type=int, default=512)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--max-len", type=int, default=12288)
    p.add_argument("--gpu-memory", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="float32", help="reads and edits (HF)")
    p.add_argument("--gen-dtype", default="bfloat16", help="the paired test answers (vLLM); the evaluation runs bfloat16")
    args = p.parse_args()
    global DTYPE
    DTYPE = args.dtype
    Path(args.out).mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    from build_pool import part_of
    tok = AutoTokenizer.from_pretrained(args.model)
    phases = args.phases.split(",")
    if "mine" in phases:
        mine(args, tok, part_of)
    if "probe" in phases:
        probe(args)
    if "mech" in phases:
        mech(args, tok)
    if "gen" in phases:
        gen(args, tok, part_of)


if __name__ == "__main__":
    main()
