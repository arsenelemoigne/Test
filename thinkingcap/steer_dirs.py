"""Three "be brief" directions in the model's internal state, found from its own replies, kept small.

The raw model's shortest-of-8 rollouts (runs/bon/probe) hold, for the same question, short and
long correct replies. We read the residual stream after every layer over the first --window
thinking tokens (where the model "decides" how much to think, before long replies start
checking themselves), then build, per layer and per kind of task:

  A  mean difference: mean over questions of (shortest correct - longest correct)
  B  within-question trend: the direction that grows as correct replies get shorter, over every
     correct reply of the question (centred per question, against log length), minus its part
     along the "gives up" direction (wrong replies - correct replies of the same question)
  C  learned: one vector added at one layer, trained (everything else frozen) so the model
     prefers its own shortest correct reply over its longest (DPO against the unpushed model,
     NLL anchor on the short reply)

The push goes into the weights as a v_proj bias (steer.py), so standard Transformers and vLLM
load the model as is. Questions are split by id: directions use the "fit" ones, the layer
filter and the fresh rollouts use the "held" ones.

    python steer_dirs.py states --rollouts runs/bon/probe/rollouts.jsonl --out runs/steer
    python steer_dirs.py directions --out runs/steer
    python steer_dirs.py learn --out runs/steer --layer 14
    python steer_dirs.py build --out runs/steer --kind A --layer 14 --strength 2 --target steered/A-14-2
    python steer_dirs.py check --model steered/A-14-2
    python steer_dirs.py held --out runs/steer --data data/bon.jsonl --per-source 25
    python steer_dirs.py compare --out runs/steer --runs runs/steer/roll-A runs/steer/roll-B
"""

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from probe_curve import read_jsonl, write_jsonl

KIND = {"gsm8k_all": "math", "mbpp_train": "code", "toolace": "tool", "ifeval_like": "text",
        "arc_easy": "choice", "commonsense_qa": "choice"}
TYPES = ["math", "code", "tool", "text", "choice"]


def held(qid):
    """About 30% of questions are held out of the direction fitting."""
    return int(hashlib.md5(qid.encode()).hexdigest(), 16) % 10 < 3


def usable(r):
    return r["closed_thinking"] and not r["truncated"] and len(r["thinking_ids"]) >= 8


def groups_of(rollouts):
    by_question = defaultdict(list)
    for r in rollouts:
        if usable(r):
            by_question[r["id"]].append(r)
    return by_question


# ---------------------------------------------------------------- states

def states(args):
    """Residual stream after each layer, averaged over the first --window thinking tokens, per reply."""
    import torch
    from transformers import AutoModelForCausalLM

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    by_question = groups_of(read_jsonl(Path(args.rollouts)))
    per_kind = defaultdict(list)
    for qid, group in by_question.items():
        per_kind[KIND[group[0]["source"]]].append(qid)
    picked = []
    for kind, ids in per_kind.items():
        rng.shuffle(ids)
        for qid in ids[:args.per_kind]:
            group = by_question[qid]
            right = sorted((r for r in group if r["correct"]), key=lambda r: r["output_tokens"])
            wrong = [r for r in group if not r["correct"]]
            # the shortest and longest correct, one middle correct (for the trend) and one wrong
            chosen = right[:1] + right[len(right) // 2:len(right) // 2 + 1] + right[-1:] + wrong[:1]
            seen = set()
            for r in chosen:
                if r["sample"] not in seen:
                    seen.add(r["sample"])
                    picked.append(r)
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device).eval()
    vectors, meta = [], []
    with torch.no_grad():
        for n, r in enumerate(picked, 1):
            window = r["thinking_ids"][:args.window]
            ids = torch.tensor([r["prompt_ids"] + window], device=device)
            hidden = model(input_ids=ids, output_hidden_states=True).hidden_states[1:]  # after layers 0..27
            vectors.append(torch.stack([h[0, len(r["prompt_ids"]):].float().mean(0) for h in hidden]).half().cpu())
            meta.append({"id": r["id"], "kind": KIND[r["source"]], "correct": r["correct"],
                         "tokens": r["output_tokens"], "held": held(r["id"])})
            if n % 500 == 0:
                print(f"read {n}/{len(picked)} replies", flush=True)
    torch.save(torch.stack(vectors), out / "states.pt")
    write_jsonl(out / "states_meta.jsonl", meta)
    print(f"states: {len(picked)} replies from {len({m['id'] for m in meta})} questions", flush=True)


# ---------------------------------------------------------------- directions A and B, layer filter

def directions(args):
    import torch

    out = Path(args.out)
    X = torch.load(out / "states.pt").float()  # replies x layers x hidden
    meta = read_jsonl(out / "states_meta.jsonl")
    layers = X.shape[1]
    rows = defaultdict(list)
    for i, m in enumerate(meta):
        rows[m["id"]].append(i)

    def fit(kinds, split_held):
        """A, B (gives-up removed), per layer, from questions of these kinds in one split."""
        diffs, trend_num, trend_den, giveup = [], 0, 0.0, []
        for qid, idx in rows.items():
            m0 = meta[idx[0]]
            if m0["kind"] not in kinds or m0["held"] != split_held:
                continue
            right = sorted((i for i in idx if meta[i]["correct"]), key=lambda i: meta[i]["tokens"])
            wrong = [i for i in idx if not meta[i]["correct"]]
            if len(right) >= 2 and meta[right[-1]]["tokens"] >= args.ratio * meta[right[0]]["tokens"]:
                diffs.append(X[right[0]] - X[right[-1]])
            if len(right) >= 2:
                z = torch.tensor([math.log(meta[i]["tokens"]) for i in right])
                z = z - z.mean()
                x = X[right] - X[right].mean(0)
                trend_num = trend_num + (-z[:, None, None] * x).sum(0)
                trend_den += float((z ** 2).sum())
            if right and wrong:
                giveup.append(X[wrong].mean(0) - X[right].mean(0))
        if not diffs:
            return None
        a = torch.stack(diffs).mean(0)
        b = trend_num / max(trend_den, 1e-9)
        if giveup:
            g = torch.stack(giveup).mean(0)
            g = g / g.norm(dim=-1, keepdim=True)
            b = b - (b * g).sum(-1, keepdim=True) * g
        return {"A": a, "B": b, "pairs": len(diffs), "giveup": len(giveup)}

    def pairwise(direction, kinds):
        """Held-out questions: how often the shorter correct reply projects higher. 0.5 = no signal."""
        hits = torch.zeros(layers)
        total = 0
        for qid, idx in rows.items():
            m0 = meta[idx[0]]
            if m0["kind"] not in kinds or not m0["held"]:
                continue
            right = sorted((i for i in idx if meta[i]["correct"]), key=lambda i: meta[i]["tokens"])
            if len(right) >= 2 and meta[right[-1]]["tokens"] >= args.ratio * meta[right[0]]["tokens"]:
                hits += ((X[right[0]] - X[right[-1]]) * direction).sum(-1).gt(0).float()
                total += 1
        return hits / max(total, 1), total

    found = {"all": fit(set(TYPES), False)}
    for kind in TYPES:
        found[kind] = fit({kind}, False)
    result = {"norm_A": found["all"]["A"].norm(dim=-1).tolist(),
              "residual_norm": X.norm(dim=-1).mean(0).tolist(), "filter": {}, "cos_math": {}}
    lines = [f"pairs used: all {found['all']['pairs']}, " + ", ".join(
        f"{k} {found[k]['pairs'] if found[k] else 0}" for k in TYPES), "",
        "held-out pairwise accuracy (shorter correct reply projects higher; 0.50 = no signal)",
        f"{'layer':>5} {'A all':>6} {'B all':>6} " + " ".join(f"{'A ' + k:>8}" for k in TYPES)
        + f" {'|A|':>6} {'|h|':>6} {'cos math/rest':>14}"]
    rest = fit({k for k in TYPES if k != "math"}, False)
    scores = {}
    for kind_dir in ("A", "B"):
        scores[kind_dir], _ = pairwise(found["all"][kind_dir], set(TYPES))
    per_type = {k: pairwise(found["all"]["A"], {k})[0] for k in TYPES}
    for layer in range(layers):
        cos = float(torch.nn.functional.cosine_similarity(found["math"]["A"][layer], rest["A"][layer], dim=0)) \
            if found["math"] else float("nan")
        result["cos_math"][layer] = cos
        lines.append(f"{layer:>5} {scores['A'][layer]:>6.2f} {scores['B'][layer]:>6.2f} "
                     + " ".join(f"{per_type[k][layer]:>8.2f}" for k in TYPES)
                     + f" {result['norm_A'][layer]:>6.2f} {result['residual_norm'][layer]:>6.1f} {cos:>14.2f}")
    # candidate layers: best A filter score, skipping the first and last quarter (word meaning, next word)
    middle = list(range(layers // 4, layers - layers // 4))
    for kind_dir in ("A", "B"):
        result["filter"][kind_dir] = sorted(middle, key=lambda l: -float(scores[kind_dir][l]))[:args.top]
    lines += ["", f"candidate layers: A {result['filter']['A']}, B {result['filter']['B']}"]
    # per kind and across kinds, at the best layer: does each kind's own direction predict "short" on
    # the other kinds (held-out pairwise accuracy), and how alike are the directions (cosine)?
    layer = result["filter"]["A"][0]
    kinds = [k for k in TYPES if found[k]]
    result["transfer"], result["cosine"] = {}, {}
    for method in ("A", "B"):
        lines += ["", f"layer {layer}, direction {method} fitted on (row) -> held-out pairwise accuracy on (column)",
                  f"{'':>8}" + "".join(f"{k:>8}" for k in kinds) + f"{'ALL':>8}"]
        for fitted in kinds + ["all"]:
            cells = [float(pairwise(found[fitted][method], {k})[0][layer]) for k in kinds]
            cells.append(float(pairwise(found[fitted][method], set(TYPES))[0][layer]))
            result["transfer"][f"{method}:{fitted}"] = cells
            lines.append(f"{fitted:>8}" + "".join(f"{c:>8.2f}" for c in cells))
        lines += [f"cosine between kinds' {method} directions at layer {layer}",
                  f"{'':>8}" + "".join(f"{k:>8}" for k in kinds)]
        for a in kinds:
            cells = [float(torch.nn.functional.cosine_similarity(found[a][method][layer], found[b][method][layer], dim=0))
                     for b in kinds]
            result["cosine"][f"{method}:{a}"] = cells
            lines.append(f"{a:>8}" + "".join(f"{c:>8.2f}" for c in cells))
    held_pairs = {k: pairwise(found["all"]["A"], {k})[1] for k in TYPES}
    lines += ["", f"held-out pairs per kind: {held_pairs} (about +/-0.1 noise at 25 pairs)"]
    torch.save({"A": found["all"]["A"], "B": found["all"]["B"],
                **{f"{m}:{k}": found[k][m] for k in kinds for m in ("A", "B")}}, out / "directions.pt")
    (out / "directions.json").write_text(json.dumps(result, indent=1))
    (out / "directions.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


# ---------------------------------------------------------------- C: a learned vector

def learn(args):
    """Train one vector at --layer (all weights frozen): prefer the shortest correct reply over the longest."""
    import torch
    from transformers import AutoModelForCausalLM

    out = Path(args.out)
    rng = random.Random(0)
    norm = json.loads((out / "directions.json").read_text())["norm_A"][args.layer]
    by_question = groups_of(read_jsonl(Path(args.rollouts)))
    pairs = defaultdict(list)
    for qid, group in by_question.items():
        if held(qid):
            continue
        right = sorted((r for r in group if r["correct"]), key=lambda r: r["output_tokens"])
        if len(right) >= 2 and right[-1]["output_tokens"] >= args.ratio * right[0]["output_tokens"] \
                and len(right[0]["prompt_ids"]) + right[-1]["output_tokens"] <= args.max_len:
            pairs[KIND[group[0]["source"]]].append((right[0]["prompt_ids"], right[0]["output_ids"], right[-1]["output_ids"]))
    data = []
    for kind, items in pairs.items():  # balanced over kinds
        rng.shuffle(items)
        data += items[:args.pairs // len(TYPES)]
    rng.shuffle(data)
    print(f"learn: {len(data)} pairs at layer {args.layer}", flush=True)

    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device)
    model.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    vector = torch.zeros(model.config.hidden_size, device=device, requires_grad=True)
    state = {"on": True}

    def push(module, inputs, output):
        if not state["on"]:
            return output
        return (output[0] + vector.to(output[0].dtype),) + tuple(output[1:])
    model.model.layers[args.layer].self_attn.register_forward_hook(push)

    def logp(prompt, reply):
        ids = torch.tensor([prompt + reply], device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids).logits[0]
        total = 0.0
        for start in range(0, len(reply), 1024):
            pos = torch.arange(len(prompt) - 1 + start, len(prompt) - 1 + min(len(reply), start + 1024), device=device)
            lp = torch.log_softmax(logits[pos].float(), -1)
            total = total + lp.gather(1, torch.tensor(reply[start:start + 1024], device=device)[:, None]).sum()
        return total

    model.eval()
    state["on"] = False
    with torch.no_grad():
        refs = [(float(logp(p, c)), float(logp(p, r))) for p, c, r in data]
    state["on"] = True
    model.train()  # gradient checkpointing only runs in training mode; Qwen3 has no dropout
    optimizer = torch.optim.Adam([vector], lr=args.lr * norm / math.sqrt(vector.numel()))
    hits, losses = [], []
    for n, ((p, c, r), (rc, rr)) in enumerate(zip(data, refs), 1):
        pc, pr = logp(p, c), logp(p, r)
        margin = args.beta * ((pc - rc) - (pr - rr))
        loss = -torch.nn.functional.logsigmoid(margin) + args.nll * (-pc / len(c))
        (loss / args.batch).backward()
        hits.append(float(margin) > 0)
        losses.append(float(loss))
        if n % args.batch == 0 or n == len(data):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if (n // args.batch) % 5 == 0 or n == len(data):
                print(f"pairs {n}/{len(data)}  loss {sum(losses) / len(losses):.3f}  ordered right "
                      f"{sum(hits) / len(hits):.0%}  |v| {float(vector.norm()):.2f} (|A| {norm:.2f})", flush=True)
                hits, losses = [], []
    torch.save(vector.detach().cpu(), out / f"C-{args.layer}.pt")
    print(f"saved {out / f'C-{args.layer}.pt'}: |v| {float(vector.norm()):.2f}", flush=True)


# ---------------------------------------------------------------- build, check

def build(args):
    """A steered copy of the model. A and B: unit direction x strength x |A| at that layer. C: strength x v."""
    import torch
    from steer import steered_model
    from transformers import AutoTokenizer

    out = Path(args.out)
    if args.kind == "C":
        vector = torch.load(out / f"C-{args.layer}.pt") * args.strength
    else:
        direction = torch.load(out / "directions.pt")[args.kind][args.layer]
        norm = json.loads((out / "directions.json").read_text())["norm_A"][args.layer]
        vector = direction / direction.norm() * norm * args.strength
    model = steered_model(args.model, args.layer, vector, torch.bfloat16)
    model.save_pretrained(args.target)
    AutoTokenizer.from_pretrained(args.model).save_pretrained(args.target)
    print(f"built {args.target}: {args.kind} at layer {args.layer}, |push| {float(vector.norm()):.2f}", flush=True)


def check(args):
    """The steered model must give the same log-probabilities in vLLM as in Transformers (and differ from raw)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    text = tokenizer.apply_chat_template([{"role": "user", "content": "What is 17 times 23? Explain."}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=True)
    ids = tokenizer(text + "Okay, let me compute 17 times 23 step by step. First, 17 times 20 is 340.")["input_ids"]
    probe = Path(args.model) / "check_ids.json"
    probe.write_text(json.dumps(ids))
    subprocess.run([sys.executable, __file__, "vllm_logprobs", "--model", args.model, "--ids", str(probe)], check=True)
    vllm = json.loads((Path(args.model) / "check_vllm.json").read_text())

    def hf(name):
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("cuda").eval()
        with torch.no_grad():
            lp = torch.log_softmax(model(input_ids=torch.tensor([ids], device="cuda")).logits[0].float(), -1)
        values = [float(lp[i - 1, ids[i]]) for i in range(1, len(ids))]
        del model
        torch.cuda.empty_cache()
        return values
    steered, raw = hf(args.model), hf(args.raw)
    gap = sum(abs(a - b) for a, b in zip(vllm, steered)) / len(steered)
    effect = sum(abs(a - b) for a, b in zip(raw, steered)) / len(steered)
    ok = gap < max(0.03, 0.25 * effect)
    print(f"check {args.model}: vLLM vs Transformers {gap:.4f} per token; steered vs raw {effect:.4f} -> "
          f"{'OK' if ok else 'MISMATCH'}", flush=True)
    if not ok:
        raise SystemExit("vLLM does not apply the steering bias like Transformers")


def vllm_logprobs(args):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    ids = json.loads(Path(args.ids).read_text())
    llm = LLM(model=args.model, gpu_memory_utilization=0.3, enforce_eager=True, dtype="bfloat16")
    result = llm.generate(TokensPrompt(prompt_token_ids=ids), SamplingParams(max_tokens=1, prompt_logprobs=0))[0]
    values = [float(d[t].logprob) for d, t in zip(result.prompt_logprobs[1:], ids[1:])]
    (Path(args.model) / "check_vllm.json").write_text(json.dumps(values))


# ---------------------------------------------------------------- fresh replies on held-out questions

def held_data(args):
    """A data file of held-out questions (--per-source per source) for fresh rollouts with probe_curve.py."""
    rows = list(read_jsonl(Path(args.data)))
    seen = defaultdict(int)
    picked = []
    for qid in sorted({r["id"] for r in read_jsonl(Path(args.rollouts))}, key=lambda q: hashlib.md5(q.encode()).hexdigest()):
        source, index = qid.rsplit("-", 1)
        if held(qid) and seen[source] < args.per_source:
            seen[source] += 1
            picked.append(rows[int(index)])
    write_jsonl(Path(args.out) / "held.jsonl", picked)
    print(f"held-out questions: {dict(seen)}", flush=True)


def compare(args):
    """Fresh replies of each steered model vs the raw model's replies to the same held-out questions."""
    def load(path, samples=None):
        by_question = defaultdict(list)
        for r in read_jsonl(Path(path)):
            by_question[(r["source"], r["question"])].append(r)
        if samples:
            by_question = {k: sorted(v, key=lambda r: r["sample"])[:samples] for k, v in by_question.items()}
        return by_question
    runs = {Path(p).name: load(Path(p) / "rollouts.jsonl") for p in args.runs}
    keys = set.intersection(*(set(v) for v in runs.values()))
    samples = min(len(v) for run in runs.values() for v in run.values())
    raw = load(args.rollouts, samples)
    keys &= set(raw)
    runs = {"raw": raw, **runs}

    def row(name, by_question, kind):
        mine = [by_question[k] for k in keys if KIND[k[0]] == kind or kind == "ALL"]
        replies = [r for g in mine for r in g]
        best = [min(r["output_tokens"] for r in g if r["correct"]) for g in mine if any(r["correct"] for r in g)]
        return (sum(r["correct"] for r in replies) / max(1, len(replies)),
                sum(r["output_tokens"] for r in replies) / max(1, len(replies)),
                sum(best) / max(1, len(best)), len(best) / max(1, len(mine)))
    lines = [f"{len(keys)} held-out questions x {samples} replies; per kind: accuracy, mean tokens, "
             f"shortest correct of {samples} (mean), questions with a correct reply"]
    for kind in TYPES + ["ALL"]:
        lines.append(f"-- {kind}")
        for name, by_question in runs.items():
            acc, mean, best, cover = row(name, by_question, kind)
            lines.append(f"   {name:<14} acc {acc:6.1%}  mean {mean:6.0f}  shortest-correct {best:6.0f}  covered {cover:5.0%}")
    text = "\n".join(lines)
    (Path(args.out) / "compare.txt").write_text(text + "\n")
    print(text, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("states", "directions", "learn", "build", "check", "vllm_logprobs",
                                            "held", "compare"))
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--raw", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--rollouts", default="runs/bon/probe/rollouts.jsonl")
    parser.add_argument("--data", default="data/bon.jsonl")
    parser.add_argument("--out", default="runs/steer")
    parser.add_argument("--per-kind", type=int, default=200, help="states: questions per kind of task")
    parser.add_argument("--window", type=int, default=64, help="states: first thinking tokens averaged")
    parser.add_argument("--ratio", type=float, default=1.5, help="longest correct at least this much longer")
    parser.add_argument("--top", type=int, default=2, help="directions: candidate layers kept")
    parser.add_argument("--layer", type=int)
    parser.add_argument("--kind", choices=("A", "B", "C"))
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--target")
    parser.add_argument("--pairs", type=int, default=200, help="learn: pairs, balanced over kinds")
    parser.add_argument("--max-len", type=int, default=5000, help="learn: longest prompt + reply")
    parser.add_argument("--lr", type=float, default=0.05, help="learn: step per coordinate, in units of |A|/sqrt(d)")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--nll", type=float, default=1.0)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--ids")
    parser.add_argument("--per-source", type=int, default=25, help="held: fresh questions per source")
    parser.add_argument("--runs", nargs="*", default=[])
    args = parser.parse_args()
    {"states": states, "directions": directions, "learn": learn, "build": build, "check": check,
     "vllm_logprobs": vllm_logprobs, "held": held_data, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
