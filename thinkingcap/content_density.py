"""Is the trained model's thinking DENSER, beyond being shorter? (user 18 Nov; CPU, light: MiniLM on 2 threads.)

Same pairs and step matching as trim_condense.py (one original and one trained trace per question and sample index;
sentences; one-to-one MiniLM matching at cosine >= 0.75). Two measures:

1. Content per token on MATCHED steps (the same step written by both models). Reasoning content markers per 100 tokens:
   numbers, arithmetic / comparison operators, code (identifiers followed by "(", Python keywords, assignments), quoted
   strings and key=value arguments (tool calls), versus hesitation / restatement words. Same length with more markers
   would mean denser steps. Also over whole traces and over the first 100 tokens (position-matched).
2. Semantic coverage (merging): each original step goes to its most similar trained step (cosine >= threshold); repeats of
   one step among them (mutually similar originals) are trimming and count once. If a trained step covers several DISTINCT
   original steps with fewer tokens than their sum, that difference is "merging", an upper bound on condensing that
   one-to-one matching books as trimming.

python content_density.py --base ~/Downloads/tc_full/base.jsonl --new ~/Downloads/tc_full/od5c16.jsonl --out runs/density_content
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from trim_condense import NAME, TYPES, load, steps, thinking

NUM = re.compile(r"(?<![A-Za-z_])-?\d+(?:[.,]\d+)?")
OP = re.compile(r"(?<![<>=!])(?:[+\-*/×÷^%]|==|<=|>=|!=|=|<|>)(?![=])")
CODE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\s*\(|\b(?:def|return|for|while|if|elif|else|in|import|lambda|range|len|list|dict)\b")
ARG = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\s*=\s*[^=\s]|\"[^\"\n]{1,60}\"|'[^'\n]{1,60}'")
HES = re.compile(r"\b(?:wait|hmm|let me (?:check|verify|make sure|think|re-?check|double[- ]check|confirm|see)|"
                 r"actually|alternatively|but wait|double[- ]check|to confirm|again|okay|so,? the answer is)\b", re.I)


def markers(text):
    return {"numbers": len(NUM.findall(text)), "operators": len(OP.findall(text)), "code": len(CODE.findall(text)),
            "arguments": len(ARG.findall(text)), "hesitation": len(HES.findall(text))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", default="runs/density_content")
    ap.add_argument("--per-type", default="svamp=200,mbpp=200,ifeval=100,bfcl_simple_python=100,bfcl_multiple=100,bfcl_irrelevance=100")
    ap.add_argument("--sim", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    quota = {k: int(v) for k, v in (x.split("=") for x in args.per_type.split(","))}
    base, new = load(args.base, quota), load(args.new, quota)
    rnd = random.Random(args.seed)
    pairs = []
    for t in TYPES:
        ids = sorted({i for (d, i, s) in base if d == t})
        rnd.shuffle(ids)
        for i in ids[:quota.get(t, 0)]:
            for s in (0, 1):
                b, n = base.get((t, i, s)), new.get((t, i, s))
                if b and n:
                    pairs.append((t, b, n))
    print(f"{len(pairs)} trace pairs", flush=True)
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(2)
    etok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    emod = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").eval()

    @torch.no_grad()
    def embed(texts):
        vecs = []
        for k in range(0, len(texts), 128):
            enc = etok(texts[k:k + 128], padding=True, truncation=True, max_length=128, return_tensors="pt")
            h = emod(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            vecs.append(torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1).numpy())
        return np.concatenate(vecs) if vecs else np.zeros((0, 384))

    def tok_share(st, total):
        c = np.array([len(x) for x in st], float)
        return c / c.sum() * total if c.sum() else c

    agg = defaultdict(lambda: defaultdict(float))
    for n_done, (t, b, n) in enumerate(pairs):
        tb, tn = thinking(b["text"]), thinking(n["text"])
        sb, sn = steps(tb), steps(tn)
        Tb, Tn = float(b.get("reasoning_tokens") or 0), float(n.get("reasoning_tokens") or 0)
        if not sb or not sn or not Tb or not Tn:
            continue
        kb, kn = tok_share(sb, Tb), tok_share(sn, Tn)
        Eb = embed(sb)
        S = Eb @ embed(sn).T
        Sbb = Eb @ Eb.T
        used_b, used_n, matched = set(), set(), []
        order = np.dstack(np.unravel_index(np.argsort(-S, axis=None), S.shape))[0]
        for a, c in order:
            if S[a, c] < args.sim:
                break
            if a in used_b or c in used_n:
                continue
            used_b.add(a); used_n.add(c); matched.append((a, c))
        # first 100 tokens of each trace (by character share)
        def head(t, total):
            return t[: max(1, int(len(t) * min(1.0, 100 / total)))]
        for key in (t, "all"):
            g = agg[key]
            g["pairs"] += 1
            for side, txt, tot in (("b_all", tb, Tb), ("n_all", tn, Tn), ("b_head", head(tb, Tb), min(100, Tb)), ("n_head", head(tn, Tn), min(100, Tn))):
                for m_, v in markers(txt).items():
                    g[f"{side}_{m_}"] += v
                g[f"{side}_tokens"] += tot
            for a, c in matched:
                for m_, v in markers(sb[a]).items():
                    g[f"b_match_{m_}"] += v
                for m_, v in markers(sn[c]).items():
                    g[f"n_match_{m_}"] += v
                g["b_match_tokens"] += kb[a]; g["n_match_tokens"] += kn[c]; g["matched"] += 1
            # many-to-one coverage: each trained step absorbs every original step it matches (each original step counted once,
            # by its best trained step)
            best_c = S.argmax(axis=1)
            cover = defaultdict(list)
            for a in range(len(sb)):
                if S[a, best_c[a]] >= args.sim:
                    cover[int(best_c[a])].append(a)
            merged = 0.0
            for c, As in cover.items():
                # repeats of one step (mutually similar originals) are trimming, not merging: keep a set of DISTINCT
                # original steps (pairwise cosine < sim), greedily from the longest
                distinct = []
                for a in sorted(As, key=lambda x: -kb[x]):
                    if all(Sbb[a, d] < args.sim for d in distinct):
                        distinct.append(a)
                if len(distinct) >= 2:
                    merged += max(0.0, sum(kb[a] for a in distinct) - kn[c])
            g["merged_tokens"] += merged
            g["saved_tokens"] += Tb - Tn
            g["covered_b_tokens"] += sum(kb[a] for As in cover.values() for a in As)
            g["b_tokens"] += Tb
        if n_done % 200 == 0:
            print(f"  {n_done}/{len(pairs)}", flush=True)

    rep = {}
    lines = ["== content markers per 100 thinking tokens (original -> trained)"]
    for key in ["all"] + list(TYPES):
        if key not in agg:
            continue
        g = agg[key]
        r = {}
        for scope in ("match", "all", "head"):
            for m_ in ("numbers", "operators", "code", "arguments", "hesitation"):
                bt, nt = g[f"b_{scope}_tokens"], g[f"n_{scope}_tokens"]
                r[f"{scope}_{m_}"] = (100 * g[f"b_{scope}_{m_}"] / bt if bt else None, 100 * g[f"n_{scope}_{m_}"] / nt if nt else None)
        r["merged_share_of_saving"] = g["merged_tokens"] / g["saved_tokens"] if g["saved_tokens"] else None
        r["tokens_per_matched_step"] = (g["b_match_tokens"] / g["matched"], g["n_match_tokens"] / g["matched"]) if g["matched"] else None
        rep[key] = r
        lines.append(f"-- {NAME.get(key, 'All')}: tokens per matched step {r['tokens_per_matched_step'][0]:.1f} -> "
                     f"{r['tokens_per_matched_step'][1]:.1f}; merging (many-to-one) = {100 * (r['merged_share_of_saving'] or 0):.1f}% of the saving")
        for scope, lab in (("match", "matched steps"), ("head", "first 100 tokens"), ("all", "whole traces")):
            lines.append(f"   {lab:16s} " + ", ".join(f"{m_} {r[f'{scope}_{m_}'][0]:.1f} -> {r[f'{scope}_{m_}'][1]:.1f}"
                                                      for m_ in ("numbers", "operators", "code", "arguments", "hesitation")))
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
