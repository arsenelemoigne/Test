"""Could condensing hide in "net trimming"? (user 18 Nov) CPU, light.

trim_condense.py books a trained step that rewrites an original step shorter AND differently (below the match threshold) as
"original trimmed + new added", so its saving lands in net trimming. This checks the UNMATCHED trained steps (no one-to-one
match at cosine >= 0.75) on the same 1,600 pairs:
  1. their best similarity to ANY original step of the same answer (distribution), and their length against that closest
     original step: many close (0.55-0.75) and much shorter would point to hidden condensing;
  2. an upper bound: for each unmatched trained step whose closest original step is ALSO unmatched and similar >= --loose,
     count (original tokens - trained tokens) as possible hidden condensing (each original step used once);
  3. content per 100 tokens (numbers, operators, code, arguments, hesitation) of unmatched trained steps against unmatched
     original steps and matched steps: condensed rewrites should carry MORE content per token.

python hidden_condense.py --base ~/Downloads/tc_full/base.jsonl --new ~/Downloads/tc_full/od5c16.jsonl --out runs/hidden_condense
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from content_density import markers
from trim_condense import NAME, TYPES, load, steps, thinking


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", default="runs/hidden_condense")
    ap.add_argument("--per-type", default="svamp=200,mbpp=200,ifeval=100,bfcl_simple_python=100,bfcl_multiple=100,bfcl_irrelevance=100")
    ap.add_argument("--sim", type=float, default=0.75)
    ap.add_argument("--loose", type=float, default=0.55)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    quota = {k: int(v) for k, v in (x.split("=") for x in args.per_type.split(","))}
    base, new = load(args.base, quota), load(args.new, quota)
    rnd = random.Random(0)
    pairs = []
    for t in TYPES:
        ids = sorted({i for (d, i, s) in base if d == t})
        rnd.shuffle(ids)
        for i in ids[:quota.get(t, 0)]:
            for s in (0, 1):
                b, n = base.get((t, i, s)), new.get((t, i, s))
                if b and n:
                    pairs.append((t, b, n))
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
        return np.concatenate(vecs)

    def share(st, total):
        c = np.array([len(x) for x in st], float)
        return c / c.sum() * total

    bins = [0, 0.35, 0.45, 0.55, 0.65, 0.75]
    agg = defaultdict(lambda: defaultdict(float))
    for n_done, (t, b, n) in enumerate(pairs):
        tb, tn = thinking(b["text"]), thinking(n["text"])
        sb, sn = steps(tb), steps(tn)
        Tb, Tn = float(b.get("reasoning_tokens") or 0), float(n.get("reasoning_tokens") or 0)
        if not sb or not sn or not Tb or not Tn:
            continue
        kb, kn = share(sb, Tb), share(sn, Tn)
        S = embed(sb) @ embed(sn).T
        used_b, used_n = set(), set()
        order = np.dstack(np.unravel_index(np.argsort(-S, axis=None), S.shape))[0]
        for a, c in order:
            if S[a, c] < args.sim:
                break
            if a in used_b or c in used_n:
                continue
            used_b.add(a); used_n.add(c)
        un_n = [c for c in range(len(sn)) if c not in used_n]
        un_b = [a for a in range(len(sb)) if a not in used_b]
        taken = set()
        for key in (t, "all"):
            g = agg[key]
            g["saved"] += Tb - Tn
            g["new_tokens"] += Tn
            g["unmatched_new_tokens"] += sum(kn[c] for c in un_n)
            for c in un_n:
                best = int(np.argmax(S[:, c]))
                sim = float(S[best, c])
                for lo, hi in zip(bins, bins[1:]):
                    if lo <= sim < hi:
                        g[f"bin_{lo}"] += kn[c]
                if sim >= args.loose:
                    g["close_new_tokens"] += kn[c]
                    g["close_orig_tokens"] += kb[best]
                if key == t and sim >= args.loose and best not in used_b and best not in taken:
                    taken.add(best)
                    gain = max(0.0, kb[best] - kn[c])
                    agg[t]["hidden_upper"] += gain
                    agg["all"]["hidden_upper"] += gain
            for side, idx, st, tok in (("un_new", un_n, sn, kn), ("un_orig", un_b, sb, kb), ("m_new", sorted(used_n), sn, kn),
                                       ("m_orig", sorted(used_b), sb, kb)):
                for i in idx:
                    for m_, v in markers(st[i]).items():
                        g[f"{side}_{m_}"] += v
                    g[f"{side}_tokens"] += tok[i]
        if n_done % 200 == 0:
            print(f"  {n_done}/{len(pairs)}", flush=True)
    rep, lines = {}, []
    for key in ["all"] + list(TYPES):
        g = agg[key]
        if not g["saved"]:
            continue
        r = {"unmatched_share_of_new": g["unmatched_new_tokens"] / g["new_tokens"],
             "best_sim_bins": {f"{lo}-{hi}": g[f"bin_{lo}"] / g["unmatched_new_tokens"] for lo, hi in zip(bins, bins[1:])},
             "close_len_ratio": g["close_new_tokens"] / g["close_orig_tokens"] if g["close_orig_tokens"] else None,
             "hidden_condense_upper_share_of_saving": g["hidden_upper"] / g["saved"]}
        for side in ("un_new", "un_orig", "m_new", "m_orig"):
            r[side] = {m_: 100 * g[f"{side}_{m_}"] / g[f"{side}_tokens"] for m_ in ("numbers", "operators", "code", "arguments", "hesitation")} if g[f"{side}_tokens"] else None
        rep[key] = r
        lines.append(f"-- {NAME.get(key, 'All')}: unmatched = {100 * r['unmatched_share_of_new']:.0f}% of the trained thinking; "
                     f"their best similarity to any original step: " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in r["best_sim_bins"].items()))
        lines.append(f"   those >= {args.loose}: length vs their closest original step {r['close_len_ratio'] or 0:.2f}x; "
                     f"UPPER BOUND of hidden condensing = {100 * r['hidden_condense_upper_share_of_saving']:.1f}% of the saving")
        for side, lab in (("un_new", "unmatched trained"), ("un_orig", "unmatched original"), ("m_new", "matched trained"), ("m_orig", "matched original")):
            if r[side]:
                lines.append(f"   content per 100 tokens, {lab:18s}: " + ", ".join(f"{m_} {v:.1f}" for m_, v in r[side].items()))
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
