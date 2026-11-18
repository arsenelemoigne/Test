"""Where do the saved thinking tokens come from: trimming (whole steps dropped) or condensing (same step, fewer tokens)?

For each sampled question we pair one original thinking trace with one trained-model trace (same sample index), split
both into steps (sentences), embed every step (all-MiniLM-L6-v2, mean pooling) and match new steps to original steps
one-to-one by cosine similarity (greedy, best pairs first, threshold --sim). Then, exactly:

    original tokens - new tokens = trimmed + condensed - added
    trimmed   = tokens of original steps with no match            (whole steps dropped)
    condensed = sum over matched pairs of (original step tokens - new step tokens)   (can be negative: a step got longer)
    added     = tokens of new steps with no match                 (new reasoning)

Step tokens are the trace's reasoning_tokens shared out by characters. A second measure, information per token, is the
mean surprisal (bits/token) of each thinking trace under a fixed reader (the ORIGINAL Qwen3-0.6B): filler is predictable
(low bits), condensed text is not. Pure trimming leaves bits/token flat; condensing raises it.

python trim_condense.py --base ~/Downloads/tc_full/base.jsonl --new ~/Downloads/tc_full/od5c16.jsonl --out runs/trimcond
"""
import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import zlib

TYPES = ["svamp", "mbpp", "ifeval", "bfcl_simple_python", "bfcl_multiple", "bfcl_irrelevance"]
NAME = {"svamp": "Math", "mbpp": "Code", "ifeval": "Instructions", "bfcl_simple_python": "One function",
        "bfcl_multiple": "Several functions", "bfcl_irrelevance": "No function fits"}


def thinking(text):
    m = re.search(r"<think>(.*?)(</think>|$)", text or "", re.S)
    return (m.group(1) if m else "").strip()


def steps(t):
    """Sentences; code lines and list items stay whole; fragments under 3 words join the next step."""
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", t) if p.strip()]
    out, buf = [], ""
    for p in parts:
        buf = (buf + " " + p).strip() if buf else p
        if len(buf.split()) >= 3:
            out.append(buf)
            buf = ""
    if buf:
        out.append(buf) if not out else out.__setitem__(-1, out[-1] + " " + buf)
    return out


def load(path, keep):
    rows = {}
    for line in open(path):
        r = json.loads(line)
        if r["dataset"] in keep and r["sample_index"] in (0, 1):
            rows[(r["dataset"], r["id"], r["sample_index"])] = r
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", default="runs/trimcond")
    ap.add_argument("--per-type", default="svamp=200,mbpp=200,ifeval=100,bfcl_simple_python=100,bfcl_multiple=100,bfcl_irrelevance=100")
    ap.add_argument("--sim", type=float, default=0.75, help="cosine threshold for 'the same step'")
    ap.add_argument("--sims", default="0.65,0.75,0.85", help="thresholds reported for sensitivity")
    ap.add_argument("--surprisal-n", type=int, default=0, help="traces per type per model scored for bits/token under Qwen3-0.6B (0 = skip; needs a GPU: it froze an 8 GB Mac)")
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
                    pairs.append((t, i, s, b, n))
    print(f"{len(pairs)} trace pairs", flush=True)

    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(2)
    dev = "cpu"  # (18 Nov) never MPS on the 8 GB Mac
    etok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    emod = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(dev).eval()

    @torch.no_grad()
    def embed(texts):
        vecs = []
        for k in range(0, len(texts), 128):
            enc = etok(texts[k:k + 128], padding=True, truncation=True, max_length=128, return_tensors="pt").to(dev)
            h = emod(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1)
            vecs.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
        return np.concatenate(vecs) if vecs else np.zeros((0, 384))

    def tokens_per_step(st, total):
        c = np.array([len(x) for x in st], float)
        return c / c.sum() * total if c.sum() else c

    sims = [float(x) for x in args.sims.split(",")]
    agg = {s: defaultdict(lambda: defaultdict(float)) for s in sims}
    examples = []
    for n_done, (t, i, s, b, n) in enumerate(pairs):
        tb, tn = thinking(b["text"]), thinking(n["text"])
        sb, sn = steps(tb), steps(tn)
        Tb, Tn = float(b.get("reasoning_tokens") or 0), float(n.get("reasoning_tokens") or 0)
        kb, kn = tokens_per_step(sb, Tb), tokens_per_step(sn, Tn)
        S = embed(sb) @ embed(sn).T if sb and sn else np.zeros((len(sb), len(sn)))
        for thr in sims:
            used_b, used_n, matched = set(), set(), []
            if S.size:
                order = np.dstack(np.unravel_index(np.argsort(-S, axis=None), S.shape))[0]
                for a, c in order:
                    if S[a, c] < thr:
                        break
                    if a in used_b or c in used_n:
                        continue
                    used_b.add(a); used_n.add(c); matched.append((a, c))
            trimmed = sum(kb[a] for a in range(len(sb)) if a not in used_b)
            added = sum(kn[c] for c in range(len(sn)) if c not in used_n)
            condensed = sum(kb[a] - kn[c] for a, c in matched)
            g = agg[thr]
            zb = len(zlib.compress(tb.encode(), 9)) * 8 / max(Tb, 1); zn = len(zlib.compress(tn.encode(), 9)) * 8 / max(Tn, 1)
            for key in (t, "all"):
                g[key]["zbits_b"] += zb * Tb; g[key]["zbits_n"] += zn * Tn
                g[key]["n"] += 1; g[key]["Tb"] += Tb; g[key]["Tn"] += Tn
                g[key]["trimmed"] += trimmed; g[key]["condensed"] += condensed; g[key]["added"] += added
                g[key]["steps_b"] += len(sb); g[key]["steps_n"] += len(sn); g[key]["matched"] += len(matched)
                g[key]["tok_matched_b"] += sum(kb[a] for a, _ in matched); g[key]["tok_matched_n"] += sum(kn[c] for _, c in matched)
            if thr == args.sim and t in ("svamp", "mbpp") and len(examples) < 6 and s == 0 and matched:
                a, c = max(matched, key=lambda p: kb[p[0]] - kn[p[1]])
                examples.append({"type": t, "id": i, "orig_step": sb[a], "new_step": sn[c], "sim": round(float(S[a, c]), 3),
                                 "trimmed_example": next((sb[x] for x in range(len(sb)) if x not in used_b), "")})
        if n_done % 200 == 0:
            print(f"  {n_done}/{len(pairs)}", flush=True)

    # ---- bits per thinking token under the original model (one trace per question, sample 0); GPU only
    if args.surprisal_n <= 0:
        TYPES_S = []
    else:
        TYPES_S = TYPES
    qtok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B") if TYPES_S else None

    from transformers import AutoModelForCausalLM
    qmod = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.float32).to("cuda").eval() if TYPES_S else None

    @torch.no_grad()
    def bits(text):
        ids = qtok("<think>\n" + text, return_tensors="pt").input_ids[:, :3000].to("cuda")
        if ids.shape[1] < 3:
            return None, 0
        logits = qmod(ids).logits[0, :-1].float()
        lp = torch.log_softmax(logits, -1).gather(1, ids[0, 1:, None])[:, 0]
        return float(-lp.sum() / math.log(2)), ids.shape[1] - 1

    surp = defaultdict(lambda: defaultdict(float))
    for t in TYPES_S:
        chosen = [p for p in pairs if p[0] == t and p[2] == 0][:args.surprisal_n]
        for (_, _, _, b, n) in chosen:
            for who, r in (("base", b), ("new", n)):
                tot, k = bits(thinking(r["text"]))
                if tot is not None:
                    for key in (t, "all"):
                        surp[key][who + "_bits"] += tot; surp[key][who + "_tok"] += k
        print(f"  surprisal {t} done", flush=True)

    report = {"sim_main": args.sim, "by_threshold": {}, "surprisal": {}, "examples": examples}
    for thr in sims:
        rep = {}
        for key, g in agg[thr].items():
            saved = g["Tb"] - g["Tn"]
            rep[key] = {"pairs": int(g["n"]), "think_orig": g["Tb"] / g["n"], "think_new": g["Tn"] / g["n"],
                        "saved": saved / g["n"], "trimmed": g["trimmed"] / g["n"], "condensed": g["condensed"] / g["n"],
                        "added": g["added"] / g["n"], "share_trim": g["trimmed"] / saved if saved else None,
                        "share_condense": g["condensed"] / saved if saved else None, "share_added": -g["added"] / saved if saved else None,
                        "steps_orig": g["steps_b"] / g["n"], "steps_new": g["steps_n"] / g["n"],
                        "matched_steps": g["matched"] / g["n"],
                        "tok_per_matched_step_orig": g["tok_matched_b"] / max(g["matched"], 1),
                        "tok_per_matched_step_new": g["tok_matched_n"] / max(g["matched"], 1),
                        "gzip_bits_per_token_orig": g["zbits_b"] / max(g["Tb"], 1), "gzip_bits_per_token_new": g["zbits_n"] / max(g["Tn"], 1)}
        report["by_threshold"][str(thr)] = rep
    for key, v in surp.items():
        report["surprisal"][key] = {"bits_per_token_orig": v["base_bits"] / max(v["base_tok"], 1),
                                    "bits_per_token_new": v["new_bits"] / max(v["new_tok"], 1)}
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    r = report["by_threshold"][str(args.sim)]
    print(f"\n== trimming vs condensing (match threshold {args.sim}; per trace pair, thinking tokens)")
    for key in ["all"] + TYPES:
        if key in r:
            x = r[key]
            print(f"   {NAME.get(key, 'All'):18s} {x['think_orig']:6.0f} -> {x['think_new']:5.0f} (saved {x['saved']:5.0f}): "
                  f"trimmed {x['trimmed']:5.0f} ({100 * x['share_trim']:3.0f}%), condensed {x['condensed']:5.0f} "
                  f"({100 * x['share_condense']:3.0f}%), added -{x['added']:4.0f}; steps {x['steps_orig']:.1f} -> {x['steps_new']:.1f}, "
                  f"tokens per matched step {x['tok_per_matched_step_orig']:.1f} -> {x['tok_per_matched_step_new']:.1f}; "
                  f"compressed bits/token {x['gzip_bits_per_token_orig']:.2f} -> {x['gzip_bits_per_token_new']:.2f}")
    print("== sensitivity (share trimmed / condensed, all types):",
          {k: (round(v['all']['share_trim'], 2), round(v['all']['share_condense'], 2)) for k, v in report["by_threshold"].items()})
    print("== bits per thinking token under the original model:")
    for key in ["all"] + TYPES:
        if key in report["surprisal"]:
            x = report["surprisal"][key]
            print(f"   {NAME.get(key, 'All'):18s} {x['bits_per_token_orig']:.3f} -> {x['bits_per_token_new']:.3f}")


if __name__ == "__main__":
    main()
