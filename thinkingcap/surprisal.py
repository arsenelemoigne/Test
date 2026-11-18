"""Information per thinking token (consult30 priority 2, user 18 Nov): measurement only, GPU.

Teacher-forced log-probabilities of every thinking token of two models' natural answers (Dev-512 rollouts from
probe_curve.py, --samples), each scored under BOTH models (the original = a fixed reader; the trained model = itself).
Reports per type, for each (trace author, reader): mean bits per thinking token, the same over the first 100 thinking
tokens only (prefix-matched, so length does not bias it), and a 10-bin profile over the trace (bits/token by position
decile: a repetitive tail shows as low bits late in the trace). Logits are computed in chunks so long traces fit on an L4.

    python surprisal.py --a base=Qwen/Qwen3-0.6B:runs/bon/dev512-base/rollouts.jsonl \
        --b od5c-best=models/od5c-best:runs/bon/dev512-od5c-best/rollouts.jsonl --out runs/eff/surprisal.json
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from build_pool import part_of
from probe_curve import read_jsonl


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--a", required=True, help="name=model:rollouts")
    ap.add_argument("--b", required=True, help="name=model:rollouts")
    ap.add_argument("--samples", default="0")
    ap.add_argument("--max-tokens", type=int, default=6000, help="prompt + thinking tokens scored per trace (longer are cut)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    keep = {int(x) for x in args.samples.split(",")}
    specs = {}
    for s in (args.a, args.b):
        name, rest = s.split("=", 1)
        model, roll = rest.split(":", 1)
        specs[name] = (model, roll)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    open_id, close_id = tok.convert_tokens_to_ids("<think>"), tok.convert_tokens_to_ids("</think>")
    traces = {}  # author -> list of (part, prompt_ids, thinking_ids)
    for author, (_, roll) in specs.items():
        per_q = defaultdict(int)
        rows = []
        for r in read_jsonl(Path(roll)):
            k = per_q[r["id"]]
            per_q[r["id"]] += 1
            if k not in keep or not r.get("prompt_ids"):
                continue
            th = [t for t in (r.get("thinking_ids") or []) if t not in (open_id, close_id)]
            if th:
                rows.append((part_of(r), list(r["prompt_ids"]) + [open_id], th))
        traces[author] = rows
        print(f"surprisal: {author}: {len(rows)} traces", flush=True)
    acc = defaultdict(lambda: defaultdict(float))  # (author, reader, part) -> sums
    for reader, (model, _) in specs.items():
        m = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16).to("cuda").eval()
        with torch.no_grad():
            for author, rows in traces.items():
                for n, (p, pr, th) in enumerate(rows):
                    ids = (pr + th)[:args.max_tokens]
                    n_th = len(ids) - len(pr)
                    if n_th <= 1:
                        continue
                    x = torch.tensor([ids], device="cuda")
                    h = m.model(x).last_hidden_state[0]
                    bits = []
                    start = len(pr) - 1  # position predicting the first thinking token
                    for a in range(start, len(ids) - 1, 1024):
                        b = min(a + 1024, len(ids) - 1)
                        lp = torch.log_softmax(m.lm_head(h[a:b]).float(), -1)
                        tgt = x[0, a + 1:b + 1]
                        bits += (-lp.gather(1, tgt[:, None])[:, 0] / math.log(2)).tolist()
                    for key in (p, "all"):
                        d = acc[(author, reader, key)]
                        d["bits"] += sum(bits)
                        d["tokens"] += len(bits)
                        d["bits100"] += sum(bits[:100])
                        d["tokens100"] += min(100, len(bits))
                        for i, v in enumerate(bits):
                            dec = min(9, i * 10 // len(bits))
                            d[f"dec{dec}"] += v
                            d[f"decn{dec}"] += 1
                    if n % 200 == 0:
                        print(f"surprisal: reader {reader}, author {author}: {n}/{len(rows)}", flush=True)
        del m
        torch.cuda.empty_cache()
    res = {}
    lines = ["bits per thinking token (author's traces, read by reader): all / first 100 tokens / profile by position decile"]
    for (author, reader, p), d in sorted(acc.items()):
        prof = [d[f"dec{i}"] / d[f"decn{i}"] if d[f"decn{i}"] else None for i in range(10)]
        res[f"{author}|{reader}|{p}"] = {"bits_per_token": d["bits"] / d["tokens"], "bits_per_token_first100": d["bits100"] / d["tokens100"],
                                         "profile": prof, "tokens": d["tokens"]}
        lines.append(f"   {p:20s} author {author:10s} reader {reader:10s}: {d['bits'] / d['tokens']:.3f} / "
                     f"{d['bits100'] / d['tokens100']:.3f} / " + " ".join(f"{v:.2f}" for v in prof if v is not None))
    Path(args.out).write_text(json.dumps(res, indent=1))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
