"""Hidden states for the probe zoo, read once per cut of a cal_cuts.py run (one forward pass, no generation).

Saves, per cut (float16, under --out, which is NOT synced to Drive: several GB):
  points.npy  [n, 28, 2, hidden]   every layer (1-28) at the last thinking token before "</think>" and at the last token
  window.npy  [n, len(--window-layers), --window, hidden]   the --window tokens just before "</think>" at a few layers

    python extract_feats.py --model models/rc1-rl-40 --cal runs/bon/calcheck-rc1rl40 --out /content/feats/calcheck-rc1rl40
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--cal", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--window-layers", default="14,21,28")
    parser.add_argument("--batch-tokens", type=int, default=12000)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    out = Path(args.out)
    if (out / "window.npy").exists():
        print(f"{out}: features exist", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    close_id = tok.convert_tokens_to_ids("</think>")
    rows = [json.loads(l) for l in (Path(args.cal) / "cuts.jsonl").read_text().splitlines() if l.strip()]
    seqs, close_pos = [], []
    for r in rows:
        seqs.append(r["prompt_ids"] + r["thinking_ids"])
        close_pos.append(len(r["prompt_ids"]) + r["thinking_ids"].index(close_id))
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype)).to("cuda").eval()
    n_layers, hidden = model.config.num_hidden_layers, model.config.hidden_size
    wl = [int(x) for x in args.window_layers.split(",")]
    points = np.lib.format.open_memmap(out / "points.tmp.npy", mode="w+", dtype=np.float16, shape=(len(rows), n_layers, 2, hidden))
    window = np.lib.format.open_memmap(out / "window.tmp.npy", mode="w+", dtype=np.float16,
                                       shape=(len(rows), len(wl), args.window, hidden))
    order = sorted(range(len(rows)), key=lambda k: len(seqs[k]))
    batch = []

    def flush():
        width = max(len(seqs[k]) for k in batch)
        ids = torch.zeros(len(batch), width, dtype=torch.long)
        mask = torch.zeros(len(batch), width, dtype=torch.long)
        for i, k in enumerate(batch):
            ids[i, :len(seqs[k])] = torch.tensor(seqs[k])
            mask[i, :len(seqs[k])] = 1
        with torch.no_grad():
            states = model.model(input_ids=ids.cuda(), attention_mask=mask.cuda(), output_hidden_states=True).hidden_states
        for i, k in enumerate(batch):
            before, last = close_pos[k] - 1, len(seqs[k]) - 1
            for layer in range(1, n_layers + 1):
                points[k, layer - 1, 0] = states[layer][i, before].float().cpu().numpy()
                points[k, layer - 1, 1] = states[layer][i, last].float().cpu().numpy()
            lo = max(0, close_pos[k] - args.window)
            for j, layer in enumerate(wl):
                w = states[layer][i, lo:close_pos[k]].float().cpu().numpy()
                window[k, j, args.window - len(w):] = w  # left-padded with zeros when the context is shorter
    for done, k in enumerate(order):
        if batch and (len(batch) + 1) * len(seqs[k]) > args.batch_tokens:
            flush()
            batch = []
        batch.append(k)
        if done % 1000 == 0:
            print(f"{out.name}: {done}/{len(order)}", flush=True)
    if batch:
        flush()
    points.flush()
    window.flush()
    del points, window
    (out / "points.tmp.npy").rename(out / "points.npy")
    (out / "window.tmp.npy").rename(out / "window.npy")
    print(f"{out}: {len(rows)} cuts, {n_layers} layers x 2 places, window {args.window} at layers {wl}", flush=True)


if __name__ == "__main__":
    main()
