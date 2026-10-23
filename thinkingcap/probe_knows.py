"""Does the model know when it already has the answer? A linear probe on its hidden state at each cut (P(IK)-style).

Reuses a cal_cuts.py run: the same correct replies, each cut at 1-80% of its thinking, every cut labelled by the real
outcome (right re-answers out of 20 from that cut). For every cut the model reads the prompt and the cut thinking once,
and its hidden state is kept at two places: the last thinking token before "</think>" (where it would decide to stop)
and the last token after "</think>\\n\\n" (where it starts answering), at a few layers.

Labels: safe = 20/20 right from the cut; unsafe = right <= --unsafe (15/20); cuts in between are left out.
A logistic regression per (layer, place) is trained with 5-fold cross-validation grouped by question (a question's cuts
never sit on both sides), and compared with three baselines that need no hidden state: the cut position alone
(fraction + data type), the model's teacher-forced answer confidence (cal_cuts.py), and the position + the probe.

The practical test, on held-out folds only: a stopper that ends each reply at its earliest cut where the probe says
"safe" with p >= t, t chosen per data type on the training folds as the most saving within the type's allowance
(math 0.5 pt, code 1, others 2, as cal_cuts.py). Its saving and real loss are set beside the same stopper driven by the
confidence, by the position alone (one fixed cut fraction per type), and by the oracle (earliest cut right 20/20).

    python probe_knows.py --model models/ca1-soup-10 --cal runs/bon/pairs-rc1/cal \\
        --rollouts runs/bon/probe-rc1/rollouts.jsonl runs/bon/kprobe-rc1/rollouts.jsonl \\
        --keep runs/bon/select-rc1/sft.jsonl runs/bon/kselect-rc1/sft.jsonl --out runs/bon/probeknow-ca1
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from cal_cuts import cut_text
from probe_curve import read_jsonl
from prune_traces import thinking_text


def replies_and_jobs(args, tokenizer):
    """The same replies and cuts, in the same order, as cal_cuts.py built them (checked against its cuts.jsonl)."""
    open_id = tokenizer.convert_tokens_to_ids("<think>")
    allowed = {r["id"] for k in args.keep for r in read_jsonl(Path(k))}
    by_question = defaultdict(list)
    for path in args.rollouts:
        for r in read_jsonl(Path(path)):
            if r["id"] in allowed and r["correct"] and r["closed_thinking"] and not r["truncated"]:
                by_question[r["id"]].append(r)
    replies = [r for group in by_question.values() for r in sorted(group, key=lambda r: -r["output_tokens"])[:1]]
    jobs = []
    for i, r in enumerate(replies):
        text = tokenizer.decode([t for t in r["thinking_ids"] if t != open_id]).strip()
        for f in args.fractions:
            jobs.append((i, f, tokenizer.encode(thinking_text([cut_text(text, f)]), add_special_tokens=False)))
    return replies, jobs


def hidden_features(model_path, sequences, positions, layers, dtype, batch_tokens):
    """[n, len(layers), len(positions[0]), hidden] float16: the hidden state at each asked position and layer."""
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=getattr(torch, dtype)).to("cuda").eval()
    order = sorted(range(len(sequences)), key=lambda k: len(sequences[k]))
    out = np.zeros((len(sequences), len(layers), len(positions[0]), model.config.hidden_size), dtype=np.float16)
    batch = []
    def flush():
        width = max(len(sequences[k]) for k in batch)
        ids = torch.zeros(len(batch), width, dtype=torch.long)
        mask = torch.zeros(len(batch), width, dtype=torch.long)
        for row, k in enumerate(batch):
            ids[row, :len(sequences[k])] = torch.tensor(sequences[k])
            mask[row, :len(sequences[k])] = 1
        with torch.no_grad():
            # the transformer body only: the vocabulary-wide logits of a 16k-token batch would take ~5 GB
            states = model.model(input_ids=ids.cuda(), attention_mask=mask.cuda(), output_hidden_states=True).hidden_states
        for row, k in enumerate(batch):
            for li, layer in enumerate(layers):
                for pi, pos in enumerate(positions[k]):
                    out[k, li, pi] = states[layer][row, pos].float().cpu().numpy()
    for done, k in enumerate(order):
        if batch and (len(batch) + 1) * len(sequences[k]) > batch_tokens:
            flush()
            batch = []
        batch.append(k)
        if done % 1000 == 0:
            print(f"hidden states: {done}/{len(order)}", flush=True)
    if batch:
        flush()
    return out


def auc(scores, labels):
    """Area under the ROC curve (ties counted half), without sklearn."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = scores[labels], scores[~labels]
    if not len(pos) or not len(neg):
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]), kind="mergesort"), kind="mergesort").astype(float)
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):  # average ranks of ties
        hit = allv == v
        if hit.sum() > 1:
            ranks[hit] = ranks[hit].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


def fit_predict(train_x, train_y, test_x, l2=1.0, steps=300):
    """Standardized L2 logistic regression (full-batch LBFGS in torch, on the GPU when there is one)."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mu, sd = train_x.mean(0), train_x.std(0) + 1e-4
    x = torch.tensor((train_x - mu) / sd, dtype=torch.float32, device=device)
    y = torch.tensor(train_y, dtype=torch.float32, device=device)
    w = torch.zeros(x.shape[1], device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(x @ w + b, y) + l2 * (w * w).sum() / len(y)
        loss.backward()
        return loss
    opt.step(closure)
    tx = torch.tensor((test_x - mu) / sd, dtype=torch.float32, device=device)
    with torch.no_grad():
        return torch.sigmoid(tx @ w + b).cpu().numpy()


def stopper(replies_cuts, key, threshold):
    """Stop each reply at its earliest cut whose `key` >= threshold (else keep it whole): (saved share, loss points)."""
    saved, success = [], []
    for total, cuts in replies_cuts:
        hit = next((c for c in cuts if c[key] >= threshold), None)
        saved.append(0.0 if hit is None else 1 - hit["size"] / total)
        success.append(1.0 if hit is None else hit["success"])
    return float(np.mean(saved)), 100 * (1 - float(np.mean(success)))


def pick(replies_cuts, key, grid, allow):
    """The threshold saving most within the allowance (on these replies), or None."""
    best = None
    for t in grid:
        s, loss = stopper(replies_cuts, key, t)
        if loss <= allow and (best is None or s > best[1]):
            best = (t, s)
    return None if best is None else best[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, help="the model whose hidden states are read (merged weights)")
    parser.add_argument("--cal", required=True, help="a cal_cuts.py output folder: cuts, verdicts, confidence")
    parser.add_argument("--rollouts", nargs="+", required=True)
    parser.add_argument("--keep", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fractions", default="0.01,0.05,0.1,0.2,0.35,0.5,0.65,0.8")
    parser.add_argument("--layers", default="7,14,21,28", help="hidden-state layers read (0 = embeddings)")
    parser.add_argument("--unsafe", type=int, default=15, help="right re-answers (of 20) at or below which a cut is unsafe")
    parser.add_argument("--allow", default="gsm_plus=0.5,gsm_ic=0.5,kodcode=1,default=2")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-tokens", type=int, default=16000)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    args.fractions = [float(f) for f in args.fractions.split(",")]
    from transformers import AutoTokenizer
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cal = Path(args.cal)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    close_id = tokenizer.convert_tokens_to_ids("</think>")

    replies, jobs = replies_and_jobs(args, tokenizer)
    cuts_file = [json.loads(l) for l in (cal / "cuts.jsonl").read_text().splitlines() if l.strip()]
    assert len(cuts_file) == len(jobs) and all(c["thinking_ids"] == j[2] for c, j in zip(cuts_file, jobs)), \
        "these replies and cuts are not the ones of --cal"
    verdict = {v["key"]: v for v in read_jsonl(cal / "verdicts.jsonl")}
    conf = {c["key"]: c["confidence"] for c in read_jsonl(cal / "confidence.jsonl")}
    of = max(v["of"] for v in verdict.values())

    sequences, positions, meta = [], [], []
    for k, (i, f, t) in enumerate(jobs):
        r, v = replies[i], verdict[k]
        close = t.index(close_id)
        sequences.append(r["prompt_ids"] + t)
        positions.append([len(r["prompt_ids"]) + close - 1, len(r["prompt_ids"]) + len(t) - 1])
        size = len(t) + len(v["right_ids"] or v["wrong_ids"] or [])
        meta.append({"reply": i, "question": r["id"], "source": r["source"], "fraction": f, "right": v["right"],
                     "success": v["right"] / max(1, v["of"]), "confidence": conf[k], "size": size,
                     "total": r["output_tokens"]})
    print(f"{len(replies)} replies, {len(jobs)} cuts; reading hidden states of {args.model}", flush=True)
    feats_path = out / "features.npy"
    layers = [int(x) for x in args.layers.split(",")]
    if feats_path.exists():
        feats = np.load(feats_path)
    else:
        feats = hidden_features(args.model, sequences, positions, layers, args.dtype, args.batch_tokens)
        np.save(feats_path, feats)

    sources = sorted({m["source"] for m in meta})
    labelled = [k for k, m in enumerate(meta) if m["right"] == of or m["right"] <= args.unsafe]
    y_all = np.array([1.0 if meta[k]["right"] == of else 0.0 for k in range(len(meta))])
    onehot = np.array([[m["source"] == s for s in sources] for m in meta], dtype=np.float32)
    position = np.concatenate([np.array([[m["fraction"], np.log(m["fraction"])] for m in meta], dtype=np.float32), onehot,
                               onehot * np.array([[m["fraction"]] for m in meta], dtype=np.float32)], axis=1)
    questions = sorted({m["question"] for m in meta})
    rng = np.random.default_rng(0)
    fold_of_q = {q: f for q, f in zip(questions, rng.permutation(len(questions)) % args.folds)}
    fold = np.array([fold_of_q[m["question"]] for m in meta])
    lab = np.zeros(len(meta), bool)
    lab[labelled] = True

    def cv(x):
        """Out-of-fold P(safe) for every cut (trained on the labelled cuts of the other folds)."""
        p = np.zeros(len(meta))
        for f in range(args.folds):
            train = lab & (fold != f)
            p[fold == f] = fit_predict(x[train], y_all[train], x[fold == f])
        return p

    lines = [f"Probe: does {args.model} know, at a cut, that it already has the answer? ({len(replies)} replies, "
             f"{len(jobs)} cuts; safe = {of}/{of} right, unsafe <= {args.unsafe}/{of}; {lab.sum()} labelled, "
             f"{int(y_all[lab].sum())} safe; {args.folds}-fold CV grouped by question)", ""]
    scores = {"position": cv(position), "confidence": np.array([m["confidence"] for m in meta])}
    grid = {}
    for li, layer in enumerate(layers):
        for pi, place in enumerate(["before </think>", "after </think>"]):
            name = f"layer {layer}, {place}"
            scores[name] = cv(feats[:, li, pi].astype(np.float32))
            grid[name] = auc(scores[name][lab], y_all[lab])
            print(f"{name}: AUC {grid[name]:.3f}", flush=True)
    best = max(grid, key=grid.get)
    li, pi = layers.index(int(best.split()[1].rstrip(","))), 0 if "before" in best else 1
    scores["probe + position"] = cv(np.concatenate([feats[:, li, pi].astype(np.float32), position], axis=1))
    scores["probe"] = scores[best]

    lines.append("AUC of P(safe) on held-out questions, every (layer, place):")
    lines += [f"  {n:<28} {a:.3f}" for n, a in grid.items()]
    lines += ["", f"best probe = {best}", "",
              f"{'AUC by data type':<18}{'cuts':>6}{'safe':>6}{'position':>10}{'confid.':>9}{'probe':>8}{'probe+pos':>11}"]
    for s in sources + ["ALL"]:
        m = lab & (np.array([x["source"] == s for x in meta]) if s != "ALL" else True)
        row = [auc(scores[n][m], y_all[m]) for n in ("position", "confidence", "probe", "probe + position")]
        lines.append(f"{s:<18}{int(m.sum()):>6}{int(y_all[m].sum()):>6}" + "".join(f"{v:>{w}.3f}" for v, w in zip(row, (10, 9, 8, 11))))

    # the practical test: stoppers chosen on the training folds, scored on the held-out ones
    grid_t = [x / 100 for x in range(5, 100, 5)] + [0.97, 0.99]
    for title, allow in (("each type's allowance", dict(spec.split("=") for spec in args.allow.split(","))),
                         ("2 points for every type", {"default": "2"})):
      lines += ["", f"Stopping at the earliest cut the signal calls safe ({title}; threshold per type chosen on the training",
                "folds; saved / real loss in points on held-out folds; oracle = earliest cut right 20/20):",
                f"{'data type':<18}{'allow':>6}{'oracle':>14}{'position':>14}{'confidence':>14}{'probe':>14}{'probe+pos':>14}"]
      for s in sources:
          budget = float(allow.get(s, allow["default"]))
          cells = []
          for key in ("oracle", "position", "confidence", "probe", "probe + position"):
              saved_all, success_all = [], []
              for f in range(args.folds):
                  def build(test):
                      groups = defaultdict(list)
                      for k, m in enumerate(meta):
                          if m["source"] == s and ((fold[k] == f) == test):
                              c = dict(m, oracle=1.0 if m["right"] == of else 0.0)
                              if key not in ("oracle",):
                                  c[key] = scores[key][k] if key != "position" else m["fraction"]
                              groups[m["reply"]].append(c)
                      return [(cs[0]["total"], sorted(cs, key=lambda c: c["fraction"])) for cs in groups.values()]
                  train, test = build(False), build(True)
                  if key == "oracle":
                      t = 1.0
                  elif key == "position":  # one fixed cut fraction for the whole type
                      t = pick(train, key, args.fractions, budget)
                  else:
                      t = pick(train, key, grid_t, budget)
                  for total, cuts in test:
                      hit = None if t is None else next((c for c in cuts if c[key] >= t), None)
                      saved_all.append(0.0 if hit is None else 1 - hit["size"] / total)
                      success_all.append(1.0 if hit is None else hit["success"])
              cells.append(f"{np.mean(saved_all):.0%} / {100 * (1 - np.mean(success_all)):.1f}")
          lines.append(f"{s:<18}{budget:>6}" + "".join(f"{c:>14}" for c in cells))
    lines += ["", "Reading: the probe carries real knowledge only if it beats the position baseline (later cuts are simply "
              "safer) and the confidence; the stopper table says how much of the oracle's saving it would recover."]
    text = "\n".join(lines)
    print(text, flush=True)
    (out / "summary.txt").write_text(text + "\n")
    np.save(out / "scores.npy", np.stack([scores[k] for k in ("position", "confidence", "probe", "probe + position")]))


if __name__ == "__main__":
    main()
