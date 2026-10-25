"""Is it predictable, from the model's hidden state at a cut, that its answer will stay right if it stops thinking there?

A linear probe (P(IK)-style) on cal_cuts.py runs: correct replies of the model (the longest right one per question),
each cut at 1-80% of its thinking, every cut labelled by its empirical outcome (right re-answers out of 20). Scope: the
replies are all right in full, so this measures when a reply that succeeds can be shortened; questions the model gets
wrong are outside it (stopping those early costs no accuracy).

Features: one forward pass over prompt + cut thinking + "</think>\\n\\n" (no generation); the hidden state at the last
thinking token before "</think>" (where a stop would be decided) and at the last token after it (where the answer
starts), at --layers. Labels: safe = 20/20 right, unsafe <= --unsafe (15/20), 16-19 left out (counted in the report);
label-threshold sensitivity is reported (20 vs <= 10, >= 19 vs <= 15).

Protocol (fixed before looking at results):
  - folds by question: every cut of a question sits in one fold;
  - the (layer, place) used is chosen by nested cross-validation inside the training folds only; the AUC of each of
    the 8 configurations is also printed, as exploration, not as the result;
  - baselines: cut position (fraction + data type) and the model's teacher-forced answer confidence;
  - the primary metric is practical: a stopper ending each reply at its earliest cut whose score >= t, t chosen per
    data type on training data only (most saving within the type's allowance: math 0.5 pt, code 1, others 2; and 2 pts
    for all), scored on held-out data as tokens saved / real loss (mean re-answer success at the chosen cuts) and
    coverage (replies stopped early); the position stopper is one fixed cut fraction per type; against the empirical
    20/20 oracle (earliest cut seen 20/20, replies with none kept whole; 20 trials are a sample, not a true maximum);
  - 95% intervals by bootstrap over questions (not cuts);
  - transfer: leave-one-data-type-out AUC (trained on the other types);
  - with --test-cal: probe, configuration and thresholds are fixed on --cal, then scored ONCE on the test set (new
    questions, e.g. rc2's cuts of the same model).

    python probe_knows.py --model models/rc1-rl-40 --cal runs/bon/calcheck-rc1rl40 \\
        --rollouts runs/bon/probe-rc1/rollouts.jsonl runs/bon/kprobe-rc1/rollouts.jsonl \\
        --keep runs/bon/select-rc1/sft.jsonl runs/bon/kselect-rc1/sft.jsonl \\
        --test-cal runs/bon/pairs-rc2/cal --test-rollouts ... --test-keep ... --out runs/bon/probeknow-rl40
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from cal_cuts import cut_text
from probe_curve import read_jsonl
from prune_traces import thinking_text

PLACES = ["before </think>", "after </think>"]
ORACLE = "oracle 20/20"


def replies_and_jobs(rollouts, keep, fractions, tokenizer):
    """The same replies and cuts, in the same order, as cal_cuts.py built them (checked against its cuts.jsonl)."""
    open_id = tokenizer.convert_tokens_to_ids("<think>")
    allowed = {r["id"] for k in keep for r in read_jsonl(Path(k))}
    by_question = defaultdict(list)
    for path in rollouts:
        for r in read_jsonl(Path(path)):
            if r["id"] in allowed and r["correct"] and r["closed_thinking"] and not r["truncated"]:
                by_question[r["id"]].append(r)
    replies = [r for group in by_question.values() for r in sorted(group, key=lambda r: -r["output_tokens"])[:1]]
    jobs = []
    for i, r in enumerate(replies):
        text = tokenizer.decode([t for t in r["thinking_ids"] if t != open_id]).strip()
        for f in fractions:
            jobs.append((i, f, tokenizer.encode(thinking_text([cut_text(text, f)]), add_special_tokens=False)))
    return replies, jobs


def hidden_features(model_path, sequences, positions, layers, dtype, batch_tokens):
    """[n, len(layers), 2, hidden] float16: the hidden state at each asked position and layer."""
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=getattr(torch, dtype)).to("cuda").eval()
    order = sorted(range(len(sequences)), key=lambda k: len(sequences[k]))
    out = np.zeros((len(sequences), len(layers), 2, model.config.hidden_size), dtype=np.float16)
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


def load_set(cal, rollouts, keep, args, tokenizer, cache_name):
    """Cuts of one cal_cuts.py run: per-cut metadata and hidden-state features (cached in --out)."""
    meta, sequences, positions = load_meta(cal, rollouts, keep, args.fractions, tokenizer)
    path = Path(args.out) / cache_name
    if path.exists():
        feats = np.load(path)
    else:
        print(f"{cal}: {len(set(m['question'] for m in meta))} replies, {len(meta)} cuts; reading hidden states of {args.model}", flush=True)
        feats = hidden_features(args.model, sequences, positions, args.layers, args.dtype, args.batch_tokens)
        np.save(path, feats)
    return meta, feats


OWN_VARIANTS = ("own_min", "own_low5", "own_first50", "own_certainty")  # own_conf.py's fairer self-confidence readings


def load_paragraph_meta(cal, close_id):
    """Per-cut metadata of a label_cuts.py run (cuts at paragraph ends, right and wrong replies)."""
    cuts = read_jsonl(cal / "cuts.jsonl")
    replies = {r["reply"]: r for r in read_jsonl(cal / "replies.jsonl")}
    verdict = {v["key"]: v for v in read_jsonl(cal / "verdicts.jsonl")}
    conf = {c["key"]: c["confidence"] for c in read_jsonl(cal / "confidence.jsonl")}
    own_path = cal / "own_confidence.jsonl"
    own_rows = {c["key"]: c for c in read_jsonl(own_path)} if own_path.exists() else {}
    tag = cal.parent.name if cal.name == "cal" else cal.name
    meta, sequences, positions = [], [], []
    for c in cuts:
        k, r, v = c["key"], replies[c["reply"]], verdict[c["key"]]
        t = c["thinking_ids"]
        sequences.append(c["prompt_ids"] + t)
        positions.append([len(c["prompt_ids"]) + t.index(close_id) - 1, len(c["prompt_ids"]) + len(t) - 1])
        # one reply per (question, right/wrong): the reply is the unit of folds and stopping
        meta.append({"question": f"{tag}/{r['id']}/{'r' if r['correct'] else 'w'}", "source": r["source"],
                     "fraction": c["frac"], "right": v["right"], "of": v["of"], "success": v["right"] / max(1, v["of"]),
                     "confidence": conf[k], "own": own_rows[k]["own_confidence"] if k in own_rows else float("nan"),
                     **{v: own_rows.get(k, {}).get(v, float("nan")) for v in OWN_VARIANTS},
                     "answer": own_rows[k].get("answer") if k in own_rows else None, "full_ok": float(r["correct"]),
                     "size": len(t) + len(v["right_ids"] or v["wrong_ids"] or []), "total": r["output_tokens"]})
    return meta, sequences, positions


def load_meta(cal, rollouts, keep, fractions, tokenizer):
    """Per-cut metadata of one cal_cuts.py run (labels, confidences, sizes), plus the token sequences and positions."""
    cal = Path(cal)
    close_id = tokenizer.convert_tokens_to_ids("</think>")
    if (cal / "replies.jsonl").exists():  # label_cuts.py: paragraph-end cuts of right and wrong replies
        return load_paragraph_meta(cal, close_id)
    replies, jobs = replies_and_jobs(rollouts, keep, fractions, tokenizer)
    cuts_file = [json.loads(l) for l in (cal / "cuts.jsonl").read_text().splitlines() if l.strip()]
    assert len(cuts_file) == len(jobs) and all(c["thinking_ids"] == j[2] for c, j in zip(cuts_file, jobs)), \
        f"the replies and cuts rebuilt here are not the ones of {cal}"
    verdict = {v["key"]: v for v in read_jsonl(cal / "verdicts.jsonl")}
    conf = {c["key"]: c["confidence"] for c in read_jsonl(cal / "confidence.jsonl")}
    own_path = cal / "own_confidence.jsonl"  # own_conf.py: the probability of the model's OWN answer at the cut
    own_rows = {c["key"]: c for c in read_jsonl(own_path)} if own_path.exists() else {}
    own = {k: c["own_confidence"] for k, c in own_rows.items()}
    sequences, positions, meta = [], [], []
    for k, (i, f, t) in enumerate(jobs):
        r, v = replies[i], verdict[k]
        close = t.index(close_id)
        sequences.append(r["prompt_ids"] + t)
        positions.append([len(r["prompt_ids"]) + close - 1, len(r["prompt_ids"]) + len(t) - 1])
        meta.append({"question": f"{cal.parent.name}/{r['id']}", "source": r["source"], "fraction": f, "right": v["right"], "of": v["of"],
                     "success": v["right"] / max(1, v["of"]), "confidence": conf[k], "own": own.get(k, float("nan")),
                     "answer": own_rows[k].get("answer") if k in own_rows else None,
                     "size": len(t) + len(v["right_ids"] or v["wrong_ids"] or []), "total": r["output_tokens"]})
    return meta, sequences, positions


def calibration(scores, success, bins=10):
    """How well predicted probabilities match the real success rate (k/20) of each cut: log loss (cross-entropy against
    the soft target), Brier score, and expected calibration error (|mean predicted - mean real| over 10 equal-width bins,
    weighted by bin size). Lower is better for all three; meaningful only for scores in [0, 1]."""
    p = np.clip(np.asarray(scores, float), 1e-6, 1 - 1e-6)
    y = np.asarray(success, float)
    log_loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    brier = float(np.mean((p - y) ** 2))
    idx = np.minimum((p * bins).astype(int), bins - 1)
    ece = float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))
    return log_loss, brier, ece


def auc(scores, labels):
    """Area under the ROC curve, ties counted half."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    npos, nneg = int(labels.sum()), int((~labels).sum())
    if not npos or not nneg:
        return float("nan")
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inverse]
    return float((ranks[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def fit(train_x, train_y, l2=1.0, steps=300):
    """Standardized L2 logistic regression (full-batch LBFGS in torch); returns a predict function."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_x = np.nan_to_num(np.asarray(train_x, dtype=np.float32), posinf=0.0, neginf=0.0)
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
    try:
        opt.step(closure)
    except RuntimeError:  # LBFGS line search blew up (near-separable data): plain Adam from zero instead
        with torch.no_grad():
            w.zero_(); b.zero_()
        adam = torch.optim.Adam([w, b], lr=0.01)
        for _ in range(500):
            adam.zero_grad()
            closure()
            adam.step()

    def predict(test_x):
        test_x = np.nan_to_num(np.asarray(test_x, dtype=np.float32), posinf=0.0, neginf=0.0)
        tx = torch.tensor((test_x - mu) / sd, dtype=torch.float32, device=device)
        with torch.no_grad():
            return torch.sigmoid(tx @ w + b).cpu().numpy()
    return predict


def fit_mlp(train_x, train_y, hidden=(256, 64), dropout=0.2, wd=1e-3, epochs=60, seed=0):
    """A small MLP probe (as Zhang et al. 2025): standardized input -> 256 -> 64 -> 1, ReLU, dropout, AdamW, early
    stopping on 15% of the training rows; returns a predict function. Targets may be soft (success rates)."""
    import torch
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mu, sd = train_x.mean(0), train_x.std(0) + 1e-4
    x = torch.tensor((train_x - mu) / sd, dtype=torch.float32, device=device)
    y = torch.tensor(train_y, dtype=torch.float32, device=device)
    perm = torch.randperm(len(y), generator=torch.Generator().manual_seed(seed)).to(device)
    n_val = max(1, int(0.15 * len(y)))
    val, tr = perm[:n_val], perm[n_val:]
    layers, d = [], x.shape[1]
    for h in hidden:
        layers += [torch.nn.Linear(d, h), torch.nn.ReLU(), torch.nn.Dropout(dropout)]
        d = h
    net = torch.nn.Sequential(*layers, torch.nn.Linear(d, 1)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        net.train()
        for idx in tr[torch.randperm(len(tr), device=device)].split(256):
            opt.zero_grad()
            loss_fn(net(x[idx]).squeeze(-1), y[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = loss_fn(net(x[val]).squeeze(-1), y[val]).item()
        if v < best - 1e-4:
            best, best_state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 6:
                break
    net.load_state_dict(best_state)
    net.eval()

    def predict(test_x):
        tx = torch.tensor((test_x - mu) / sd, dtype=torch.float32, device=device)
        with torch.no_grad():
            return torch.sigmoid(net(tx).squeeze(-1)).cpu().numpy()
    return predict


def stability_scores(meta):
    """Answer stability (answer-convergence baseline): at each cut, how many consecutive earlier cuts of the same reply gave
    the same own answer (normalized), scaled to 0-1; NaN when own answers are missing."""
    norm = lambda s: " ".join(str(s).lower().split())[-120:]
    out = np.full(len(meta), np.nan)
    by_q = defaultdict(list)
    for k, m in enumerate(meta):
        by_q[m["question"]].append(k)
    for ks in by_q.values():
        ks.sort(key=lambda k: meta[k]["fraction"])
        run = 0
        for i, k in enumerate(ks):
            if meta[k].get("answer") is None:
                break
            run = run + 1 if i and norm(meta[k]["answer"]) == norm(meta[ks[i - 1]]["answer"]) else 0
            out[k] = run / max(1, len(ks) - 1)
    return out


class Data:
    """One set of cuts with its labels and feature views."""

    def __init__(self, meta, feats, sources, unsafe, safe=None):
        self.meta, self.feats, self.n = meta, feats, len(meta)
        of = max(m["of"] for m in meta)
        right = np.array([m["right"] for m in meta])
        safe = of if safe is None else safe
        self.label = (right >= safe).astype(float)
        self.labelled = (right >= safe) | (right <= unsafe)
        self.source = np.array([m["source"] for m in meta])
        self.question = np.array([m["question"] for m in meta])
        self.fraction = np.array([m["fraction"] for m in meta])
        self.oracle = np.array([1.0 if m["right"] == m["of"] else 0.0 for m in meta])
        frac = self.fraction.astype(np.float32)
        onehot = np.array([[s == x for x in sources] for s in self.source], dtype=np.float32)
        self.position = np.concatenate([frac[:, None], np.log(np.maximum(frac, 1e-3))[:, None], onehot,
                                        onehot * frac[:, None]], axis=1)
        self.confidence = np.array([m["confidence"] for m in meta])
        self.own = np.array([m["own"] for m in meta])
        self.own_variants = {v: np.array([m.get(v, float("nan")) for m in meta]) for v in OWN_VARIANTS}
        self.stability = stability_scores(meta)
        self.success = np.array([m["success"] for m in meta])

    def view(self, config):
        if config == "position":
            return self.position
        li, pi = config
        return self.feats[:, li, pi].astype(np.float32)


def folds_by_question(questions, k, seed):
    uniq = sorted(set(questions))
    fold_of = {q: f for q, f in zip(uniq, np.random.default_rng(seed).permutation(len(uniq)) % k)}
    return np.array([fold_of[q] for q in questions])


def oof(data, config, mask, folds, k, soft=False):
    """Out-of-fold P(safe) for the cuts in `mask` (each fold predicted by a probe fit on the other folds' labelled cuts);
    soft: fit to the real success rate (right/20) of every cut instead, so the output reads as a calibrated success rate."""
    p = np.full(data.n, np.nan)
    x = data.view(config)
    for f in range(k):
        train, test = mask & (folds != f) & (True if soft else data.labelled), mask & (folds == f)
        if test.any() and train.any():
            p[test] = fit(x[train], (data.success if soft else data.label)[train])(x[test])
    return p


def choose_config(data, configs, mask, k, seed):
    """The configuration with the best cross-validated AUC using only the cuts in `mask` (the training side)."""
    folds = np.full(data.n, -1)
    folds[mask] = folds_by_question(data.question[mask], k, seed)
    m = mask & data.labelled
    return max(configs, key=lambda c: auc(oof(data, c, mask, folds, k)[m], data.label[m]))


def reply_groups(data, score, mask):
    """{question: (reply total tokens, [cut dicts in fraction order, each with its stopping score])}."""
    groups = defaultdict(list)
    for k in np.flatnonzero(mask):
        groups[data.question[k]].append(dict(data.meta[k], score=score[k]))
    return {q: (cs[0]["total"], sorted(cs, key=lambda c: c["fraction"])) for q, cs in groups.items()}


def stop_one(group, threshold):
    """One reply stopped at its earliest cut with score >= threshold: (saved share, success, stopped early).
    Paragraph sets (label_cuts.py) hold the full thinking as their last cut: the natural stop is measured the same way as
    every cut (its re-answers' success, its length), so the replies' selection (longest right / longest wrong) cancels."""
    total, cuts = group
    if "full_ok" in cuts[-1]:
        nat = cuts[-1]
        hit = None if threshold is None else next((c for c in cuts[:-1] if c["score"] >= threshold), None)
        return (0.0, nat["success"], 0) if hit is None else (1 - hit["size"] / max(1, nat["size"]), hit["success"], 1)
    hit = None if threshold is None else next((c for c in cuts if c["score"] >= threshold), None)
    return (0.0, 1.0, 0) if hit is None else (1 - hit["size"] / total, hit["success"], 1)


def summarize(items):
    """items: [(group, threshold)] -> saved share, real loss in points, coverage."""
    if not items:
        return 0.0, 0.0, 0.0
    r = np.array([stop_one(g, t) for g, t in items])
    # loss against the NATURAL stop (the full reply: right = 1, wrong = 0), so wrong replies do not count as a loss
    natural = np.array([g[1][-1]["success"] if "full_ok" in g[1][-1] else 1.0 for g, _ in items])
    return float(r[:, 0].mean()), 100 * (float(natural.mean()) - float(r[:, 1].mean())), float(r[:, 2].mean())


def pooled(data, score, threshold, sources):
    """One threshold for every data type: (mean saved share over types, mean real loss in points over types), each type
    weighted equally as in the benchmark average."""
    # every reply counts once (a type with a handful of held-out replies cannot swing the pooled loss by 10+ points)
    items = [(g, threshold) for g in reply_groups(data, score, np.ones(data.n, bool)).values()]
    s, l, _ = summarize(items)
    return s, l


def tradeoff_lines(dev, dev_scores, test, test_scores, names, sources, budgets=(0.5, 1, 2, 3, 5), title=""):
    """The savings / accuracy trade-off with ONE threshold per classifier for all data types together (chosen on the
    development out-of-fold scores as the most saving whose pooled loss stays within the budget), scored on the test set.
    Pooling uses every question at once, so a budget is not eaten by one or two mistakes in a small type."""
    grid = [x / 100 for x in range(2, 100, 2)] + [0.97, 0.98, 0.99, 0.995]
    width = max(len(n) for n in names) + 2
    out = ["", f"Trade-off, one threshold for all types{title}: TEST tokens saved / real loss (points, mean over types) "
           "at a pooled loss budget chosen on development", f"{'classifier':<{width}}" + "".join(f"{f'<= {b} pt':>16}" for b in budgets)]
    for n in names:
        dsc, tsc = dev_scores[n], test_scores[n]
        if n == "position":
            dsc, tsc = dev.fraction, test.fraction
        curve = [(t, *pooled(dev, dsc, t, sources)) for t in grid]
        cells = []
        for b in budgets:
            ok = [c for c in curve if c[2] <= b]
            if not ok:
                cells.append("-")
                continue
            t = max(ok, key=lambda c: c[1])[0]
            s, l = pooled(test, tsc, t, sources)
            cells.append(f"{s:.0%} / {l:.1f}")
        out.append(f"{n:<{width}}" + "".join(f"{c:>16}" for c in cells))
    return out


def pick_threshold(groups, grid, allow):
    """The threshold saving most within the allowance on these replies (None: never stop early)."""
    best = None
    for t in grid:
        saved, loss, _ = summarize([(g, t) for g in groups])
        if loss <= allow and (best is None or saved > best[1]):
            best = (t, saved)
    return None if best is None else best[0]


def bootstrap(fn, keys, reps, seed):
    """95% interval of fn(resampled keys) over `reps` resamples of whole questions."""
    rng = np.random.default_rng(seed)
    keys = list(keys)
    if not keys:
        return float("nan"), float("nan")
    vals = [fn([keys[i] for i in rng.integers(0, len(keys), len(keys))]) for _ in range(reps)]
    vals = [v for v in vals if v == v]
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"), float("nan"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, help="the model whose hidden states are read (merged weights)")
    parser.add_argument("--cal", required=True, help="development set: a cal_cuts.py output folder")
    parser.add_argument("--rollouts", nargs="+", required=True)
    parser.add_argument("--keep", nargs="+", required=True)
    parser.add_argument("--extra-runs", nargs="*", default=[],
                        help="more development sets (same model), by run name under --bon: pairs-NAME/cal, probe-/kprobe-NAME "
                             "rollouts, select-/kselect-NAME keep files")
    parser.add_argument("--bon", default="runs/bon")
    parser.add_argument("--test-cal", help="held-out set (new questions, same model): scored once, everything fixed")
    parser.add_argument("--test-rollouts", nargs="+")
    parser.add_argument("--test-keep", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--fractions", default="0.01,0.05,0.1,0.2,0.35,0.5,0.65,0.8")
    parser.add_argument("--layers", default="7,14,21,28", help="hidden-state layers read (0 = embeddings)")
    parser.add_argument("--unsafe", type=int, default=15, help="right re-answers (of 20) at or below which a cut is unsafe")
    parser.add_argument("--allow", default="gsm_plus=0.5,gsm_ic=0.5,kodcode=1,default=2")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--batch-tokens", type=int, default=16000)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.fractions = [float(f) for f in args.fractions.split(",")]
    args.layers = [int(x) for x in args.layers.split(",")]
    from transformers import AutoTokenizer
    Path(args.out).mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    meta, feats = load_set(args.cal, args.rollouts, args.keep, args, tokenizer, "features.npy")
    for run in args.extra_runs:
        bon = Path(args.bon)
        rollouts = [p for p in (bon / f"probe-{run}" / "rollouts.jsonl", bon / f"kprobe-{run}" / "rollouts.jsonl") if p.exists()]
        keep = [p for p in (bon / f"select-{run}" / "sft.jsonl", bon / f"kselect-{run}" / "sft.jsonl") if p.exists()]
        emeta, efeats = load_set(bon / f"pairs-{run}" / "cal", rollouts, keep, args, tokenizer, f"features-{run}.npy")
        meta, feats = meta + emeta, np.concatenate([feats, efeats])
    sources = sorted({m["source"] for m in meta})
    dev = Data(meta, feats, sources, args.unsafe)
    configs = [(li, pi) for li in range(len(args.layers)) for pi in range(2)]
    name = lambda c: f"layer {args.layers[c[0]]}, {PLACES[c[1]]}"
    allows = [("each type's allowance", dict(s.split("=") for s in args.allow.split(","))),
              ("2 points for every type", {"default": "2"})]
    grid = [x / 100 for x in range(5, 100, 5)] + [0.97, 0.99]
    everything = np.ones(dev.n, bool)
    of = max(m["of"] for m in meta)
    n_mid = int((~dev.labelled).sum())
    keys = ["position", "confidence", "probe", "probe/type"]

    lines = [f"Probe on {args.model}: is 'stopping here keeps the answer right' predictable from the hidden state at a cut?",
             f"Development set {args.cal}{''.join(' + ' + r for r in args.extra_runs)}: {len(set(dev.question))} questions (one correct reply each), {dev.n} cuts; "
             f"safe = {of}/{of}: {int(dev.label.sum())}; unsafe <= {args.unsafe}/{of}: "
             f"{int((dev.labelled & (dev.label == 0)).sum())}; left out ({args.unsafe + 1}-{of - 1}/{of}): {n_mid} "
             f"({n_mid / dev.n:.0%})",
             "Scope: replies that are right in full; the question is when they can be shortened. A good probe shows "
             "that predictive information is linearly readable, not that the model 'represents knowing'.", ""]

    # ---- development set: nested cross-validation ----
    folds = folds_by_question(dev.question, args.folds, args.seed)
    explore = {c: oof(dev, c, everything, folds, args.folds) for c in configs}
    lines.append("Exploration (each configuration's out-of-fold AUC; NOT the result: picking the best one is optimistic):")
    lines += [f"  {name(c):<28} {auc(p[dev.labelled], dev.label[dev.labelled]):.3f}" for c, p in explore.items()]
    probe, probe_t, probe_s, chosen = np.full(dev.n, np.nan), np.full(dev.n, np.nan), np.full(dev.n, np.nan), []
    chosen_cfgs = []
    for f in range(args.folds):
        train, test = folds != f, folds == f
        c = choose_config(dev, configs, train, args.folds - 1, args.seed + 1 + f)
        chosen.append(name(c))
        chosen_cfgs.append(c)
        x = dev.view(c)
        probe[test] = fit(x[train & dev.labelled], dev.label[train & dev.labelled])(x[test])
        probe_s[test] = fit(x[train], dev.success[train])(x[test])  # trained on the real success rate of every cut
        for s in sources:  # one probe per data type (the pooled one transfers poorly between types)
            src = dev.source == s
            probe_t[test & src] = fit(x[train & dev.labelled & src], dev.label[train & dev.labelled & src])(x[test & src])
    best_cfg = max(set(chosen_cfgs), key=chosen_cfgs.count)  # the configuration the nested choice picked most often
    dev_scores = {"position": oof(dev, "position", everything, folds, args.folds), "confidence": dev.confidence,
                  "probe": probe, "probe/type": probe_t, "probe/soft": probe_s}
    allx = feats.reshape(dev.n, -1).astype(np.float32)
    mlp, mlp_all = np.full(dev.n, np.nan), np.full(dev.n, np.nan)
    for f in range(args.folds):
        train, test = folds != f, folds == f
        lab_tr = train & dev.labelled
        x = dev.view(best_cfg)
        mlp[test] = fit_mlp(x[lab_tr], dev.label[lab_tr], seed=f)(x[test])
        mlp_all[test] = fit_mlp(allx[lab_tr], dev.label[lab_tr], hidden=(512, 128), seed=f)(allx[test])
    dev_scores["mlp"], dev_scores["mlp/all layers"] = mlp, mlp_all
    keys += ["mlp", "mlp/all layers"]
    if np.isfinite(dev.stability).all():
        dev_scores["answer stability"] = dev.stability
        keys.append("answer stability")
    has_own = bool(np.isfinite(dev.own).all())
    if has_own:
        dev_scores["own confidence"] = dev.own
    keys += ["probe/soft"] + (["own confidence"] if has_own else [])
    lines += ["", f"Nested choice of configuration, per outer fold: {'; '.join(chosen)}", ""]

    def auc_table(data, scores, title):
        ks = [k for k in keys if k in scores]
        out = [title, f"{'data type':<16}{'cuts':>6}{'safe':>6}" + "".join(f"{k:>22}" for k in ks)]
        for s in sources + ["ALL"]:
            m = data.labelled & ((data.source == s) if s != "ALL" else np.ones(data.n, bool))
            by_q = defaultdict(list)
            for k in np.flatnonzero(m):
                by_q[data.question[k]].append(k)
            cells = []
            for key in ks:
                sc = scores[key]
                pick = lambda sample: [k for q in sample for k in by_q[q]]
                lo, hi = bootstrap(lambda sample: auc(sc[pick(sample)], data.label[pick(sample)]), by_q.keys(),
                                   args.bootstrap, args.seed)
                cells.append(f"{auc(sc[m], data.label[m]):.3f} [{lo:.2f},{hi:.2f}]")
            out.append(f"{s:<16}{int(m.sum()):>6}{int(data.label[m].sum()):>6}" + "".join(f"{c:>22}" for c in cells))
        return out

    def stopper_table(data, scores, threshold_for, title, allow):
        """threshold_for(key, source, budget, question) -> the threshold fixed for that question's reply (no peeking)."""
        ks = [k for k in keys if k in scores]
        out = ["", title, f"{'data type':<16}{'allow':>6}" + "".join(f"{k:>28}" for k in [ORACLE] + ks)]
        for s in sources:
            budget = float(allow.get(s, allow["default"]))
            cells = []
            for key in [ORACLE] + ks:
                sc = data.oracle if key == ORACLE else (data.fraction if key == "position" else scores[key])
                groups = reply_groups(data, sc, data.source == s)
                items = {q: (g, 1.0 if key == ORACLE else threshold_for(key, s, budget, q)) for q, g in groups.items()}
                saved, loss, cover = summarize(list(items.values()))
                lo, hi = bootstrap(lambda sample: summarize([items[q] for q in sample])[1], items.keys(),
                                   args.bootstrap, args.seed)
                cells.append(f"{saved:.0%}/{loss:.1f} [{lo:.1f},{hi:.1f}] c{cover:.0%}")
            out.append(f"{s:<16}{budget:>6}" + "".join(f"{c:>28}" for c in cells))
        return out

    def tranche_table(data, scores, title):
        """Per data type and cut point: real success vs the soft probe's and the model's own predicted success (mean),
        and the AUC of each AT THAT CUT POINT (questions ranked against each other at the same depth: no position effect)."""
        cols = [k for k in ("probe/soft", "own confidence") if k in scores and np.isfinite(scores[k]).all()]
        out = ["", title, "  cell = real success / " + " / ".join(f"{c} mean" for c in cols) + "  |  AUC " +
               " / ".join(cols) + " among the cuts at that depth (n/a: one label only)"]
        for s in sources:
            out.append(f"  {s}")
            for f in args.fractions:
                lo = ([0.0] + list(args.fractions))[list(args.fractions).index(f)]
                m = (data.source == s) & (data.fraction > (lo if f != args.fractions[0] else -1)) & (data.fraction <= f)
                ml = m & data.labelled
                means = " / ".join(f"{np.nanmean(scores[c][m]):.2f}" for c in cols)
                aucs = " / ".join((lambda a: "n/a" if a != a else f"{a:.2f}")(auc(scores[c][ml], data.label[ml])) for c in cols)
                out.append(f"    {f:>4.0%}: n {int(m.sum()):>4}  real {data.success[m].mean():.2f} / {means}  |  AUC {aucs}")
        return out

    lines += auc_table(dev, dev_scores, "AUC on held-out questions (development set, nested CV; 95% bootstrap over "
                                        "questions):")
    lines += tranche_table(dev, dev_scores, "Cut point by cut point (development set, out-of-fold):")

    best_c = max(configs, key=lambda c: auc(explore[c][dev.labelled], dev.label[dev.labelled]))
    # would more questions help? held-out AUC when the probe is fit on a share of the training questions only
    lines += ["", f"Learning curve ({name(best_c)}; out-of-fold AUC, probe fit on a share of each training side's questions; "
              "still rising at 100% = more questions would help):"]
    x = dev.view(best_c)
    rng = np.random.default_rng(args.seed)
    for share in (0.25, 0.5, 0.75, 1.0):
        p = np.full(dev.n, np.nan)
        for f in range(args.folds):
            qs = sorted(set(dev.question[folds != f]))
            sub = set(rng.permutation(qs)[:max(2, int(share * len(qs)))])
            train = dev.labelled & (folds != f) & np.array([q in sub for q in dev.question])
            p[folds == f] = fit(x[train], dev.label[train])(x[folds == f])
        per = "; ".join(f"{s} {auc(p[dev.labelled & (dev.source == s)], dev.label[dev.labelled & (dev.source == s)]):.3f}"
                        for s in sources)
        lines.append(f"  {share:>4.0%} of the questions: ALL {auc(p[dev.labelled], dev.label[dev.labelled]):.3f} | {per}")
    lines += ["", f"Label sensitivity (exploration, {name(best_c)}; out-of-fold AUC, probe | position):"]
    for safe, unsafe in ((of, 10), (of - 1, args.unsafe), (of, args.unsafe)):
        alt = Data(meta, feats, sources, unsafe, safe)
        p = oof(alt, best_c, everything, folds, args.folds)
        q = oof(alt, "position", everything, folds, args.folds)
        lines.append(f"  safe >= {safe}/{of}, unsafe <= {unsafe}/{of} ({int(alt.labelled.sum())} cuts): "
                     f"{auc(p[alt.labelled], alt.label[alt.labelled]):.3f} | {auc(q[alt.labelled], alt.label[alt.labelled]):.3f}")

    lines += ["", f"Transfer (exploration, {name(best_c)}): AUC on one data type, probe trained on the other types only:"]
    x = dev.view(best_c)
    for s in sources:
        test = dev.source == s
        train = ~test & dev.labelled
        p = fit(x[train], dev.label[train])(x[test])
        lines.append(f"  {s:<16} {auc(p[dev.labelled[test]], dev.label[test][dev.labelled[test]]):.3f}")

    # the practical test on the development set: for each held-out fold, thresholds from the other folds
    fold_of_q = dict(zip(dev.question, folds))
    cache = {}

    def dev_threshold(key, s, budget, q):
        f = fold_of_q[q]
        if (key, s, budget, f) not in cache:
            sc = dev.fraction if key == "position" else dev_scores[key]
            g = reply_groups(dev, sc, (dev.source == s) & (folds != f))
            cache[(key, s, budget, f)] = pick_threshold(list(g.values()), args.fractions if key == "position" else grid,
                                                        budget)
        return cache[(key, s, budget, f)]
    for title, allow in allows:
        lines += stopper_table(dev, dev_scores, dev_threshold, f"Stopper, development set ({title}): saved / real loss "
                               "[95% CI of the loss] c coverage; thresholds from the training folds only", allow)

    # ---- held-out test set: everything fixed on the development set, scored once ----
    if args.test_cal:
        tmeta, tfeats = load_set(args.test_cal, args.test_rollouts, args.test_keep, args, tokenizer, "test_features.npy")
        test = Data(tmeta, tfeats, sources, args.unsafe)
        final_c = choose_config(dev, configs, everything, args.folds, args.seed + 99)
        x = dev.view(final_c)
        lab = dev.labelled
        test_scores = {"position": fit(dev.position[lab], dev.label[lab])(test.position), "confidence": test.confidence,
                       "probe": fit(x[lab], dev.label[lab])(test.view(final_c)), "probe/type": np.full(test.n, np.nan),
                       "probe/soft": fit(x, dev.success)(test.view(final_c)), "own confidence": test.own,
                       "mlp": fit_mlp(x[lab], dev.label[lab])(test.view(final_c)),
                       "mlp/all layers": fit_mlp(feats.reshape(dev.n, -1).astype(np.float32)[lab], dev.label[lab],
                                                 hidden=(512, 128))(tfeats.reshape(test.n, -1).astype(np.float32)),
                       "answer stability": test.stability}
        dev_oof = {"position": dev.fraction, "confidence": dev.confidence,
                   "probe": oof(dev, final_c, everything, folds, args.folds), "probe/type": np.full(dev.n, np.nan),
                   "probe/soft": oof(dev, final_c, everything, folds, args.folds, soft=True), "own confidence": dev.own,
                   "answer stability": dev.stability}
        # the MLPs' out-of-fold scores on the development set (for their thresholds), with the final configuration
        dev_oof["mlp"], dev_oof["mlp/all layers"] = np.full(dev.n, np.nan), np.full(dev.n, np.nan)
        xf, allx = dev.view(final_c), feats.reshape(dev.n, -1).astype(np.float32)
        for f in range(args.folds):
            tr, te = (folds != f) & lab, folds == f
            dev_oof["mlp"][te] = fit_mlp(xf[tr], dev.label[tr], seed=f)(xf[te])
            dev_oof["mlp/all layers"][te] = fit_mlp(allx[tr], dev.label[tr], hidden=(512, 128), seed=f)(allx[te])
        tx = test.view(final_c)
        for s in sources:
            src, tsrc = dev.source == s, test.source == s
            if tsrc.any():
                test_scores["probe/type"][tsrc] = fit(x[lab & src], dev.label[lab & src])(tx[tsrc])
            dev_oof["probe/type"][src] = oof(dev, final_c, src, folds, args.folds)[src]
        fixed = {}

        def test_threshold(key, s, budget, q):
            if (key, s, budget) not in fixed:
                g = reply_groups(dev, dev_oof[key], dev.source == s)
                fixed[(key, s, budget)] = pick_threshold(list(g.values()),
                                                         args.fractions if key == "position" else grid, budget)
            return fixed[(key, s, budget)]
        lines += ["", "=" * 110,
                  f"HELD-OUT TEST SET {args.test_cal}: {len(set(test.question))} new questions, {test.n} cuts; probe = "
                  f"{name(final_c)}, chosen and fit on the development set only; thresholds from the development set's "
                  "out-of-fold scores. Scored once.", ""]
        if not np.isfinite(test.own).all():
            test_scores.pop("own confidence")
        if not np.isfinite(test.stability).all():
            test_scores.pop("answer stability")
        lines += auc_table(test, test_scores, "AUC on the test set (95% bootstrap over questions):")
        lines += tranche_table(test, test_scores, "Cut point by cut point (test set):")
        names_t = [k for k in keys if k in test_scores and k in dev_oof]
        lines += tradeoff_lines(dev, dev_oof, test, test_scores, names_t, sources)
        for title, allow in allows:
            lines += stopper_table(test, test_scores, test_threshold, f"Stopper on the test set ({title}): saved / real "
                                   "loss [95% CI of the loss] c coverage", allow)

    lines += ["", "Caveat on 'confidence': it is the teacher-forced probability of the answer the full (right) reply wrote, i.e. "
              "of the RIGHT answer: it uses the reference, so it is an optimistic baseline a deployed stopper could not compute."]
    lines += ["", "Reading: the probe adds information only if it beats the position baseline (later cuts are simply "
              "safer) and the confidence on held-out questions, in the stopper table (tokens saved at bounded real loss), "
              "not only in AUC. A failure shows that THIS linear probe finds no usable signal, not that none exists."]
    text = "\n".join(lines)
    print(text, flush=True)
    (Path(args.out) / "summary.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
