"""Probe zoo: every stopping classifier on the same labelled cuts, compared on the same untouched test set.

Development = one or more cal_cuts.py runs of the model (--cal + --extra-runs); test = --test-cal (new questions, never
used for any choice). Features come from extract_feats.py (--feats-root/<set>/points.npy and window.npy).

Classifiers (each gives out-of-fold scores on the development set, 5 folds by question, and test scores from a fit on
the whole development set):
  baselines   position (cut point + data type), the model's own confidence (probability of its own greedy answer),
              answer stability (same own answer as at the previous cut), and the reference-based confidence (shown
              for comparison only: it uses the right answer)
  sweep       a linear probe on every layer (1-28) at both places; the best (development out-of-fold AUC) is "linear"
  linear      best layer/place; also one per data type, and one fit to the real success rate (calibrated)
  mlp         2-layer MLP on the best layer/place; and on all 28 layers x 2 places concatenated
  attention   attention-pooling probes over the last --window thinking tokens (one per window layer, and all window
              layers concatenated): LayerNorm -> 256-d projection + learned position -> 4-head attention pooling with a
              learned query -> MLP -> P(safe)

Reported for every classifier: AUC (development OOF and test, per data type), and on the test set the stopping
simulation (earliest cut whose score >= a threshold chosen per data type on the development out-of-fold scores):
tokens saved / real loss / coverage, at each type's allowance and at 2 points for all; plus cut-point-by-cut-point
real vs predicted success for the calibrated probes.

    python probe_zoo.py --cal runs/bon/calcheck-rc1rl40 --rollouts ... --keep ... --extra-runs lot1 lot2 \\
        --test-cal runs/bon/pairs-rc2/cal --test-rollouts ... --test-keep ... --feats-root /content/feats --out runs/bon/zoo
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from probe_knows import (Data, auc, fit, fit_mlp, folds_by_question, load_meta, pick_threshold, reply_groups,
                         summarize, tradeoff_lines)


def attention_fit(get, train_idx, train_y, dim=256, heads=4, epochs=30, seed=0):
    """Attention-pooling probe; get(indices) -> float16 [b, T, d] windows (gathered batch by batch, so the multi-GB
    window array is never copied). Returns predict(get_test, indices)."""
    import torch
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    T, d = get(train_idx[:1]).shape[1:]
    n = len(train_idx)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.LayerNorm(d)
            self.proj = torch.nn.Linear(d, dim)
            self.pos = torch.nn.Parameter(torch.zeros(T, dim))
            self.query = torch.nn.Parameter(torch.randn(1, 1, dim) * 0.02)
            self.attn = torch.nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.1)
            self.head = torch.nn.Sequential(torch.nn.Linear(dim, 64), torch.nn.ReLU(), torch.nn.Dropout(0.2),
                                            torch.nn.Linear(64, 1))

        def forward(self, x):
            pad = (x.abs().sum(-1) == 0)  # left padding of short contexts
            pad[:, -1] = False
            h = self.proj(self.norm(x)) + self.pos
            pooled, _ = self.attn(self.query.expand(len(x), -1, -1), h, h, key_padding_mask=pad)
            return self.head(pooled.squeeze(1)).squeeze(-1)

    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    y = torch.tensor(train_y, dtype=torch.float32)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_val = max(1, int(0.15 * n))
    val, tr = perm[:n_val], perm[n_val:]
    batch = lambda idx: torch.tensor(np.asarray(get(train_idx[idx.numpy()])), dtype=torch.float32, device=dev)
    best, state, bad = float("inf"), None, 0
    for _ in range(epochs):
        net.train()
        for idx in tr[torch.randperm(len(tr))].split(128):
            opt.zero_grad()
            loss_fn(net(batch(idx)), y[idx].to(dev)).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = sum(loss_fn(net(batch(idx)), y[idx].to(dev)).item() * len(idx) for idx in val.split(512)) / len(val)
        if v < best - 1e-4:
            best, state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 5:
                break
    net.load_state_dict(state)
    net.eval()

    def predict(get_test, idx):
        out = []
        with torch.no_grad():
            for i in range(0, len(idx), 512):
                x = torch.tensor(np.asarray(get_test(idx[i:i + 512])), dtype=torch.float32, device=dev)
                out.append(torch.sigmoid(net(x)).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)
    return predict


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cal", required=True)
    parser.add_argument("--rollouts", nargs="+", required=True)
    parser.add_argument("--keep", nargs="+", required=True)
    parser.add_argument("--extra-runs", nargs="*", default=[])
    parser.add_argument("--bon", default="runs/bon")
    parser.add_argument("--test-cal", help="held-out set; without it, --test-share of the development QUESTIONS are held out")
    parser.add_argument("--test-rollouts", nargs="*", default=[])
    parser.add_argument("--test-keep", nargs="*", default=[])
    parser.add_argument("--test-share", type=float, default=0.3)
    parser.add_argument("--feats-root", default="/content/feats")
    parser.add_argument("--out", required=True)
    parser.add_argument("--fractions", default="0.01,0.05,0.1,0.2,0.35,0.5,0.65,0.8")
    parser.add_argument("--unsafe", type=int, default=15)
    parser.add_argument("--allow", default="gsm_plus=0.5,gsm_ic=0.5,kodcode=1,default=2")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--window-layers", default="14,21,28")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    fractions = [float(f) for f in args.fractions.split(",")]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bon, root = Path(args.bon), Path(args.feats_root)
    feat_name = lambda cal: cal.name if cal.name != "cal" else cal.parent.name

    def load(cal, rollouts, keep):
        meta = load_meta(cal, rollouts, keep, fractions, tok)[0]
        d = root / feat_name(Path(cal))
        return meta, np.load(d / "points.npy", mmap_mode="r"), np.load(d / "window.npy", mmap_mode="r")

    parts = [load(Path(args.cal), args.rollouts, args.keep)]
    for run in args.extra_runs:
        rollouts = [p for p in (bon / f"probe-{run}" / "rollouts.jsonl", bon / f"kprobe-{run}" / "rollouts.jsonl") if p.exists()]
        keep = [p for p in (bon / f"select-{run}" / "sft.jsonl", bon / f"kselect-{run}" / "sft.jsonl") if p.exists()]
        parts.append(load(bon / f"pairs-{run}" / "cal", rollouts, keep))
    meta = sum((p[0] for p in parts), [])
    points = np.concatenate([np.asarray(p[1]) for p in parts])
    window = np.concatenate([np.asarray(p[2]) for p in parts])  # float16, a few GB in RAM
    if args.test_cal:
        tmeta, tpoints, twindow = load(Path(args.test_cal), args.test_rollouts, args.test_keep)
        tpoints, twindow = np.asarray(tpoints), np.asarray(twindow)
    else:  # hold out a share of the questions (both replies of a question on the same side), fixed before any fit
        import hashlib
        qid = lambda m: m["question"].rsplit("/", 1)[0] if m["question"].endswith(("/r", "/w")) else m["question"]
        is_test = np.array([int(hashlib.md5(qid(m).encode()).hexdigest(), 16) % 1000 < 1000 * args.test_share for m in meta])
        tmeta = [m for m, x in zip(meta, is_test) if x]
        meta = [m for m, x in zip(meta, is_test) if not x]
        tpoints, twindow = points[is_test], window[is_test]
        points, window = points[~is_test], window[~is_test]
        args.test_cal = f"{args.test_share:.0%} of the development questions, held out"
    sources = sorted({m["source"] for m in meta})
    dev, test = Data(meta, points, sources, args.unsafe), Data(tmeta, tpoints, sources, args.unsafe)
    folds = folds_by_question(dev.question, args.folds, args.seed)
    lab = dev.labelled
    n_layers = points.shape[1]
    wl = [int(x) for x in args.window_layers.split(",")]
    print(f"zoo: development {dev.n} cuts ({len(set(dev.question))} questions), test {test.n} cuts "
          f"({len(set(test.question))} questions); {n_layers} layers", flush=True)

    dev_s, test_s = {}, {}

    def run(name, make_x_dev, make_x_test, fitter, soft=False):
        """Out-of-fold development scores and test scores for one classifier."""
        p = np.full(dev.n, np.nan)
        xd = make_x_dev()
        for f in range(args.folds):
            tr = (folds != f) & (np.ones(dev.n, bool) if soft else lab)
            p[folds == f] = fitter(xd[tr], (dev.success if soft else dev.label)[tr], f)(xd[folds == f])
        tr = np.ones(dev.n, bool) if soft else lab
        dev_s[name] = p
        test_s[name] = fitter(xd[tr], (dev.success if soft else dev.label)[tr], 99)(make_x_test())
        print(f"zoo: {name}: dev AUC {auc(p[lab], dev.label[lab]):.3f}, test AUC "
              f"{auc(test_s[name][test.labelled], test.label[test.labelled]):.3f}", flush=True)

    linear = lambda x, y, s: fit(x, y)
    # baselines
    run("position", lambda: dev.position, lambda: test.position, linear)
    for name, d, t in (("own confidence", dev.own, test.own), ("answer stability", dev.stability, test.stability),
                       ("confidence (uses the right answer)", dev.confidence, test.confidence)):
        if np.isfinite(d).all() and np.isfinite(t).all():
            dev_s[name], test_s[name] = d, t
    # layer sweep (development out-of-fold AUC only; the best one is carried forward)
    sweep = {}
    for layer in range(n_layers):
        for place in range(2):
            x = points[:, layer, place].astype(np.float32)
            p = np.full(dev.n, np.nan)
            for f in range(args.folds):
                tr = (folds != f) & lab
                p[folds == f] = fit(x[tr], dev.label[tr])(x[folds == f])
            sweep[(layer, place)] = auc(p[lab], dev.label[lab])
    best = max(sweep, key=sweep.get)
    bl, bp = best
    xbest = lambda data, pts: pts[:, bl, bp].astype(np.float32)
    run("linear", lambda: xbest(dev, points), lambda: xbest(test, tpoints), linear)
    run("linear, real success rate", lambda: xbest(dev, points), lambda: xbest(test, tpoints), linear, soft=True)
    # one linear probe per data type
    p, t = np.full(dev.n, np.nan), np.full(test.n, np.nan)
    xd, xt = xbest(dev, points), xbest(test, tpoints)
    for s in sources:
        src, tsrc = dev.source == s, test.source == s
        for f in range(args.folds):
            tr = (folds != f) & lab & src
            p[(folds == f) & src] = fit(xd[tr], dev.label[tr])(xd[(folds == f) & src])
        if tsrc.any():
            t[tsrc] = fit(xd[lab & src], dev.label[lab & src])(xt[tsrc])
    dev_s["linear, one per type"], test_s["linear, one per type"] = p, t
    mlp = lambda hidden: (lambda x, y, s: fit_mlp(x, y, hidden=hidden, seed=s))
    run("mlp", lambda: xbest(dev, points), lambda: xbest(test, tpoints), mlp((256, 64)))
    run(f"mlp, all {n_layers} layers x 2 places", lambda: points.reshape(dev.n, -1).astype(np.float32),
        lambda: tpoints.reshape(test.n, -1).astype(np.float32), mlp((512, 128)))
    def run_attention(name, js):
        """Attention probe over the window at layers js (concatenated along the feature axis)."""
        def getter(arr):
            return lambda idx: np.concatenate([arr[idx, j] for j in js], -1) if len(js) > 1 else arr[idx, js[0]]
        gd, gt = getter(window), getter(twindow)
        p = np.full(dev.n, np.nan)
        for f in range(args.folds):
            tr = np.flatnonzero((folds != f) & lab)
            te = np.flatnonzero(folds == f)
            p[te] = attention_fit(gd, tr, dev.label[tr], seed=f)(gd, te)
        tr = np.flatnonzero(lab)
        dev_s[name] = p
        test_s[name] = attention_fit(gd, tr, dev.label[tr], seed=99)(gt, np.arange(test.n))
        print(f"zoo: {name}: dev AUC {auc(p[lab], dev.label[lab]):.3f}, test AUC "
              f"{auc(test_s[name][test.labelled], test.label[test.labelled]):.3f}", flush=True)
    for j, layer in enumerate(wl):
        run_attention(f"attention, last {window.shape[2]} tokens, layer {layer}", [j])
    run_attention(f"attention, last {window.shape[2]} tokens, layers {'+'.join(map(str, wl))}", list(range(len(wl))))

    # ---- report ----
    names = list(dev_s)
    places = ["before </think>", "after </think>"]
    lines = [f"PROBE ZOO. Development: {args.cal} + {', '.join(args.extra_runs) or '-'} ({dev.n} cuts, "
             f"{len(set(dev.question))} questions); TEST (untouched): {args.test_cal} ({test.n} cuts, "
             f"{len(set(test.question))} questions). safe = 20/20, unsafe <= {args.unsafe}/20.", "",
             "Layer sweep (linear probe, development out-of-fold AUC; place 'before' | 'after' </think>):"]
    for layer in range(n_layers):
        lines.append(f"  layer {layer + 1:>2}: {sweep[(layer, 0)]:.3f} | {sweep[(layer, 1)]:.3f}")
    lines += [f"best: layer {bl + 1}, {places[bp]}", ""]
    width = max(len(n) for n in names) + 2
    for title, scores, data in (("AUC, development (out-of-fold)", dev_s, dev), ("AUC, TEST", test_s, test)):
        lines += [title, f"{'classifier':<{width}}" + "".join(f"{s[:12]:>13}" for s in sources) + f"{'ALL':>8}"]
        for n in names:
            cells = []
            for s in sources + ["ALL"]:
                m = data.labelled & ((data.source == s) if s != "ALL" else np.ones(data.n, bool))
                cells.append(auc(scores[n][m], data.label[m]))
            lines.append(f"{n:<{width}}" + "".join(f"{c:>13.3f}" for c in cells[:-1]) + f"{cells[-1]:>8.3f}")
        lines.append("")
    lines += tradeoff_lines(dev, dev_s, test, test_s, names, sources) + [""]
    for title, allow in (("each type's allowance", dict(s.split("=") for s in args.allow.split(","))),
                         ("2 points for every type", {"default": "2"})):
        lines += [f"Stopping simulation on the TEST set ({title}): tokens saved / real loss in points / coverage; "
                  "threshold per type chosen on the development out-of-fold scores",
                  f"{'classifier':<{width}}" + "".join(f"{s[:12]:>20}" for s in sources)]
        rows = [("oracle 20/20 (bound)", None)] + [(n, n) for n in names]
        for label, n in rows:
            cells = []
            for s in sources:
                budget = float(allow.get(s, allow["default"]))
                if n is None:
                    groups = reply_groups(test, test.oracle, test.source == s)
                    saved, loss, cover = summarize([(g, 1.0) for g in groups.values()])
                else:
                    dsc = dev.fraction if n == "position" else dev_s[n]
                    tsc = test.fraction if n == "position" else test_s[n]
                    grid = fractions if n == "position" else [x / 100 for x in range(5, 100, 5)] + [0.97, 0.99]
                    th = pick_threshold(list(reply_groups(dev, dsc, dev.source == s).values()), grid, budget)
                    saved, loss, cover = summarize([(g, th) for g in reply_groups(test, tsc, test.source == s).values()])
                cells.append(f"{saved:.0%}/{loss:.1f}/c{cover:.0%}")
            lines.append(f"{label:<{width}}" + "".join(f"{c:>20}" for c in cells))
        lines.append("")
    lines += ["Cut point by cut point, TEST (real success vs the calibrated probe's and the model's own predicted success):"]
    for s in sources:
        lines.append(f"  {s}")
        for f in fractions:
            lo = -1 if f == fractions[0] else fractions[fractions.index(f) - 1]
            m = (test.source == s) & (test.fraction > lo) & (test.fraction <= f)
            if not m.any():
                continue
            own = f"{np.nanmean(test.own[m]):.2f}" if np.isfinite(test.own[m]).all() else "-"
            lines.append(f"    {f:>4.0%}: real {test.success[m].mean():.2f} | probe (real success rate) "
                         f"{np.mean(test_s['linear, real success rate'][m]):.2f} | own confidence {own}")
    lines += ["", "Caveat: 'confidence (uses the right answer)' is teacher-forced on the right answer; shown for reference only."]
    text = "\n".join(lines)
    print(text, flush=True)
    (out / "summary.txt").write_text(text + "\n")
    np.savez_compressed(out / "scores.npz", **{f"dev::{k}": v for k, v in dev_s.items()},
                        **{f"test::{k}": v for k, v in test_s.items()})


if __name__ == "__main__":
    main()
