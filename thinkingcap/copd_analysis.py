"""COPD Runs A and C, from the data (user, 10 Nov: "verify the data of the COPD runs C and A, analyze it and explain it").

1. Is "right up / wrong down" also "short up / long down"? On each run's training questions, from the ORIGINAL model's 10
   stored answers: on questions with both right and wrong answers, mean tokens of the right vs the wrong ones, the share of
   such questions where the wrong answers are longer, and the share of wrong answers cut at the limit, per part.
2. The 720 screens of both runs at the same steps against the original: per part, accuracy, mean tokens of RIGHT and of
   WRONG answers, the share cut at the limit, and the share of wrong answers that stopped early (thinking under half the
   original's median thinking on its right answers to the same question).
3. The training logs (rl_log.jsonl of every block): per step, practice accuracy, tokens, and the coach's verdict.

    python copd_analysis.py --answers ANSWERS10.jsonl --questions-a data/tp-cp1a.jsonl --questions-c data/tp-cp1c2.jsonl \\
        --base runs/mvtf/s4-base --runs-a runs/mvtf/s4-cp1a-10 ... --runs-c runs/mvtf/s4-cp1c2-10 ... \\
        --logs-a adapters/cp1a-rl --logs-c adapters/cp1c2-rl
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def read(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def right(r):
    return bool(r.get("correct")) and not r.get("truncated")


def length_link(answers, questions, label):
    part = {(q.get("pool_id") or q.get("id")): q["part"] for q in questions}
    by_q = defaultdict(list)
    for r in answers:
        if r["id"] in part:
            ok = bool(r["correct"]) and r.get("closed_thinking", True) and not r["truncated"]
            by_q[r["id"]].append((ok, r["output_tokens"], bool(r["truncated"])))
    print(f"\n== 1. {label}: on mixed questions (the original right AND wrong among its 10 answers), right vs wrong lengths")
    rows = defaultdict(lambda: {"q": 0, "wrong_longer": 0, "r": [], "w": [], "cut": 0, "nw": 0})
    for qid, got in by_q.items():
        rs = [n for ok, n, _ in got if ok]
        ws = [(n, c) for ok, n, c in got if not ok]
        if not rs or not ws:
            continue
        d = rows[part[qid]]
        d["q"] += 1
        d["wrong_longer"] += statistics.fmean(n for n, _ in ws) > statistics.fmean(rs)
        d["r"] += rs
        d["w"] += [n for n, _ in ws]
        d["cut"] += sum(c for _, c in ws)
        d["nw"] += len(ws)
    for p, d in sorted(rows.items()):
        print(f"  {p:<20} {d['q']:>4} mixed questions; wrong answers longer on {100 * d['wrong_longer'] / d['q']:3.0f}% of them; "
              f"mean tokens right {statistics.fmean(d['r']):5.0f} / wrong {statistics.fmean(d['w']):5.0f}; "
              f"{100 * d['cut'] / max(1, d['nw']):3.0f}% of wrong answers cut at the limit")
    return rows


def screen(run, base_right_think):
    out = defaultdict(lambda: {"n": 0, "ok": 0, "rt": [], "wt": [], "cut": 0, "early": 0, "nw": 0})
    for r in read(Path(run) / "records.jsonl"):
        d = out[r["dataset"]]
        d["n"] += 1
        if right(r):
            d["ok"] += 1
            d["rt"].append(r["output_tokens"])
        else:
            d["nw"] += 1
            d["wt"].append(r["output_tokens"])
            d["cut"] += bool(r.get("truncated"))
            med = base_right_think.get((r["dataset"], r["id"]))
            d["early"] += bool(med) and (r.get("reasoning_tokens") or 0) < med / 2
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--answers", required=True)
    ap.add_argument("--questions-a", required=True)
    ap.add_argument("--questions-c", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--runs-a", nargs="+", required=True)
    ap.add_argument("--runs-c", nargs="+", required=True)
    ap.add_argument("--logs-a", required=True)
    ap.add_argument("--logs-c", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    answers = read(args.answers)
    report = {"length_link_a": length_link(answers, read(args.questions_a), "Run A's training questions"),
              "length_link_c": length_link(answers, read(args.questions_c), "Run C's training questions")}
    base_think = defaultdict(list)
    for r in read(Path(args.base) / "records.jsonl"):
        if right(r):
            base_think[(r["dataset"], r["id"])].append(r.get("reasoning_tokens") or 0)
    base_med = {k: statistics.median(v) for k, v in base_think.items()}
    base = screen(args.base, base_med)
    print("\n== 2. 720 screens, per part: accuracy | mean tokens right / wrong | cut at the limit | wrong answers that stopped early")
    for label, runs in (("A", args.runs_a), ("C", args.runs_c)):
        for run in runs:
            s = screen(run, base_med)
            print(f"  Run {label} {Path(run).name}:")
            for p in sorted(s):
                d, b = s[p], base[p]
                f = lambda xs: f"{statistics.fmean(xs):5.0f}" if xs else "    -"  # noqa: E731
                print(f"    {p:<20} acc {100 * b['ok'] / b['n']:5.1f} -> {100 * d['ok'] / d['n']:5.1f} | right {f(b['rt'])} -> {f(d['rt'])}"
                      f" | wrong {f(b['wt'])} -> {f(d['wt'])} | cut {100 * b['cut'] / b['n']:4.1f}% -> {100 * d['cut'] / d['n']:4.1f}%"
                      f" | stopped early {d['early']}/{d['nw']} wrong")
    print("\n== 3. training logs, per step: practice accuracy, tokens, share of tokens the coach pushes down, coach's mean push")
    for label, logs in (("A", args.logs_a), ("C", args.logs_c)):
        seen = {}
        for f in sorted(Path(logs).glob("step-*/rl_log.jsonl"), key=lambda p: int(p.parent.name[5:])):
            first = int(f.parent.name[5:])
            entries = read(f)
            for i, e in enumerate(entries):  # each block's log restarts its step count: place it after the block's start
                block = len(entries)
                g = first - block + e["step"]
                seen[g] = e
        print(f"  Run {label}: " + "; ".join(f"{g}: {e['accuracy']:.2f}/{e['mean_tokens']:.0f}/"
                                           f"{(e.get('copd') or {}).get('negative', 0):.2f}/{(e.get('copd') or {}).get('mean', 0):+.3f}"
                                           for g, e in sorted(seen.items())))
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "copd_analysis.json").write_text(json.dumps({k: {p: {kk: (len(vv) if isinstance(vv, list) else vv)
                                                                                 for kk, vv in d.items()} for p, d in v.items()}
                                                                        for k, v in report.items()}, indent=1))


if __name__ == "__main__":
    main()
