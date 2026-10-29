"""Topic check (user, 28 Oct: "does the type of question, materially speaking, match the eval?"). Report only.

The pool was matched on behaviour (accuracy, thinking length), not on subject. Here, per evaluation part, every question
is embedded (all-MiniLM-L6-v2, the user turn only, with each part's template lines removed: lines found in more than 30%
of a part's questions, e.g. MBPP's instruction) and two nearest-neighbour similarities are compared:
  eval -> rest of eval   each evaluation question's closest OTHER evaluation question of its part
  eval -> pool           its closest pool question, the pool subsampled to the same size as the evaluation part
If the pool covers the evaluation's subjects as densely as the evaluation covers itself, the two distributions match.
Also reported: pool -> eval (is a typical pool question as close to the evaluation as an evaluation question is), and the
share of evaluation questions whose closest pool question is less similar than the evaluation's own 10th percentile
(subjects the pool misses). The fresh set's questions are left out (scores only).

    python topic_check.py --pool data/pool-dp10.jsonl --questions questions --exclude questions_fresh20261022 --out runs/bon/topic-dp10
"""

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

PARTS = ("svamp", "mbpp", "bfcl_simple_python", "bfcl_multiple", "bfcl_irrelevance", "ifeval")


def read(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def user_text(messages):
    return "\n".join(m["content"] for m in messages if m["role"] == "user")


def strip_templates(texts, share=0.3):
    lines = Counter(l.strip() for t in texts for l in set(t.splitlines()) if l.strip())
    common = {l for l, n in lines.items() if n > share * len(texts)}
    return ["\n".join(l for l in t.splitlines() if l.strip() not in common) or t for t in texts]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pool", required=True)
    parser.add_argument("--questions", default="questions", help="every evaluation question")
    parser.add_argument("--exclude", nargs="*", default=[], help="question sets left out (the fresh set)")
    parser.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    import numpy as np
    from sentence_transformers import SentenceTransformer

    rng = random.Random(args.seed)
    skip = {q["id"] for e in args.exclude for q in read(Path(e) / "questions.jsonl")}
    evals = defaultdict(list)
    for q in read(Path(args.questions) / "questions.jsonl"):
        if q["id"] not in skip:
            evals[q["dataset"]].append(user_text(q["messages"]))
    pool = defaultdict(list)
    for r in read(args.pool):
        pool[r["part"]].append(user_text(r["prompt"]))
    encoder = SentenceTransformer(args.encoder)
    report, lines = {}, ["topic check: median nearest-neighbour cosine per part (higher = closer subjects)",
                         "part: eval->rest-of-eval | eval->pool (same size) | pool->eval vs eval->eval | eval questions the pool misses"]
    for part in PARTS:
        ev = evals.get(part, [])
        if len(ev) < 20 or not pool.get(part):
            lines.append(f"  {part}: not enough questions")
            continue
        sample = rng.sample(pool[part], min(len(ev), len(pool[part])))
        texts = strip_templates(ev + sample)
        vec = encoder.encode(texts, normalize_embeddings=True, batch_size=128, show_progress_bar=False)
        e, p = vec[:len(ev)], vec[len(ev):]
        ee = e @ e.T
        np.fill_diagonal(ee, -1)
        e_e = ee.max(1)
        e_p = (e @ p.T).max(1)
        p_e = (p @ e.T).max(1)
        floor = float(np.quantile(e_e, 0.1))
        missed = float((e_p < floor).mean())
        med = lambda a: float(statistics.median(a.tolist()))
        report[part] = {"eval_n": len(ev), "pool_n": len(sample), "eval_to_eval": med(e_e), "eval_to_pool": med(e_p),
                        "pool_to_eval": med(p_e), "eval_p10": floor, "eval_missed_share": missed}
        verdict = "OK" if med(e_p) >= med(e_e) - 0.05 and missed <= 0.2 else "CHECK"
        report[part]["verdict"] = verdict
        lines.append(f"  {part}: {med(e_e):.2f} | {med(e_p):.2f} | {med(p_e):.2f} vs {med(e_e):.2f} | "
                     f"{missed:.0%} below the eval's own 10th percentile ({floor:.2f}) -> {verdict}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
