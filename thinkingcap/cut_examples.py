"""A few readable examples of calibrated cuts: per data type, one question shown at every cut point, with the end of the
kept thinking, how many of the 20 re-answers were right, and one answer written from there.

    python cut_examples.py runs/bon/pairs-rc2/cal --out runs/viz/cut_examples.txt
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

FRACTIONS = [0.01, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8]


def read(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def short(text, n):
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + " ..."


def tail(text, n):
    text = " ".join(text.split())
    return text if len(text) <= n else "... " + text[-n:]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("cal")
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-type", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    cal = Path(args.cal)
    cuts = read(cal / "cuts.jsonl")
    verdicts = {v["key"]: v for v in read(cal / "verdicts.jsonl")}
    replies = defaultdict(list)  # question id -> [(fraction, cut row, verdict)]
    for k, c in enumerate(cuts):
        replies[c["row"]["id"]].append((FRACTIONS[k % len(FRACTIONS)], c, verdicts[k]))
    by_source = defaultdict(list)
    for qid, rows in replies.items():
        by_source[qid.rsplit("-", 1)[0]].append(qid)
    rng = random.Random(args.seed)
    lines = ["Examples of cuts (model: the one that wrote and re-answered these replies). For each cut: share of the thinking",
             "kept, its length in tokens, the last words kept before '</think>' was inserted, right re-answers out of 20, and",
             "one answer the model wrote from that point.", ""]
    for source in sorted(by_source):
        qids = by_source[source]
        # prefer questions whose cuts go from failing to safe: they show where the answer appears
        mixed = [q for q in qids if min(v["right"] for _, _, v in replies[q]) <= 5 and max(v["right"] for _, _, v in replies[q]) == 20]
        for qid in rng.sample(mixed or qids, min(args.per_type, len(mixed or qids))):
            rows = sorted(replies[qid], key=lambda r: r[0])
            prompt = tok.decode(rows[0][1]["prompt_ids"], skip_special_tokens=True)
            lines += ["=" * 110, f"{source}  ({qid})", "QUESTION: " + short(prompt.split("user", 1)[-1], 400), ""]
            for f, c, v in rows:
                think = tok.decode(c["thinking_ids"], skip_special_tokens=False).replace("<think>", "").replace("</think>", "")
                answer = tok.decode(v["right_ids"] or v["wrong_ids"] or [], skip_special_tokens=True)
                label = "SAFE" if v["right"] == v["of"] else ("unsafe" if v["right"] <= 15 else "in between")
                lines.append(f"-- cut at {f:.0%}: {len(c['thinking_ids'])} thinking tokens kept | right {v['right']}/{v['of']} ({label})")
                lines.append("   last words kept : " + tail(think, 220))
                lines.append(f"   an answer from here ({'right' if v['right_ids'] else 'wrong'}): " + short(answer, 220))
            lines.append("")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
