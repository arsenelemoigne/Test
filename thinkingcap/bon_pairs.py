"""DPO pairs from the best-of-N rollouts already sampled: no new generation.

Per question (only those kept by select_shortest.py and the contamination check):
  shorten  shortest correct reply  beats  the longest correct reply (at least --ratio times longer)
  guard    shortest correct reply  beats  the shortest wrong reply (finished, not cut by the cap):
           brevity must not come from giving up

    python bon_pairs.py --rollouts runs/bon/probe/rollouts.jsonl --keep runs/bon/select-r1/sft.jsonl --out runs/bon/pairs
    python dpo.py --pairs runs/bon/pairs/pairs.jsonl --init-adapter adapters/bon-r1 --out adapters/bon-dpo --mix 0.6,0.4,0
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from probe_curve import read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--keep", required=True, help="sft.jsonl whose question ids are allowed")
    parser.add_argument("--out", required=True)
    parser.add_argument("--ratio", type=float, default=1.5)
    args = parser.parse_args()

    allowed = {r["id"] for r in read_jsonl(Path(args.keep))}
    by_question = defaultdict(list)
    for r in read_jsonl(Path(args.rollouts)):
        if r["id"] in allowed:
            by_question[r["id"]].append(r)
    pairs, kinds, lengths = [], Counter(), defaultdict(list)
    for qid, group in by_question.items():
        done = [r for r in group if r["closed_thinking"] and not r["truncated"]]
        right = sorted((r for r in done if r["correct"]), key=lambda r: r["output_tokens"])
        wrong = sorted((r for r in done if not r["correct"]), key=lambda r: r["output_tokens"])
        if not right:
            continue
        base = {"id": qid, "source": group[0]["source"], "prompt_ids": right[0]["prompt_ids"],
                "chosen_ids": right[0]["output_ids"]}
        if len(right) >= 2 and right[-1]["output_tokens"] >= args.ratio * right[0]["output_tokens"]:
            pairs.append(base | {"kind": "shorten", "rejected_ids": right[-1]["output_ids"]})
            kinds["shorten", base["source"]] += 1
            lengths[base["source"]].append((right[0]["output_tokens"], right[-1]["output_tokens"]))
        if wrong:
            pairs.append(base | {"kind": "guard", "rejected_ids": wrong[0]["output_ids"]})
            kinds["guard", base["source"]] += 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "pairs.jsonl", pairs)
    lines = [f"{'source':<16}{'shorten':>9}{'guard':>7}{'chosen':>8}{'rejected':>10}"]
    for source in sorted(lengths.keys() | {s for _, s in kinds}):
        ls = lengths[source]
        lines.append(f"{source:<16}{kinds['shorten', source]:>9}{kinds['guard', source]:>7}"
                     f"{sum(c for c, _ in ls) / max(1, len(ls)):>8.0f}{sum(r for _, r in ls) / max(1, len(ls)):>10.0f}")
    lines.append(f"total {len(pairs)} pairs; chosen/rejected: mean tokens of shorten pairs")
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text + "\n")
    print(text, flush=True)


if __name__ == "__main__":
    main()
