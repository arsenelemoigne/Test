"""Where the Marginal Value Theorem cut falls against the settle cut, for several travel costs.

Reads the log.jsonl prune_traces.py writes (every correct rollout's p(gold) curve per step and
the thinking tokens after each step), so the sweep needs no GPU:

    python mvt_report.py runs/prune-mvt256/log.jsonl --travel 32,128,256,512,1024,2048
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from prune_traces import median, mvt_index, settle_index


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("log")
    parser.add_argument("--travel", default="32,128,256,512,1024,2048")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--smooth", type=int, default=2)
    args = parser.parse_args()
    entries = [json.loads(line) for line in Path(args.log).read_text().splitlines() if line.strip()]
    entries = [e for e in entries if "positions" in e]
    print(f"{'travel':>7}{'source':>16}{'n':>6}{'full':>7}{'settle':>8}{'mvt':>7}{'mvt-settle':>12}"
          f"{'earlier':>9}{'later':>7}{'mvt p':>7}")
    for travel in [float(t) for t in args.travel.split(",")]:
        by_source = defaultdict(list)
        for e in entries:
            settle = settle_index(e["curve"], args.threshold)
            if settle is None:
                continue
            mvt = mvt_index(e["curve"], e["positions"], travel, args.smooth)
            row = (e["positions"][-1], e["positions"][settle], e["positions"][mvt], max(e["curve"][:mvt + 1]))
            by_source[e["source"]].append(row)
            by_source["ALL"].append(row)
        for source in sorted(by_source, key=lambda s: (s == "ALL", s)):
            rows = by_source[source]
            print(f"{travel:>7.0f}{source:>16}{len(rows):>6}{median([r[0] for r in rows]):>7.0f}"
                  f"{median([r[1] for r in rows]):>8.0f}{median([r[2] for r in rows]):>7.0f}"
                  f"{median([r[2] - r[1] for r in rows]):>+12.0f}"
                  f"{sum(r[2] < r[1] for r in rows) / len(rows):>9.0%}"
                  f"{sum(r[2] > r[1] for r in rows) / len(rows):>7.0%}"
                  f"{median([r[3] for r in rows]):>7.2f}")
    print("\nmedians of thinking tokens kept; earlier/later: share of rollouts where the MVT cut comes "
          "before/after the settle cut; mvt p: median p(gold) secured at the MVT cut")


if __name__ == "__main__":
    main()
