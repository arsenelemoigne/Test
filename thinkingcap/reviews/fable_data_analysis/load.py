import pickle, json, statistics as st, sys
from collections import defaultdict, Counter
from pathlib import Path
S = Path("/private/tmp/claude-501/-Users-arsenelemoigne-Claude-Build/a053225a-bb13-4ed1-b9d0-c21572b835cb/scratchpad")
M = {n: pickle.load(open(S / f"an/{n}.pkl", "rb")) for n in ("base", "nothink", "rc1", "rpE80", "rpI120")}
Q = {}
for l in open(S / "tct/questions/questions.jsonl"):
    q = json.loads(l); Q[q["id"]] = q
def by_q(rows, ds):
    d = defaultdict(list)
    for r in rows:
        if r["ds"] == ds: d[r["id"]].append(r)
    return d
def mean(x): return sum(x) / max(1, len(x))
def med(x): return st.median(x) if x else float("nan")
