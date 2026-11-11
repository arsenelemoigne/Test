```python
# Let's check even more details in `repair_pairs.py`, `dev_score.py`, `track_ptrepair_odpo_dev.py`, and `queued_lines.txt`!
# Look at `repair_pairs.py` Line 189:
#     def take(part, kind, pool, n):
#         rng.shuffle(pool)
#         got = []
#         for qid, c, w in pool:
#             if len(got) >= n:
#                 break
#             key = (id(c), id(w))
#             if key in used or per_q[qid] >= 3 or (kind != "guard" and right_q[qid] >= 2):
#                 continue
#             used.add(key)
#             per_q[qid] += 1
#             right_q[qid] += kind != "guard"
#             ...
# WAIT A SECOND!
# Look at Lines 124-126 in `repair_pairs.py`:
#   if (len(rights) >= 2 and len(rights) / len(q) >= args.shorten_min_right and not is_cr
#           and size(rights[-1]) >= args.shorten_ratio * size(rights[0])):
#       cand[part]["shorten"].append((qid, rights[0], rights[-1]))
# How many `shorten` candidates can a single question `qid` contribute?
# AT MOST ONE (`(qid, rights[0], rights[-1])`)!
# Now let's check the question counts in `rpH` (used by `rpJ`), `p1src` (used by `p1r1`), and `p1r2`!
# 1) In `rpJ` (reusing `rpH`'s rollouts):
#    How many questions per part did `rpH` answer?
#    Look at `track_ptrepair_odpo_dev.py` Line 203 (`sizes0`):
#      `sizes0 = {"svamp": 2400, "mbpp": 900, "ifeval": 1500, "bfcl_simple_python": 300, "bfcl_multiple": 300, "bfcl_irrelevance": 300}`
#    Now look at `rpJ`'s quota in `queued_lines.txt`:
#      `--rp-quota svamp=320,mbpp=320,ifeval=320,bfcl_simple_python=100,bfcl_multiple=80,bfcl_irrelevance=140`
#    Wait: for `bfcl_irrelevance` (`no-fit`), `rpH` only answered **300 questions**!
#    In `rpJ`, `bfcl_irrelevance` has quota `n = 140`:
#    - `n_sh = round(0.30 * 140) = 42` shorten pairs (requires `>= 6/8` right AND `rights[-1] >= 1.3 * rights[0]`).
#    - `n_g = round(0.15 * 140) = 21` capped guard pairs (requires mixed outcome AND `w < 0.5 * med`).
#    - `n_r = 140 - len(sh) - len(g) >= 77` equal-length right pairs (requires mixed outcome AND `max/min <= 1.2`).
#    Wait: in `rpH` (300 `no-fit` questions answered by `rc1-soup-30`), how many of those 300 `no-fit` questions are mixed-outcome (`1/8 <= p <= 7/8`)?
#    Earlier we saw that in `rpI` (300 `no-fit` questions), only **59 questions** were mixed-outcome (`no-fit 59/80`, L5172), and in `rpF`, `tools 60-71/80` (L5188)!
#    Even if a mixed-outcome question gives up to 2 `right` pairs (`right_q[qid] >= 2`), how many of those ~59 mixed-outcome `no-fit` questions have a right and wrong answer within `--ratio 1.2` (`±20%` length)? Probably only ~25–35 questions!
#    And what about `capguard` (`w < 0.5 * med`)? On `no-fit` (where `rc1-soup-30` median thinking is already ~90 tokens) and `simple`/`multiple` (~100 tokens), very few finished wrong answers are `< 0.5 * med` (~45 tokens)!
#
# 2) Now let's check `p1src` (Round 1 from stored 10 answers of raw Qwen, 60% hash split) in `track_ptrepair_odpo_dev.py` Lines 408-464!
#    WAIT A SECOND! Let's check how many questions per part exist in `P1_ANSWERS` (`lupin-all-dp13-nofit1-tools1-code1-nofit2-nofit4-math1-math2.jsonl`) after `p1src` drops:
#    - `metamath_*`
#    - `nofit4`, `math1`, `math2`
#    - `Dev-512` (`140 svamp, 100 mbpp, 132 ifeval, 60 simple, 35 multiple, 45 no-fit`)
#    - AND drops 40% of the remaining questions (`hash % 10 >= 6`)!
#    HOLD EVERYTHING! Let's check from `EXPERIMENT_LOG.md` how many questions per part are in `pool-dp13` + `nofit1, tools1, code1, nofit2` (without `metamath`, `nofit4`, `math1`, `math2`)!
#    Let's search `EXPERIMENT_LOG.md` for `dp13`, `nofit1`, `tools1`, `code1`, `nofit2`, and the pool sizes in `L5080`!

for i, l in enumerate(log_lines):
    if any(k in l for k in ["dp13", "nofit1", "tools1", "code1", "nofit2", "Available per part"]):
        print(f"L{i+1}: {l.strip()[:130]}")
```