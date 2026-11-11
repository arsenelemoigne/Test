# Full-test record analysis behind REPAIR_IFEVAL_TOOLS.md (11 Nov 2026)

Inputs (downloaded from Drive, kept on the Mac in ~/Downloads/tc_full, not in git: 200 MB - 1.8 GB each):
- `runs/mvtf/full-base`, `full-base-nothink`, `full-rc1-soup-30`, `full-rpE-80`, `full-rpI-120` : `records.jsonl` (22,072 answers each)
- `runs/bon/rp-rpH`, `rp-rpI` : `rollouts.jsonl` (rc1-soup-30's and rpE-80's own answers, 5,700 questions x 8)
- `runs/bon/lupin-all-dp13-nofit1-tools1-code1-nofit2-nofit4-math1-math2.jsonl` (the original's 10 stored answers)
- the 2,759 test questions: `python utils/eval/benchmarks.py` in a clone of BottleCapAI/ThinkingCap-Test

`build.py SCRATCH RECORDS_DIR base nothink rc1 rpE80 rpI120` re-grades every IFEval answer rule by rule with Google's official
checkers (needs absl-py, langdetect, nltk, immutabledict) and classes every tool-call reply (call / empty list / prose);
`load.py` loads the tables. The tables in REPAIR_IFEVAL_TOOLS.md were printed from these with short inline scripts (per-rule
pass rates, the four IFEval classes, words / N on "at least N words", thinking-length percentiles, the 28 near-fit no-fit
questions, accuracy among parseable calls with thinking off).
