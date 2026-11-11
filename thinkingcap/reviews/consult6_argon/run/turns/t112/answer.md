```python
# Let's carefully check EVERY claim in `CLAUDE_REVIEW.md` against `EXPERIMENT_LOG.md` and `all_runs.json`!
# Look at the claims in `CLAUDE_REVIEW.md`:
#
# Claim C1: "Round 1 is a correctness-heavy, moderate-dose DPO from the original model; the log's closest precedents land near 30% saved, not 45-50%:
#   - lpc-runa2 (original's own answers, per-type shorten/guard/right mix, 4,442 pairs, 275 steps, 3 seeds) plateaued at 31-33% (L3553-3617; runb / runac 23-30%);
#   - D6 simright (similar-length right>wrong pairs) halved the shortening (34 -> 28%, L4566-4574);
#   - fp32 alone cost ~11 points of saving at equal pairs (bidax6p, L4662-4668).
#   - Round 1's preset has only ~39% shorten pairs. Expected round 1: ~25-35% saved."
#
# Let's check `EXPERIMENT_LOG.md` at L3550-3625 (`lpc-runa2`, `runb`, `runac`), L4560-4580 (`D6 simright`), and also check ALL runs from the original Qwen to see what REALLY determines whether Round 1 lands at 30% saved vs 50-55% saved!
print("--- L3550-3620 (lpc-runa2, runb, runac, lpc-bida) ---")
print("".join(log_lines[3549:3620]))
print("--- L4560-4585 (D6 simright, D2, D3, D4, D5) ---")
print("".join(log_lines[4559:4585]))
```