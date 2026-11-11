```python
# Let's also check `dev_score.py` for how it compares `rpJ` against `rpE-80` (Open Question 1 & Claude's point 5)!
# Look at `dev_score.py`:
#   `python dev_score.py --base runs/bon/dev512-base/rollouts.jsonl --cand runs/bon/dev512-rpJ-40/rollouts.jsonl --name rpJ-40`
# What does `dev_score.py` compute?
# It computes the paired bootstrap 95% interval of `(cand - base)`!
# Wait: what is the half-width of `(cand - base)` on Dev-512 (512 questions x 8 answers)?
# Let's calculate the exact standard error and 95% CI half-width of `(cand - base)` vs `(cand - ref_model)` (i.e. `rpJ - rpE-80`) on Dev-512!
# In `EXPERIMENT_LOG.md` L3523-3530, we have the exact 5-95% ranges for 40 questions per part (240 questions total):
# Let's check L3520-3536 of `EXPERIMENT_LOG.md`!
print("".join(log_lines[3520:3537]))
```