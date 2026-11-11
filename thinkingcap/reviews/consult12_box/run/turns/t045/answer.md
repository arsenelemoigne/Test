```python
# Look at `code/track_excerpts.py`!
# Look at how `ptrepair` works when `--rp-rollouts-from` and `--rp-extra-questions` or own questions are used:
# WAIT A SECOND!
# In `rpM` (currently running on GPU 2 right after `devsvamp` and `devscore`):
# `rpM` has `--rp-rollouts-from rpI --rp-extra-questions svamp=1200 --rp-seed 31`.
# So `rpM` saves its 1,200 SVAMP-like rollouts to `runs/bon/rp-rpM-extra/rollouts.jsonl`!
# AND what if we run `rpN` on top of `rpL-60` (or `rpK-60`)?
# Let's check TWO HUGE QUESTIONS for `rpN`:
# Question A: Should `rpN` be trained on top of **`rpL-60` (`59.27% saved / -1.69 official, -1.55 finished-only`)** OR on top of **`rpK-60` (`54.43% saved / -1.44 official, -1.44 finished-only`)**?
# Question B: For `rpN`, should `rpL-60` (or `rpK-60`) generate its OWN fresh answers (`3,000-3,200 questions x 8`, taking ~0.95 GPU-hours for `rpL-60` since `rpL-60` is 59.3% shorter than base!) OR reuse `rpI` + `rpM-extra` rollouts?
# Let's analyze BOTH Question A and Question B with hard numbers from the tables!

# First, let's check what happens when a shortened model (`rpE-80`) is repaired with accuracy-only pairs on its OWN fresh answers (`rpI-120` on `rpE-80`) vs what happens on `rpL-60` vs `rpK-60`:
# In `rpI-120`, `rpE-80` generated `4,500 x 8` fresh answers on unseen questions (`rp-rpI/rollouts.jsonl`), built accuracy-only pairs (`guard` = shortest right > shortest finished wrong), and trained for 120 steps (`lr=1e-5`).
# Let's check the exact full-test transition of `rpE-80 -> rpI-120` in `all_stats`:
print("=== Exact Full-Test Transition of rpE-80 -> rpI-120 (Accuracy-Only Repair on Own Answers) ===")
for p in ['svamp', 'mbpp', 'ifeval', 'bfcl_simple_python', 'bfcl_multiple', 'bfcl_irrelevance', 'bfcl_pooled']:
    e = all_stats['rpE80'][p]
    i = all_stats['rpI120'][p]
    if p == 'ifeval':
        e_acc = -2.56; i_acc = -1.57
    else:
        e_acc = e['d_acc']; i_acc = i['d_acc']
    print(f"{p:18s} | rpE80: {e['saved']:5.2f}% / {e_acc:+5.2f} (thk {e['thk']:6.1f}+rsp {e['rsp']:5.1f}) -> rpI120: {i['saved']:5.2f}% / {i_acc:+5.2f} (thk {i['thk']:6.1f}+rsp {i['rsp']:5.1f}) | Delta: Sav {i['saved']-e['saved']:+5.2f}%, Acc {i_acc-e_acc:+5.2f}")
print(f"Headline           | rpE80: 47.04% / -0.95 -> rpI120: 41.91% / +0.13 | Delta: Sav -5.13%, Acc +1.08 (Slope: +0.211 acc per -1% saved)")
```