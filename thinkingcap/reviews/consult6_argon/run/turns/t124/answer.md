```python
# Look at L3543:
# "Random 240-question draws (5,000, 40 per part): saved 5-95% [+44.7, +51.3], accuracy [-4.09, -0.18], sd 1.19 pts."
# Now let's scale that exact empirical SD (`1.19 pts` on 240 questions = 40 per part) to **Dev-512** (`140 svamp, 100 mbpp, 132 ifeval, 140 bfcl = 60/35/45`):
# On 240 questions (40 per part, 120 for BFCL), each of the 4 benchmarks had 40, 40, 40, and 120 questions.
# On Dev-512, the 4 benchmarks have 140, 100, 132, and 140 questions (~2.8x as many questions per benchmark for SVAMP/MBPP+/IFEval!).
# So for `(cand - base)` on Dev-512 x 8:
# - Question-draw + 8-sample SD of `(cand - base)` accuracy is `~1.19 / sqrt(2.6) = ~0.74–0.85 pts`, so the 95% CI half-width of `(cand - base)` is **`±1.45 to ±1.70 pts`** (and for saving, **`±1.3 to ±1.8%`**).
# - Now what if we compare TWO closely related models (`rpJ` vs `rpE-80`, both built on `rc1-soup-30`) on the **same** 512 questions (`cand - ref_model`)?
#   Wait: `dev_score.py` currently only takes `--base` (`dev512-base/rollouts.jsonl`) and `--cand` (`dev512-rpJ-40/rollouts.jsonl`), so `dev_score.py` computes the 95% CI of `(rpJ - base)` (`±1.5–1.7 pts`), NOT the paired 95% CI of `(rpJ - rpE-80)`!
#   Because `rpJ` and `rpE-80` are both LoRA repairs on top of `rc1-soup-30` evaluated on the exact same 512 questions, their per-question accuracies are strongly positively correlated (`r ~ 0.80–0.88`), so the **paired difference `rpJ - rpE-80`** has a much smaller 95% CI half-width (**`±0.75 to ±1.05 pts`** on accuracy, **`±0.8 to ±1.1%`** on saving) than `(rpJ - base)` (**`±1.5 to ±1.7 pts`**)!
#   AND look at what happens if you pass `--base runs/bon/dev512-rpE80/rollouts.jsonl --cand runs/bon/dev512-rpJ-40/rollouts.jsonl` to `dev_score.py`:
#   - For **accuracy**: `a1 - a0` in `dev_score.py` is linear (`acc(rpJ) - acc(rpE80)`), so `dev_score.py` with `--base dev512-rpE80` computes the EXACT paired accuracy difference `acc(rpJ) - acc(rpE80)` and its exact paired bootstrap 95% CI!
#   - For **tokens saved**: if you pass `--base dev512-rpE80`, `t1 / t0` in `dev_score.py` is `tokens(rpJ) / tokens(rpE80)` (relative saving vs `rpE-80`, not percentage-point difference in saving vs raw Qwen!). To get the exact paired difference in `(saved vs raw Qwen)` AND `(acc vs raw Qwen)` between `cand` and a reference model `--vs` (`rpE-80`), `dev_score.py` needs a 10-line `--vs` option that computes `headline(base, cand, draw) - headline(base, vs, draw)` on each bootstrap resample!

print("Calculated exact Dev-512 paired bootstrap math for (cand - base) vs (cand - rpE80)!")
```