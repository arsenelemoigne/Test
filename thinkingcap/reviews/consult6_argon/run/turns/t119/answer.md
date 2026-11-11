```python
# STOP AND LOOK AT LINE 4755 OF `EXPERIMENT_LOG.md` ABOVE!
# Read Line 4755 ten times:
#   `D12 lpd-chain1 step 100: +21.7 / +0.4 ... the chain-1 rule (shortest right > longest right >= 1.5x; guard = shortest right > shortest finished wrong; balanced 40:60) trains slowly, far below 40% at step 100. Steps 200 / final next.`
#   `D12 lpd-chain1 step 200 +18.4 / +0.4 ... final +17.8 / +0.8 ... Chain-1 keeps accuracy but saves under 20%: far from the 40% bar, below D11 push2 in saving.`
#
# WAIT A MINUTE!
# Why did `D12 lpd-chain1` save ONLY **17.8%–21.7%**, whereas `D10 lpd-l2s50e` (right above it at L4708-4713, with the EXACT same `fp32+fp16` `bidax5q` settings!) saved **54.4% at step 100 and 53.5% at final**?!
# Let's compare `D10 lpd-l2s50e` (54.4% / 53.5% saved) vs `D12 lpd-chain1` (21.7% / 17.8% saved) at L4708 and L4755:
# 1. Both used the EXACT SAME training settings (`bidax5q s0` settings: `fp32 + fp16 autocast`, `LoRA r32`, `beta 0.1`, `NLL 0`, `cosine lr`).
# 2. Why did `D10 lpd-l2s50e` save **54.4% / 53.5%**, while `D12 lpd-chain1` saved only **21.7% / 17.8%**?!
#    Look at the two differences in the pair rule:
#    (a) **Shorten share:**
#        - In `D10 lpd-l2s50e`, the shorten cap was **2x guard pairs (`67% shorten / 33% guard`, i.e., 2:1 shorten:guard, L3230)**, with easy ratio `1.2x` on always-solved questions!
#        - In `D12 lpd-chain1`, the share was **40% shorten / 60% guard** (and in `PATH1_PLAN.md`'s `--mix-preset round1`, shorten is **39.4% shorten / 46.9% equal-length right / 13.7% capped guard**)!
#    (b) **Rejected answer in Guard pairs:**
#        - In `D10 lpd-l2s50e`, `guard` was `shortest right > longest wrong` (`long2short` on `p < 0.5`).
#        - In `D12 lpd-chain1`, `guard` was `shortest right > shortest finished wrong` (and in `PATH1_PLAN.md` `--kinds mix`, `right` is equal-length `right_long`/`right_short` and `capguard` has `chosen > 2x rejected`!).
#
# Let's check what happened to `D12 lpd-chain1r2` (Round 2 of `D12 lpd-chain1`)! Where is `chain1r2` in `EXPERIMENT_LOG.md`? Let's search!
for i, l in enumerate(log_lines):
    if "chain1" in l:
        print(f"L{i+1}: {l.strip()[:130]}")
```