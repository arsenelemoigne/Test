```python
# Look closely at the numbers across steps 20, 40, and 60 of rpK!
# This is a critical insight that separates REAL dose-response mechanics from MC sampling luck:
# Look at what happens from rpE80 -> rpK-20 -> rpK-40 -> rpK-60:
# 1. Saving:
#    - From step 0 (rpE80: 47.04%) to step 20 (49.88%): +2.83% saving (mostly SVAMP +3.44%, MBPP+ +4.82%, BFCL +2.90%; IFEval -0.14%).
#    - From step 20 (49.88%) to step 40 (51.17%): +1.30% saving (SVAMP +0.73%, MBPP+ +1.32%, IFEval +0.38%, BFCL +2.85%).
#    - From step 40 (51.17%) to step 60 (54.43%): +3.26% saving (SVAMP +2.81%, MBPP+ +2.98%, IFEval +3.69%, BFCL +3.67%).
#    Wait: why did step 40 -> 60 compress TWICE as fast (+3.26% saving) as step 20 -> 40 (+1.30% saving)?
#    Look at `raw_logs/gpu2_night_key_lines.txt`:
#    In rpK training:
#    step 20/60: loss 0.6724, pairs ordered right 69%
#    step 40/60: loss 0.6370, pairs ordered right 68%
#    step 55/60: loss 0.5717, pairs ordered right 80%
#    step 60/60: loss 0.5973, pairs ordered right 72%
#    And notice on Dev-512 (`gpu2_night_key_lines.txt`):
#    rpK-20 Dev-512: 50.52% saved (vs rpE80 +3.59%)
#    rpK-40 Dev-512: 50.40% saved (vs rpE80 +3.47%)
#    rpK-60 Dev-512: 53.78% saved (vs rpE80 +6.84%)
#    On BOTH Dev-512 AND the Full Test, `rpK-40` sits at only ~30-38% of the way from `rpK-20` to `rpK-60` in compression (`51.17%` vs `49.88%` and `54.43%`), NOT 50% of the way (`52.16%`)!
# 2. Now look at WHY `rpK-40`'s accuracy (`-0.77`, or `-0.75` depending on rounding) is `+0.51` pts above `rpK-20` (`-1.28`) and `+0.67` pts above `rpK-60` (`-1.44`), i.e., `+0.59` pts above the midpoint `-1.36`:
#    Let's break that `+0.59` pt midpoint bump (`-0.77` vs `-1.36`) into its 4 benchmarks:
#    (a) BFCL Pooled (`+0.44` pts bump = `+0.11` headline pts):
#        This is 100% REAL (structural, not luck)! Why? Look at the 3 sub-parts of BFCL:
#        - `bfcl_irrelevance` rises monotonically with step as the 140 non-call pairs take effect: `+3.44` (rpE80) -> `+3.65` (K20) -> `+4.69` (K40) -> `+6.20` (K60).
#        - `bfcl_multiple` recovers after step 20: `+0.06` (rpE80) -> `-1.50` (K20) -> `-0.81` (K40) -> `-1.00` (K60).
#        - `bfcl_simple_python` holds flat up to step 40 (`+0.09` rpE80 -> `+0.28` K20 -> `+0.25` K40, call rate `97.00% -> 97.19% -> 96.66%`), and ONLY collapses between step 40 and step 60 (`+0.25 -> -1.47`, as call rate drops to `95.75%` when the 140 irrelevance pairs over-shift the shared call threshold)!
#        So at step 40, `bfcl_irrelevance` has already gained `+1.04` pts over step 20 (`+3.65 -> +4.69`) and `bfcl_multiple` has recovered `+0.69` pts (`-1.50 -> -0.81`), BEFORE `bfcl_simple_python` pays its `-1.72` pt non-call penalty at step 60!
#    (b) MBPP+ Code (`+0.61` pts bump = `+0.15` headline pts):
#        Partly real, partly noise! From step 20 to step 40, MBPP+ thinking barely moved (`991.2 -> 962.7` tokens, `-2.9%`) and accuracy stayed right at `-0.26` / `-0.13` (vs `-1.03` in rpE80, helped by the 114 `right_cr` code pairs). Between step 40 and step 60, cross-type shortening pulled MBPP+ thinking down another `-65.9` tokens (`962.7 -> 896.8`, `-6.8%`), where MBPP+ dropped `-1.09` pts (`-0.13 -> -1.22`).
#    (c) SVAMP Math (`+0.13` pts bump = `+0.03` headline pts):
#        SVAMP is completely flat from step 20 to 40 to 60 (`-2.59 -> -2.34 -> -2.35`).
#    (d) IFEval (`+1.17` pts bump = `+0.29` headline pts: `-3.07` at step 20, `-1.87` at step 40, `-3.00` at step 60):
#        Wait: why did IFEval jump `+1.20` pts at step 40 (`-3.07 -> -1.87`) and then fall `-1.13` pts at step 60 (`-1.87 -> -3.00`), whereas on Dev-512 IFEval moved monotonically `-0.7 -> -2.4 -> -3.4`?
#        Let's check IFEval in rpK-40 in detail! Why did `rpK-40` score `-1.87` on Full-Test IFEval?

# Let's check which questions in IFEval differed between rpK-40 and (rpK-20, rpK-60), and how much of IFEval's +1.17 bump is response length (`241.5` vs `229.7` and `229.6`) vs pure MC sampling noise!
if_ans = ft_ans[ft_ans['dataset'] == 'ifeval'].copy()
for m in ['rpE80', 'rpK20', 'rpK40', 'rpK60', 'rpL60']:
    g = if_ans[if_ans['model'] == m]
    # MC SE of IFEval official accuracy (fixed 541 questions, 8 samples per question):
    q_var = g.groupby('id')['right_off'].apply(lambda x: np.var(x * 100.0, ddof=1) / 8.0)
    mc_se = np.sqrt(q_var.sum()) / len(q_var)
    print(f"{m:6s} | IFEval off_acc: {g['right_off'].mean()*100:.2f}% | Pure 8-sample MC SE: {mc_se:.3f} pts | rsp_tokens: {g['response_tokens'].mean():.1f}")
```