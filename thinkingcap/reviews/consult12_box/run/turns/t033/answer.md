```python
# Look at `Dev-512 vs Full Test Comparison` for `rpL-60`:
# Dev-512: 58.86% / +0.92 (vs rpE80: +11.93% / -0.44)
# Full Test: 59.27% / -1.69 (vs rpE80: +12.23% / -0.74)
# And for `rpK-60`:
# Dev-512: 53.78% / +0.76 (vs rpE80: +6.84% / -0.60)
# Full Test: 54.43% / -1.44 (vs rpE80: +7.39% / -0.49)
# Notice how remarkably consistent the Full-Dev offset is across `rpE80`, `rpK-20`, `rpK-60`, `rpL-60`:
# - Saving offset (Full - Dev): +0.11%, -0.64%, +0.76%, +0.66%, +0.41% (mean +0.26%!)
# - Accuracy offset (Full - Dev):
#   rpE80: -2.31
#   rpK-20: -2.79
#   rpK-60: -2.20
#   rpL-60: -2.61
#   Mean (excluding rpK-40): -2.48 pts!
#   And for `rpK-40`, Full - Dev offset is -1.72 pts (+0.76 pts higher than the -2.48 mean!).

# Let's do a complete statistical analysis of Question 1:
# "1. How much to trust rpK-40 (51.2 / -0.8): luck versus real, given its neighbours (steps 20 and 60) and the noise of one full test. What would confirm it (a second seed? a re-run of the full test with another sampling seed is NOT allowed: the evaluation is fixed), and is it a submittable result as is?"

# Wait: let's decompose `rpK-40`'s `51.17% saved / -0.77 (-0.75 before rounding? wait: let's check exact rounding of rpK-40)` into:
# (a) What part of rpK-40 vs rpK-20 and rpK-60 is REAL structural dose dynamics between step 20, step 40, and step 60?
# (b) What part of rpK-40 is finite-sampling (8 MC draws per question) luck?
# Let's check the exact numbers for rpK-20, rpK-40, rpK-60 across all 6 sub-benchmarks!

for ds in ['svamp', 'mbpp', 'ifeval', 'bfcl_simple_python', 'bfcl_multiple', 'bfcl_irrelevance', 'bfcl_pooled']:
    s20 = all_stats['rpK20'][ds]
    s40 = all_stats['rpK40'][ds]
    s60 = all_stats['rpK60'][ds]
    if ds == 'ifeval':
        # use official
        a20 = -3.07; a40 = -1.87; a60 = -3.00
    else:
        a20 = s20['d_acc']; a40 = s40['d_acc']; a60 = s60['d_acc']
    print(f"{ds:18s} | Sav: 20={s20['saved']:5.2f}%, 40={s40['saved']:5.2f}%, 60={s60['saved']:5.2f}% | dAcc: 20={a20:+6.2f}, 40={a40:+6.2f}, 60={a60:+6.2f} | 40 vs avg(20,60): Sav {s40['saved']-0.5*(s20['saved']+s60['saved']):+5.2f}%, Acc {a40-0.5*(a20+a60):+5.2f}")
```