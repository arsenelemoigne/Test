```python
# Look closely at what happened in rpK-40 vs rpK-20 and rpK-60!
# Why did rpK-40 have +0.84 pts bump over 0.5*(rpK-20 + rpK-60) on bfcl_simple_python, +1.17 pts on ifeval, +0.61 pts on mbpp, +0.44 pts on bfcl_multiple?
# Let's inspect each of those mechanisms in detail!
# 1) On bfcl_simple_python:
#    Look at call rate and accuracy:
#    rpE80: acc 82.25%, call 97.00%
#    rpK20: acc 82.44% (+0.28), call 97.19%
#    rpK40: acc 82.41% (+0.25), call 96.66%
#    rpK60: acc 80.69% (-1.47), call 95.75%
#    Wait: between step 20 and step 40, bfcl_simple_python did NOT drop at all (+0.28 -> +0.25)! The -1.72 pt drop on bfcl_simple_python happened ONLY between step 40 and step 60 (as the 140 irrelevance non-call pairs over-dosed the model into declining borderline calls between step 40 and step 60)!
# 2) On MBPP+:
#    rpE80: acc 51.42% (-1.03), thk 1089.8
#    rpK20: acc 52.18% (-0.26), thk 991.2
#    rpK40: acc 52.31% (-0.13), thk 962.7
#    rpK60: acc 51.22% (-1.22), thk 896.8
#    Wait: between step 20 and step 40, MBPP+ held steady at -0.26 / -0.13 while thinking was ~963-991 tokens! Then between step 40 and step 60, cross-type shortening pushed MBPP+ thinking down another -66 tokens (962.7 -> 896.8), and MBPP+ accuracy dropped by -1.09 pts (-0.13 -> -1.22)!
# 3) What about IFEval? Let's break down IFEval by `ifeval_class` and rule across rpE80, rpK20, rpK40, rpK60, rpL60!

if_q = ft_q[ft_q['dataset'] == 'ifeval'].copy()
# Let's merge official prompt_pass per question for each model
off_q = ft_if_full.groupby(['model', 'id', 'sample'])['passed'].all().groupby(level=['model', 'id']).mean() * 100.0
off_q = off_q.unstack('model')
off_q.index = off_q.index.astype(int)
if_q['id_int'] = if_q['id'].astype(int)
if_q = if_q.merge(off_q, left_on='id_int', right_index=True, suffixes=('', '_off'))

print("=== IFEval by ifeval_class (Official Prompt Accuracy & Thinking+Response Tokens) ===")
for cls, g in if_q.groupby('ifeval_class'):
    print(f"\nClass: {cls} (N={len(g)})")
    for m in ['base', 'rpE80', 'rpK20', 'rpK40', 'rpK60', 'rpL60', 'p1r1s0']:
        acc_off = g[m].mean()
        d_off = acc_off - g['base'].mean()
        acc_fin = (g[f'{m}:right_of_8'] / 8.0 * 100.0).mean()
        thk = g[f'{m}:thinking'].mean()
        rsp = g[f'{m}:response'].mean()
        run = (g[f'{m}:runaway_of_8'] / 8.0 * 100.0).mean()
        print(f"  {m:7s}: off {acc_off:5.2f}% ({d_off:+6.2f}) | fin {acc_fin:5.2f}% | thk {thk:5.1f} + rsp {rsp:5.1f} | run {run:4.2f}%")
```