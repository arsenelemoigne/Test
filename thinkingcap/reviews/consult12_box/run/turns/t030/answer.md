```python
# Look at `df_rules`:
# Over 90% of the IFEval deficit in rpL60 (and rpK60) sits in just FIVE rules:
# 1. `length_constraints:number_words` (n=52): -18.51% in rpL60 (-12.50% in rpK60, -10.34% in rpK40, -14.42% in rpE80, +12.02% in p1r1s0) -> impact -1.78 pts on IFEval!
# 2. `keywords:frequency` (n=42): -11.01% in rpL60 (-9.52% in rpK60, -8.63% in rpK40, -9.82% in rpE80) -> impact -0.85 pts!
# 3. `length_constraints:number_sentences` (n=52): -5.77% in rpL60 (-7.45% in rpK60, -5.53% in rpK40, -5.53% in rpE80) -> impact -0.55 pts!
# 4. `detectable_format:number_highlighted_sections` (n=48): -5.21% in rpL60 (-2.60% in rpK60, -2.86% in rpK40, -2.60% in rpE80) -> impact -0.46 pts!
# 5. `detectable_content:number_placeholders` (n=27): -8.33% in rpL60 (-7.87% in rpK60, -7.41% in rpK40, -4.63% in rpE80) -> impact -0.42 pts!
# Notice that those 5 rules ALONE sum to -4.06 pts on IFEval in rpL60 (while all other 20 rules NET to +0.69 pts ABOVE base in rpL60)!
# Even more striking: the top 3 rules (`number_words`, `keywords:frequency`, `number_sentences`) are the exact `length_rule()` rules that `rpM` and `rpN` target with `--count-mode length` (`--count-resp-min 1.10 --count-resp-cap 1.75 --count-resp-pad 60`)!

# Now let's check SVAMP and BFCL and MBPP+ in rpL-60 vs rpK-60, rpK-40, rpE-80!
sv_q = ft_q[ft_q['dataset'] == 'svamp'].copy()
sv_q['q_band'] = pd.qcut(sv_q['rpE80:thinking'], 4, labels=['Q1_short', 'Q2_med_short', 'Q3_med_long', 'Q4_long'])
print("=== SVAMP by rpE80 thinking quartiles ===")
for qb, g in sv_q.groupby('q_band', observed=False):
    print(f"\nBand: {qb} (N={len(g)}, rpE80 thk range {g['rpE80:thinking'].min():.0f}-{g['rpE80:thinking'].max():.0f})")
    for m in ['base', 'rpE80', 'rpI120', 'rpK20', 'rpK40', 'rpK60', 'rpL60', 'p1r1s0']:
        acc = (g[f'{m}:right_of_8'] / 8.0 * 100.0).mean()
        d_acc = acc - (g['base:right_of_8'] / 8.0 * 100.0).mean()
        thk = g[f'{m}:thinking'].mean()
        run = (g[f'{m}:runaway_of_8'] / 8.0 * 100.0).mean()
        print(f"  {m:7s}: acc {acc:5.2f}% ({d_acc:+6.2f}) | thk {thk:6.1f} | run {run:4.2f}%")
```