```python
# Look at `at_least_N`, `fewer_than_N`, `no_counting`, `other_counting` across rpK20 -> rpK40 -> rpK60 -> rpL60!
# Look at `at_least_N` in rpK-40:
# In rpK-40, `at_least_N` response tokens peaked at 560.6 tokens (vs 475.3 at step 20 and 525.5 at step 60), so `at_least_N` accuracy peaked at 21.34% (-12.93 vs base, compared to -17.24 at step 20 and -14.87 at step 60)!
# Meanwhile in rpL-60:
# Look at what happened to `at_least_N` in rpL-60: response tokens dropped to 440.5 tokens, and `at_least_N` accuracy dropped to 13.58% (-20.69 vs base, -5.39 pts worse than rpE-80)!
# Look at `no_counting` in rpL-60: `no_counting` (264 questions = 48.8% of IFEval!) reached 62.78% (+0.99 over base! best of ANY model!) while cutting thinking to 121.8 and response to 117.2 and runaways to 1.14%!
# Wait: let's decompose rpL-60's -3.37 IFEval loss by class!
for m in ['rpE80', 'rpK20', 'rpK40', 'rpK60', 'rpL60']:
    print(f"=== {m} IFEval breakdown vs base (weighted by N/541) ===")
    tot = 0.0
    for cls, g in if_q.groupby('ifeval_class'):
        w = len(g) / 541.0
        d = g[m].mean() - g['base'].mean()
        contrib = w * d
        tot += contrib
        print(f"  {cls:15s} (w={w:.3f}): d_acc = {d:+6.2f} -> contrib to IFEval d_acc = {contrib:+6.2f} (contrib to Headline = {contrib/4:+6.3f})")
    print(f"  TOTAL IFEval d_acc = {tot:+6.2f} (Headline contrib = {tot/4:+6.3f})")
```