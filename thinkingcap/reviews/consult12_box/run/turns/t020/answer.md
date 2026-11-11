```python
# Let's see how official IFEval accuracy is computed from fulltest_ifeval_rule_verdicts.csv or fulltest_per_answer.csv!
for m, g in ft_if_full.groupby('model'):
    # prompt-level strict accuracy (all rules passed for a given (id, sample)):
    prompt_pass = g.groupby(['id', 'sample'])['passed'].all().mean() * 100.0
    # inst-level strict accuracy:
    inst_pass = g['passed'].mean() * 100.0
    # prompt-level finished-only (runaway == 0):
    g_fin = g[g['runaway'] == 0]
    prompt_pass_fin = g.groupby(['id', 'sample']).apply(lambda x: x['passed'].all() and (x['runaway'] == 0).all()).mean() * 100.0
    print(f"{m:8s} | prompt_pass (all): {prompt_pass:6.2f}% | prompt_pass (no runaway): {prompt_pass_fin:6.2f}% | inst_pass: {inst_pass:6.2f}%")
```