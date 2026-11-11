```python
# Notice the slight difference in IFEval d_acc between finished-only (in fulltest_per_question.csv where base IFEval is 57.76% vs official where base IFEval is slightly different or runaway is treated differently).
# Wait! Let's check how official IFEval accuracy is defined vs fulltest_per_question.csv!
# In NEW_RESULTS.md:
# rpE-80: 47.0 / -0.9 (IFEval -2.6, vs -2.38 in ft_q)
# rpK-20: 49.9 / -1.3 (IFEval -3.1, vs -2.93 in ft_q)
# rpK-40: 51.2 / -0.8 (-0.75) (IFEval -1.9, vs -1.87 in ft_q)
# rpK-60: 54.4 / -1.4 (IFEval -3.0, vs -3.00 in ft_q)
# rpL-60: 59.3 / -1.7 (IFEval -3.4, vs -2.82 in ft_q)
# Let's check fulltest_ifeval_rule_verdicts.csv to see why NEW_RESULTS.md has IFEval -3.4 for rpL-60 vs -2.82 in ft_q!
ft_if_full = pd.read_csv(os.path.join(pack_dir, "tables/fulltest_ifeval_rule_verdicts.csv"))
print(ft_if_full.head(10))
```