```python
# Check fulltest_per_answer.csv and fulltest_ifeval_rule_verdicts.csv as well!
ft_a = pd.read_csv(os.path.join(pack_dir, "tables/fulltest_per_answer.csv"), nrows=5)
ft_if = pd.read_csv(os.path.join(pack_dir, "tables/fulltest_ifeval_rule_verdicts.csv"), nrows=5)
print("fulltest_per_answer cols:", ft_a.columns.tolist())
print("fulltest_ifeval_rule_verdicts cols:", ft_if.columns.tolist())
```