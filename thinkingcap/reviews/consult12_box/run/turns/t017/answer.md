```python
ft_a_models = pd.read_csv(os.path.join(pack_dir, "tables/fulltest_per_answer.csv"), usecols=['model'])['model'].unique()
ft_if_models = pd.read_csv(os.path.join(pack_dir, "tables/fulltest_ifeval_rule_verdicts.csv"), usecols=['model'])['model'].unique()
print("ft_a_models:", ft_a_models)
print("ft_if_models:", ft_if_models)
```