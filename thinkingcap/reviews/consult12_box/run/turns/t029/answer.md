```python
# Let's inspect what rules inside `other_counting` and `at_least_N` lost accuracy in rpK-60 and rpL-60 vs base and rpE-80!
rule_pass = ft_if_full.groupby(['model', 'rule'])['passed'].agg(['mean', 'count']).unstack('model')
rule_pass_mean = rule_pass['mean'] * 100.0
rule_counts = rule_pass['count']['base'] // 8
df_rules = pd.DataFrame({
    'n_q': rule_counts,
    'base': rule_pass_mean['base'],
    'rpE80': rule_pass_mean['rpE80'] - rule_pass_mean['base'],
    'rpK20': rule_pass_mean['rpK20'] - rule_pass_mean['base'],
    'rpK40': rule_pass_mean['rpK40'] - rule_pass_mean['base'],
    'rpK60': rule_pass_mean['rpK60'] - rule_pass_mean['base'],
    'rpL60': rule_pass_mean['rpL60'] - rule_pass_mean['base'],
    'p1r1s0': rule_pass_mean['p1r1s0'] - rule_pass_mean['base'],
})
df_rules['L60_impact'] = df_rules['n_q'] * df_rules['rpL60'] / 541.0
print("=== Rules sorted by rpL60 impact vs base ===")
print(df_rules.sort_values('L60_impact').round(2).to_string())
```