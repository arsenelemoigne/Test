```python
# Check if rpK40 and rpL60 have non-null entries in fulltest_per_question.csv!
for m in ['base', 'rpE80', 'rpI120', 'rpK20', 'rpK40', 'rpK60', 'rpL60', 'p1r1s0']:
    nn = ft_q[f"{m}:right_of_8"].notna().sum()
    print(f"{m}: {nn} non-null rows out of {len(ft_q)}")
```