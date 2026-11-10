```python
# Let's see the full what string for rpe-80 and rpi-120
for k in ['rpe-80', 'rpe-80full', 'rpi-120', 'rpi-120full']:
    row = df_runs[df_runs['key'] == k].iloc[0]
    print(k, "->", row['what'])
```