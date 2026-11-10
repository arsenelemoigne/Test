```python
import pandas as pd
import numpy as np

# Let's inspect all_runs.json in detail
df_runs = pd.DataFrame(all_runs)
print("Columns in all_runs:", df_runs.columns.tolist())
print("Kinds:", df_runs['kind'].value_counts().to_dict())

full_runs = df_runs[df_runs['kind'] == 'full'].copy()
print("\n--- ALL FULL RUNS ---")
for idx, r in full_runs.iterrows():
    p = r['parts']
    p_str = " | ".join([f"{k}:{v['saved']}%/{v['acc']:+.1f}" for k, v in p.items()]) if isinstance(p, dict) else ""
    print(f"{r['key']:18s} | {r['name']:28s} | saved={r['saved']:5.1f}% | acc={r['acc']:+5.2f} | {p_str}")
```