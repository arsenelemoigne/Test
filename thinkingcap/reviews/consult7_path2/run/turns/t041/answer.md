```python
import json

with open(f"{pack_dir}/all_runs.json") as f:
    all_runs = json.load(f)

print(f"Type of all_runs: {type(all_runs)}, length: {len(all_runs)}")
if isinstance(all_runs, list):
    print("Keys of first item:", all_runs[0].keys())
    for r in all_runs:
        name = str(r.get("name", "") or r.get("id", "") or r.get("run", ""))
        if any(k in name.lower() for k in ["gr", "tp", "drpo", "sb", "cp", "copd", "rp", "rc1", "ca1", "od"]):
            print(f"{name:25s} | eval: {str(r.get('eval', r.get('set', ''))):10s} | saved: {r.get('saved', r.get('tokens_saved'))} | acc: {r.get('acc', r.get('accuracy_change'))} | vs_curve: {r.get('vs_curve')}")
elif isinstance(all_runs, dict):
    print("Keys:", list(all_runs.keys())[:20])
```