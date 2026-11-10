```python
import json, re

with open(os.path.join(pack_dir, "all_runs.json"), "r") as f:
    all_runs = json.load(f)

print(f"Total entries in all_runs.json: {len(all_runs)}")
if isinstance(all_runs, list):
    print("Sample entry:", json.dumps(all_runs[0], indent=2))
elif isinstance(all_runs, dict):
    k0 = list(all_runs.keys())[0]
    print("Sample entry:", k0, json.dumps(all_runs[k0], indent=2))

log_path = os.path.join(pack_dir, "EXPERIMENT_LOG.md")
with open(log_path, "r") as f:
    log_lines = f.readlines()

print(f"EXPERIMENT_LOG.md total lines: {len(log_lines)}")

# Print all markdown headers in EXPERIMENT_LOG.md with line numbers
headers = [(i+1, line.strip()) for i, line in enumerate(log_lines) if line.startswith("#")]
for ln, h in headers:
    print(f"L{ln}: {h}")
```