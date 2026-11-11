```python
import pandas as pd
import numpy as np
import json

with open(os.path.join(pack_dir, "all_runs.json"), "r") as f:
    all_runs = json.load(f)

print(f"Total entries in all_runs.json: {len(all_runs)}")
for r in all_runs[-25:]:
    print(r.get("name"), r.get("kind"), r.get("saved"), r.get("acc_delta"), r.get("notes", "")[:80])
```