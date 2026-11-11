```python
import os

pack_dir = "/Users/arsenelemoigne/Claude Build/Test-mvt-fading/thinkingcap/reviews/consult12_box/pack"
work_dir = "/Users/arsenelemoigne/Claude Build/Test-mvt-fading/thinkingcap/reviews/consult12_box/work"
os.makedirs(work_dir, exist_ok=True)

with open(os.path.join(pack_dir, "NEW_RESULTS.md"), "r") as f:
    print("=== NEW_RESULTS.md ===")
    print(f.read()[:9000])
```