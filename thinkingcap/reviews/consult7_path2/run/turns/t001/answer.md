```python
import os

pack_dir = "/Users/arsenelemoigne/Claude Build/Test-mvt-fading/thinkingcap/reviews/consult7_path2/pack"
work_dir = "/Users/arsenelemoigne/Claude Build/Test-mvt-fading/thinkingcap/reviews/consult7_path2/work"
os.makedirs(work_dir, exist_ok=True)

for root, dirs, files in os.walk(pack_dir):
    for f in sorted(files):
        p = os.path.join(root, f)
        print(f"{os.path.relpath(p, pack_dir)}: {os.path.getsize(p)} bytes")
```