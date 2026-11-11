```python
import os
work_dir = "/Users/arsenelemoigne/Claude Build/Test-mvt-fading/thinkingcap/reviews/consult6_argon/work"
pack_dir = "/Users/arsenelemoigne/Claude Build/Test-mvt-fading/thinkingcap/reviews/consult6_argon/pack"
print("Work dir:", os.listdir(work_dir))
print("Pack dir:", os.listdir(pack_dir))
for f in os.listdir(work_dir):
    p = os.path.join(work_dir, f)
    print(f, os.path.getsize(p))
```