```python
import os, json

pack_dir = "/Users/arsenelemoigne/Claude Build/Test-mvt-fading/thinkingcap/reviews/consult6_argon/pack"
work_dir = "/Users/arsenelemoigne/Claude Build/Test-mvt-fading/thinkingcap/reviews/consult6_argon/work"
os.makedirs(work_dir, exist_ok=True)

files = os.listdir(pack_dir)
print("Files in pack:", files)

ladder_path = os.path.join(pack_dir, "LADDER_DESIGN_FOR_REVIEW.md")
with open(ladder_path, "r") as f:
    ladder_text = f.read()

print(f"LADDER_DESIGN_FOR_REVIEW.md length: {len(ladder_text)} chars")
print("--- PART 1 ---")
print(ladder_text[:9000])
```