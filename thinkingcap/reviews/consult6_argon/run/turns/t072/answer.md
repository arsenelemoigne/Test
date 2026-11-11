```python
import os, json

pack_dir = "/Users/arsenelemoigne/Claude Build/Test-mvt-fading/thinkingcap/reviews/consult6_argon/pack"
work_dir = "/Users/arsenelemoigne/Claude Build/Test-mvt-fading/thinkingcap/reviews/consult6_argon/work"
os.makedirs(work_dir, exist_ok=True)

print("Pack contents:", os.listdir(pack_dir))
code_dir = os.path.join(pack_dir, "code")
if os.path.isdir(code_dir):
    print("code/ contents:", os.listdir(code_dir))

fc_path = os.path.join(pack_dir, "FACT_CHECK.md")
plan_path = os.path.join(pack_dir, "PATH1_PLAN.md")

with open(fc_path, "r") as f:
    fc_text = f.read()
with open(plan_path, "r") as f:
    plan_text = f.read()

print(f"FACT_CHECK.md length: {len(fc_text)} chars")
print(f"PATH1_PLAN.md length: {len(plan_text)} chars")

print("=== FACT_CHECK.md (Part 1) ===")
print(fc_text[:9000])
```