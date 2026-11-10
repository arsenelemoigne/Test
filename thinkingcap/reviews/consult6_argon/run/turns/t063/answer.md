```python
import os, json

pack_dir = "/Users/arsenelemoigne/Claude Build/Test-mvt-fading/thinkingcap/reviews/consult6_argon/pack"
work_dir = "/Users/arsenelemoigne/Claude Build/Test-mvt-fading/thinkingcap/reviews/consult6_argon/work"
os.makedirs(work_dir, exist_ok=True)

print("Pack contents:", os.listdir(pack_dir))
code_dir = os.path.join(pack_dir, "code")
if os.path.exists(code_dir):
    print("Code dir contents:", os.listdir(code_dir))

with open(os.path.join(pack_dir, "FACT_CHECK.md"), "r") as f:
    fact_check = f.read()

print(f"FACT_CHECK.md length: {len(fact_check)} chars")
print(fact_check[:9500])
```