```python
# Let's check dpo.py and od5g/od5f monitors & dev scores in the pack
import os, json
pack = "/Users/arsenelemoigne/Claude Build/Test-mvt-fading/thinkingcap/reviews/consult32_od5h/pack"
print("Files in pack:", os.listdir(pack))
print("Files in pack/code:", os.listdir(os.path.join(pack, "code")))
print("Files in pack/monitors:", os.listdir(os.path.join(pack, "monitors")))
```