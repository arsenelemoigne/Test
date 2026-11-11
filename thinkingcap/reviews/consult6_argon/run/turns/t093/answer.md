```python
import random
from collections import Counter

ps_demo = [{"kind": "shorten", "i": i} for i in range(96)] + [{"kind": "guard", "i": i} for i in range(48)] + [{"kind": "right", "i": i} for i in range(176)]
rng = random.Random(0)
q_sorted = sorted(ps_demo, key=lambda p: (p["kind"], rng.random()))
g = [p for p in q_sorted if p["kind"] == "guard"]
r = [p for p in q_sorted if p["kind"] != "guard"]
q_final = [x for pair in zip(g, r) for x in pair] + g[len(r):] + r[len(g):]

for step, frac in [(20, 0.25), (40, 0.50), (60, 0.75), (80, 1.00)]:
    n_seen = int(len(q_final) * frac)
    c_seen = Counter(p["kind"] for p in q_final[:n_seen])
    print(f"At step {step:2d} ({frac*100:3.0f}% through pairs): {dict(c_seen)}")
```