```python
# LOOK AT THAT OUTPUT ABOVE:
# At step 20 (25% through pairs): {'guard': 40, 'right': 40} -> ZERO shorten pairs! (0% shorten!)
# At step 40 (50% through pairs): {'guard': 48, 'right': 112} -> ZERO shorten pairs! (0% shorten!)
# At step 60 (75% through pairs): {'guard': 48, 'right': 176, 'shorten': 16} -> only 16 of 96 shorten pairs (6.7% shorten)!
# All the rest of the shorten pairs are sitting at steps 61-80, which `rpJ` NEVER EVEN TRAINS ON because `rpJ` has `--rp-steps 60` on a 1,280-pair dataset (80 steps)!
#
# AND WHAT ABOUT ROUND 1 (`p1r1s0` / `p1r1s1`)?!
# Let's check Round 1 (`--mix-preset round1`, 200 steps, 3,200 pairs, screened at steps 100 and 200):
# In Round 1, for `svamp` (800 pairs: 35% shorten = 280, 50% right = 400, 15% guard = 120):
ps_r1_svamp = [{"kind": "shorten", "i": i} for i in range(280)] + [{"kind": "guard", "i": i} for i in range(120)] + [{"kind": "right", "i": i} for i in range(400)]
q_r1 = sorted(ps_r1_svamp, key=lambda p: (p["kind"], rng.random()))
g1 = [p for p in q_r1 if p["kind"] == "guard"]
r1 = [p for p in q_r1 if p["kind"] != "guard"]
q_r1_final = [x for pair in zip(g1, r1) for x in pair] + g1[len(r1):] + r1[len(g1):]

for step, frac in [(50, 0.25), (100, 0.50), (130, 0.65), (150, 0.75), (200, 1.00)]:
    n_seen = int(len(q_r1_final) * frac)
    c_seen = Counter(p["kind"] for p in q_r1_final[:n_seen])
    print(f"Round 1 SVAMP at step {step:3d} ({frac*100:3.0f}%): {dict(c_seen)}")
```