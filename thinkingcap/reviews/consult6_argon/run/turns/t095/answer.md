```python
# Let's check what happens on IFEval in Round 1 (`--mix-preset round1`: 27.5% shorten = 220, 72.5% right = 580, 0% guard = 0):
ps_r1_ifeval = [{"kind": "shorten", "i": i} for i in range(220)] + [{"kind": "right", "i": i} for i in range(580)]
q_r1_if = sorted(ps_r1_ifeval, key=lambda p: (p["kind"], rng.random()))
g_if = [p for p in q_r1_if if p["kind"] == "guard"]
r_if = [p for p in q_r1_if if p["kind"] != "guard"]
q_r1_if_final = [x for pair in zip(g_if, r_if) for x in pair] + g_if[len(r_if):] + r_if[len(g_if):]

for step, frac in [(100, 0.50), (145, 0.725), (200, 1.00)]:
    n_seen = int(len(q_r1_if_final) * frac)
    c_seen = Counter(p["kind"] for p in q_r1_if_final[:n_seen])
    print(f"Round 1 IFEval at step {step:3d} ({frac*100:5.1f}%): {dict(c_seen)}")
```