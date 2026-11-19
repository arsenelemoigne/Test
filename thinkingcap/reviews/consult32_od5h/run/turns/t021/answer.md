Aha! Look closely at lines 455-475 and lines 680-731 of `repair_pairs.py`! Let's also read the rest of `repair_pairs.py` (from line 785 to the end) and `track.py` around `odpo_fixed`, `fullpick`, and `ptrepair` to see how pairs are sampled, shuffled, and trained on! Let's run a script to inspect the end of `repair_pairs.py` and `track.py`.
```python
with open(
    "/Users/arsenelemoigne/Claude Build/Test-mvt-fading/thinkingcap/reviews/consult32_od5h/pack/code/repair_pairs.py"
) as f:
    lines = f.readlines()
for i, line in enumerate(lines[820:], start=821):
    print(f"{i:4d}: {line}", end="")
```