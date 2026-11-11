```python
# Let's check why the handoff happened at turn 127! Look at `turns/` in `run/`!
turns_dir = os.path.join(parent_dir, "run", "turns")
turn_files = sorted(os.listdir(turns_dir))
print("Last 10 turn files:", turn_files[-10:])
for tf in turn_files[-5:]:
    with open(os.path.join(turns_dir, tf)) as f:
        t = json.load(f)
    print(tf, t.keys(), {k: (len(str(v)) if k=='reply' else v) for k, v in t.items()})
```