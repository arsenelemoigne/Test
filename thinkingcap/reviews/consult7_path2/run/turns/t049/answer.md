```python
with open(f"{pack_dir}/code/rl_train.py") as f:
    rl_text = f.read()
for i, line in enumerate(rl_text.splitlines()):
    if any(w in line for w in ["SAMPLING", "LOGIT_TEMP", "clip", "ratio", "updates", "old_logps", "dr_grpo"]):
        print(f"L{i+1}: {line}")
```