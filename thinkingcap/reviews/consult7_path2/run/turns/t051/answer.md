```python
for i, line in enumerate(exp_lines):
    if any(k in line for k in ["gr2e", "gr2f", "gr2n", "tp_dr_grpo", "tp_lr", "tp_kl", "tp_reward"]):
        print(f"L{i+1}: {line.strip()[:140]}")
```