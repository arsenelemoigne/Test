```python
# Let's check EXPERIMENT_LOG.md tail and THINKING_OFF.md / pertype_best_full.md
with open(os.path.join(pack, "EXPERIMENT_LOG.md")) as f:
    exp_log = f.read()
print("--- EXPERIMENT_LOG.md last 6000 chars ---")
print(exp_log[-6000:])
```