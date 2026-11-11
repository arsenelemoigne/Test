```python
# Let's check EVERY point in FACT_CHECK.md against EXPERIMENT_LOG.md and all_runs.json so we can give an honest, exact, rigorous response on all 10 fact-check items!
with open(os.path.join(pack_dir, "EXPERIMENT_LOG.md"), "r") as f:
    log_lines = f.readlines()
with open(os.path.join(pack_dir, "all_runs.json"), "r") as f:
    all_runs = json.load(f)

# Let's check the lines cited in FACT_CHECK.md:
# 1. L4131-4136 (D4 lpd-budget)
print("--- Fact check 1: L4128-4145 ---")
print("".join(log_lines[4127:4145]))

# 2. rpI drift and rpH count-rules: let's check L5030-5050 and L5165-5205
print("--- Fact check 2 & 5: L5025-5055 ---")
print("".join(log_lines[5024:5055]))
print("--- Fact check 2: L5165-5205 ---")
print("".join(log_lines[5164:5205]))
```