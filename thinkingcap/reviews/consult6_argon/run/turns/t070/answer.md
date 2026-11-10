```python
# Let's carefully verify every single item in FACT_CHECK.md against EXPERIMENT_LOG.md, and check the code line by line!
with open(os.path.join(pack_dir, "EXPERIMENT_LOG.md"), "r") as f:
    log_lines = f.readlines()
with open(os.path.join(pack_dir, "all_runs.json"), "r") as f:
    all_runs = json.load() if False else json.load(open(os.path.join(pack_dir, "all_runs.json")))

def print_lines(start, end):
    for i in range(max(1, start), min(len(log_lines)+1, end+1)):
        print(f"L{i}: {log_lines[i-1]}", end="")

# Let's check Fact-Check items 1 to 5 and "Other corrections":
print("--- CHECK 1: D4 (lpd-budget) at L4131-4138, L4621-4628 ---")
print_lines(4130, 4140)
print_lines(4620, 4630)

print("\n--- CHECK 2: rpH and rpI at L5120-5205 ---")
print_lines(5120, 5205)
```