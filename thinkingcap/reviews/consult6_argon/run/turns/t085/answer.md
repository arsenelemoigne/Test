```python
# Let's also check Fact Check items 3, 4, 5 in EXPERIMENT_LOG.md:
# - L4662-4686 (bidax6p)
# - L1540-1630 (zbench)
# - L5055-5078 (ir-full ifeval_rules.json diagnostic!)
# - L3620-3630 (rc1-rl-40 full test)
print("--- Fact check 3: L4660-4688 ---")
print("".join(log_lines[4659:4688]))
print("--- Fact check 4: L1538-1628 ---")
print("".join(log_lines[1537:1628]))
print("--- Fact check 5: L5055-5078 ---")
print("".join(log_lines[5054:5078]))
print("--- rc1-rl-40 full test: L3618-3630 ---")
print("".join(log_lines[3617:3630]))
```