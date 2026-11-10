```python
# Let's check how "the curve" (yardstick) is defined in EXPERIMENT_LOG.md!
m_curve = search_log("yardstick|curve.*predict")
for ln, txt in m_curve:
    print(f"L{ln}: {txt}")
```