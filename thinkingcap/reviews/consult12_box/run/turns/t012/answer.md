```python
# Let's inspect gpu2_night_full.txt or gpu1_latest_raw.txt or dev512_per_question.csv to see exact full-test numbers if printed in logs, or exact numbers in NEW_RESULTS.md
with open(os.path.join(pack_dir, "raw_logs/gpu2_night_full.txt"), "r") as f:
    g2_full = f.read()
print("Length of gpu2_night_full.txt:", len(g2_full))
print(g2_full[-5000:])
```