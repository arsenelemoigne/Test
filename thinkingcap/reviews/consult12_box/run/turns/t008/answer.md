```python
print("=== raw_logs/gpu2_night_key_lines.txt (tail) ===")
print(read_file("raw_logs/gpu2_night_key_lines.txt", 4000, from_end=True))
print("\n=== raw_logs/gpu1_latest_raw.txt (tail) ===")
print(read_file("raw_logs/gpu1_latest_raw.txt", 4000, from_end=True))
print("\n=== code/queued_lines.txt ===")
print(read_file("code/queued_lines.txt", 4000))
```