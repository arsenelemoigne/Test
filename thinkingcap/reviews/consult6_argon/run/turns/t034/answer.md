```python
# Let's analyze per-part accuracy vs saving across all runs in all_runs.json (kind=='720' and kind=='full').
# Specifically, let's see what happens to each part (v, c, f, s, m, i, and pooled bfcl) as its own saving increases!

parts_info = [
    ('v', 'SVAMP (Math)', 0.25),
    ('c', 'MBPP+ (Code)', 0.25),
    ('f', 'IFEval', 0.25),
    ('bfcl', 'BFCL (Pooled)', 0.25),
    ('s', 'BFCL Simple', 400/3360),
    ('m', 'BFCL Multiple', 200/3360),
    ('i', 'BFCL Irrelevance', 240/3360),
]

print("--- PER-PART ACCURACY AT DIFFERENT SAVING BINS (Screen-720 runs, n=197) ---")
df720 = df[df['kind'] == '720'].copy()

for pcode, pname, w in parts_info:
    scol = f'{pcode}_saved'
    acol = f'{pcode}_acc'
    # Bins: 20-35%, 35-45%, 45-55%, 55%+
    b1 = df720[(df720[scol] >= 20) & (df720[scol] < 35)][acol]
    b2 = df720[(df720[scol] >= 35) & (df720[scol] < 45)][acol]
    b3 = df720[(df720[scol] >= 45) & (df720[scol] < 55)][acol]
    b4 = df720[(df720[scol] >= 55)][acol]
    print(f"{pname:18s} | 20-35%: {b1.mean():+5.2f} (n={len(b1):2d}) | 35-45%: {b2.mean():+5.2f} (n={len(b2):2d}) | 45-55%: {b3.mean():+5.2f} (n={len(b3):2d}) | >=55%: {b4.mean():+5.2f} (n={len(b4):2d})")
```