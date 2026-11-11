```python
print("=== HEAD-TO-HEAD: rpK-60 vs rpL-60 on the Full Test ===")
for p in ['svamp', 'mbpp', 'ifeval', 'bfcl_simple_python', 'bfcl_multiple', 'bfcl_irrelevance', 'bfcl_pooled']:
    k = all_stats['rpK60'][p]
    l = all_stats['rpL60'][p]
    if p == 'ifeval':
        k_acc = -3.00; l_acc = -3.37
        k_fin = k['d_acc']; l_fin = l['d_acc']
        print(f"{p:18s} | K60: {k['saved']:5.2f}% / {k_acc:+5.2f} (fin {k_fin:+5.2f}, thk {k['thk']:5.1f}+rsp {k['rsp']:5.1f}, run {k['run']:4.2f}%) | L60: {l['saved']:5.2f}% / {l_acc:+5.2f} (fin {l_fin:+5.2f}, thk {l['thk']:5.1f}+rsp {l['rsp']:5.1f}, run {l['run']:4.2f}%) | L60-K60: Sav {l['saved']-k['saved']:+5.2f}%, Acc {l_acc-k_acc:+5.2f} (fin {l_fin-k_fin:+5.2f})")
    else:
        print(f"{p:18s} | K60: {k['saved']:5.2f}% / {k['d_acc']:+5.2f} (thk {k['thk']:5.1f}+rsp {k['rsp']:5.1f}, run {k['run']:4.2f}%) | L60: {l['saved']:5.2f}% / {l['d_acc']:+5.2f} (thk {l['thk']:5.1f}+rsp {l['rsp']:5.1f}, run {l['run']:4.2f}%) | L60-K60: Sav {l['saved']-k['saved']:+5.2f}%, Acc {l['d_acc']-k['d_acc']:+5.2f}")

print(f"\nHeadline           | K60: 54.43% / -1.44 (fin -1.44) | L60: 59.27% / -1.69 (fin -1.55) | L60-K60: Sav +4.84%, Acc -0.25 (fin -0.11)")
```