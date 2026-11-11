```python
for r in all_runs:
    if r.get("kind") == "full":
        print(f"{r['key']:20s} | saved: {r['saved']:5.1f}% | acc: {r['acc']:+5.2f}")
        for p, v in sorted(r.get("parts", {}).items()):
            print(f"   {p:22s}: saved {v.get('saved'):5.1f}%, acc {v.get('acc'):+5.2f}")
```