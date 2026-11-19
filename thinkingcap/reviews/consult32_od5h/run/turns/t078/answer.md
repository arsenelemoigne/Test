```python
# Let's check fulltests.json and EXPERIMENT_LOG.md (tail) to see od5g, od5f, rpL-80, etc.
with open(os.path.join(pack, "fulltests.json")) as f:
    ft = json.load(f)
for k, v in ft.items():
    if any(x in k for x in ["od5", "rpL", "r2s"]):
        print(k, v.get("all"), {p: v.get(p) for p in ["svamp", "mbpp", "ifeval", "bfcl_simple_python", "bfcl_multiple", "bfcl_irrelevance"] if p in v})
```