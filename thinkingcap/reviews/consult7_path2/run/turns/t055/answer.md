```python
draft = """# PATH 2: Post-DPO Reinforcement Learning Strategy for Qwen3-0.6B

## Executive Summary
Path 1 (2-round DPO from raw Qwen3-0.6B) is designed to compress reasoning in Round 1 (~53–57% tokens saved, -2.2 to -3.2 pts accuracy) and repair accuracy in Round 2 (~49.5–52.5% saved, -0.7 to -1.3 pts accuracy). However, across the entire experiment log (`rpE-80`: 47.0% / -0.9; `rpE-60`: 47.8% / -1.2; `rpE-40`: 50.6% / -1.9; `rpI-120`: 41.9% / +0.1; `od1`: 51.5% / -1.2 -> 53.3% / -1.8), DPO repair and online DPO exhibit a structural trade-off: **repairing accuracy via DPO inevitably gives back 3–7 points of token savings** because shortest-right-over-shortest-wrong pairs push up longer right answers when short ones fail, while adding more DPO "shorten" pairs (`od1`) pushes the model back down along the same accuracy-length curve rather than lifting the frontier.

**Path 2** is a short, surgical, on-policy Reinforcement Learning stage (40–60 steps, ~3.5–4.5 GPU-hours per seed on one L4) that starts **from Path 1's Round-2 final model** (merged into base weights, with a fresh LoRA r16 on top) and pushes it across the `> 50% saved / > -1.0 pt` finish line.
"""
with open(f"{work_dir}/report_draft.md", "w") as f:
    f.write(draft)
print("Draft initialized.")
```