```python
bt = "`" * 3
with open(f"{work_dir}/notes.md", "a") as f:
    f.write("\n## 2. Path 2 Design Completed\n- Starting point: Path 1 Round 2 merged model (`models/p1r2-soup-best`).\n- Reward: `p2budget` with per-question P35 target budget B(q), P90 tail cap C(q), K2.5 difficulty toggle at solve_rate >= 0.50, IFEval thinking-only measurement + counting-rule response guard, and -0.20 early-stop penalty on wrong answers with thinking < 0.5 * base_think.\n- Optimizer: DLER batch-wise advantage std (`--advantage batch`), actual active token denominator (no `--dr-grpo` / no `--fixed-denominator`), dynamic sampling 1.5x with `--drop-flat`, clip-higher [0.20, 0.28], lr 1.5e-5 -> 5e-6 over 50 steps, KL 0.002.\n")

with open(f"{work_dir}/report_draft.md", "w") as f:
    f.write("# PATH 2: Post-DPO Reinforcement Learning Strategy for Qwen3-0.6B\n\nSee final report output for full details.\n")

print("Updated notes.md and report_draft.md successfully.")
```