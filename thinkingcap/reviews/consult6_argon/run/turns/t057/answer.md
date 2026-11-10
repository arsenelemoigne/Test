```python
# The team specifically asked us to add a Section 6:
# "the 2-3 strategy paths you think are most likely to reach > 50% saved at better than -1 on the full test with one reproducible recipe
#  (within our constraints: the model's own answers, no teacher-written answers, no prompt tricks, no copied competitor recipe, two L4 GPUs).
#  For each path: why you believe it can win (the evidence from our log), its full design (data and question selection per type, pair rules
#  or reward, every hyperparameter, steps, seeds, how checkpoints are chosen without touching the test), the GPU-hours, the expected result
#  with a range, and the first cheap experiment that would confirm or kill it. Rank them. Use your code runs to check the evidence."

import os, json, re
import numpy as np
import pandas as pd

pack_dir = "/Users/arsenelemoigne/Claude Build/Test-mvt-fading/thinkingcap/reviews/consult6_argon/pack"
work_dir = "/Users/arsenelemoigne/Claude Build/Test-mvt-fading/thinkingcap/reviews/consult6_argon/work"
notes_path = os.path.join(work_dir, "notes.md")
draft_path = os.path.join(work_dir, "report_draft.md")

with open(os.path.join(pack_dir, "all_runs.json"), "r") as f:
    all_runs = json.load(f)
with open(os.path.join(pack_dir, "EXPERIMENT_LOG.md"), "r") as f:
    log_lines = f.readlines()

df_runs = pd.DataFrame(all_runs)
records = []
for idx, r in df_runs.iterrows():
    p = r['parts']
    row = {
        'key': r['key'],
        'name': r['name'],
        'kind': r['kind'],
        'what': r['what'],
        'saved': r['saved'],
        'acc': r['acc'],
        'yardstick': -0.388 + 0.0899 * r['saved'] - 0.0028 * (r['saved']**2),
    }
    row['vs_curve'] = row['acc'] - row['yardstick']
    for k in ['v', 'c', 'f', 's', 'm', 'i']:
        if isinstance(p, dict) and k in p:
            row[f'{k}_saved'] = p[k]['saved']
            row[f'{k}_acc'] = p[k]['acc']
    row['bfcl_acc'] = (400*row['s_acc'] + 200*row['m_acc'] + 240*row['i_acc'])/840.0
    row['bfcl_saved'] = 100.0 * (1.0 - (400*(1-row['s_saved']/100.0) + 200*(1-row['m_saved']/100.0) + 240*(1-row['i_saved']/100.0))/840.0)
    records.append(row)
df = pd.DataFrame(records)

# Let's check ALL evidence in the log across different families of methods so we can design and verify the top 3 winning paths quantitatively!
# Wait: what are all the distinct methodological families in the log that respect the constraints (model's own answers, no teacher-written answers, no prompt tricks, no copied competitor recipe)?
# Let's list and check the evidence for each candidate path:
#
# PATH 1: 2-Stage On-Policy "Compact-and-Verify" DPO (or Iterative On-Policy DPO with Equal-Length Re-Read + Counting-Rule Calibration + Shorten Replay)
#   - Evidence in log:
#     * `cleanA-rc1-full` (57.4% / -3.50) -> `rpe-40full` (50.6% / -1.90) -> `rpe-60full` (47.8% / -1.20) -> `rpe-80full` (47.0% / -0.90) -> `rpi-120full` (41.9% / +0.10, +1.64 vs curve).
#     * `rpB-40` (43.4% / +0.10, +1.86 vs curve) and `rpC-80` (43.4% / +0.43, +2.19 vs curve) on `pt-mid` (53.6% / -3.30 full).
#     * `D6 lpd-simright` (similar-length right>wrong + shorten: +1.8 vs curve at 28.1% / +1.43 pp).
#     * `D4 lpd-budget` (difficulty-aware budget shorten target: 49.0% / -1.42 pp, +1.29 vs curve).
#     * `od1` / `od2` (online DPO rounds with 50% guard [re-read + IFEval counting-rule] + 50% shorten [>=6/8, >=1.5x]).
#
# PATH 2: Per-Type Specialist LoRA Task-Arithmetic Merge + 1 On-Policy Balanced Calibration Round (Clean Own Data, `fp32+fp16`)
#   - Wait: let's check the evidence for Per-Type LoRA merges (`pt-mid`, `pt-soft`, `lpt6mix-Cnf`, `rpB`, `rpC`, `lpk`)!
#   - Let's check what happened with `pt-mid` + `rpB`/`rpC` vs `lpk`!
#   - Look at `pt-mid` (4 per-type adapters: math 0.30, code 0.45, ifeval 0.60, tools 0.45):
#     * On Screen-720: `pt-mid` scored 55.0% / -2.58 pp (+1.33 vs curve); on Full Test: 53.6% / -3.30 pp.
#     * When repaired with `rpC` (lr 1e-5, 80 steps, equal-length right>wrong on math/code/no-fit):
#       - `rpC-20`: **50.5% / -1.83 pp** (+1.16 vs curve)
#       - `rpC-40`: **51.2% / -2.19 pp**
#       - `rpC-80`: **43.4% / +0.43 pp** (+2.19 vs curve!)
#       - `rpB-20`: **48.7% / -1.20 pp** (+1.45 vs curve!)
#     * And look at `lpt6mix-Cnf` (instructions 0.5 + simple 0.5 + multiple 0.5 + weak no-fit 0.25, ZERO math adapter!):
#       - `lpt6mix-Cnf` scored **43.3% / -1.20 pp** (+0.5 vs curve), with **SVAMP +42% saved / -1.3 pp**, **MBPP+ +48% saved / -0.4 pp**, **simple +37% / 0.0 pp**, **multiple +41% / +1.1 pp**, **no-fit +41% / -2.1 pp**!
#       - Why did `lpt6mix-Cnf` save 42% on math and 48% on code with only -1.3 pp and -0.4 pp loss despite having NO math or code adapter?! Because compression spills over globally (rank-1), while leaving out the math shortening adapter avoided direct over-cutting of hard math!
#
# PATH 3: What about RL (DRPO / Correctness-Gated Budget GRPO) or Self-Distilled Promptless Compaction / Cut-Verified On-Policy DPO?
#   - Let's check EVERY RL run and every cut-verified / self-distilled run in `EXPERIMENT_LOG.md` to see which 3rd path has the strongest quantitative backing!

m_drpo = [i+1 for i, l in enumerate(log_lines) if "drpo" in l.lower() or "self-replay" in l.lower() or "v2r" in l.lower()]
print("Matches for drpo/v2r/self-replay:", len(m_drpo))
```