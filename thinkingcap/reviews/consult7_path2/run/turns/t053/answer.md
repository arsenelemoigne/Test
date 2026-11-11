```python
notes = """# Consultation 7 (Path 2 RL Design) - Research Notes

## 1. Diagnosis of Past GRPO / RL Failures
- **grA / grB / grC (3 Nov)**:
  - Started from raw Qwen3-0.6B, 150 steps, 12 questions x 8 replies (96 rollouts/step).
  - Used `--reward lenpen` (khudgins reproduction): `R = (1 - lam_t * min(1, thinking/2048)) * correct + 0.1 * formed`.
  - Why it failed (grA -5.8%/-0.83, grC -6.7%/-0.37, grB -0.6%/-0.72):
    1. **Tiny length contrast vs correctness + format bonus**: In grA, `lam_t` ramped from 1e-3 to 0.05 over 300 steps (or 100 steps). Halving a 300-token thinking trace changed reward by `0.05 * 150 / 2048 = 0.0036`, which is ~275x smaller than the 1.0 correctness signal and 27x smaller than the 0.1 format bonus (which wrong answers also received!).
    2. **Dr. GRPO (`--dr-grpo`) disabled std scaling AND divided by `len(chunk) * max_tokens`**:
       - Without std scaling (`scale=False` in `group_advantages`), an all-correct group with a reward spread of `0.0036` has raw advantages of `~ ±0.0018`, whereas a mixed-correctness group has advantages of `~ ±0.5`. In standard GRPO (`scale=True`), dividing by group std normalizes all-correct groups to `~ ±1.0` so length differences actually drive updates! Under `--dr-grpo`, all-correct groups had ~300x smaller gradients than mixed groups.
       - Dividing loss by `len(chunk) * 4096` instead of actual token count (~500 tokens) shrank the effective learning rate by ~8x! Combined with `lr = 2e-6` (a full-FT lr, 5-10x too low for LoRA), the effective step size was microscopic.
       - Furthermore, Dr. GRPO removes the `1/|o_i|` response-level length normalization, which Arora & Zanette (2502.04463) and Dr. GRPO (2503.20783) showed is the implicit shortening force on positive-advantage responses.
       - And because wrong answers are naturally longer than right answers (copdanalysis: SVAMP wrong 1773 vs right 880), wait — why did grA/grC get LONGER (-5.8%, -6.7%)? Because in mixed groups on raw Qwen, on hard/medium questions, longer thinking occasionally succeeds or format bonus (+0.1) rewards formed wrong answers, and without `1/|o_i|` penalty on positive advantages or strong length penalty, the model drifted longer.
    3. **Optimizer reset every 10 steps**: `track_thinkprune.py` ran 10-step blocks without `--tp-persist` (`--init-state`), resetting AdamW momentum/variance every 10 steps.
    4. **Thinking-only penalty & 4,096 code cap**: `lenpen` only penalized thinking up to 2,048 (ignoring answer-side loops in IFEval and >2,048 code thinking), and cut code rollouts at 4,096 instead of 8,192.

- **gr2e / gr2f (6 Nov)**:
  - Started from raw Qwen3-0.6B, 40 steps, 32 questions x 8 replies, `--reward gated --gr-form linear` (`R = max(0, 1 - 0.3 * L / B_part)` if right & formed & finished else 0), `lr = 1e-5` warmup 12 cosine to `1e-6`, `KL = 0.005`, `--dr-grpo --fixed-denominator`, `--tp-persist`.
  - Why gr2e failed: `--gr-norms-from` re-derived part labels with `build_pool.part_of` and set `svamp=82`, so any SVAMP answer >273 tokens got 0 reward.
  - Why gr2f barely moved (+4.0% @ 10, +11.4% @ 20, +6.8% @ 30, +11.3% / +0.3 @ 40):
    1. **Dr. GRPO + Fixed Denominator crushed the length gradient**: With `R = 1 - 0.3 * L / B_part`, two right answers differing by 20% of `B_part` differ in reward by only `0.06`. With `--dr-grpo` (`scale=False`), the advantage in an all-right group is `±0.03`, which is **15x smaller** than a right-vs-wrong advantage (`~0.5` to `0.8`). And `--fixed-denominator` divided the loss by `256 * 8192 = 2,097,152` tokens (because `--max-tokens-code 8192` makes `max(args.max_tokens, args.max_tokens_code) = 8192`), whereas actual active tokens per step were `~120,000`! That diluted the gradient by **17.5x**!
    2. **Cosine LR decayed to 1e-6 over just 40 steps**: By step 25-40, `lr` was down to `1e-6..3e-6`, divided by the 17.5x fixed-denominator dilution and 0.03 unscaled advantages = zero effective movement after step 20 (+11.4% -> +6.8% -> +11.3%).
    3. **No dynamic sampling of all-wrong or uninformative groups**, and starting from the **raw model** (where smooth linear penalty `0.3 L/B` is too weak to prune 50%+ of tokens in 40 steps, as confirmed by matousek-martin's `no-replay` ablation which got only 1.85% saved in 128 steps!).

- **Other past RL runs in the log**:
  - `rc1-rl` (23 Oct, Arora reward, alpha 0.15, from rc1-soup-30 on 782 repeated questions): thinking grew from 200-250 to 400-500 tokens because on an already-shortened model with repeated questions, the 1.0 correctness reward outweighed `alpha=0.15` and bought accuracy back by lengthening!
  - `drpo1` (lambda 0.2, lr 1e-5, from rc1-rl-40) & `sb1`/`sb2` (ShorterBetter, beta 0.005 / 0.001, from rc1-rl-40): collapsed/overshot to 63-73% saved / -3.2 to -6.9 pts because DRPO and ShorterBetter apply aggressive group-relative length pressure *without* a safe per-question budget floor, driving already-short answers on `rc1` into premature stopping.
  - `COPD` (cp1a, cp1c, cp1d): 4B teacher pushed useful re-reads down twice as hard as wasted checks; adding `--copd-outcome` (Run C) caused early-stopping collapse (57.1% / -4.7) because wrong answers are naturally longer, so unscaled `right - mean_right` acted as a blind length penalty.
"""
with open(f"{work_dir}/notes.md", "w") as f:
    f.write(notes)
print("Saved notes.md")
```