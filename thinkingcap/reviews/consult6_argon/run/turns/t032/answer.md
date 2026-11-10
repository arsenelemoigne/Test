```python
# Let's start writing our quantitative findings to work/notes.md right now, and also do a comprehensive statistical analysis of all_runs.json and the log data.

with open(notes_path, "w") as f:
    f.write("""# Consult 6 (Argon) - Quantitative Research Notes

## 1. Task & Official Evaluation Math
- Model: Qwen3-0.6B, thinking mode ON (`<think>...</think>` + final answer).
- Target: > 50.0% tokens saved (thinking + answer combined) AND > -1.00 pp accuracy change on the full test (2,759 questions x 8 samples = 22,072 generations), using ONE clean, reproducible recipe (single training line, own answers only, no competitor copying, no test-time prompt/switch).
- Official evaluation breakdown (2,759 questions):
  - SVAMP (v): 1,000 questions (weight 0.25)
  - MBPP+ (c): 378 questions (weight 0.25)
  - IFEval (f): 541 questions (weight 0.25)
  - BFCL (pooled weight 0.25): 840 questions = simple (s: 400, weight 400/3360 = 11.90%), multiple (m: 200, weight 200/3360 = 5.95%), irrelevance/no-fit (i: 240, weight 240/3360 = 7.14%).
- Headline accuracy change = `(acc_v + acc_c + acc_f + (400*acc_s + 200*acc_m + 240*acc_i)/840) / 4`.
- Headline tokens saved = `1 - (ratio_v * ratio_c * ratio_f * ratio_bfcl)**0.25`.

## 2. Original Model (Raw Qwen3-0.6B) Full Test Baseline (`full-base`, 2,759 x 8) & Thinking-Off (`ntfull2`, 2,759 x 8)
- Raw thinking ON vs OFF accuracy and mean thinking tokens:
  - SVAMP (1,000): ON 88.7% (677 think tok) -> OFF 20.6% (-68.1 pp, 89-98% saved)
  - MBPP+ (378): ON 52.4% (2,079 think tok) -> OFF 40.8% (-11.6 pp, 89-98% saved)
  - IFEval (541): ON 58.7% (433 think tok, ~247 resp tok) -> OFF 57.5% (-1.2 pp, 62% total tok saved because response stays ~247 tok!)
  - BFCL simple (400): ON 82.2% (171 think tok) -> OFF 71.4% (-10.8 pp)
  - BFCL multiple (200): ON 76.1% (192 think tok) -> OFF 55.3% (-20.8 pp)
  - BFCL irrelevance / no-fit (240): ON 83.4% (186 think tok) -> OFF 90.3% (+6.9 pp! Thinking hurts no-fit by talking model into calling a function).
  - Pooled BFCL (840): ON 81.09% -> OFF 72.97% (-8.12 pp).
  - Overall 4-benchmark mean accuracy: ON 70.22% -> OFF 47.97% (-22.25 pp).
""")

print("Initial notes written.")
```