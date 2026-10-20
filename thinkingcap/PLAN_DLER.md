# Plan: retrain with DLER's proven recipe, at a proper scale

Why: our RL runs were small (20-60 steps x 32 questions), changed several things at once, used
untested settings (lr restarts) and lacked standard stabilisers. Long C's entropy ran away after
an lr restart (0.58 -> 2.84 in 23 steps) and was stopped at step 23. DLER (NVIDIA, arXiv
2510.15110) gets -77% tokens at +2.6 points average accuracy on DeepSeek-R1-1.5B with a simple
recipe. This plan copies that recipe as exactly as our setup allows and changes one thing at a time.

## 1. The DLER recipe and how we adapt it

| component | DLER (paper / code) | ours |
|---|---|---|
| reward | 1 if correct, 0 if wrong; any reply longer than L gets 0 ("truncation penalty"), implemented as the generation cap | same. L per source (see below), one fixed L per source for the whole run, no schedule |
| advantage | group-mean baseline (no std), then normalised over the **whole batch**: A = (A - mean_batch) / std_batch (REINFORCE++ baseline) | same, replaces our per-group std normalisation |
| clipped update | PPO ratio clip, low 0.2 / high 0.28 ("clip-higher"), dual clip c = 10, grad clip 1.0 | same. Needs the old log-probs: computed once per step before the first update |
| updates per step | 512 prompts, mini-batch 64: 8 updates per step, 1 epoch | 64 prompts, mini-batch 16: 4 updates per step |
| dynamic sampling | drop prompts whose rollouts are all 0 or all 1; sample more until the batch is full | same: sample 96, keep up to 64 informative prompts, top up if short |
| rollouts per prompt | 16 | 8 (budget); 16 if the pilot shows too few informative prompts |
| KL | MSE KL loss to the base model, coefficient 0.0005 | same (LoRA can disable the adapter for the reference) |
| entropy bonus | none | none |
| learning rate | 1e-6, constant, full fine-tuning | 1e-5 constant for LoRA r16 (LoRA needs ~10x a full-FT lr); **no restarts** |
| loss aggregation | token-mean | token-mean, as DLER (logged per source to watch long code replies) |
| sampling | temperature 1.0, top-p 1.0 for training | same (our runs used the eval's 0.6 / 0.95) |
| steps | 450 (7B), not stated for 1.5B | 100 for the main run, after pilots |
| difficulty-aware L (DA-DLER) | second stage, 150 steps: easy questions (correct ratio > 0.5) get L = 2000, the rest 4000 | optional stage 2 |
| merge | keep the top 25% largest weight deltas, scale 0.7, add to base | optional, after the main run; we already have exact LoRA blending |

**Choosing L.** DLER caps at 4,000 tokens for a model that writes ~10,500 on average (~40%). Our
raw model's training answers average about 500 (ARC, CSQA), 1,400 (GSM8K), 1,900 (Skywork) and
2,000+ (HumanEval). Proposed caps, to be checked by pilot: ARC/CSQA 512, GSM8K 1,024, Skywork
2,048, HumanEval 3,072. A reply that has not finished by its cap scores 0, exactly as in DLER.

**Healthy dynamics (DLER fig. 6), to monitor:** length drops fast, then rises gradually toward
the cap; entropy may rise *slowly* with clip-higher. Unhealthy: entropy jumping step to step (our
long C), length collapsing below ~30% of the cap, the share of all-0 prompts growing.

## 2. Scale and data

- **64 informative prompts x 8 rollouts = 512 replies per step** (we used 256, not filtered).
- **Sources per step, balanced:** GSM8K, ARC-Easy, CommonsenseQA, Skywork (kept; dynamic sampling
  drops the hopeless ones), HumanEval.
- **Code that never runs out:** HumanEval has 164 problems (124 train, 40 held out). At ~13 code
  prompts per step, 100 steps need ~1,300 code prompts. Step 1: reuse the 124 problems in a loop
  (each seen ~10 times, with fresh samples; watch the held-out 40 for memorisation). Step 2, if
  the pilot shows code over-fitting: add a larger coding set with unit tests (candidate:
  KodCode, to be checked for format and licence before use; never MBPP, which MBPP+ is built on).
- **No question repeats** for the non-code sources (pool of ~40,000).

## 3. Measurement

- **Every step, per source:** accuracy, mean length, entropy, share of prompts dropped as all-0 /
  all-1, clip fraction, KL. (Per-source accuracy/length/entropy logging is already in
  `rl_train.py`.)
- **Guard every 10 steps** on held-out maths, quiz and the 40 held-out coding problems, per
  source, as a monitor only (no automatic rollback), with an agreed stop rule: stop and analyse
  together if the guard drops more than 3 points below the raw model, or entropy rises more than
  0.3 in 5 steps.
- **Benchmarks:** checkpoints every 25 steps scored on the 240-question subset with **4 answers
  per question** for anything we might keep (2 for quick looks), always against the raw model run
  on the **same VM**.

## 4. Saving

- `--mirror` to `MyDrive/thinkingcap_runs/<run>` from step 1: every checkpoint, the per-step log,
  the guard file. Evaluation folders copied to Drive as they finish.
- One run per Colab cell; nothing queued only in the browser tab.

## 5. Order of work

1. **Implement** in `rl_train.py` (behind flags, old behaviour kept): truncation reward with
   per-source caps; batch-level advantage normalisation; PPO clipped loss with old log-probs and
   mini-batches; dynamic sampling with top-up; MSE KL; extra logs (drop rates, clip fraction, KL).
   Code reuse in a loop for HumanEval. Unit tests on the toy model for each piece.
2. **Pilot, 10 steps (~1.5 h):** check the curves (length, entropy, drop rates, per source) and the
   time per step. Adjust L or the group size if needed. One change at a time.
3. **Main run, 100 steps:** monitor every 10 steps against the stop rule; benchmark at 25/50/75/100.
4. **If accuracy is short of the limit:** merge (blend with base, or DLER's top-25% x 0.7) before
   training more.
5. **Optional:** DA-DLER stage (easy questions get half the cap) for another 50 steps.

## 6. Cost (L4, to be confirmed by the pilot)

Our current speed: ~256 replies of ~1,000 tokens in ~2.5 min, plus training. With 512 replies
(plus ~50% extra for dynamic sampling) but shorter caps, expect ~6-8 min per step: the pilot
~1.5 h, the 100-step run ~10-13 h of L4 (~20-25 compute units), plus ~15 min per benchmark check.

## Not known from the paper

Number of steps for 1.5B; whether wrong answers score 0 or -1 (we use 0); whether the merge's
top 25% is per tensor or global; any limit on dynamic-sampling retries; how to choose L for other
model sizes.
