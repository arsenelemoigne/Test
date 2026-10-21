# Public entry: "Lightning McQwen3" (github.com/basta/lightning-mcqwen3, blog.basta.one/blog/thinkingcap, 14 Oct 2026)

Self-reported, 8 samples per question with the challenge's evaluate.py/compare.py (full set implied, not stated).

| benchmark | with think gate | without gate (clean) |
|---|---|---|
| SVAMP | 66.3% / -3.8 | 66% / -3.8 |
| MBPP+ | 39.1% / -1.9 | 39% / -1.9 |
| IFEval | 32.0% / +1.4 | 32% / +1.4 |
| BFCL | 44.6% / +0.8 | 25% / +0.9 |
| **all** | **47.3% / -0.88** (composite run; full gated rerun 46.6% / -1.33) | **43.2% / -0.85** |

The gate starts BFCL replies with an **empty think block** (30% of BFCL prompts): against "not
expected to turn thinking off". The clean number is 43.2% / -0.85.

## Pipeline (each step a LoRA r16/alpha 32 on all linear layers, continued from the previous)

1. **Chain-of-Draft distillation**: questions answered twice under a "max 5 words per thinking
   step" prompt with two examples; one correct trace per question, prompt removed, 11,247 SFT
   targets, lr 1e-4, 1 epoch. Alone: 26.5% / -1.27 (SVAMP 52% / -2.9).
2. **On-policy truncation oracle**: the distilled model answers ~9.3k other questions 4x; every
   correct trace cut at sentence ends near 10..90% of its thinking, closed with `</think>` and
   re-answered 4x (197k cuts). Safe cut = all 4 re-answers right; "stable run" = safe cuts that
   stay safe to the end.
3. **Oracle DPO**: 9,034 pairs, 40% shorten (stable run's second cut over the full trace), 40%
   guard (same chosen over the unsafe cut before it + wrong answer), 20% hard (full trace over a
   middle cut + wrong answer for questions with no stable run). lr 2e-5, beta 0.1, batch 16, no
   anchor, stopped at step 100 of 561: 42.5% / -1.45. **Biggest jump** (from 26.5%).
4. **Anchored correctness DPO**: own right maths trace over own wrong trace, lengths within 25%,
   732 pairs, 1 epoch, plus NLL on the chosen trace (weight 1): 43.2% / -0.85 (+0.6 pp).
5. Optional gate: logistic probe on layer 21 at the last prompt token, fitted on ~1.5k ToolACE
   prompts; BFCL only.

Data: only the challenge's suggested train.jsonl (+ ToolACE for the gate). Compute: sampling on
40 GB cluster GPUs (8 shards, a few hours each); DPO and evals on a 12 GB RTX 4070 SUPER.

## What did not work for them

Naive DPO shortest-vs-longest correct (90% saved, -27 pp); unanchored oracle DPO collapsed
(emits end-of-text after `</think>`); correctness DPO without anchor -5.2 pp; more SFT data or
epochs (no gain); merging the two LoRAs 51.0% / -3.93; dropping maths shorten pairs (SVAMP still
shortened); gating maths/code; training on traces from a bigger model (distribution shift).
Offline oracle SFT on base traces: only 11.4% / -0.34 (close to our offline MVT SFT experience).

## Lessons for us

- On-policy beats offline: cutting the *current* model's traces and checking each cut by
  re-answering is what made the big jump; same lesson as the iterated on-policy SFT literature.
- Preference pairs with **guard pairs** (cut too early -> wrong) teach where not to cut.
- An anchored correctness step recovers accuracy cheaply.
- Their clean 43.2% / -0.85 (8 samples) is the bar; MVT track 37.1% / -0.6 (8 samples); our RL
  blend 33.5% / -0.5 (2 samples, optimistic by ~0.5-1 pp).
- Originality: copying this pipeline would be "copying others"; our distinct pieces are the MVT
  stopping rule, per-type handling, RL with a correctness-safe length reward and adapter blending.
