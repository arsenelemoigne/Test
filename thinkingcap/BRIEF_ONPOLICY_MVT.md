# Brief for the MVT track: what the public entry shows, and on-policy, verified MVT (paste-ready)

## 1. The challenge, in a few lines

- **Task:** BottleCapAI ThinkingCap (github.com/BottleCapAI/ThinkingCap-Test). Make Qwen3-0.6B write
  fewer tokens, thinking included, while keeping accuracy. The working limit is at most -3 points
  against the untouched model.
- **Scoring:** the official `evaluate.py` / `compare.py` on SVAMP (maths), MBPP+ (code), IFEval
  (instructions) and BFCL (function calling).
  - Tokens saved = 1 minus the geometric mean of the per-benchmark token ratios, BFCL subsets pooled.
  - Accuracy change = the mean of the per-benchmark changes.
  - The full evaluation is 2,759 questions x 8 samples; we use a 240-question subset (40 per
    benchmark).
- **Rules:**
  - The model must still be ~0.6B and think before answering.
  - It must load in standard Transformers and vLLM with no custom code.
  - No training on the evaluation benchmarks.
  - The evaluation's prompt, sampling and caps stay unchanged.
  - The README: "not expected to … turn thinking off"; "interested in your own ideas, not how well
    you can copy others"; "send several checkpoints along the way".
- **Hardware:** Colab Pro, one L4 (24 GB).

## 2. Where we are (240-question subset)

| track | method | tokens saved | accuracy | samples |
|---|---|---|---|---|
| MVT (branch mvt-fading) | SFT on own traces cut by the Marginal Value Theorem on easy multiple-choice questions; maths traces kept whole and shown twice | 37.1% | -0.6 | 8 |
| RL (branch ccr-f7e14615-o4qu8r) | GRPO, 8 samples, "short and correct" reward (ShorterBetter); exact LoRA blend of a strong and a gentle run | 33.5% | -0.5 | 2 (about 0.5-1 pp optimistic) |

**Lessons so far**
- One-shot imitation of cut traces teaches stopping too early. Maths must stay whole.
- Shortening learned on maths and quiz questions transfers poorly to code and tool calls.
- TokenProbe and steering do not transfer.

**Headroom from our saved evaluations**
- Answers are tiny (10-200 tokens) and 72-99% of tokens are thinking.
- Per question, the shortest correct reply any of 8 of our models produced gives 44% saved.
- So short correct thinking already exists in the models; it just is not the habit.

## 3. The one public entry: "Lightning McQwen3"

Sources: github.com/basta/lightning-mcqwen3 and blog.basta.one/blog/thinkingcap (14 Oct 2026).
The numbers are self-reported, with 8 samples per question.

- **Headline:** 47.3% / -0.88.
- **Clean result: 43.2% / -0.85.** The headline includes a "think gate" that starts 30% of BFCL
  replies with an empty think block, which is probably against the rules. The clean result is the
  one to compare against.
- **Per benchmark (clean):** SVAMP 66% / -3.8, MBPP+ 39% / -1.9, IFEval 32% / +1.4, BFCL 25% / +0.9.

### What each step added (cumulative totals)

| step | total saved / accuracy | added |
|---|---|---|
| 1. Chain-of-Draft distillation: answers generated under a "max 5 words per thinking step" prompt with two examples; prompt removed; SFT on 11,247 correct traces (LoRA r16, lr 1e-4, 1 epoch) | 26.5% / -1.27 | +26.5%, mostly maths (SVAMP 52% shorter) |
| 2+3. **On-policy truncation oracle + DPO** (details below) | 42.5% / -1.45 | **+16.0% for -0.18 pp**, mostly BFCL and IFEval |
| 4. Anchored correctness DPO: own right maths trace over own wrong trace, lengths within 25%, 732 pairs, NLL anchor weight 1 | 43.2% / -0.85 | +0.7%, **+0.6 pp back** |
| (5. gate, BFCL only, empty think) | 47.3% / -0.88 | +4.1% |

### The oracle step

1. **Sampling:** the *current* (distilled) model answers ~9.3k new questions, 4 times each.
2. **Cutting:** every correct trace is cut at the sentence ends nearest 10%, 20% … 90% of its
   thinking and closed with `</think>`.
3. **Re-answering:** the model re-answers 4 times from each cut (197k cuts).
4. **Safe cut:** all 4 re-answers are right. **Stable run:** the safe cuts that stay safe to the end.
5. **DPO pairs** (9,034 in total):
   - 40% shorten: the stable run's second cut over the full trace.
   - 40% guard: the same chosen side over the unsafe cut just before it, with a wrong answer.
   - 20% hard: for questions with no stable run, the full trace over a middle cut with a wrong
     answer.
6. **Training:** DPO with lr 2e-5, beta 0.1, batch 16, reference = the starting adapter, no anchor,
   **stopped at step 100 of 561**.
   - Unanchored DPO trained longer collapsed: it emitted end-of-text after `</think>`.

### Proof that "on the current model" matters

The same idea applied to the original model's traces:
- SFT on cuts: 11.4% / -0.34.
- DPO on pairs: 28.1% / -0.78.

On the distilled model's traces it gives 42.5%.

### What failed for them

- Naive DPO, shortest vs longest correct: 90% saved but -27 pp.
- Correctness DPO without the anchor: -5.2 pp.
- More SFT data or epochs: no gain.
- Merging the two LoRAs: 51.0% / -3.93.
- Dropping the maths shorten pairs: SVAMP still shortened.
- Gating maths or code.
- Training on a bigger model's traces: distribution shift.

**Compute:** sampling on 40 GB cluster GPUs (8 shards, a few hours each); DPO and evaluation on a
12 GB RTX 4070 SUPER (~1.5 h per 8-sample evaluation).

## 4. How this maps onto MVT

| | MVT now | their oracle |
|---|---|---|
| traces | the original model's, sampled once | the current model's, freshly sampled |
| where to cut | MVT rule: stop when the next paragraph adds less certainty than its cost | try 10..90% |
| check the cut | p(correct) from the probe + natural re-answers | **4 re-answers from the cut must all be right** |
| training signal | SFT on the cut traces | **DPO with shorten pairs and guard pairs** (too early = wrong loses) |
| rounds | one | one (distill, then oracle) |
| maths | kept whole | shortened (SVAMP -3.8), their weak spot |

**Our edge:**
- Keeping maths whole is the opposite of their weak spot.
- The MVT rule picks cut points by marginal value instead of brute force, so far fewer
  re-answers are needed (one candidate cut instead of nine).
- Per-type handling.

## 5. Proposal to discuss: on-policy, verified MVT in rounds

1. Start from the current MVT winner, not the raw model.
2. **Sample** the current model on new questions (easy MC, maths, code, tool-like, none from the
   benchmarks), 4 times each.
3. **Cut** each correct trace at its MVT point, plus one earlier and one later candidate.
4. **Verify** each cut by re-answering K times, with K = 2-4 depending on budget. Safe means all
   right.
5. **Build pairs:**
   - shorten: safe MVT cut over the full trace.
   - guard: too-early cut with a wrong re-answer, losing to the safe cut.
   - maths: keep whole, or allow only verified-safe cuts with a stricter K.
6. **Train** with DPO (beta ~0.1, reference = the start adapter, short, stopped early with
   checkpoints) or SFT on the safe cuts. Compare both.
7. **Optional accuracy repair:** own right vs own wrong trace, with an NLL anchor.
8. **Repeat** steps 2-6 for 2-3 rounds, on fresh samples from the new model each round.
9. **Evaluate** each round (2 samples to screen, 8 to confirm) and keep every checkpoint.

**Budget on one L4 (rough):**
- 3k questions x 4 samples, plus ~3 cuts each x K re-answers: roughly 2-4 h of vLLM sampling per
  round.
- DPO or SFT: under 1 h.
- Evaluation: ~15 min with 2 samples, ~1 h with 8.

## 6. Open questions for the discussion

- Is MVT's cut point already close to the oracle's "second safe cut", or earlier or later? Measure
  on a few hundred traces.
- DPO or SFT on verified cuts: which fits best with MVT's existing SFT data?
- How many re-answers are needed at 0.6B (2 vs 4)?
- Can verified cuts make maths safely shorter, or should maths stay whole?
- Does the gain transfer to code and BFCL? Their oracle step helped BFCL and IFEval most.
- Originality: the write-up should credit Lightning McQwen3, ThoughtFold and Chain-of-Draft for
  the overlap, and present MVT cut selection, per-type handling and keeping maths whole as our
  own contribution.
