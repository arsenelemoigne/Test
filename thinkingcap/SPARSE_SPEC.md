# SPARSE: pruning Qwen3-0.6B and teaching the leftover weights to think short

**An add-on project, separate from LUPIN, run continuously on spare GPU time.** It never delays LUPIN: LUPIN comes
first, and SPARSE reuses what LUPIN produces (the checked pool, the 10 answers per question, the LUPIN v1 pairs). No new
training data is collected for it. Decided with the user on 28 Oct 2026; nothing is built yet.

## The question

How many of Qwen3-0.6B's weights does it really need, and once the "main path" is kept, is it easier or harder to teach the
model to think shorter? The user's goal is fewer active parameters at (nearly) the same accuracy.

## Rules (same as LUPIN)

- Scored like LUPIN: the fresh test set (240 questions x 8), accuracy and thinking tokens **per data type** (math, code,
  instructions, tools). Accuracy limits apply to the overall score (as the competition scores it), reported per type too.
- Accuracy budgets: **v1 = less than 3 points lost, v2 = less than 1 point lost** versus the raw model. One point is close
  to run-to-run noise (~0.8 pt measured), so final candidates are also scored on the full evaluation (2,759 x 8).
- Training data comes from the LUPIN pool only (never the evaluation sets). The fresh set is only for scoring.

## The arms

| arm | model | target | question |
|---|---|---|---|
| A | raw Qwen | reference | thinking tokens per type (done) |
| B | LUPIN (our fine-tune) | < 1 pt | how much shorter training makes it |
| C | raw Qwen pruned with **Wanda** | v1 < 3 pts, v2 < 1 pt | how much can go without retraining; does thinking get longer? |
| D | raw Qwen pruned with **Lottery Ticket** | v1 < 3, v2 < 1 | the same with retraining: can more go? |
| E | pruned model + one LoRA round on the LUPIN v1 pairs ("think shorter") | within the pruned model's budget | does an emptier model learn short thinking more easily or less easily? |
| F | pruned model, the kept ("winning") weights frozen, the removed weights trained to think shorter | same | can the freed weights carry the new habit, and how much free space is needed? |

## How each step works

**C. Wanda (one-shot, no retraining).** Score every weight by |weight| x the strength of its input on calibration text,
remove the lowest X% inside each output neuron, done in minutes. Sweep X = 10, 20, 30, 40, 50%; keep the largest X within
3 pts (v1) and within 1 pt (v2). Calibrate on Qwen's OWN reasoning answers from the LUPIN pool, not generic web text
(on reasoning models this mattered a lot in published work: Qwen3-8B at 50%, MATH 0.56 with web-text calibration vs 0.86
with the model's own reasoning). The official Wanda code needs small fixes for Qwen3 (newer transformers, sequence length
2048 instead of 40,960, pass the rotary position inputs).
Optional, cheap: one Wanda map per data type (calibrate on math only, code only, instructions only, tools only) and measure
how much the maps overlap and where (early vs late layers): the "routes" per data type.

**D. Lottery Ticket (iterative, with retraining).** Each round: remove the smallest 20% of the remaining weights, reset the
survivors to Qwen's original weights, retrain. About 10 rounds reach ~90% removed; evaluate after each round and keep the
largest cut within 3 / 1 pts. "Retrain" = train the pruned model to reproduce Qwen's own RIGHT answers from the LUPIN pool
(self-distillation: keeps the model's behaviour, reuses LUPIN data). Control from the original paper: the same cut chosen at
random. Masks are kept during training; optimizer reset each round. Only the linear layers of the 28 blocks are pruned
(not the tied word embedding, not the norms).

**E. Pruned + LoRA "think shorter".** One LoRA round on the LUPIN v1 pairs (reused as they are, so no new generation;
they come from the original Qwen, a known imperfection of this cheap test). Compared with B given the same single round,
so the comparison is like for like. Measured against the pruned model's own starting point (pruning alone tends to make
thinking longer).

**F. Freeze the winning weights, train the freed ones.** Start the freed weights from ZERO (option 2, chosen: the pruned
model "grows back" capacity only for the new habit), train them on the LUPIN v1 pairs with the winning weights frozen.
Variants: train 1% / 5% / 20% / all of the freed weights. Control: train a RANDOM set of the same size (it may include
winning weights) to see whether WHICH weights are trained matters. (Option 1, freed weights starting from their original
values = full Qwen with a partial freeze, is a possible later variant.)

## Order and cost (one L4 / A100)

1. **Pilot on the Wanda cut** (E and F use Wanda's mask instead of the Lottery Ticket's): C sweep ~3 GPU-h; E + matched B
   ~4 GPU-h; F variants + control ~10 GPU-h. **Pilot total ~15-20 GPU-hours.**
2. **Only if the pilot is interesting:** D (Lottery Ticket, ~15-30 GPU-h) and E / F on its mask (+10-15 GPU-h).

## What published work says (research of 28 Oct 2026, sources in the conversation log / EXPERIMENT_LOG)

- Lottery Ticket: small image networks keep accuracy with 10-20% of their weights (Frankle & Carbin 2019); deep ones needed
  rewinding to an early training point (2020); BERT-base (110M) keeps it with 30-60% of weights depending on the task
  (Chen et al. 2020). No published true Lottery Ticket run on a decoder LLM: Qwen3-0.6B would be the first and the largest.
- Wanda: ~50% of weights removed on 7B-70B Llama models with little loss; collapses near 80%; small models break sooner and
  SparseGPT beats Wanda there (0.5B-3.8B study, 2025).
- Pruned reasoning models lose accuracy and think longer (Qwen3-8B, R1-Distill); calibrating on the model's own reasoning
  recovers much of it.
- Training only a small or even random fraction of weights can be enough to learn new behaviour in RL (2025-2026 papers),
  which is why F may work.
- Unstructured sparsity (C, D) does not make the model faster or smaller in memory on normal GPUs; structured pruning
  (whole neurons / layers, e.g. Minitron, Sheared-LLaMA) plus distillation is the route to a truly smaller model, a possible
  later extension.
