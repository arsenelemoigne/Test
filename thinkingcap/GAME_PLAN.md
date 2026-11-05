# ThinkingCap game plan — beyond v1 (DRAFT, 5 Nov 2026, not committed to the GPUs)

Status: draft for the user's review, then for the consultant's critique (consult2 follow-up). Nothing here is queued.

## 0. Goal and bar
- Target (full test, 2,759 x 8): **> 50% tokens saved at better than -1.0 accuracy point**.
- Screen target (720 x 4), with margin for screen bias (+0.6 saving, -0.3 accuracy on two checked models) and seed noise
  (+-0.7 accuracy between seeds): **>= 52% saved and >= -0.5**.
- Bar to beat: v1 = lpc-bidax5q s0, 46.7 / -0.5 on the 720 (+1.8 vs the curve). A candidate must beat v1 at equal saving,
  i.e. sit further above the curve acc(s) = -0.388 + 0.0899 s - 0.0028 s^2.

## 1. Rules for every run
- One change at a time against a named reference; the reference is re-scored only if its code path changed.
- Score on the 720 x 4 at fixed checkpoints, always per type (svamp, mbpp, ifeval, bfcl simple / multiple / no-fit).
- Before training, write the pair summary: counts per part x kind, median chosen / rejected length per part x kind,
  share of pairs whose chosen answer is the LONGER one (the hidden length pressure the consultant found).
- Decision rules are written before the run (section 6). No full test without the user's OK.
- Everything saved (section 7) so a later analysis (or the consultant) can re-use it.

## 2. Shared settings S0 (unless a track says otherwise)
| setting | value | why |
|---|---|---|
| start | original Qwen3-0.6B (Track A: merged pt-mid) | clean, comparable |
| LoRA | r32, alpha 64, dropout 0, all 7 projections | v1's; the per-type adapters used the same rank |
| DPO | beta 0.1, reference = the start model, no NLL / SFT term | v1's; NLL anchor did not help length DPO before |
| optimizer | AdamW, lr 2e-5 peak, warm-up 5%, cosine to 0, grad clip 1.0 | v1's |
| precision | float32 weights, float16 autocast + GradScaler | v1's; bf16 vs fp32 decided by bidax6p (running) |
| batch | 16 pairs / step (Track B adapters: 8, see B2) | v1's |
| length | one pass; pairs over 8,704 tokens skipped | v1's |
| seeds | 1 for screening; 2 for any finalist | seed spread +-0.7 pts |

## 3. Track A — repair pt-mid (fast test of the consultant's alternative 1)
**Hypothesis.** pt-mid (55.0 / -2.6 on the 720, ~+1.3 vs curve) loses almost only on math (-5.6) and no-fit (-7.1);
a short correctness round on those two parts can recover >= 1.5 points while keeping >= 50% saved.

**pt-mid, exactly:** 4 adapters (math, code, ifeval, tools), each trained alone on its type's pairs from lpv-base
(l2s50e rule + no-fit protection; LUPIN settings: bf16, NLL anchor 1.0, constant lr 2e-5, 16 pairs / step, 1 pass),
merged at strengths math 0.30 / code 0.45 / ifeval 0.60 / tools 0.45. Adapters on Drive: adapters/lpw-pertype-*.

| step | content |
|---|---|
| A1 data | merge pt-mid to a full model; it answers the training pool's math (svamp-like) and no-fit questions x4 (~5,400 q), plus a 1,500-question replay slice of the other parts x4 |
| A2 pairs | math + no-fit: 50% similar-length right > wrong (length ratio <= 1.25, half with the right one longer), 50% guard = shortest right > SHORTEST finished wrong; NO shorten; other parts: replay pairs of the same two kinds (20% of the total); ~1,500 pairs |
| A3 training | fresh LoRA on merged pt-mid, reference = pt-mid, S0 but lr 1e-5, 60 steps, checkpoints 20 / 40 / 60 |
| A4 test | 720 x 4 at 20 / 40 / 60 |
| decision | KEEP if a checkpoint has >= 50% saved and >= -1.0 (screen); STOP at step 40 if saving < 48% or accuracy gained < 0.5 |
| cost | ~3 h answers + 0.5 h training + 3 h screens on one GPU |

## 4. Track B — the consultant's per-type recipe (consult2_dpo_strategy.md), adapted
**Hypothesis.** Separately trained per-type adapters, each with its own pair mix, merged at per-type strengths, reach a
better trade-off than any single joint run, because types interfere (D7, D8 show protection leaking across types).

| step | content |
|---|---|
| B1 pairs | new pair rule (lupin_pairs.py --recipe pertypeq) with the consultant's quotas: math 720 (10 / 40 / 50 shorten / guard / right), code 1,440 (65 / 20 / 15), ifeval 1,440 (65 / 20 / 15), simple 480 (45 / 25 / 30), multiple 240 (20 / 35 / 45), no-fit 480 (10 / 50 / 40); eligibility, length windows, per-question caps and the guard half-and-half rule as in his section 2B; from the 4-answer store of the original model |
| B2 adapters | 6 adapters from the original, S0 except batch 8; steps math 90, code 180, ifeval 180, simple 60, multiple 30, no-fit 60; checkpoints at 1/3, 2/3, end |
| B3 merge search | start strengths math 0.35 / code 0.80 / ifeval 0.90 / simple 0.65 / multiple 0.50 / no-fit 0.85 at the 2/3 checkpoints; global x 0.6 / 0.8 / 1.0 (3 screens), then +-0.15 on math, code, ifeval, no-fit around the best (up to 8 screens) |
| B4 control | the same 4,800 pairs in ONE joint adapter (S0, ~300 steps), 720 at 100 / 200 / final: does separation help? |
| B5 gated round | only if the best merge is >= 46% and better than -0.7: the merge answers ~3,000 questions x8, the consultant's second-round pool (2,400 pairs, no math / no-fit shorten), lr 1e-5, batch 64, ~38 steps, checkpoints 10 / 20 / 30 / 38 |
| decision | finalist if >= 52% / >= -0.5 on the 720; then seed 2 and the fresh 240 x 8 before any full test |
| cost | ~14 h adapters + ~11 h merge screens + 6 h control + ~13 h round 2 |

## 5. Checks during training (every run)
- Pairs ordered right %: rises above 60% by mid-run; flat at ~50% = no signal; > 95% early = too easy (check pairs).
- Loss and the tripwire line (dpo.py) every 5 steps; NaN / loss spike -> stop.
- Screens: per-type cut rate, thinking-closed rate, mean answer (after </think>) length (the IFEval runaway), accuracy
  per type; any type below -3.0 or cut rate up > 1 point -> flag before continuing.

## 6. Tests and decisions
- Every checkpoint: 720 x 4, per type, vs curve.
- Finalists (max 2): seed 2 + fresh 240 x 8 (independent of the screen). The consultant asked for an independent
  ~2,000 x 8 validation set: we do not have one; open question (section 9).
- Full test only for the final candidate, with the user's OK.

## 7. Saved for later analysis
Pair files + pair summary (section 1), training logs (loss, ordered-right, lr), every checkpoint adapter, merge
weights, per-question records of every screen (s4-*/records.jsonl), all on Drive; a log entry per result in
EXPERIMENT_LOG.md; the visuals updated.

## 8. Order (after the current queue: D6, D4, D5, D10, bidax6p / 6s, D12 chain1 rounds 1-2)
1. Track A (one GPU, ~7 h) while B1 is built and tested on the Mac.
2. B2 adapters (both GPUs, ~7 h each), then B3 merge screens; B4 control in parallel.
3. B5 only if its gate passes. Finalists -> section 6.

## 9. Open questions (for the consultant)
- Justify each hyperparameter with the data (pair counts per kind, step counts, batch 8, lr, merge strengths, round-2 size).
- Track A: is the repair spec right (kinds, size, lr, steps)? What would make it fail?
- bf16 vs float32 / cosine to 0 vs 2e-6: wait for bidax6p, or does it not matter for per-type adapters?
- 4 adapters (pt-mid's grouping) or 6 (his)? BFCL's three parts have few pairs each.
- Validation: is the 720 x 4 + fresh 240 x 8 enough instead of a new 2,000 x 8 set?
