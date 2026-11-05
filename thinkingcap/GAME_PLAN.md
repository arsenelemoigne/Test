# ThinkingCap game plan — beyond v1 (v2, 5 Nov 2026; NOT committed to the GPUs)

v1 draft -> grounded by the consultant (Ivan, consult2 follow-up: consult2_game_plan_review.md, every number with its
reason, evidence and confidence). This v2 applies its corrections. Awaiting the user's approval; nothing is queued.

## 0. Goal and acceptance
- Target (full test 2,759 x 8): > 50% saved AND better than -1.0 (both strict).
- Screen acceptance (720 x 4): >= 52% saved and >= -0.5 — an engineering margin, not a guarantee (screen/full offset
  +0.6 / -0.3 seen on only two models; seed spread +-0.7).
- Selection on actual saving and accuracy (Pareto vs v1 = 46.7 / -0.5 on the 720), NOT on the curve residual; the curve
  stays a diagnostic.
- Labels: "repair signal" (credible gain vs its start), "useful checkpoint" (improves the observed frontier),
  "finalist" (>= 52 / >= -0.5, then validation).

## 1. Rules
- One change at a time for causal comparisons; Track B is a BUNDLED candidate (grouping + pairs + settings), judged as such.
- Pair manifest before training (section 6); decisions written before the run; no full test without the user's OK.

## 2. Shared settings
| setting | Track A | Track B | evidence / confidence |
|---|---|---|---|
| start / reference | merged pt-mid | original Qwen | essential / high |
| LoRA | r32, a64, dropout 0, 7 projections, embeddings frozen | same | family default / medium |
| DPO beta | 0.1 | 0.1 | D-runs and pt-family / medium |
| NLL anchor | 0 (close to D7's correctness runs) | 1.0, historical implementation (every pt merge had it) | not proven either way / medium |
| precision | fp32 weights + fp16 autocast | bf16 (historical pt path); re-check after bidax6p | confound, log stability / medium |
| lr / schedule | 1e-5 peak, 5-step warm-up, cosine to 0 | constant 2e-5 (historical) | low / medium |
| batch | 16 | 16 | pt-family / medium |
| length filter | prompt+chosen and prompt+rejected each <= 8,704 tokens; never truncate (max seen 8,391) | same | high |
| DPO log-probs | sequence-level (unchanged); NLL reduction logged | same | high |
| AdamW | all settings (betas, eps, weight decay) written to the log; grad clip 1.0 | same | medium/low |
| seeds | 1 screening, 2 for finalists (stability check only) | same | high |

## 3. Track A — repair pt-mid (fast, ~8-9 GPU-hours)
**Hypothesis (corrected).** pt-mid = 55.0 / -2.6; its loss comes from math (-1.40 of the overall), CODE (-0.88) and
no-fit (-0.51). Repairing all three while keeping most compression can reach ~52.7 / -0.45 (consultant's working-point
budget, not a prediction). Needs about +2.1 points, not +1.5.

pt-mid exactly: 4 adapters (math, code, ifeval, tools) trained alone on the baseline pair set (l2s50e rule + no-fit
protection; bf16, NLL 1.0, constant lr 2e-5, batch 16, one pass), merged at 0.30 / 0.45 / 0.60 / 0.45.

| step | value |
|---|---|
| A1 answers | merged pt-mid answers 5,400 training questions x4 (21,600 answers): math 2,400, code 900, no-fit 1,500, ifeval 300, simple 150, multiple 150; mixed-outcome questions first, then all-right and all-wrong |
| A2 pairs | 1,600 pairs, 50% shortest right > shortest FINISHED wrong + 50% similar-length right > wrong (ratio <= 1.25, exactly half with the right one longer); math 640, code 400, no-fit 240, ifeval 160, simple 96, multiple 64 (80% repair, 20% maintenance); no shorten, no wrong>wrong, no cut rejections; <= 3 pairs per question (<= 2 similar-length); dedup across kinds |
| A3 training | fresh LoRA on merged pt-mid, settings column A, one pass = 100 steps max, checkpoints 20 / 40 / 60 / 100 |
| A4 test | 720 x 4 at each checkpoint |
| stop | saving < 48% at step 40 -> stop A; accuracy gain < 0.5 at 40 = warning only; past 60 only if still improving the frontier |
| fails if | too few right answers on damaged questions; accuracy comes back only by long thinking; shortest-right guards still push shorter; no-fit repair turns into refusing valid calls; chosen likelihood collapses (-> NLL ablation) |

## 4. Track B — per-type adapters + merge (~25-40 GPU-hours, bundled candidate)
**Hypothesis.** Separately trained per-type adapters with their own pair mix, merged per type, beat one joint run
because types interfere (D7, D8). 4 adapters (math, code, ifeval, tools), not 6: the only proven merges used 4, BFCL
subsets are small, and one tools adapter learns call vs no-call jointly.

| step | value |
|---|---|
| B1 pairs (CPU) | 4,800 pairs from the existing 4-answer store (verified feasible jointly by the consultant): math 720 (shorten / guard / right 10 / 40 / 50), code 1,440 (65 / 20 / 15), ifeval 1,440 (65 / 20 / 15), tools 1,200 = simple 480 (45 / 25 / 30) + multiple 240 (20 / 35 / 45) + no-fit 480 (10 / 50 / 40) |
| shorten rule | math: 4/4 right, second-shortest right, rejected 1.3-4x; code: >= 3/4, shortest, 1.5-6x; ifeval: >= 3/4, thinking 1.5-6x and total >= 1.15x, >= 80% of the gap from thinking; simple / no-fit: >= 3/4, 1.5-6x; multiple: 4/4, second-shortest, 1.3-4x; >= 50% of the gap from thinking; <= 2 shorten pairs per question; length windows kept but nearly inactive (<= 4% exclusions) |
| guard rule | half shortest right > shortest finished wrong, half shortest right > longest wrong/cut; cuts <= 25%, rejected > 4x chosen <= 25% |
| right rule | right > finished wrong, ratio <= 1.25, exactly half with the right one longer |
| global | <= 3 pairs per question, <= 2 right pairs, no duplicate sample pairs, no wrong>wrong; tight spots: ifeval shorten needs second pairs; no-fit right 192 of 218 available |
| B2 adapters | settings column B; one pass: math 45 steps (save 15/30/45), code 90 (30/60/90), ifeval 90 (30/60/90), tools 75 (25/50/75) |
| B3 merge | final adapters; first the 3 historical vectors softer 0.20/0.25/0.35/0.25, soft 0.25/0.35/0.45/0.35, mid 0.30/0.45/0.60/0.45; then at most 8 one-coordinate moves around the best (math +-0.05, code +-0.05, ifeval +-0.10, tools +-0.05); merge weight deltas, one model |
| B4 control | same 4,800 pairs, ONE joint adapter, column B settings, 300 steps (= the four runs' total), screens 100 / 200 / 300 |
| B5 round 2 | only if a merge is at 48-52% and >= -0.5 with no part collapsed (>= 52 / -0.5 -> validate instead; worse accuracy -> repair; < 48% -> fix round 1): the merge answers ~3,000 questions x8 (math 600, code 800, ifeval 800, simple 300, multiple 200, no-fit 300); 2,400 pairs (shorten / guard / right: math 0/200/200, code 352/160/128, ifeval 384/160/96, simple 72/84/84, multiple 16/64/80, no-fit 0/192/128), 75% new + 25% replay; shorten needs >= 6/8 right (multiple >= 7/8); lr 1e-5, beta 0.1, NLL 1.0, batch 64, 38 steps, warm-up 2, cosine to 0, save 10/20/30/38 |

## 5. Tests
- 720 x 4: development and checkpoint selection.
- Fresh 240 x 8: confirmation only (note: it was used to choose models in October, so it is a secondary set, not pristine).
- 2 seeds for the frozen recipe; full test only for the final candidate with the user's OK.
- Paired question-level bootstrap (all answers of a question together, stratified by part, BFCL pooled and geometric token
  score inside each replicate).

## 6. Measurements and data saved
- Before training: hashes (base, adapter, tokenizer); full pair manifest (question id, source, sample ids, token hashes,
  kind, solve count, correctness, cut); prompt / chosen / rejected / thinking / answer lengths; filters and exclusion
  reasons; dedup; per-question reuse; allocation seed; grader version; exact loss reductions and optimizer settings.
- Every step (summary every 5): DPO and NLL loss separately; chosen / rejected policy and reference log-probs; margins
  and ordered-pair share by part and kind; chosen likelihood change; grad norm and clip rate; lr, loss scale, overflow;
  adapter delta norm. Stop only on persistent non-finite values.
- Every screen: per question and sample: correct, failure category, total / thinking / answer tokens, cut, thinking
  closed, malformed, tool action class (right call, wrong call, false refusal, right refusal), code failure kind,
  IFEval failure by constraint family incl. answer runaway.
- Alarms: NEW degradation vs the track's start model; cut rate +1 point = warning; disappearing thinking, systematic
  false refusals, grader exploitation = manual review. (The training "ordered right" % is a dashboard, not a rule.)

## 7. What would change the plan (consultant, section 8)
A lengthens math/code and drops below 48% -> stop A. Joint control >= separate adapters -> prefer the joint model. Tools
adapter trades no-fit against valid calls -> split no-fit off. Code fails sharply past ~57% -> stop raising code. IFEval
savings from breaking required lengths -> reject. Round 2 repeats the ~1-point debt -> stop at its earliest useful
checkpoint. Screen gain gone on fresh / seed 2 -> treat as noise.

## 8. Order (after the current queue: D6, D4, D5, D10, bidax6p / 6s, D12 chain1 rounds 1-2)
1. CPU: pair builders (A2, B1), feasibility and audits; check bidax6p for the precision choice.
2. Track A on one GPU; Track B adapters on the other.
3. B3 merges (3 screens), B4 control alongside, up to 8 coordinate screens.
4. B5 only if its gate passes. Freeze; seed 2 + fresh 240; full test with the user's OK.
GPU estimates are reservations: A ~8-9 h; B adapters <= 14 h; merges 3-11 h; control 6-7 h; round 2 <= 13 h.
