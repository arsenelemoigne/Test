# The team's GRPO runs (3 Nov 2026), everything they did: setup, settings, data, results

## Setup common to every run
- Start: the ORIGINAL Qwen3-0.6B (no SFT), a NEW LoRA on all linear layers, rank 32, alpha 64, dropout 0.
- Rollouts: vLLM, the test's sampling (temperature 0.6, top_p 0.95, top_k 20), max 4,096 new tokens for EVERY type (code included, while the test gives code 8,192), seed = run seed + step.
- Batch: 12 questions x group 8 = 96 completions per optimizer step; 150 steps (= 14,400 completions in total); one optimizer update per batch.
- Optimizer: AdamW, weight decay 0, CONSTANT learning rate 2e-6 (no warm-up, no decay), gradient-norm clip 1.0. The run is executed in blocks of 10 steps; each block reloads the previous adapter but starts a FRESH AdamW (optimizer moments reset every 10 steps).
- KL to the original model: per-token coefficient (0.04 in most runs, 0 in grC).
- Advantages: group mean subtracted; "Dr. GRPO" flag = NOT divided by the group standard deviation and the loss divided by a constant (replies x max tokens); without the flag = standard GRPO (divided by the group std, per-reply mean loss).
- Data: the same questions as the team's DPO data: a pool of ~18,000 training questions (the same raw_meta questions you analysed), mixed like the test (25% each benchmark; BFCL 400/200/240), interleaved by those shares, shuffled; --tp-min-solve 0 = every question kept (including never-solved ones). 150 steps x 12 = 1,800 questions seen once each.
- Grading during RL: the training questions' own checkers (typed graders, IFEval with langdetect for language constraints).
- Screening: a 720-question screen of the test-like distribution, 4 answers each, at steps 50 / 100 / 150; results are tokens saved % vs the original (NEGATIVE = LONGER) / accuracy change in points.

## Reward ("lenpen", copied from a public reproduction: khudgins / ornith-thinking-cap)
right (and finished): 1 - lam * min(1, thinking_tokens / 2048); wrong or cut: 0; PLUS a format bonus of 0.1 to EVERY reply whose thinking is closed with an answer after it (right or wrong).
lam ramps linearly from 1e-3 to lam_max over --lp-ramp steps. Optional gate: lam = 0 in a group right less than G of the time (unused).
```python
            elif args.reward == "lenpen":
                lam = args.lp_lam_start + (args.lp_lam_max - args.lp_lam_start) * min(1.0, (args.offset + step) / max(1, args.lp_ramp))
                if sum(correct) / len(correct) < args.lp_gate:
                    lam = 0.0
                think = [r.index(close_id) if close_id in r else len(r) for r in group]
                formed = [close_id in r and len(r) > r.index(close_id) + 2 for r in group]
                rewards = [(1.0 - lam * min(1.0, n / args.lp_norm) if ok else 0.0) + (args.format_bonus if fm else 0.0)
                           for n, ok, fm in zip(think, correct, formed)]
```
```python
def group_advantages(rewards, scale=True):
    """GRPO: reward minus the group mean, over the group's standard deviation (0 if all equal); scale=False (Dr. GRPO) keeps
    the plain difference to the mean."""
    mean = sum(rewards) / len(rewards)
    spread = statistics.pstdev(rewards)
    if spread < 1e-8:
        return [0.0] * len(rewards)
    return [(r - mean) / spread if scale else r - mean for r in rewards]
```
Loss aggregation (Dr. GRPO branch), excerpt:
```python
ntered)
        jobs = [tuple(j) for j in jobs]

        # on-policy update(s): loss = -sum_t A_t log pi(token_t), averaged over the chunk's tokens
        lr = learning_rate(step, args.steps, args.lr, args.lr_end, args.restarts)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        model.train()
        loss_sum, entropy_sum, entropy_count = 0.0, 0.0, 0
        active = [j for j in jobs if j[1] and not (j[3] == 0 and not j[2])]
        for chunk in split_groups(active, args.updates, rng):
            total_tokens = max(1, sum(len(j[1]) for j in chunk))
            if args.dr_grpo:  # a constant denominator: long wrong answers no longer get a smaller per-token push
                total_tokens = max(1, len(chunk) * args.max_tokens)
            n_questions = max(1, len({id(j[5]) for j in chunk}))
            if args.objective == "drpo":
                kl_sum, kl_terms = 0.0, []
                for batch in token_batches(chunk, args.batch_tokens):
                    with torch.autocast(device_type=device, dtype=dtype, enabled=dtype != torch.float32):
                        batch_lp = batch_logps(model, [(j[0], j[1]) for j in batch], device)
                    loss = 0.0
                    for j, logps in zip(batch, batch_lp):
                        loss = loss - j[3] * logps.float().mean() / n_questions
                        kl_terms.append(float((old_logps[id(j[1])].to(logps.device) - logps.float()).sum()))
                    sc
```

## The runs (exact settings)
| run | lam_max | ramp | KL | advantage | other | status |
|---|---|---|---|---|---|---|
| grA | 0.05 | 100 | 0.04 | Dr. GRPO | the public recipe | 150 steps |
| grB | 0.2 | 100 | 0.04 | Dr. GRPO | | 150 steps |
| grC | 0.05 | 100 | 0 | Dr. GRPO | | 150 steps |
| grS | 0.05 | 100 | 0.04 | standard (std-scaled) | | stopped before scoring |
| grP | 0.5 | 50 | 0.04 | standard | | stopped before scoring |
| grF | SmartThinker reward | | 0 | standard | only questions solved <= 75% | stopped before scoring |

## Results on the 720 screen (tokens saved % / accuracy points; negative saving = longer than the original)
| run | step 50 | step 100 | step 150 |
|---|---|---|---|
| grA | +1.5 / -1.0 | -2.3 / -1.2 | -5.8 / -0.9 |
| grB | -2.0 / -0.5 | +1.2 / 0.0 | -0.6 / -0.7 |
| grC | -0.3 / +0.4 | -3.0 / -1.0 | -6.7 / -0.3 |
Training logs: batch accuracy 83-95%, mean thinking 400-730 tokens per step with no downward trend; every question group had some signal.

## The team's own log entries about these runs (verbatim)
- ~02:00 (user): three tasks. (1) a stronger v1 from pair techniques compared at equal accuracy (pending the user's
  list); (2) GRPO from the raw Qwen to learn how it behaves vs DPO; (3) RL on v1 with the winning reward.
  Built for (2): rl_train.py --reward lenpen (khudgins' ThinkingCap reproduction, read from github.com/khudgins/
  ornith-thinking-cap: right = 1 - lam * min(1, thinking / 2048), wrong = 0, + 0.1 format bonus; lam ramps 1e-3 ->
  lam_max), --lp-gate (GRPO-lambda: no length penalty where a group is right < G of the time), --dr-grpo (advantages not
  divided by the group spread; loss divided by replies x max tokens). His settings: G 8, 12 prompts, lr 2e-6, KL 0.04,
  temperature 0.9 (we keep the evaluation's sampling), LoRA r32 a64, 500-1000 steps. Papers: lr ~1e-6 typical at
  0.5-1.5B full fine-tune (LoRA usually higher); KL 0 (TRL default, Dr. GRPO / DAPO) to 0.04; GRPO-lambda reports a
  uniform length penalty losing accuracy after ~40 steps.
  Queued: GPU 1 after bidax5q seed 1: gr0 smoke, grA (his recipe), grB (penalty up to 0.2); GPU 2 after bidax7: grC (no
  KL), grD (gate 0.5). All from raw, 150 steps, the 720 at 50/100/150, ~4-4.5 h each.
- ~02:30 (user): GRPO from the raw Qwen as RL, 5 variations, ONE setting changed each from grA (lr 2e-6, KL 0.04,
  penalty up to 0.05, no gate): grB penalty 0.2, grC KL 0, grD gate 0.5 (GRPO-lambda), grE lr 5e-6. 150 steps each,
  12 questions x 8 answers, Dr. GRPO normalization, the 720 at 50/100/150. GPU 1: gr0 smoke -> grA -> grB -> grE;
  GPU 2 after bidax7: grC -> grD. Then adapt.
- ~02:40: GRPO screens now draw the SAME questions as v1's DPO data: pool dp13 + nofit4 + math1 + math2 (thinkprune
  --tp-extra, default nofit4,math1,math2; answers from bidax5q's 4-answer file, used for the eligibility filter only).
  The plan (user): 10 DPO pair strategies (all else = bidax5q) -> v1; 5 GRPO variations from the raw Qwen; then v1
  answers its own questions and both DPO and GRPO are tried on it. All judged on the 720 with the yardstick.
- 01:30 (real clock; earlier "~02:xx" stamps tonight were estimates): bidax5q SEED 1 final +48.5 / -2.2 (svamp
  +58/-2.9, mbpp +39/-4.6, ifeval +51/-0.4, simple +40/-2.5, multiple +44/-0.4, no-fit +45/+0.8): +0.4 vs the curve
  (seed 0 final +46.7 / -0.5, +1.8). Seed 1 step 300 also scored (line truncated in the log view). The two seeds'
  finals differ by 1.7 pts of accuracy: the recipe is ~+1.1 above the curve on average, seed luck +-0.7; the 1-point
  bar holds at step 200 for both seeds (+1.7 / +1.3), at the final only for seed 0. lpw-2stage step 50 re-scored on the
  720: +29.5 / +1.6 (+1.8 vs the curve; a dp13-era "be right first, then shorten" two-stage run, 29 Oct).
  GPU 1 had pulled the plan before the GRPO lines and started bidax6s: interrupted and restarted so gr0 / grA / grB /
  grE run first.
- ~01:45 (user): sixth GRPO variation grF = SmartThinker reward (arXiv 2603.08000). rl_train.py --reward smart: per
  group, l* = (s1^2 mu2 - s2^2 mu1)/(s1^2 - s2^2) from all lengths ~N(mu1,s1^2) and right lengths ~N(mu2,s2^2), clipped
  to the group's range; only right answers longer than l* penalized (-ReLU(l - l*)); coefficient p_err / (mean(r_len) -
  min(r_len)), the largest that keeps every right answer at or above the group mean; groups with < 2 right answers or
  all right get plain 0/1. Unit check on a toy group: right answers >= mean, wrong <= mean, the 900-token right answer
  lands exactly on the mean. Everything else = grA (the one change is the reward). Queued on GPU 2 after grD, behind a
  5-step smoke test (gs0).

### 3 Nov: grF aligned with the SmartThinker paper (read in full)
Before it ran, grF had three settings borrowed from grA that the paper does not use. Now: plain GRPO advantages
(reward minus group mean, divided by the group's spread; token-mean loss, no Dr. GRPO), no KL leash (the paper omits
it), constant lr. New `track.py --tp-max-solve`: grF keeps only questions the original Qwen misses at least once in 4,
because an all-right group gets λ = p_err/... = 0, so no signal, and the paper uses 64 questions a step against our 12
(the paper itself says difficulty-based sampling would cut the steps needed). lr stays 2e-6 like grA (LoRA r32; the
paper's 1e-6 is a full fine-tune). Paper facts worth keeping: the longest right answer always gets advantage 0 (never
pushed down); the fixed-λ ablation saved more (−63%) but lost 3.2 pts on the 1.5B model, so the dynamic λ is the
accuracy guard; SmartThinker on top of an already-pruned model (ThinkPrune-4k → SmartThinker) was its best result,
which fits the plan to try GRPO on v1.

**3 Nov ~02:00:** only 2 GPUs on the Colab account (user), so runtimes 3/4 are dropped (plan_line_c/d and the two notebooks
removed) and the DPO screen alternates with GRPO on the two queues, scored at steps 100 / 200 / final (step 300 dropped:
too close to the final at 339). GPU 1: grA -> D11 push2 -> grB -> D1 dast -> grE -> D2 hard, D3 route, D4 budget, D5 margin,
D10 l2s50e -> old re-scores, bidax6s. GPU 2: code2 -> bidax7 -> grC -> D9 adar1 -> grD -> gs0, grF -> D7 st1 + 2stage,
D8 math, D6 simright -> bidax6p.
### 3 Nov: idea parked: weight-space editing (Astra) and a contrastive Wanda test (discussion only, nothing built)
Source: "Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang, Khattab, MIT, 2 Nov 2026,
rikiyat.github.io/blog/weight-space-programming; logs github.com/RikiyaT/lm_understand-logs). A frontier LLM ("GPT-6
Astra"), with a Python sandbox and up to 50 forward-pass batches, reads and edits weights by reasoning, no gradient descent.
Closest to us, Exp 8: Qwen3.5-0.8B looped to the 32k cap on 93% of MATH-500; Astra traced the loop to one direction written
by the MLPs of layers 15 and 19 and removed it from 64 output columns each (0.016% of weights): cap hits 93 -> 33%, MATH-500
7.9 -> 48.1, IFEval x3. On Gemma 3 270M (rarely loops) no gain: "editing could remove an obstacle that hid an ability, but
it could not add an ability the model lacked". So: it removes a bad habit, it does not add skill; math is spread over the
weights (no "math weights" to pick out), but a habit (re-checking a settled answer) is a direction / a few heads.
How it relates to ours: closer to Track 2d steering (October: 'short vs long' direction, ~18% saved at -0.1, stronger pushes
broke accuracy) than to Wanda. Wanda (30 Oct: 10% pruned free, 20-30% make the model think LONGER) keeps what the model
does on its usual answers; Astra finds the part that CAUSES a behaviour and removes only that. Our per-type length
directions are nearly unrelated (cosines -0.18 to 0.39), so a math-only edit might spare the other types (DPO spills over).
Hard part: some re-checks catch real errors; a column removal acts on every question alike, so keeping accuracy needs the
model to "know" inside when it is already right (cf. "Reasoning Models Know When They're Right", 2025).
How much waste is there to remove (our 25-26 Oct cut measurements, original Qwen, benchmark questions, small samples:
51-148 replies): earliest cut whose 20 forced answers are as right as finishing, right replies only: SVAMP 41%, IFEval
49%, MBPP+ 72%, BFCL 49% of the thinking. But on SVAMP the answer builds up LATE (success 0.34 at 20-40% kept vs 0.73
full), so math has the least "re-check after a settled answer" and code the most; we never labelled re-checks directly.
Cheap tests first, no API cost, ~1 GPU-h each, after the GRPO / DPO screens: (1) label re-checks in our math answers
(did the answer change?) and fit a probe on hidden states: can the 0.6B tell useful from wasted re-checks? (2)
contrastive Wanda: score weights on wasted vs useful stretches (wanda_prune.py), dampen those used only by wasted ones.
Only if (1) shows a signal: an Astra harness (OpenRouter key; cost unknown, user's call), judged on the 720 / full test.
### 3 Nov ~04:00: grA step 50 (720 x4) + grA crash fixed
**grA-50** (GRPO from the raw Qwen, khudgins/Ornith length penalty, λ ramping to 0.05 over 100 steps, so ~0.025 at step 50;
leash 0.04, lr 2e-6): **+1.5% / -1.0** (vs curve about -0.7; saving under 15%, below the yardstick's range). svamp +1.7 /
-0.2, mbpp +3.3 / 0.0, ifeval +0.4 / -2.1, one function -4.1 / -1.2, several +1.6 / -2.9, no fit +2.8 / -1.0. Training: batch
accuracy 83-95%, thinking 400-730 tokens per step with no trend, every question with signal (12/12): at this penalty
strength the reward is almost all "be right", so 50 steps change nearly nothing. As expected for this recipe (khudgins
trains 500-1,000 steps); the stronger half of the ramp (steps 50-100) is where saving should show.
**Crash:** at step 51 rl_train's IFEval grader hit a language constraint and `import langdetect` failed (the package
was missing on this runtime). Fixed: rl_train.py installs langdetect if missing (same fix protects grC / grD / grF).
grA re-queued on GPU 1 right after D11 lpd-push2 (now training), resuming from its saved step 50. GPU 2: code2 answers
done (1,500 questions), bidax7 training.
- ~08:00 **bidax7 step 100** (bidax5q + instr2 + code2, michal-bida's own instruction and code sources): **+41.3 / -1.9,
  -0.5 vs the curve**: svamp +49 / -4.6, mbpp +38 / -1.7, ifeval +40 / -0.8, one function +36 / -1.9, several +39 / +0.8, no
  fit +41 / +0.2. Below bidax5q s0 at step 100 (41.8 / +0.2, +1.7) at the same saving; steps 200 / 300 / final scoring.
  GPU 1 moved on to grB (GRPO, 4x penalty).
- ~09:00 **bidax7** (bidax5q + instr2 + code2): step 200 **+43.1 / -0.6 (+1.1 vs curve)**, step 300 **+45.3 / -0.8 (+1.3)**,
  final +43.5 / -2.5 (-0.7). Step 300: svamp +49 / -1.3, mbpp +39 / +0.8, ifeval +51 / -2.9, one function +38 / -1.0,
  several +42 / +0.6, no fit +46 / +1.5. Steps 200-300 clear the 1-point bar (like bidax5q s0: +1.7 / +1.0 at 200 / 300) but at
  ~3 points less saving, and the final checkpoint drops (mbpp +30 / -4.4, svamp -3.5). His code / instruction questions do not
  improve on bidax5q; one seed, so within seed noise. GPU 2 moved on to grC (GRPO, no leash).

### 3 Nov: "Chirurgical" edits (built, tested on CPU, queued after the DPO / GRPO screens)
User: replicate "Sparks of Weight-Space Programming" Exp 8 ourselves (no Astra, no API) on the raw Qwen, then v1, then a
subtler habit than looping. Technique, from the run notes (logs github.com/RikiyaT/lm_understand-logs):
**3 Nov ~12:30, Chirurgical revised (user): generate from scratch, like the paper, but ask the questions we already know
loop.** mine now fixes dev 256 / holdout 512 first, then the ORIGINAL model answers 1,500 fresh practice questions (75% of
them questions where a stored answer was capped with >= 30% repeated 8-grams, 25% random for the normal answers), graded;
loops and pairs come from those fresh answers. Inserted on GPU 1 between grB and D1 (chs0 smoke, then ch1). Tests:
helpers, probe / candidates / mech on a tiny Qwen3, and mine's selection with a stand-in generator all pass.
- ~11:25 **grB step 50** (GRPO, 4x length penalty: λ ramping to 0.2 over 100 steps, ~0.1 at step 50): **-2.0% / -0.5**
  (thinks 2% LONGER; below the yardstick's range): svamp -2 / -1.0, mbpp 0 / +2.9, ifeval -8 / -2.5, one function +1 / -1.7,
  several +4 / -0.6, no fit 0 / -1.5. Like grA at step 50 (+1.5 / -1.0): 50 GRPO steps at 12 questions x 8 answers move
  nothing yet, even at 4x the penalty. Steps 100 / 150 next. GPU 2: grC at step 50, scoring.
- ~13:40 **grC step 50** (GRPO, no leash: KL 0): **-0.3% / +0.4** (svamp +2 / -0.2, mbpp +3 / +3.5, ifeval -8 / -1.3, one
  function -1 / +0.6, several +4 / -2.1, no fit -1 / -0.2). Like grA / grB at step 50: unchanged from the raw Qwen. Both
  runtimes resumed after the ~12:40 reconnect (grA from step 50 on GPU 1, grC from its last block on GPU 2).
- ~15:30 **grA step 100: -2.3% / -1.2** (svamp -1 / -2.5, mbpp -4 / +2.5, ifeval -3 / -2.5, one function -4 / -0.8, several 0 / -3.6,
  no fit 0 / -2.3); **grC step 100: -3.0% / -1.0** (svamp +1 / +0.4, mbpp -5 / -1.2, ifeval -4 / -1.7, one function -3 / -1.9,
  several -1 / -4.2, no fit -7 / +1.2). At the full penalty (λ 0.05 reached at step 100) both think slightly LONGER and lose
  ~1 point. Reading: the khudgins penalty is tiny next to the right / wrong reward (at most 0.05 x min(1, thinking / 2048):
  ~0.01 for a typical 400-token answer, against 1.0 for being right), so 12 questions x 8 answers a step give almost no
  length signal; the reward is nearly pure "be right" and GRPO drifts. grB (4x penalty) and grE (lr x2.5) test exactly
  this lever; step 150 of grA / grC next.
- ~15:45 (user: don't run useless GRPO): queued GRPO changed. grD (gate) -> **grS** = standard GRPO scaling (no Dr. GRPO),
  penalty 0.05 (rescaling alone); grE (lr x2.5) -> **grP** = standard scaling + penalty 0.5 reached at step 50. grB (0.2,
  running), grF (SmartThinker, standard scaling) unchanged; grA / grC finish their 150 steps as the confirmation.
- ~17:00 **grA step 150 (final): -5.8% / -0.9** (svamp -5 / -1.0, mbpp -1 / +1.0, ifeval -13 / -3.3, one function -3 / 0.0,
  several -5 / -1.3, no fit -4 / +1.0); **grC step 150 (final): -6.7% / -0.3** (svamp -5 / -1.5, mbpp -14 / +1.5, ifeval -7 / -1.7,
  one function -2 / 0.0, several +2 / 0.0, no fit -5 / +0.8). Confirmed: with Dr. GRPO and the 0.05 penalty, GRPO from the
  raw Qwen makes thinking LONGER over training (step 50 / 100 / 150: grA +1.5 / -2.3 / -5.8%, grC -0.3 / -3.0 / -6.7%) at
  about -0.5 to -1 point: the "be right" reward dominates and longer thinking is what it rewards. The leash (grA KL 0.04 vs
  grC 0) changes little. grS / grP (standard scaling; 10x penalty) test the fix. GPU 1: grB resumed; GPU 2: D9 Ada-R1 DPO.
- ~18:00 **grB step 100** (4x penalty, full at step 100): **+1.2% / 0.0** (svamp +2 / +0.2, mbpp +1 / +2.3, ifeval +3 / -1.5, one
  function +1 / +0.4, several -2 / -3.4, no fit -3 / +0.2). The 4x penalty stops the drift to longer thinking seen in grA / grC
  (-2.3 / -3.0% at step 100) but does not shorten either. D9 lpd-adar1 (GPU 2): 30,049 pairs (17,870 questions; 2,844 never
  solved, 11,152 always solved), ~89 pairs a step over 339 steps, training.
- ~18:15 (user): **GRPO stopped.** No GRPO run shortened thinking (grA / grC finals 6-7% longer, grB +1.2% at step 100).
  grB finishes its 150 steps; grS, grP, gs0 and grF (SmartThinker) are taken out of the queues (kept as comments). The
  GPUs now run the DPO pair-strategy screen only.
- ~19:30 **grB step 150 (final): -0.6% / -0.7** (svamp 0 / -1.9, mbpp -4 / +0.6, ifeval +1 / -1.0, one function -2 / -0.6,
  several +2 / -0.8, no fit -1 / -0.4). GRPO screen closed: no run shortened thinking. **Chirurgical smoke chs0 failed** in its
  gen phase: save_candidate built the edited copy on the GPU and kept ~3.7 GB, so vLLM found 18.28 < 18.73 GB free. Before
  that, every phase worked (200 fresh answers: 102 capped, 63 right; 10 loops -> 5 pairs; probe; mech; the edited head
  L21H8 survives bfloat16 100%). Fixed (copies built on CPU, memory freed). ch1 started with the old script (its mine phase:
  617 loop-prone questions + 2,383 others); a resume line with the fixed script follows it in the queue.
