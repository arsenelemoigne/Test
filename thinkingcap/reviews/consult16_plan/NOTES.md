# Consult 16 Executable Plan — Persistent Working Notes

## 1. Objective & Deliverables
Turn the earlier Consult 15 report (`reviews/consult16_plan/pack/YOUR_REPORT.md`) into a FULL, EXECUTABLE PLAN (up to ~9,000 words, no code block in final delivery) for all three recipes, starting from the ORIGINAL `Qwen/Qwen3-0.6B`, achieving on the full test (2,759 questions x 8 = 22,072 rollouts):
- **>50% tokens saved** (4-benchmark geometric-mean token reduction across SVAMP, MBPP+, IFEval, BFCL)
- **Accuracy change no worse than -1.0 point** (4-benchmark equal-weight mean accuracy change vs. original 67.09%)
- Each recipe has 1 to 3 learning steps (one method and one job per step, fixed order, no zigzag)
- Strictly GENERAL "think less" rules (per-question measured quantities, never benchmark/dataset names)

Required 5 Sections of Deliverable:
1. For EACH of the 3 recipes and EACH step: (a) Data (prompt sources & counts per question type, how drawn, disjointness from test & Dev-512 / Dev-SVAMP-200 via 8-gram near-duplicate rule, generating model, sampling settings, samples/prompt, token caps); (b) Pair rules or reward (exact thresholds, expected pair counts per rule and per type, fallback when a rule runs short); (c) Training (method, LoRA rank/alpha/dropout/target modules, precision, lr, schedule/warmup, beta or KL/clipping/group size, batch size, max seq len, steps & epochs, seeds, reference model); (d) Checkpoints & evaluation (Dev-512, Dev-SVAMP-200, full test) with numerical decision gates and failure actions; (e) Wall-clock & GPU-hours on 2x Colab L4s.
2. WHY for every hyperparameter and design choice: pack evidence (run name + numbers) or paper (arXiv id + finding), counterfactuals (e.g., lr 2e-5 vs 5e-5, beta 0.1 vs 0.2, 120 vs 240 steps, gate thresholds), and explicit `[EXTRAPOLATION]` tags where extrapolating beyond direct runs.
3. Fix inconsistencies in earlier report (`YOUR_REPORT.md`) and list every change explicitly.
4. Implementation blueprint: what exists in `code/` (`prepare_types.py`, `repair_pairs.py`, `dpo.py`, `merge.py`, `dev_score.py`, `transfer_score.py`, `rl_train.py`, `track.py`) and can be reused as-is, what needs a new CLI option, what must be written, build & test order, and cheapest pre-flight checks.
5. Run schedule for the 2x Colab L4 GPUs (which GPU runs what, in what order, where files/results land) and the updated side-by-side comparison table of the three recipes.

## 2. Files Read So Far
- `reviews/consult16_plan/pack/YOUR_REPORT.md` (complete, 5,841 words)
- `reviews/consult16_plan/pack/YOUR_NOTES.md` (complete, 2,419 words)
- `reviews/consult16_plan/pack/code/dpo.py` (lines 1-170 read; reading remainder next)

## 3. Key Quantitative Anchors from Pack (Verified in Consult 15)
- `original` (Qwen3-0.6B): 0.0% saved / 67.09% 4-type equal-weight mean accuracy
  - SVAMP (1,000q x 8): 88.7% acc, 686.1 mean tok (676.5 think / 9.6 resp), >2000 tok = 7.25% (+227.0 tok)
  - MBPP+ (378q x 8): 52.4% acc, 2125.3 mean tok (2078.7 think / 46.7 resp), >4000 tok = 16.90% (+1172.5 tok, 7.80% cap hits at 8192)
  - IFEval (541q x 8): 58.7% acc, 696.0 mean tok (432.8 think / 263.1 resp), resp >1500 tok = 2.91% (+99.1 tok)
  - BFCL Simple (400q x 8): 82.2% acc, 194.9 mean tok (170.6 think / 24.2 resp)
  - BFCL Multiple (200q x 8): 76.1% acc, 215.9 mean tok (192.2 think / 23.6 resp)
  - BFCL No-Fit (240q x 8): 83.4% acc, 212.8 mean tok (185.8 think / 27.0 resp)
- `rpM-60`: **54.1% saved / -0.93 acc** (SVAMP 57.0% / -1.20, MBPP+ 58.2% / -0.86, IFEval 50.5% / -2.31, Simple 45.1% / -0.34, Multiple 46.9% / -1.56, No-Fit 60.7% / +4.17).
- `rpM-20`: **50.1% saved / -0.94 acc**
- `rpK-40`: **51.2% saved / -0.80 acc**
- `rpL-40`: 55.1% / -1.37; `rpL-60`: 59.3% / -1.70; `rpL-80`: 62.8% / -2.06
- `rpE-40 / 60 / 80`: 50.6% / -1.90; 47.8% / -1.20; 47.0% / -0.90 (from `rc1-soup-30` 57.4% / -3.50)
- `p1r1s0-200`: 43.4% / -1.50 (SVAMP 59.4% / -3.24, MBPP+ 44.0% / -3.31, IFEval 22.5% / +0.42 [think 193.7, resp 345.3], Simple 39.9% / -0.25, Multiple 42.1% / +1.25, No-Fit 44.7% / +0.42)
- `p1cr1s0`: Dev-512 34.6% / +2.03 (step 100), 32.4% / +2.51 (step 150), 34.4% / +2.90 (step 200)
- `bidax2`: 51.2% / -3.00
- `rpN` (hold repair on `rpL-60`): Dev-512 58.9% -> 59.9% saved (no recovery once over-shortened past ~59%)

## 4. Candidate Recipes Being Specified into Full Executable Plans
1. **Recipe 1 (2 Steps, On-Policy DPO):**
   - Step 1: Confidence-Gated Thinking Compression DPO from `original` -> ~54-57% saved
   - Step 2: Length-Matched Verification & Loop Calibration DPO on Step 1 rollouts -> 52.5-54.8% saved / -0.5 to -0.9 acc
2. **Recipe 2 (3 Steps, On-Policy DPO):**
   - Step 1: Round-1 Compression DPO from `original` -> ~47% saved
   - Step 2: Round-2 On-Policy Compression DPO on Step 1 rollouts -> ~57% saved
   - Step 3: Pure Verification & Constraint Repair DPO on Step 2 rollouts -> 53.0-55.5% saved / -0.6 to -0.95 acc
3. **Recipe 3 (2 Steps, On-Policy RL + DPO Hybrid):**
   - Step 1: Confidence-Gated ShorterBetter Dr. GRPO from `original` -> ~55% saved
   - Step 2: Length-Matched Verification DPO on Step 1 RL rollouts -> 52.0-54.5% saved / -0.6 to -0.95 acc

## 5. Open Questions to Verify in Code & Tables Next
1. Finish reading `code/dpo.py`, `code/merge.py`, `code/dev_score.py`, `code/transfer_score.py`, `code/queued_lines.txt`, `code/prepare_types.py`, `code/repair_pairs.py`, `code/rl_train.py`, `code/track.py`.
2. Verify exact prompt pool counts in `prepare_types.py` and `DATASET.md` (how many math, code, IFEval, tool prompts actually exist in the codebase, how the 8-gram near-duplicate filter against Test, Dev-512, and Dev-SVAMP-200 works).
3. Audit `YOUR_REPORT.md` for any internal inconsistencies (e.g., prompt pool counts vs existing pool in `prepare_types.py`, step counts vs checkpoint lists, `dpo.py` flags like `--max-len`, `--nll`, `--bida-precision`, `--save-every`, adapter continuation vs merge, Dev-512 vs Dev-SVAMP-200 usage).

## 6. Code Inspection Findings (Turn 9)
- `dpo.py`:
  - Takes `--model`, `--pairs`, `--init-adapter` (both starting LoRA AND reference model, scored once before training and cached in `ref.json`), `--out`, `--mix` (`0.4,0.4,0.2` or `all`), `--beta` (default 0.1), `--nll` (default 1.0, RPO chosen token NLL anchor), `--lr` (default 2e-5), `--lr-schedule` (`constant` or `cosine`), `--warmup` (default 10), `--lr-min` (default 2e-6), `--batch` (default 16), `--target-steps` (0), `--epochs` (1), `--max-steps` (0), `--save-every` (25), `--save-at` (list of step numbers), `--held-out` (48 pairs for tripwire), `--tripwire` (0.2), `--max-len` (9000), `--dtype` (`bfloat16`), `--bida-precision` (float32 base + float16 autocast with GradScaler, LoRA trainable params cast to float32), `--bida-schedule` (warmup 5% of steps, cosine down to 0), `--bida-exact` (`max_len=8704, held_out=0, lr_schedule=cosine, lr_min=0.0` + bida_precision + bida_schedule), `--keep-order`, `--seed`.
  - IMPORTANT INCONSISTENCY FOUND vs `YOUR_REPORT.md`: `YOUR_REPORT.md` described `dpo.py` using `--fp16` and `max_length=2048` and merging into base weights between steps. In actual `dpo.py`:
    1. Precision flag is `--bida-precision` (or `--bida-exact`), which loads base model in `torch.float32`, casts LoRA trainable weights to `float32`, and runs forward in `torch.float16` autocast with `torch.amp.GradScaler("cuda")`.
    2. Max sequence length flag is `--max-len 8704` (under `--bida-exact`) or `--max-len 9000` (default), NOT `2048`! (Because code rollouts go up to 8,192 tokens!). Setting `max_len=2048` would drop long rejected rollouts (>2048 tokens), defeating anti-loop pair training on MBPP+/SVAMP!
    3. Reference model handling: `dpo.py` takes `--init-adapter` as a PEFT LoRA adapter directory, scores `ref_c` and `ref_r` once with that starting adapter before training, and continues training that adapter in-place (or if starting from a merged base model `--model models/step1_merged`, needs an initial empty/zero LoRA adapter or `--init-adapter adapters/init`). Let's check `queued_lines.txt` and `track.py` to see how `p1r1s0`, `rpE`, `rpK`, `rpL`, `rpM` were actually launched!
- `merge.py`:
  - Sums adapters in `fp32` (`W = W0 + sum_i w_i * dW_i`) via `merge_and_unload()` and saves in `--dtype` (default `bfloat16` or `float16`).
- `dev_score.py`:
  - Scores candidate rollouts vs base rollouts on `Dev-512` (or with `--label Dev-SVAMP-200` on SVAMP-200) and supports `--vs` for paired bootstrap CI against a reference model's rollouts (`1,000` bootstrap draws).

## 7. Code Inspection Findings (Turn 14)
- `repair_pairs.py`:
  - Supports `--rollouts`, `--out`, `--quota`, `--ratio`, `--block` (80), `--seed`, `--think-only` (comma-separated parts), `--count-rules`, `--prefer-reread`, `--shorten-ratio`, `--shorten-min-right`, `--mix-preset` (`round1`, `round1b`, `round1c`, `typed`, `typed2`, `typed3`, `hold`), `--mix-shares`, `--resp-guard`, `--count-mode` (`all`, `atleast`, `length`), `--count-share`, `--count-resp-cap`, `--count-resp-min`, `--count-resp-pad`, `--no-shorten-counting`, `--resp-near-long`, `--loop-response-only`, `--easy-math-prefix`, `--guard-wide`, `--right-hold`, `--resp-median`, `--resp-max`, `--right-resp-cap`, `--kinds` (`guard,right`, `guard`, `guard,shorten`, `mix`).
  - Has a built-in stratified interleaving credit scheme (`queues` per part and per kind) so that when trained with `dpo.py --keep-order --mix all`, every prefix of training steps sees the exact part x kind proportions!
  - Writes `shortfall.json` and `track.py:ptrepair` implements `fill until full` (up to 2 extra batches of questions generated and added if any part falls short by >10% of its quota!).
  - CRITICAL OBSERVATION ON GENERALITY: In `repair_pairs.py`, presets like `typed3` and flags like `--think-only ifeval`, `--guard-wide svamp`, `--right-hold svamp`, and `length_rule(r)` (which reads `r["instruction_id_list"]`) branch on `part` (dataset/benchmark type) or IFEval metadata! To satisfy the strict requirement "a GENERAL 'think less' (per-question measured rules, never benchmark names)", we must add a `--general-mode` (`general_step1`, `general_step2`, `general_r2_step2`, `general_repair`) to `repair_pairs.py` that:
    1. Measures thinking length `thinking_len(r)` and response length `response_len(r)` on EVERY question regardless of `part` (i.e. `--think-only` applies universally!).
    2. Detects minimum-length constraints via prompt text regex (`PROMPT_MIN_LEN_RE`) on `r["prompt_ids"]` decoded text (never `instruction_id_list`).
    3. Detects tool-abstention prompts via prompt tool schema presence + majority correct rollouts emitting no tool call (never `part == "bfcl_irrelevance"`).
    4. Gates shortening vs length-matched `be_right` and widened premature-cut guards strictly by per-question clean solve count `c = len(rights)` out of `K` rollouts and per-question `tmed` (median right thinking tokens), NEVER `if base_part == "svamp"`.
- `track.py`:
  - `own_repair_questions(want, seed, exclude, public, avoid_grams)`:
    - Drops any question sharing an 8-word sequence (`eval_grams`, where frequency < 3 across `questions/questions.jsonl`) with the full test (`questions/questions.jsonl`).
    - Drops any question where `>= 30%` of its 8-grams are in `avoid_grams` (`dev_gram_set()`, rare 8-grams with count < 3 in `data/dev512.jsonl`, and we should include `data/dev-svamp200.jsonl` rare 8-grams too!).
    - Excludes exact question IDs in `dev_ids() | dev_svamp_ids() | excluded_ids(args)`.
  - `p1src`:
    - Uses stored 10-sample rollouts of `Qwen/Qwen3-0.6B` on the full pool (`lupin-all-dp13-nofit1-tools1-code1-nofit2-nofit4-math1-math2.jsonl`), filters out `Dev-512` IDs and 8-gram near-duplicates (`>=30%` rare 8-gram overlap), and reserves 25% of questions (33% IFEval, 35% BFCL multiple) via MD5 hash of `qid` for Round 2 (`data/p1-round1-ids.json`).
  - `ptrepair`:
    - Initializes a fresh zero LoRA (`sft.py --init-only --model <merged> --out adapters/fresh-<name>`), runs `dpo.py` with `--mix all --keep-order --nll 0.0 --beta 0.1 --batch 16 --lr <rp_lr> --lr-schedule cosine --warmup <rp_warmup> --lr-min <rp_lr_min> --max-steps <steps> --epochs <rp_epochs> --save-every 5 --held-out 0 --max-len 8704 --bida-precision --dtype bfloat16`.
    - Then screens `--rp-screen` checkpoints on `Dev-512` (`--rp-dev`) and `Dev-SVAMP-200` (`--rp-dev-svamp`) with paired bootstrap against `--rp-dev-vs`.

## 8. Additional Findings from Docs, Logs & Tables (Turn 18)
- `DATASET.md` & `PATH1_FINAL.md`:
  - The full training question pool is `data/pool-dp13.jsonl` (11,999 questions: Math 2,999, Code 3,001, IFEval 2,998, BFCL Simple 1,426, BFCL Multiple 716, BFCL Irrelevance/No-Fit 859) plus top-ups (`nofit1` 189, `tools1` 724, `code1` 800, `nofit2` 552, `nofit4` 1,871, `math1` 2,000, `math2` 2,000) = **17,870 total questions**.
  - On all 17,870 questions, the original `Qwen/Qwen3-0.6B` already has **10 stored rollouts per question** in `runs/bon/lupin-all-dp13-nofit1-tools1-code1-nofit2-nofit4-math1-math2.jsonl` (`P1_ANSWERS`).
  - `p1src` filters `P1_ANSWERS` by:
    1. Dropping all `Dev-512` IDs (`data/dev512.jsonl`, 512 questions: 140 math, 100 code, 132 IFEval [66 counting, 66 plain], 60 simple, 35 multiple, 45 no-fit) and `Dev-SVAMP-200` IDs (`data/dev-svamp200.jsonl`, 200 SVAMP-like math questions).
    2. Dropping any question with `>= 30%` rare 8-gram overlap with `Dev-512` (`dev_gram_set()`, where rare means appearing in `< 3` dev questions; and we extend `dev_gram_set()` to also include rare 8-grams of `Dev-SVAMP-200`!). Note: `pool-dp13` was already filtered by `build_pool.py` to drop any question sharing an 8-word sequence (`eval_grams` with count `< 3`) with the 2,759 test questions (`questions/questions.jsonl`).
    3. Splitting into Round 1 (`p1src`: 75% of math/code/simple/nofit, 67% of IFEval, 65% of multiple -> **13,082 questions**, 10 stored answers each, zero GPU generation cost!) and Round 2 unseen reserve (`P1_RESERVE`: 25% default, 33% IFEval, 35% BFCL multiple -> **~4,200 unseen questions** reserved for Step 2 / Round 2!).
- Why `Dev-SVAMP-200` exists and how `Dev-512` calibrates to Full Test (`NEW_RESULTS_12nov_morning.md` & `EXPERIMENT_LOG.md`):
  - `Dev-512` saving is within `~0.8-1.0` pt of Full Test for multi-stage models (`rpM-20` 49.3% Dev vs 50.1% Full; `rpM-40` 52.6% Dev vs 52.9% Full; `rpM-60` 54.9% Dev vs 54.1% Full; `rpL-80` 62.1% Dev vs 62.8% Full), while for 1-round models straight from `original`, `Dev-512` over-reads saving by `~2.5` pts (`p1r1s0-200` 45.9% Dev vs 43.4% Full).
  - `Dev-512` accuracy is `+1.4 to +3.2` pts more generous than Full Test (`rpM-20` +0.49 Dev vs -0.94 Full; `rpM-40` +0.91 Dev vs -1.10 Full; `rpM-60` +2.24 Dev vs -0.93 Full). Crucially, **paired differences on Dev-512 (`--vs`)** track Full Test differences much more tightly (`rpM-40` vs `rpE-80`: +5.7% / -0.46 on Dev -> 52.9% / -1.10 Full; `rpM-60` vs `rpE-80`: +8.0% / +0.87 on Dev -> 54.1% / -0.93 Full).
  - `Dev-SVAMP-200` (200 SVAMP-like math questions) measures math token saving on SVAMP-difficulty problems accurately (`rpE-80` 49.4%, `rpM-40` 53.8%, `rpM-60` 51.4%, `p1r1s0-200` 54.7%), complementing `Dev-512`'s harder GSM8K math slice (140 questions).

## 9. RL & Full-Test Diagnostics Checked (Turn 20)
- `consult7_gemini_path2_RL.md` & `GRPO_gr2f_summary.txt`:
  - Diagnosed why earlier RL runs failed (`grA`/`grB`/`grC` drifted longer due to microscopic length penalty `0.00366` vs `1.0` correctness + `+0.1` wrong-answer format bonus, `lr=2e-6` too small for LoRA, AdamW state reset every 10 steps before `--init-state`, and `--dr-grpo` `scale=False` silencing all-correct groups by ~275x relative to mixed groups).
  - `gr2f` (40 steps x 32 questions x 8 rollouts from original, `lr=1e-5 -> 1e-6`, `KL=0.005`, `--reward gated --gr-form linear`, `--dr-grpo --fixed-denominator --max-tokens-code 8192`) reached only `+11.3% saved / +0.3 acc` because `--fixed-denominator` divided the loss by `256 x 8192 = 2,097,152` tokens (16-20x larger than actual active tokens) and `--dr-grpo` (`scale=False`) did not normalize group variance or batch variance!
  - Conversely, `rl_train.py` ALREADY has `--advantage batch` (DLER batch-wise advantage normalization: center per group, divide by batch std `statistics.pstdev`), `--dynamic-sampling` (DAPO/DLER dynamic filtering of all-wrong groups), `--objective drpo --drpo-solve-scale` (DRPO `2510.04474` + ALP `2506.05256`), and `--reward shorterbetter` (`2504.21370`).
  - IMPORTANT INCONSISTENCY IN `YOUR_REPORT.md` FOR RECIPE 3:
    - In `YOUR_REPORT.md`, Recipe 3 recommended "Unnormalized Dr. GRPO token loss (no `1/|y_i|` division, `2503.20783`)" with `B=8` prompts x `G=8` rollouts and `max_new_tokens=2048`.
    - Checking `rl_train.py` and `GRPO_gr2f_summary.txt` + `consult7`:
      1. If `--dr-grpo` is passed in `rl_train.py`, line 798 sets `group_advantages(rewards, scale=not args.dr_grpo)` which turns OFF variance scaling (`scale=False`) AND divides by `len(chunk) * args.max_tokens`! As `consult7` proved, `scale=False` silences all-right groups (where the length spread is 0.10-0.30 vs 1.0 in mixed groups) and `len(chunk) * max_tokens` dilutes the gradient by 8-16x! Instead, using `--advantage batch` (DLER batch-wise std normalization, `2510.15110`, already implemented at line 889 of `rl_train.py`) with actual active token normalization (or mean per-group token normalization) keeps all-right and mixed groups balanced without length bias!
      2. Setting `max_new_tokens=2048` on code questions truncates normal MBPP+/KodCode solutions (where base mean is 2,125 tokens and test cap is 8,192), grading them wrong (`grade_all` marks any reply not ending in `eos` as `False`). `rl_train.py` already has `--max-tokens 4096 --max-tokens-code 8192` to match evaluation caps!
      3. In `rl_train.py`, `--questions 16 --group 8 --updates 2` (16 informative prompts x 8 rollouts = 128 rollouts per sampling step, split into 2 optimizer updates of 8 groups = 64 rollouts each) or `--questions 32 --group 8 --updates 4` uses vLLM batched generation much more efficiently than `B=8` with 1 update!

## 10. Full Audit of Inconsistencies in Earlier Report (`YOUR_REPORT.md`) & Exact Fixes (Turn 25)
1. **Prompt Pool Counts & Source Disjointness vs Actual Repository (`track.py`, `prepare_types.py`, `DATASET.md`):**
   - *Earlier report*: Described a generic "3,800 public training prompts (1,000 GSM8K, 1,000 MBPP+EvalPlus, 1,000 IFEval, 800 tools: 320/240/240)" and "generate 3,800 x 8 = 30,400 rollouts from original in Step 1".
   - *Fix*: Ground every step in the exact repository files and `track.py` functions:
     - For Step 1 from `original` (Recipes 1 & 2): `runs/bon/lupin-all-dp13-nofit1-tools1-code1-nofit2-nofit4-math1-math2.jsonl` (`P1_ANSWERS`) ALREADY stores 10 rollouts per question from `Qwen/Qwen3-0.6B` across 17,870 pool questions (`data/pool-dp13.jsonl` 11,999 + top-ups 5,871). `track.py:p1src` filters out all `Dev-512` IDs (318), `Dev-SVAMP-200` IDs (200), and near-duplicates sharing `>=30%` rare 8-grams (`count < 3`) with `Dev-512` or `Dev-SVAMP-200` (~289), and holds back the MD5 hash reserve (`P1_RESERVE`: 25% default, 33% IFEval, 35% BFCL multiple) for Step 2/3, leaving **12,468 to 12,668 Step 1 questions (x 10 stored rollouts = ~125,000 answers, or first 8 rollouts per question if standardizing on K=8) at 0.0 GPU-hours of rollout generation!** (If run from scratch without stored rollouts, drawing 3,800 questions x 8 via `own_repair_questions(..., public=True)` takes ~1.6 GPU-hours).
     - For Step 2 (and Recipe 2 Step 3): Draw **3,600 unseen questions** from the reserved pool via `own_repair_questions(want={"svamp": 1000, "mbpp": 600, "ifeval": 960, "bfcl_simple_python": 360, "bfcl_multiple": 280, "bfcl_irrelevance": 400}, exclude=dev_ids() | dev_svamp_ids() | p1_round1_ids, public=True, avoid_grams=dev_gram_set() | dev_svamp_gram_set())` and sample `K = 8` rollouts per question (`28,800` rollouts) with `probe_curve.py --generate-only`.
2. **DPO CLI Flags, Sequence Length Cap, and Reference Model Mechanics (`code/dpo.py`):**
   - *Earlier report*: Cited `--fp16` (and "never `--bf16`"), `lora_dropout=0.05`, `max_length=2048`, and `beta=0.15 / 0.20` without noting how `dpo.py` and `sft.py` actually work.
   - *Fix*:
     - In `sft.py --init-only`, `LoraConfig` uses `r=16, lora_alpha=32, lora_dropout=0.0` (not `0.05`: `dropout=0.0` ensures deterministic reference log-probs when `dpo.py` scores `ref_c, ref_r` before training).
     - In `dpo.py`, the precision flag is `--bida-precision` (loads base model in `torch.float32`, casts LoRA trainable weights to `float32`, runs forward in `torch.float16` autocast with `GradScaler("cuda")`), NOT `--fp16`.
     - In `dpo.py`, the sequence length cap is `--max-len 8704` (NOT `2048`! Code rollouts reach 8,192 tokens + prompt up to 512 tokens = 8,704; setting `--max-len 2048` would discard all MBPP+ and SVAMP anti-loop pairs!).
     - In `track.py:ptrepair`, `dpo.py` hardcodes `--beta 0.1` unless we pass `--rp-beta` from `track.py` to `dpo.py --beta`. We specify adding `--rp-beta` to `track.py` and justify `beta=0.10` (Step 1) vs `beta=0.15` / `0.20` (Step 2/3).
3. **Step Counts vs Checkpoint Lists & Quota-to-Epoch Math:**
   - *Earlier report*: In Recipe 1 Step 1, stated "Train for 240 steps ... Save checkpoints at steps 160, 200, 240, 280" (step 280 cannot be saved if training stops at 240!). Similarly in Recipe 2 Step 3, stated "Train for 60 steps ... checkpoints at 30, 45, 60, 75" (step 75 exceeds 60 steps!). Also in Recipe 1 Step 2, stated "2,600 pairs ... 80 steps, batch 16" (80 steps x 16 = 1,280 pairs, which would only read 49% of a 2,600-pair file!).
   - *Fix*: Make every step's pair quota, batch size (`16`), epochs, `--max-steps`, and `--rp-screen` checkpoints 100% mathematically consistent:
     - **Recipe 1 Step 1**: `3,200` pairs (`svamp=800,mbpp=800,ifeval=800,bfcl_simple_python=280,bfcl_multiple=240,bfcl_irrelevance=280`), `batch=16` -> `1 epoch = 200 steps`, `2 epochs = 400 steps`. Set `--max-steps 240 --epochs 2 --save-every 20`, screen on Dev-512 + Dev-SVAMP-200 at **steps `160, 200, 240`**.
     - **Recipe 1 Step 2**: `1,280` pairs (`svamp=320,mbpp=240,ifeval=360,bfcl_simple_python=140,bfcl_multiple=100,bfcl_irrelevance=120`), `batch=16` -> `1 epoch = 80 steps` (`2 epochs = 160 steps`). Set `--max-steps 80 --epochs 2 --save-every 20`, screen at **steps `20, 40, 60, 80`** (1.0 full stratified pass at step 80!).
     - **Recipe 2 Step 1**: `2,880` pairs (`svamp=720,mbpp=720,ifeval=720,bfcl_simple_python=260,bfcl_multiple=220,bfcl_irrelevance=240`), `batch=16` -> `1 epoch = 180 steps`. Set `--max-steps 180 --epochs 1 --save-every 30`, screen at **steps `120, 150, 180`**.
     - **Recipe 2 Step 2**: `1,600` pairs (`svamp=400,mbpp=400,ifeval=400,bfcl_simple_python=150,bfcl_multiple=110,bfcl_irrelevance=140`), `batch=16` -> `1 epoch = 100 steps`. Set `--max-steps 100 --epochs 1 --save-every 20`, screen at **steps `60, 80, 100`**.
     - **Recipe 2 Step 3**: `960` pairs (`svamp=240,mbpp=200,ifeval=280,bfcl_simple_python=90,bfcl_multiple=70,bfcl_irrelevance=80`), `batch=16` -> `1 epoch = 60 steps` (`2 epochs = 120 steps`). Set `--max-steps 60 --epochs 2 --save-every 15`, screen at **steps `30, 45, 60`**.
     - **Recipe 3 Step 1 (RL)**: `120` RL steps x `16` informative questions/step (`24` drawn with `--dynamic-sampling 1.5`) x `G=8` rollouts = `1,920` informative groups (`15,360` rollouts, `2` optimizer updates/step = `240` gradient updates), screen at **steps `60, 90, 120`**.
     - **Recipe 3 Step 2 (DPO)**: `960` pairs, `batch=16`, `--max-steps 60 --epochs 1 --save-every 15`, screen at **steps `30, 45, 60`**.
4. **RL Normalization & Token Caps in Recipe 3 (`code/rl_train.py`):**
   - *Earlier report*: Recommended `--dr-grpo` with `max_new_tokens=2048`.
   - *Fix*: As proved in `consult7` and `GRPO_gr2f_summary.txt`, `--dr-grpo` in `rl_train.py` disables group std scaling (`scale=False`) AND divides by `len(chunk) * max_tokens` (`8192`), silencing all-right groups by 275x and shrinking gradients by 16x. Instead, use `--advantage batch` (DLER batch-wise std normalization, already in `rl_train.py`) with actual active token normalization (or mean group token normalization) and `--max-tokens 4096 --max-tokens-code 8192` so code rollouts are never falsely truncated at 2,048!

## 11. Exact Candidate Pool & Timing Verification from Raw GPU Logs (Turn 27)
- From `gpu2_night_key_lines.txt`:
  - `p1src` exact numbers on the GPU: `126,680 of 178,700 stored answers kept (12,668 questions); dropped {'kept for round 2': 45,950, 'Dev-512': 3,180, 'shares an 8-gram with Dev-512': 2,890}`.
  - Candidate pools in `p1src` (from `p1r1s0` and `p1cr1s0` summaries):
    - `svamp`: `3,878 + 172` shorten candidates, `255` narrow guard (`1,013` wide guard), `1,839` be-right, `539` anti-loop.
    - `mbpp`: `967 + 9` shorten candidates, `134` guard, `857` be-right, `195` anti-loop.
    - `ifeval` (1,873 questions, 1,111 counting): `351 + 12` shorten, `39` (`92` wide) guard, `220` (`436`) be-right, `322` (`141` true length) counting, `101` anti-loop.
    - `bfcl_simple_python`: `688 + 100` shorten, `16` guard, `244` be-right, `0` anti-loop.
    - `bfcl_multiple`: `276 + 23` shorten, `10` guard, `162` be-right, `0` anti-loop.
    - `bfcl_irrelevance`: `1,648 + 144` shorten, `26` guard, `291` be-right, `2` anti-loop.
  - Candidate pools on `rpE-80`'s 5,700 x 8 rollouts (`rpI` rollouts, from `rpK`/`rpL`/`rpM` summaries):
    - `svamp`: `1,565 + 52` shorten, `174` narrow guard (`598` wide guard), `1,004` (`739` length-held) be-right, `40` anti-loop.
    - `mbpp`: `330 + 6` shorten, `64` guard, `430` be-right, `14` anti-loop.
    - `ifeval` (1,500 questions, 865 counting): `284 + 21` (`128 + 13` non-counting) shorten, `17` (`20`) guard, `265` (`304`) be-right, `273` (`169` atleast / `121` length) counting, `94` (`90` resp-only) anti-loop.
    - `bfcl_simple_python`: `131 + 42` shorten, `1` guard, `83` be-right.
    - `bfcl_multiple`: `140 + 41` shorten, `3` guard, `99` be-right.
    - `bfcl_irrelevance`: `250 + 17` shorten, `1` guard, `59` be-right.

## 12. Wall-Clock & Throughput Calibration on Colab L4 (Turn 32)
- From `EXPERIMENT_LOG.md` & `gpu2_night_full.txt`:
  - vLLM output throughput on L4 (`bfloat16`, `gpu_memory_utilization=0.75-0.85`):
    - At base lengths (~740 mean tokens, with 4,096 / 8,192 caps): `~950-1,200` output tokens/sec (`~150` questions x 8 in ~8-10 min; Full Test 2,759 x 8 = 22,072 rollouts takes `~2.3-2.5 hours` for compressed models (~330 mean tok) and `~4.0-4.5 hours` for base-length models).
    - For compressed models (~300-350 mean tokens):
      - `Dev-512` (512 x 8 = 4,096 rollouts): `~18-22 minutes` (0.33 GPU-h) per checkpoint including merge (`~2 min`) + vLLM generation (`~15 min`) + grading (`~2 min`).
      - `Dev-SVAMP-200` (200 x 8 = 1,600 rollouts): `~5-6 minutes` (0.09 GPU-h) per checkpoint.
      - Unseen Step 2/3 pool rollout generation (`3,600 x 8 = 28,800` rollouts from a ~47-55% compressed model, ~350 tok/reply = ~10.1M tokens): `~1.8-2.0 hours` (1.9 GPU-h) on 1 L4, or `~0.95 hours` wall-clock if split across 2 L4s (`--half a` / `--half b`).
      - Full Test (`2,759 x 8 = 22,072` rollouts from a ~53-55% compressed model, ~330 tok/reply = ~7.3M tokens): `~2.1-2.4 hours` (2.25 GPU-h) on 1 L4.
  - DPO training throughput (`dpo.py --bida-precision --batch 16 --max-len 8704`):
    - Reference log-prob scoring (`ref.json`): `~4-6 minutes` for 1,000-3,200 pairs.
    - Training steps: `~5.5-6.5 seconds/step` for batch 16 (`60` steps = `~6 min`, `80` steps = `~8 min`, `100` steps = `~10 min`, `180` steps = `~18 min`, `240` steps = `~24 min`).
  - RL training throughput (`rl_train.py` with vLLM engine + HF training step):
    - `120` RL steps x `24` candidate prompts (`16` active after dynamic sampling) x `8` rollouts = `23,040` generated rollouts (~1.6 GPU-h) + `240` optimizer steps (~0.5 GPU-h) = `~2.1 GPU-h` total.

## 13. Final Deliverable Readiness Check (Turn 38)
- `reviews/consult16_plan/work/report_draft.md` is complete (all 5 required sections, ~8,850 words after final tightening, zero code blocks).
- Verified all 5 deliverables:
  1. For each of the 3 recipes and each step: (a) data sources, counts per question type, drawing method, disjointness from Full Test, Dev-512, and Dev-SVAMP-200 via exact ID + 8-gram near-duplicate rule, generating model, sampling settings, samples/prompt, token caps; (b) pair rules or reward with exact thresholds, expected pairs per rule and per type, and shortfall fallbacks; (c) training method, LoRA rank/alpha/dropout/target modules, precision (`--bida-precision --dtype bfloat16`), lr, schedule/warmup, beta or KL/clipping/group size, batch size, max seq len (`8704`), steps & epochs, seeds, reference model; (d) checkpoints & evaluation (`Dev-512`, `Dev-SVAMP-200`, `Full Test`) with numerical decision gates and failure actions; (e) wall-clock and GPU-hours on 2x Colab L4s.
  2. Evidence-based justification (`WHY`) for every hyperparameter and design choice with exact run names + scores and arXiv IDs + findings, counterfactuals (`lr 2e-5 vs 5e-5`, `beta 0.10 vs 0.15/0.20`, `120 vs 240 steps`, gate thresholds), and `[EXTRAPOLATION]` tags.
  3. All 6 inconsistencies in `YOUR_REPORT.md` fixed and explicitly listed in Section 3.
  4. Implementation blueprint (`prepare_types.py`, `repair_pairs.py`, `dpo.py`, `merge.py`, `dev_score.py`, `rl_train.py`, `track.py`: what is reused as-is, what gets a new option, build/test order, and cheapest pre-flight checks).
  5. Two-GPU run schedule (`GPU 1` vs `GPU 2`, phases, Drive paths) and updated side-by-side comparison table of the 3 recipes.
