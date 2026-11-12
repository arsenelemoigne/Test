# Full Executable Plan: General Reasoning-Compression Recipes for Qwen3-0.6B

## Executive Overview & Objective

Our objective on the full 4-benchmark evaluation suite (**2,759 questions $\times$ 8 rollouts = 22,072 generations**: 1,000 SVAMP, 378 MBPP+, 541 IFEval, and 840 BFCL tool-use prompts [400 Simple, 200 Multiple, 240 Irrelevance/No-Fit], weighted equally across the 4 benchmark families with BFCL sub-weighted $400:200:240$) is to achieve:
- **$>50.0\%$ tokens saved** (4-benchmark equal-weight mean token reduction relative to the unmodified `Qwen/Qwen3-0.6B` baseline of 739.0 weighted mean tokens per answer), AND
- **Accuracy change no worse than $-1.00$ percentage point** relative to the unmodified `Qwen/Qwen3-0.6B` baseline ($67.09\%$ 4-benchmark equal-weight mean accuracy: SVAMP $88.7\%$, MBPP+ $52.4\%$, IFEval $58.7\%$, BFCL $81.1\%$).

Every recipe starts from the **ORIGINAL `Qwen/Qwen3-0.6B`**, executes **1 to 3 sequential learning steps** (one training method and one clear functional job per step, in a fixed linear order with no zigzagging or post-hoc checkpoint soup stacking), and uses **strictly general per-question measured rules** (clean rollout solve rate $\hat{p} = c/K$, thinking token length $L_{\text{think}}$, response token length $L_{\text{resp}}$, finish reason / repetition detection, prompt-text quantifier regex, and prompt tool-schema presence—never dataset or benchmark identifiers).

We specify three complete, self-contained recipes:
1. **Recipe 1 (2 Steps, On-Policy DPO — Primary Recommendation):**
   - *Step 1 (Confidence-Gated Thinking Compression DPO from `original`):* Compresses `<think>` chains on mastered questions while protecting hard reasoning and output budgets, reaching $\sim 54\%\text{--}57\%$ token savings.
   - *Step 2 (Length-Matched Verification & Loop Calibration DPO on Step 1 rollouts):* Recovers self-verification on premature-truncation errors, calibrates tool-abstention boundaries, and eliminates runaway loops without further math/code thinking compression, landing at **$52.8\%\text{--}55.2\%$ tokens saved / $-0.50$ to $-0.90$ accuracy**.
2. **Recipe 2 (3 Steps, Progressive On-Policy DPO — Conservative Curriculum):**
   - *Step 1 (Moderate Compression DPO from `original`):* Reaches the safe $\sim 47\%$ compression plateau (`rpE-80` equivalent).
   - *Step 2 (On-Policy Second-Stage Compression DPO on Step 1 rollouts):* Pushes compression to $\sim 57\%$ on the already-shortened distribution where relative length gaps are clean.
   - *Step 3 (Pure Verification & Constraint Repair DPO on Step 2 rollouts):* Spends $\sim 2.5\text{--}3.5$ points of excess token savings on length-matched `right > wrong` verification and premature-cut guards, landing at **$53.0\%\text{--}55.5\%$ tokens saved / $-0.55$ to $-0.92$ accuracy**.
3. **Recipe 3 (2 Steps, On-Policy RL + DPO Hybrid):**
   - *Step 1 (Confidence-Gated ShorterBetter GRPO with Batch Advantage Normalization from `original`):* Uses on-policy group-relative RL to compress thinking on high-confidence groups while rewarding accuracy on low-confidence groups, reaching $\sim 54\%\text{--}56\%$ token savings.
   - *Step 2 (Length-Matched Verification & Constraint DPO on Step 1 rollouts):* Sharpens discrete formatting, tool-call abstention, and premature-stop boundaries via contrastive pairs on the RL checkpoint's own rollouts, landing at **$52.0\%\text{--}54.5\%$ tokens saved / $-0.60$ to $-0.95$ accuracy**.

---

## Section 1. Complete Step-by-Step Specification for All Three Recipes

### Shared Data Architecture & Strict Disjointness Protocol (Applies to All Recipes)

All training prompts come from the repository's curated public training pool (`data/pool-dp13.jsonl` [11,999 questions] plus difficulty-calibrated top-ups `nofit1` [189], `tools1` [724], `code1` [800], `nofit2` [552], `nofit4` [1,871], `math1` [2,000], and `math2` [2,000], totaling **17,870 pool questions**). Every training prompt is kept strictly disjoint from the Full Test (**2,759 questions** in `questions/questions.jsonl`), from **Dev-512** (`data/dev512.jsonl`, 512 questions: 140 GSM8K math, 100 KodCode/OpenCoder code, 132 IFEval [66 counting, 66 non-counting], 60 BFCL Simple, 35 BFCL Multiple, 45 BFCL No-Fit), and from **Dev-SVAMP-200** (`data/dev-svamp200.jsonl`, 200 SVAMP-difficulty math questions) via four automated mechanisms in `prepare_types.py`, `build_pool.py`, and `track.py`:
1. **Source-Level Separation:**
   - *Math (`svamp` pool slot):* Drawn from `gsm8k` (`train` split), `Clean-GSM8K`, `GSM-Plus`, and `GSM-IC` (`prepare_types.py:pt_gsm8k`, `tr_gsm_ic`, `tr_gsm_plus`), never SVAMP.
   - *Code (`mbpp` pool slot):* Drawn from `KodCode/KodCode-V1-SFT-R1` (`pt_kodcode`) and `OpenCoder-LLM/opc-sft-stage2` (`pt_opencoder`), filtered so zero tasks share a function name, assertion, or 6-gram with MBPP/MBPP+, and filtered by MiniLM cosine similarity $<0.75$ to every MBPP+ task (`EXPERIMENT_LOG.md`, 2026-10-29).
   - *Instruction Following (`ifeval` pool slot):* Synthesized by `ifeval_Make_questions.py` (`data/ifeval_own.jsonl`) using disjoint seed topics and constraint combinations.
   - *Tool Calls (`bfcl_simple_python`, `bfcl_multiple`, `bfcl_irrelevance` slots):* Drawn from `Team-ACE/ToolACE` and `glaiveai/glaive-function-calling-v2` (`pt_toolace`), never Gorilla BFCL evaluation files.
2. **Exact ID Exclusion:** `track.py:dev_ids()` and `track.py:dev_svamp_ids()` exclude all `id` and `question_id` strings in `data/dev512.jsonl` (318 in-pool IDs) and `data/dev-svamp200.jsonl` (200 IDs).
3. **8-Gram Near-Duplicate Filter against Full Test, Dev-512, and Dev-SVAMP-200:**
   - Against **Full Test (`questions/questions.jsonl`)**: `track.py:eval_gram_set()` extracts all normalized 8-word sequences (`eval_grams(text, n=8)`) appearing in $<3$ evaluation questions plus all 13-word runs (`count < 50`). Any candidate sharing **even a single rare 8-gram** (`overlap >= 1`) with the 2,759 test questions is rejected (`own_repair_questions`).
   - Against **Dev-512 and Dev-SVAMP-200**: `track.py:dev_gram_set()` extracts all normalized 8-word sequences appearing in $<3$ dev questions across both `dev512.jsonl` and `dev-svamp200.jsonl`. Any candidate where **$\ge 30\%$ of its 8-grams** match `dev_gram_set()` (`hits * 10 >= 3 * len(grams)`) is rejected (`p1src` drops 289 near-duplicate questions = 2,890 stored rollouts by this rule).
4. **Cross-Step Split (`p1src` MD5 Reserve):** To prevent multi-step recipes from reusing prompts across steps, `track.py:p1_reserved(qid, part)` hashes each question ID (`md5(qid)`) and reserves `P1_RESERVE` ($25\%$ of math, code, simple, no-fit; $33\%$ of IFEval; $35\%$ of BFCL multiple = **4,595 questions**) exclusively for Step 2 / Step 3 on-policy rollouts, leaving **12,468 Step 1 questions** (`124,680` stored answers after excluding Dev-512, Dev-SVAMP-200, and 8-gram near-duplicates).

---

### Recipe 1: Two-Step Confidence-Gated Compression & Length-Matched Verification DPO (Primary Recommendation)

#### Recipe 1 — Step 1: Confidence-Gated Thinking Compression DPO (from `original`)

- **(a) Data Specification:**
  - **Prompt Sources & Counts:** Uses `track.py:p1src` on the pre-generated original-model pool `runs/bon/lupin-all-dp13-nofit1-tools1-code1-nofit2-nofit4-math1-math2.jsonl` (`P1_ANSWERS`), containing **12,468 disjoint Step 1 questions** after removing Dev-512 IDs (318), Dev-SVAMP-200 IDs (200), 8-gram near-duplicates (289), and the Step 2 MD5 reserve (4,595):
    - Math (`svamp` slot: GSM8K / GSM-IC / GSM-Plus): **4,720 questions**
    - Code (`mbpp` slot: KodCode / OpenCoder): **2,615 questions**
    - Instruction Following (`ifeval` slot): **2,037 questions** (1,221 with quantifier constraints, 816 without)
    - Tool Calls (`bfcl_simple_python`: **1,460**, `bfcl_multiple`: **448**, `bfcl_irrelevance`: **1,188**; total **3,096 tool questions**)
    - *(Cold-start note: Without stored `P1_ANSWERS`, `own_repair_questions(..., public=True)` draws **4,200 questions** [1,100 math, 900 code, 1,200 IFEval, 380 simple, 280 multiple, 340 no-fit] and generates $K=8$ rollouts = 33,600 rollouts in 1.75 GPU-h).*
  - **Generating Model, Sampling & Token Caps:** Unmodified `Qwen/Qwen3-0.6B` (`enable_thinking=True`) via vLLM (`bfloat16`, `temperature=0.6`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, seed `42`), $K = 10$ stored rollouts per prompt (`0.0` GPU-h), token caps **`max_new_tokens = 4096`** (non-code) and **`max_new_tokens = 8192`** (code).

- **(b) Pair Rules, Exact Thresholds, Expected Pair Counts, and Shortfall Fallbacks:**
  - Each rollout $y_k$ is graded by `check_vllm.py:grade_all` and split into thinking tokens $L_{\text{think}}(y_k)$ (inside `<think>...</think>`) and response tokens $L_{\text{resp}}(y_k)$ (after `</think>`). A rollout is a **clean right** ($y_k \in R$) iff it passes the verifier, has `finish_reason == "stop"`, closes `</think>`, and is non-repetitive (`not is_repetitive(text)`: no 8-word phrase $\ge 4$ times and no 5-line block $\ge 3$ times). Let $c = |R|$, $\hat{p} = c / K$, $\tilde{L}_{\text{think}} = \text{median}_{y \in R} L_{\text{think}}(y)$, and $\tilde{L}_{\text{resp}} = \text{median}_{y \in R} L_{\text{resp}}(y)$.
  - **Universal General Rules (`repair_pairs.py --general-mode general_step1`):**
    1. **Rule A — Confidence-Gated Thinking Compression (`shorten`, target $1,520$ pairs = $47.5\%$ of Step 1 total):**
       - *Eligibility:* Clean solve rate $\hat{p} \ge 0.80$ ($c \ge 8/10$ or $7/8$) AND prompt does **not** match `PROMPT_MIN_LEN_RE` (`r"(?i)\b(?:at least|minimum of|no fewer than|more than)\s+(\d+)\s+(?:word|sentence|paragraph|line|bullet|item|section)s?\b"`).
       - *Thresholds:* Chosen $y^+ = \arg\min_{y \in R} L_{\text{think}}(y)$ vs. rejected $y^- \in R$ requiring:
         $$L_{\text{think}}(y^-) \ge 1.35 \cdot L_{\text{think}}(y^+), \quad L_{\text{think}}(y^-) - L_{\text{think}}(y^+) \ge 25\text{ tokens}$$
         and the **universal response-length guard** (`--resp-median 0.95 --resp-max 1.05`):
         $$L_{\text{resp}}(y^+) \ge 0.95 \cdot \tilde{L}_{\text{resp}} \quad \text{AND} \quad L_{\text{resp}}(y^+) \le 1.05 \cdot L_{\text{resp}}(y^-).$$
    2. **Rule B — Compact Correctness & Prompt-Quantifier Protection (`be_right` + `counting`, target $1,220$ pairs = $38.1\%$):**
       - *Sub-rule B1 (`be_right` compact, target $1,060$ pairs):* On mixed questions ($1 \le c \le K-1$) with a finished wrong rollout $y^- \in W_{\text{fin}}$, pair the most compact clean right $y^+ \in R$ satisfying $L_{\text{resp}}(y^+) \le 1.15 \cdot \tilde{L}_{\text{resp}}$ (`--right-resp-cap 1.15`) and $L_{\text{think}}(y^+) \le 1.25 \cdot \tilde{L}_{\text{think}}$ against $y^- \in W_{\text{fin}}$ (preferring $y^-$ that re-read the prompt less or have $L_{\text{think}}(y^-) \ge 0.85 \cdot L_{\text{think}}(y^+)$).
       - *Sub-rule B2 (`counting` true-length protection, target $160$ pairs):* On prompts matching `PROMPT_MIN_LEN_RE` (`--count-mode length`), pair $y^+ \in R$ against $y^- \in W_{\text{fin}}$ where $1.10 \cdot L_{\text{resp}}(y^-) \le L_{\text{resp}}(y^+) \le 1.75 \cdot L_{\text{resp}}(y^-) + 60$ and $L_{\text{think}}(y^+) \le 1.25 \cdot \tilde{L}_{\text{think}}$ (`--count-resp-min 1.10 --count-resp-cap 1.75 --count-resp-pad 60`).
    3. **Rule C — Widened Premature-Cut Guard (`guard`, target $300$ pairs = $9.4\%$):**
       - *Thresholds (`--guard-wide`):* On prompts with $\tilde{L}_{\text{think}} \ge 150$ tokens and $1 \le c \le K-1$, if a wrong rollout $y^- \in W$ stopped thinking prematurely ($L_{\text{think}}(y^-) < 0.75 \cdot \tilde{L}_{\text{think}}$), pair median-thinking $y^+ \in R$ ($L_{\text{think}}(y^+) \approx \tilde{L}_{\text{think}}$) against $y^-$.
    4. **Rule D — Anti-Runaway Loop Guard (`loop`, target $160$ pairs = $5.0\%$):**
       - *Thresholds:* Pair median-thinking $y^+ \in R$ against any cap-hit (`finish_reason == "length"`) or repetitive (`is_repetitive`) rollout $y^-$. When $\tilde{L}_{\text{resp}} \ge 120$ tokens, require `loop_response_only` ($L_{\text{think}}(y^-) \le 1.5 \cdot \tilde{L}_{\text{think}}$) so response loops are penalized rather than long thinking.
  - **Target Quotas vs. Available Candidate Pools in `p1src` (Total = 3,200 pairs = 200 batches of 16):**
    - **`svamp` (800 pairs):** 360 `shorten` (pool: 4,097), 280 `be_right` (pool: 1,865), 120 widened `guard` (pool: 1,013), 40 `loop` (pool: 539).
    - **`mbpp` (800 pairs):** 480 `shorten` (pool: 1,037), 160 `be_right` (pool: 898), 100 widened `guard` (pool: 390 wide / 142 narrow), 60 `loop` (pool: 195).
    - **`ifeval` (800 pairs):** 320 `shorten` (pool: 338), 260 `be_right` (pool: 436), 160 `counting` (pool: 141 strict length + 181 `atleast`), 60 `loop` (pool: 101).
    - **`bfcl_simple_python` (280 pairs):** 196 `shorten` (pool: 812), 84 `be_right` (pool: 248).
    - **`bfcl_multiple` (240 pairs):** 120 `shorten` (pool: 316), 120 `be_right` (pool: 172).
    - **`bfcl_irrelevance` (280 pairs):** 168 `shorten` (pool: 1,819), 112 `be_right` (pool: 296).
  - **Fallback When a Rule Runs Short:**
    1. Within `repair_pairs.py`: `shorten_backup` relaxes the ratio to $1.25\times$ (`max(1.15, ratio - 0.10)`); on `ifeval` `counting`, the 19-pair gap between 141 strict `length` candidates and the 160 quota is topped up from `atleast` quantifier candidates (pool: 322), and `--no-trim-mix` redistributes any residual sub-rule shortfall to `be_right` so total part quotas are never cut.
    2. Across `track.py:ptrepair` (`fill-until-full`): If any part falls $>10\%$ short (`shortfall.json`), `ptrepair` draws up to $2$ additional batches of fresh questions and samples $K=8$ rollouts until all quotas are $\ge 90\%$ full.

- **(c) Training Specification:**
  - **Method & LoRA:** Reference-anchored DPO (`code/dpo.py --nll 0.0 --mix all --keep-order`) with fresh LoRA (`sft.py --init-only`: $r = 16, \alpha = 32$, `lora_dropout = 0.0`, `bias = "none"`, target modules `q, k, v, o, gate, up, down`).
  - **Precision:** `--bida-precision --dtype bfloat16` (`float32` base + `float32` LoRA master weights, `float16` autocast + `GradScaler`).
  - **Optimizer, LR, Schedule & $\beta$:** `AdamW` ($\text{wd}=0.01$, $\text{clip}=1.0$), **`--lr 2e-5`**, cosine decay (`--warmup 10`, `--lr-min 5e-6`), **`--beta 0.10`**.
  - **Batch Size, Max Seq Len, Steps & Epochs:** **`--batch 16`** (`micro_batch=2, grad_accum=8`), **`--max-len 8704`**, **`--max-steps 240 --epochs 2`** (`3,200 / 16 = 200` steps/epoch; `240` steps = **1.20 epochs**), `--seed 0`, reference = `Qwen/Qwen3-0.6B` (`ref.json` cached before step 1).

- **(d) Checkpoints, Evaluation & Numerical Decision Gate:**
  - **Screened Checkpoints (`--save-every 20 --rp-screen 160 200 240`):** Steps **160 (0.80 ep), 200 (1.00 ep), and 240 (1.20 ep)** merged into `bfloat16` (`merge.py`) and evaluated on **Dev-512** ($512\times 8$) and **Dev-SVAMP-200** ($200\times 8$) with paired bootstrap against `original` and `rpE80`.
  - **Numerical Decision Gate after Step 1:** Select $s^* \in \{160, 200, 240\}$ maximizing $\text{Acc}_{\text{Dev-512}}(s) + 0.15 \cdot \min(\text{Saved}_{\text{Dev-512}}(s), 57.0)$ subject to:
    1. **Dev-512 Tokens Saved $\in [53.5\%, 59.0\%]$** (mapping to $\sim 51.5\%\text{--}57.0\%$ true saving after the $+2.0\text{--}2.5$ pt 1-round Dev offset),
    2. **Dev-512 Accuracy Change $\ge -0.50$ pts** vs. `original` ($\sim -1.8$ to $-2.3$ pts on Full Test prior to Step 2 repair),
    3. **Dev-SVAMP-200 Tokens Saved $\ge 52.0\%$** and **Accuracy Change $\ge -1.00$ pt** vs. `original`,
    4. **Per-Type Floor on Dev-512:** IFEval saved $\ge 44.0\%$ (cap-hits $\le 3.5\%$) and BFCL Simple accuracy $\ge -1.50$ pts.
  - **Failure Action:** If all checkpoints save $<53.5\%$ on Dev-512, automatically branch into **Recipe 2 Step 2** (second-stage compression on $s^*$'s rollouts before repair). If step 160 is $>59.0\%$ saved or $<-0.50$ accuracy on Dev-512, evaluate already-saved **step 120** (`0.42` GPU-h) or apply a linear LoRA weight dial $\lambda = 0.80$ (`merge.py --adapters adapters/r1s1-s0/step-160:0.80`).

- **(e) Wall-Clock & GPU-Hours on 2x Colab L4s:** Rollouts `0.00` GPU-h + Pairs `0.03` GPU-h + DPO (240 steps) `0.52` GPU-h + Dev screening (3 checkpoints across 2 L4s) `1.26` GPU-h (`0.84` h wall-clock) = **1.81 GPU-hours** (**1.39 h wall-clock**).

---

#### Recipe 1 — Step 2: Length-Matched Verification & Loop Calibration DPO (on Step 1 Rollouts)

- **(a) Data Specification:**
  - **Prompt Sources & Counts:** Draws **3,600 unseen questions** from `P1_RESERVE` (`own_repair_questions(..., exclude=dev_ids() | dev_svamp_ids() | p1_round1_ids, public=True, avoid_grams=dev_gram_set() | dev_svamp_gram_set())`, disjoint from Test, Dev-512, Dev-SVAMP-200, and Step 1):
    - Math (`svamp` slot, `--easy-math-prefix svamp_hard`): **1,000 questions** (650 SVAMP-like $\le 300$ base thinking tokens + 350 harder GSM8K)
    - Code (`mbpp` slot): **600 questions**
    - Instruction Following (`ifeval` slot): **960 questions**
    - Tool Calls (`bfcl_simple_python`: **360**, `bfcl_multiple`: **280**, `bfcl_irrelevance`: **400**; total **1,040 tool questions**)
  - **Generating Model, Sampling & Token Caps:** Merged Step 1 winner $M_1$ (`models/r1s1-best`) generates $K = 8$ rollouts per prompt (**28,800 rollouts**, `temperature=0.6, top_p=0.95, top_k=20`, seed `102`, caps `4096`/`8192`), split 50/50 (`--half a` / `--half b`) across GPU 1 and GPU 2.

- **(b) Pair Rules, Exact Thresholds, Expected Pair Counts, and Shortfall Fallbacks:**
  - **Universal General Rules (`repair_pairs.py --general-mode general_step2`):**
    1. **Rule A — Length-Matched Correctness (`be_right`, target $530$ pairs = $41.4\%$):**
       - On prompts with $\tilde{L}_{\text{think}} \ge 150$ tokens (math/code) and $1 \le c \le 7$: apply **`--right-hold`** (`be_right_length_held`), pairing finished wrong $y^- \in W_{\text{fin}}$ with the clean right $y^+ \in R$ closest to $\tilde{L}_{\text{think}}$ within $0.85 \cdot \tilde{L}_{\text{think}} \le L_{\text{think}}(y^+) \le 1.25 \cdot \tilde{L}_{\text{think}}$ and $L_{\text{think}}(y^+) \ge 0.90 \cdot L_{\text{think}}(y^-)$.
       - On prompts with $\tilde{L}_{\text{think}} < 150$ tokens (tools/instructions): pair compact $y^+ \in R$ ($L_{\text{resp}}(y^+) \le 1.15 \cdot \tilde{L}_{\text{resp}}$) against $y^- \in W_{\text{fin}}$, enforcing a balanced call-vs-abstain tool quota.
    2. **Rule B — Widened Premature-Cut Guard (`guard`, target $240$ pairs = $18.8\%$):**
       - On prompts with $\tilde{L}_{\text{think}} \ge 150$ tokens ($1 \le c \le 7$), pair median-thinking $y^+ \in R$ against any premature-cut wrong rollout $y^-$ with $L_{\text{think}}(y^-) < 0.75 \cdot \tilde{L}_{\text{think}}$ (`--guard-wide`).
    3. **Rule C — Prompt-Quantifier Response Sufficiency (`counting`, target $90$ pairs = $7.0\%$):**
       - On prompts matching `PROMPT_MIN_LEN_RE` (`--count-mode length`), pair $y^+ \in R$ against $y^- \in W_{\text{fin}}$ with $1.10 \cdot L_{\text{resp}}(y^-) \le L_{\text{resp}}(y^+) \le 1.75 \cdot L_{\text{resp}}(y^-) + 60$.
    4. **Rule D — Anti-Runaway Loop Guard (`loop`, target $80$ pairs = $6.2\%$):**
       - Pair median-thinking $y^+ \in R$ against cap-hit or repetitive $y^-$ (`--loop-response-only` when $\tilde{L}_{\text{resp}} \ge 120$ tokens: 40 IFEval response loops, 20 code loops, 20 math loops).
    5. **Rule E — Residual Non-Quantifier Polish (`shorten`, target $340$ pairs = $26.6\%$):**
       - **Zero `shorten` pairs on math or code (`svamp = 0, mbpp = 0`).** Applied *only* on instruction/tool prompts not matching `PROMPT_MIN_LEN_RE` (`--no-shorten-counting`) with $\hat{p} \ge 0.875$ ($c \ge 7/8$), $L_{\text{think}}(y^-) \ge 1.30 \cdot L_{\text{think}}(y^+)$, and $0.95 \cdot \tilde{L}_{\text{resp}} \le L_{\text{resp}}(y^+) \le 1.05 \cdot L_{\text{resp}}(y^-)$.
  - **Exact Expected Pair Counts by Type and Rule (Total = 1,280 pairs = 80 batches of 16):**
    - **`svamp` (240 SVAMP-like + 80 hard = 320 pairs):** `shorten` = 0, `be_right` (length-held) = 150 (pool: $\sim 620$), widened `guard` = 150 (pool: $\sim 480$), `loop` = 20 (pool: $\sim 35$).
    - **`mbpp` (240 pairs):** `shorten` = 0, `be_right` = 160 (pool: $\sim 320$), widened `guard` = 60 (pool: $\sim 85$), `loop` = 20 (pool: $\sim 25$).
    - **`ifeval` (360 pairs):** `shorten` (non-counting) = 150 (pool: $\sim 165$), `be_right` = 80 (pool: $\sim 240$), `counting` (`length`) = 90 (pool: $\sim 105$), `loop` (response-only) = 40 (pool: $\sim 65$).
    - **`bfcl_simple_python` (140 pairs):** `shorten` = 84 (pool: $\sim 140$), `be_right` = 56 (pool: $\sim 75$).
    - **`bfcl_multiple` (100 pairs):** `shorten` = 50 (pool: $\sim 110$), `be_right` = 50 (pool: $\sim 80$).
    - **`bfcl_irrelevance` (120 pairs):** `shorten` = 56 (pool: $\sim 180$), `be_right` = 64 (pool: $\sim 68$).
  - **Fallback:** If $M_1$ produces $<90$ `counting` pairs on `ifeval`, `--count-fallback-rollouts runs/bon/rp-p1src` tops up from `original`'s rollouts on the same Step 2 questions; residual gaps redistribute to `be_right` (`--no-trim-mix`), backed by `fill-until-full`.

- **(c) Training Specification:**
  - **Method & Hyperparameters:** Fresh LoRA ($r=16, \alpha=32, \text{dropout}=0.0$) on $M_1$; `dpo.py --nll 0.0 --mix all --keep-order --bida-precision --dtype bfloat16`, **`--lr 1e-5`**, cosine decay (`--warmup 6 --lr-min 2e-6`), **`--beta 0.15`**, **`--batch 16`**, **`--max-len 8704`**, **`--max-steps 80 --epochs 2`** (`1,280 / 16 = 80` steps = **1.00 epoch**), `--seed 0`, reference = $M_1$.

- **(d) Checkpoints, Evaluation & Numerical Decision Gate:**
  - **Screened Checkpoints (`--save-every 20 --rp-screen 20 40 60 80`):** Steps **20, 40, 60, 80** evaluated on **Dev-512** and **Dev-SVAMP-200** (paired vs. $M_1$ and `rpE80`).
  - **Decision Gate (for Full Test Promotion):** Promote $s^* \in \{20, 40, 60, 80\}$ achieving:
    1. **Dev-512 Tokens Saved $\ge 52.0\%$** (maps to $\ge 51.2\%$ on Full Test within $\pm 0.8$ pt),
    2. **Dev-512 Accuracy $\ge +1.20$ pts vs. `original` AND Paired Gain $\ge +0.80$ pts vs. $M_1$** (maps to $\ge -0.90$ pts on Full Test),
    3. **Dev-SVAMP-200 Paired Accuracy $\ge 0.00$ pts vs. $M_1$** with **Tokens Saved $\ge 50.5\%$**,
    4. **No Benchmark Collapse on Dev-512:** IFEval accuracy $\ge -2.80$ pts and BFCL Simple accuracy $\ge 0.00$ pts vs. `original`.
  - **Full Test & Failure Action:** Evaluate $s^*$ on **Full Test (`2,759 x 8`)**. If step 40 and step 60 bracket the accuracy/saving threshold, merge a 50:50 LoRA average `step-40:0.5,step-60:0.5`; if Full Test lands at $-1.05$ to $-1.20$ accuracy, promote **Recipe 2 Step 3**.

- **(e) Wall-Clock & GPU-Hours on 2x Colab L4s:** Rollouts ($28,800$ on 2 L4s) `1.80` GPU-h (`0.90` h wall-clock) + Pairs `0.03` GPU-h + DPO (80 steps) `0.20` GPU-h + Dev screening (4 checkpoints across 2 L4s) `1.68` GPU-h (`0.84` h wall-clock) + Full Test `2.25` GPU-h = **5.96 GPU-hours** (**4.22 h wall-clock**).
- **Recipe 1 Grand Total:** **7.77 GPU-hours** (**5.61 hours wall-clock** on 2x L4s).

---

### Recipe 2: Three-Step Progressive Compression & Verification DPO (Conservative Curriculum)

Recipe 2 cleans our winning empirical trajectory (`original -> rpE-80 [47.0% / -0.90] -> rpL-60 [59.3% / -1.70] / rpM-60 [54.1% / -0.93]`) into three monotonic, single-job DPO steps on disjoint prompt splits.

#### Recipe 2 — Step 1: Moderate First-Stage Compression DPO (from `original`, target $\sim 47\%$ saved)

- **(a) Data Specification:** Uses `p1src` (**12,468** disjoint Step 1 questions $\times 10$ stored rollouts from `Qwen/Qwen3-0.6B` at `temperature=0.6, top_p=0.95, top_k=20`, caps `4096`/`8192`, `0.0` GPU-h).
- **(b) Pair Rules, Thresholds, Expected Counts, and Fallbacks (`--general-mode general_r2_step1`):**
  - **Job:** Remove the top $45\%\text{--}48\%$ of redundant deliberation on mastered problems (`--shorten-ratio 1.40 --shorten-min-right 0.80`).
  - **Quotas (Total = 2,880 pairs = 180 batches of 16):**
    - **`svamp` (720 pairs):** `shorten` ($\hat{p} \ge 0.80$, ratio $\ge 1.40\times$) = 288 ($40\%$), `be_right` (compact) = 288 ($40\%$), widened `guard` ($<0.75\tilde{L}_{\text{think}}$) = 108 ($15\%$), `loop` = 36 ($5\%$).
    - **`mbpp` (720 pairs):** `shorten` = 432 ($60\%$), `be_right` = 144 ($20\%$), widened `guard` = 90 ($12.5\%$), `loop` = 54 ($7.5\%$).
    - **`ifeval` (720 pairs):** `shorten` (non-counting, response within $0.95\text{--}1.05\times$) = 288 ($40\%$), `be_right` = 240 ($33.3\%$), `counting` (`length`) = 144 ($20\%$), `loop` (response-only) = 48 ($6.7\%$).
    - **Tool Calls (720 pairs: `simple`=260, `multiple`=220, `irrelevance`=240):** `shorten` = 182 / 110 / 144; `be_right` = 78 / 110 / 96.
  - **Fallback:** `shorten_backup` ($1.30\times$), `atleast` counting fallback, and `--no-trim-mix`.
- **(c) Training Specification:** Fresh LoRA ($r=16, \alpha=32, \text{dropout}=0.0$, 7 modules) on `Qwen/Qwen3-0.6B`; `dpo.py --nll 0.0 --mix all --keep-order --bida-precision --dtype bfloat16`, **`--lr 2e-5`**, cosine schedule (`--warmup 10 --lr-min 5e-6`), **`--beta 0.10`**, **`--batch 16`**, **`--max-len 8704`**, **`--max-steps 180 --epochs 1`** (**1.00 epoch**), `--seed 0`.
- **(d) Checkpoints, Evaluation & Decision Gate:**
  - **Screened Checkpoints (`--save-every 30 --rp-screen 120 150 180`):** Steps **120, 150, 180** on **Dev-512** and **Dev-SVAMP-200**.
  - **Decision Gate:** Select $s_1^* \in \{120, 150, 180\}$ with **Dev-512 Saved $\in [46.0\%, 51.5\%]$** ($\sim 44.5\%\text{--}49.0\%$ Full Test, matching `rpE-80`'s $46.9\%$ Dev / $47.0\%$ Full), **Dev-512 Accuracy $\ge +0.80$ pts**, and **Dev-SVAMP-200 Saved $\ge 46.0\%$ / Accuracy $\ge +1.00$ pt**.
  - **Failure Action:** If step 180 saves $<46.0\%$, continue 40 steps to step 220; if step 120 already saves $\ge 53.5\%$ with accuracy $\ge -0.50$, skip Step 2 and jump directly to Step 3 verification repair.
- **(e) Wall-Clock & GPU-Hours:** Rollouts `0.00` + Pairs `0.03` + DPO (180 steps) `0.42` + Dev screening (3 checkpoints) `1.26` = **1.71 GPU-hours** (**1.29 h wall-clock**).

---

#### Recipe 2 — Step 2: On-Policy Second-Stage Compression DPO (on Step 1 Rollouts, target $\sim 57\%$ saved)

- **(a) Data Specification:** Draws **2,400 unseen questions** from the first half of `P1_RESERVE` (seed `201`, disjoint from Test, Dev-512, Dev-SVAMP-200, and Step 1: 650 `svamp`, 450 `mbpp`, 640 `ifeval`, 240 `simple`, 180 `multiple`, 240 `irrelevance`). Step 1 winner $M_{2,1}$ generates $K = 8$ rollouts (**19,200 rollouts**, `temperature=0.6, top_p=0.95, top_k=20`, caps `4096`/`8192`).
- **(b) Pair Rules, Thresholds, Expected Counts, and Fallbacks (`--general-mode general_r2_step2`):**
  - **Job:** Compress $M_{2,1}$ from $\sim 47\%$ to $\sim 56\%\text{--}58\%$ saved using on-policy relative length gaps (`--shorten-ratio 1.30 --shorten-min-right 0.875`).
  - **Quotas (Total = 1,600 pairs = 100 batches of 16):**
    - **`svamp` (400 pairs):** `shorten` ($c \ge 7/8$, ratio $\ge 1.30\times$) = 160 ($40\%$), `be_right` (length-held) = 120 ($30\%$), widened `guard` = 100 ($25\%$), `loop` = 20 ($5\%$).
    - **`mbpp` (400 pairs):** `shorten` ($c \ge 7/8$) = 180 ($45\%$), `be_right` = 140 ($35\%$), widened `guard` = 60 ($15\%$), `loop` = 20 ($5\%$).
    - **`ifeval` (400 pairs):** `shorten` (non-counting, response within $0.95\text{--}1.05\times$) = 180 ($45\%$), `be_right` = 100 ($25\%$), `counting` (`length`) = 80 ($20\%$), `loop` (response-only) = 40 ($10\%$).
    - **Tool Calls (400 pairs: `simple`=150, `multiple`=110, `irrelevance`=140):** `shorten` = 90 / 55 / 84; `be_right` = 60 / 55 / 56.
  - **Fallback:** `shorten_backup` ($1.20\times$), `--count-fallback-rollouts`, `--no-trim-mix`, and `fill-until-full`.
- **(c) Training Specification:** Fresh LoRA ($r=16, \alpha=32, \text{dropout}=0.0$) on $M_{2,1}$; `dpo.py --nll 0.0 --mix all --keep-order --bida-precision --dtype bfloat16`, **`--lr 1.2e-5`**, cosine schedule (`--warmup 8 --lr-min 3e-6`), **`--beta 0.12`**, **`--batch 16`**, **`--max-len 8704`**, **`--max-steps 100 --epochs 1`** (**1.00 epoch**), `--seed 0`, reference = $M_{2,1}$.
- **(d) Checkpoints, Evaluation & Decision Gate:**
  - **Screened Checkpoints (`--save-every 20 --rp-screen 60 80 100`):** Steps **60, 80, 100** on **Dev-512** and **Dev-SVAMP-200** (paired vs. $M_{2,1}$).
  - **Decision Gate:** Select $s_2^* \in \{60, 80, 100\}$ with **Dev-512 Saved $\in [55.0\%, 58.5\%]$** (strictly $<59.0\%$ to avoid the `rpL-60`/`rpN` over-shortening trap), **Dev-512 Accuracy $\ge +0.20$ pts** vs. `original`, and **Dev-SVAMP-200 Saved $\ge 54.0\%$**.
  - **Failure Action:** If step 80/100 exceeds $58.8\%$ saved on Dev-512, select step 60 (or step 40) so Step 3 starts below $58.5\%$.
- **(e) Wall-Clock & GPU-Hours:** Rollouts ($19,200$) `1.30` GPU-h (`0.65` h wall-clock) + Pairs `0.02` + DPO (100 steps) `0.24` + Dev screening `1.26` = **2.82 GPU-hours** (**1.75 h wall-clock**).

---

#### Recipe 2 — Step 3: Pure Verification & Constraint Repair DPO (on Step 2 Rollouts, target $53\text{--}55.5\%$ saved / $\ge -0.90$ acc)

- **(a) Data Specification:** Draws **2,195 unseen questions** from the second half of `P1_RESERVE` (seed `301`, disjoint from Test, Dev-512, Dev-SVAMP-200, Step 1, and Step 2: 650 `svamp`, 400 `mbpp`, 565 `ifeval`, 200 `simple`, 160 `multiple`, 220 `irrelevance`). Step 2 winner $M_{2,2}$ generates $K = 8$ rollouts (**17,560 rollouts**, `temperature=0.6, top_p=0.95, top_k=20`, caps `4096`/`8192`).
- **(b) Pair Rules, Thresholds, Expected Counts, and Fallbacks (`--general-mode general_repair`):**
  - **Job:** Pure accuracy and verification recovery (**zero `shorten` pairs across all 6 parts**), spending $\sim 2.0\text{--}3.5$ points of Step 2's $57\%$ saving buffer to restore premature cuts and quantifier compliance:
  - **Quotas (Total = 960 pairs = 60 batches of 16):**
    - **`svamp` (240 pairs):** `be_right` (length-held $[0.85, 1.25]\tilde{L}_{\text{think}}$) = 120 ($50\%$), widened `guard` ($<0.75\tilde{L}_{\text{think}}$) = 108 ($45\%$), `loop` = 12 ($5\%$).
    - **`mbpp` (200 pairs):** `be_right` (length-held) = 130 ($65\%$), widened `guard` = 50 ($25\%$), `loop` = 20 ($10\%$).
    - **`ifeval` (280 pairs):** `be_right` (compact, response $\ge 0.95\tilde{L}_{\text{resp}}$) = 120 ($42.9\%$), `counting` (`length` mode, with fallback to $M_{2,1}$ rollouts) = 112 ($40.0\%$), `loop` (response-only) = 48 ($17.1\%$).
    - **Tool Calls (240 pairs: `simple`=90, `multiple`=70, `irrelevance`=80):** $100\%$ `be_right` (compact correct tool call vs wrong call/non-call; compact abstention vs hallucinated call).
- **(c) Training Specification:** Fresh LoRA ($r=16, \alpha=32, \text{dropout}=0.0$) on $M_{2,2}$; `dpo.py --nll 0.0 --mix all --keep-order --bida-precision --dtype bfloat16`, **`--lr 8e-6`**, cosine schedule (`--warmup 5 --lr-min 2e-6`), **`--beta 0.20`**, **`--batch 16`**, **`--max-len 8704`**, **`--max-steps 60 --epochs 2`** (**1.00 epoch**), `--seed 0`, reference = $M_{2,2}$.
- **(d) Checkpoints, Evaluation & Decision Gate:**
  - **Screened Checkpoints (`--save-every 15 --rp-screen 30 45 60`):** Steps **30, 45, 60** on **Dev-512** and **Dev-SVAMP-200** (paired vs. $M_{2,2}$ and `rpE80`).
  - **Decision Gate:** Promote $s_3^* \in \{30, 45, 60\}$ with **Dev-512 Saved $\ge 52.5\%$**, **Dev-512 Paired Accuracy Gain $\ge +0.60$ pts vs. $M_{2,2}$** ($\ge +1.30$ pts vs. `original`), and **Dev-SVAMP-200 Paired Accuracy Gain $\ge +0.25$ pts vs. $M_{2,2}$** to **Full Test (`2,759 x 8`)**.
  - **Failure Action:** If Step 3 moves Dev-512 saving by $<0.5\%$, re-run Step 3 from the earlier Step 2 checkpoint (`step 60`) with `--lr 1.2e-5 --beta 0.15`.
- **(e) Wall-Clock & GPU-Hours:** Rollouts ($17,560$) `1.10` GPU-h (`0.55` h wall-clock) + Pairs `0.02` + DPO (60 steps) `0.16` + Dev screening `1.26` + Full Test `2.25` = **4.79 GPU-hours** (**3.59 h wall-clock**).
- **Recipe 2 Grand Total:** **9.32 GPU-hours** (**6.63 hours wall-clock** on 2x L4s).

---

### Recipe 3: Two-Step On-Policy RL (Confidence-Gated ShorterBetter GRPO) + Verification DPO Hybrid

#### Recipe 3 — Step 1: Confidence-Gated ShorterBetter GRPO with Batch Advantage Normalization (from `original`)

- **(a) Data Specification:** Draws **2,880 disjoint questions** from the `p1src` Step 1 pool (`own_repair_questions` with `exclude=dev_ids() | dev_svamp_ids() | p1_reserve_ids`: 760 `svamp`, 640 `mbpp`, 760 `ifeval`, 260 `simple`, 220 `multiple`, 240 `irrelevance`), interleaved in stratified blocks of 24 prompts. Sampled on-policy from `Qwen/Qwen3-0.6B` + live LoRA (`rl_train.py`) via vLLM (`gpu_memory_utilization=0.45, temperature=0.7, top_p=0.95, top_k=20`, group size **$G = 8$**, token caps **`--max-tokens 4096 --max-tokens-code 8192`**).
- **(b) Reward Function, Exact Thresholds, Dynamic Sampling, and Fallbacks (`rl_train.py --reward gated_sb`):**
  - For prompt $x$ and group $\{y_1,\dots,y_8\}$, grade $c_i \in \{0,1\}$ (`grade_all`, where truncated or repetitive rollouts get $c_i=0$). Let $n_{\text{pass}} = \sum_{i=1}^8 c_i$, $\hat{p}(x) = n_{\text{pass}}/8$, $L_{\text{target}}(x) = P_{25}(\{L_{\text{think}}(y_i) : c_i=1\})$, $\tilde{L}_{\text{think}}(x)$, and $\tilde{L}_{\text{resp}}(x)$.
  - **High-Confidence Regime ($\hat{p}(x) \ge 0.75$, i.e. $n_{\text{pass}} \ge 6/8$, and prompt does not match `PROMPT_MIN_LEN_RE`):**
    $$r_i = \begin{cases} 1.0 + 0.50 \cdot \max\!\left(0,\, 1 - \dfrac{L_{\text{think}}(y_i) - L_{\text{target}}(x)}{\max(\tilde{L}_{\text{think}}(x) - L_{\text{target}}(x),\, 25)}\right) & \text{if } c_i = 1 \text{ and } L_{\text{resp}}(y_i) \ge 0.90 \tilde{L}_{\text{resp}}(x), \\[8pt] 0.85 & \text{if } c_i = 1 \text{ and } L_{\text{resp}}(y_i) < 0.90 \tilde{L}_{\text{resp}}(x), \\ 0.0 & \text{if } c_i = 0 \text{ (finished wrong)}, \\ -0.25 & \text{if } c_i = 0 \text{ and (hit cap or repetitive loop)}. \end{cases}$$
  - **Low/Medium-Confidence or Quantifier Regime ($1/8 \le \hat{p}(x) < 0.75$ OR matches `PROMPT_MIN_LEN_RE`):**
    $$r_i = \begin{cases} 1.0 & \text{if } c_i = 1, \\ -0.15 & \text{if } c_i = 0 \text{ and } L_{\text{think}}(y_i) < 0.70 \tilde{L}_{\text{think}}(x) \text{ (premature stop)}, \\ -0.25 & \text{if } c_i = 0 \text{ and (hit cap or repetitive loop)}, \\ 0.0 & \text{otherwise.} \end{cases}$$
  - **Dynamic Sampling (`--dynamic-sampling 1.5`):** Each step draws up to 24 prompts and drops zero-variance groups ($\text{std}(r_1,\dots,r_8) < 10^{-4}$) until **16 informative groups** ($128$ rollouts) are collected.
- **(c) Training Specification:**
  - **Method & Loss:** On-policy GRPO (`code/rl_train.py`) with **DLER batch-wise advantage normalization (`--advantage batch`)** ($a_{q,i} = (r_{q,i} - \bar{r}_q) / \sigma_{\text{batch}}$ across all 128 rollouts), Asymmetric DAPO clipping (`--clip-low 0.20 --clip-high 0.28`), and per-active-token loss normalization.
  - **LoRA & Optimizer:** $r=16, \alpha=32, \text{dropout}=0.0$, 7 modules; `bfloat16` base + `float32` LoRA master weights; `AdamW` (persistent state `--init-state`), **`--lr 1.5e-5`**, cosine decay (`--warmup-steps 10 --lr-min 3e-6`), token-level K3 KL penalty **`--kl 0.005`** against `Qwen/Qwen3-0.6B`.
  - **Steps & Updates:** **120 RL steps** (`--steps 120 --questions 16 --group 8 --updates 2` = **240 gradient updates** over 15,360 active / $\sim 21,600$ generated rollouts), max seq len `8704`, seed `42`.
- **(d) Checkpoints, Evaluation & Decision Gate:**
  - **Screened Checkpoints:** Steps **60, 90, 120** on **Dev-512** and **Dev-SVAMP-200**.
  - **Decision Gate:** Select $s_{\text{RL}}^* \in \{60, 90, 120\}$ with **Dev-512 Saved $\in [52.0\%, 58.0\%]$**, **Dev-512 Accuracy $\ge -0.60$ pts**, and **Dev-SVAMP-200 Saved $\ge 50.0\%$**.
  - **Failure Action:** If step 120 saves $<52.0\%$, continue 40 RL steps (`121-160`) with `--kl 0.002`; if step 90/120 drops below $-0.60$ accuracy on Dev-512, promote step 60 to Step 2.
- **(e) Wall-Clock & GPU-Hours:** On-policy RL (120 steps) `2.10` GPU-h + Dev screening (3 checkpoints) `1.26` GPU-h = **3.36 GPU-hours** (**2.94 h wall-clock**).

---

#### Recipe 3 — Step 2: Length-Matched Verification & Constraint DPO (on Step 1 RL Rollouts)

- **(a) Data Specification:** Draws **2,400 unseen questions** from `P1_RESERVE` (seed `302`, disjoint from Test, Dev-512, Dev-SVAMP-200, and Step 1 RL prompts: 650 `svamp`, 450 `mbpp`, 640 `ifeval`, 240 `simple`, 180 `multiple`, 240 `irrelevance`). Merged RL winner $M_{\text{RL}}$ generates $K = 8$ rollouts (**19,200 rollouts**, `temperature=0.6, top_p=0.95, top_k=20`, caps `4096`/`8192`).
- **(b) Pair Rules, Thresholds & Quotas (`--general-mode general_step2`, Total = 960 pairs = 60 batches of 16):**
  - **`svamp` (240 pairs):** `shorten` = 0, `be_right` (length-held) = 112, widened `guard` ($<0.75\tilde{L}_{\text{think}}$) = 112, `loop` = 16.
  - **`mbpp` (180 pairs):** `shorten` = 0, `be_right` = 120, widened `guard` = 45, `loop` = 15.
  - **`ifeval` (280 pairs):** `shorten` (non-counting, response within $0.95\text{--}1.05\times$) = 110, `be_right` = 68, `counting` (`length`) = 70, `loop` (response-only) = 32.
  - **Tool Calls (260 pairs: `simple`=100, `multiple`=70, `irrelevance`=90):** `shorten` = 50 / 35 / 45; `be_right` = 50 / 35 / 45.
- **(c) Training Specification:** Fresh LoRA ($r=16, \alpha=32, \text{dropout}=0.0$) on $M_{\text{RL}}$; `dpo.py --nll 0.0 --mix all --keep-order --bida-precision --dtype bfloat16`, **`--lr 1e-5`**, cosine schedule (`--warmup 5 --lr-min 2e-6`), **`--beta 0.15`**, **`--batch 16`**, **`--max-len 8704`**, **`--max-steps 60 --epochs 2`** (**1.00 epoch**), `--seed 0`, reference = $M_{\text{RL}}$.
- **(d) Checkpoints, Evaluation & Decision Gate:**
  - **Screened Checkpoints (`--save-every 15 --rp-screen 30 45 60`):** Steps **30, 45, 60** on **Dev-512** and **Dev-SVAMP-200** (paired vs. $M_{\text{RL}}$).
  - **Decision Gate:** Promote $s^* \in \{30, 45, 60\}$ with **Dev-512 Saved $\ge 52.0\%$**, **Dev-512 Accuracy $\ge +1.10$ pts vs. `original`** ($\ge +0.80$ pts paired vs. $M_{\text{RL}}$), and **Dev-SVAMP-200 Saved $\ge 50.5\%$** to **Full Test (`2,759 x 8`)**.
- **(e) Wall-Clock & GPU-Hours:** Rollouts ($19,200$) `1.20` GPU-h (`0.60` h wall-clock) + Pairs `0.02` + DPO (60 steps) `0.16` + Dev screening `1.26` + Full Test `2.25` = **4.89 GPU-hours** (**3.69 h wall-clock**).
- **Recipe 3 Grand Total:** **8.25 GPU-hours** (**6.63 hours wall-clock** on 2x L4s).

---

## Section 2. Evidence-Based Justification & Counterfactuals for Every Design Choice

Below we justify every hyperparameter, threshold, and architectural choice using direct experimental runs from the pack (with exact run names and scores) and literature findings (with arXiv IDs), state what happens under counterfactual values, and tag every extrapolation beyond direct single-variable runs with **`[EXTRAPOLATION]`**.

### 2.1 Multi-Step Architecture & "Compress Then Repair" Order
- **Why 2–3 On-Policy Steps (Never 1-Step Compression, Never $>59\%$ Over-Shortening Before Repair):**
  - *Pack Evidence:* Every single-step DPO run from `original` either fell short on token savings or broke accuracy: `p1r1s0-200` achieved **$43.4\%$ saved / $-1.50$ acc** on Full Test (SVAMP $-3.24$, MBPP+ $-3.31$, IFEval $22.5\%$ saved because response tokens ballooned from $263.1 \to 345.3$), `p1cr1s0` (with widened guards and anti-loop pairs) plateaued at **$34.4\%$ saved / $+2.90$ acc** on Dev-512 at step 200, and `bidax2` (competitor 1:1 shortest-right vs longest-right DPO) hit **$51.2\%$ saved / $-3.00$ acc** (SVAMP $-4.88$, MBPP+ $-4.23$). Conversely, two-stage compress-then-repair runs (`rpK-40`: **$51.2\%$ / $-0.80$**; `rpM-20`: **$50.1\%$ / $-0.94$**; `rpM-60`: **$54.1\%$ / $-0.93$**) are the *only* models in the project to enter the $>50\%$ saved / $\ge -1.0$ accuracy goal box.
  - *Literature Evidence:* **LC-R1 (`arXiv:2506.13607`)**, **Laser (`arXiv:2505.15612`)**, and **Thinkless (`arXiv:2505.13379`)** demonstrate that decoupling length compression from correctness verification prevents the policy gradient from collapsing the model's verification circuit on medium-difficulty problems.
  - *Counterfactual (Repairing after Over-Shortening past $\sim 59\%$ vs. Repairing from $47\%\text{--}57\%$):* In `rpN`, applying a 45-step accuracy-only repair (`--mix-preset hold`, zero shorten pairs) on top of `rpL-60` ($59.3\%$ Full Test / $58.9\%$ Dev-512) produced **zero recovery** (`rpN-15`: $58.5\% / +1.16$; `rpN-30`: $59.7\% / +0.71$; `rpN-45`: $59.9\% / +1.28$ on Dev-512, paired difference $\approx 0.0$ on both axes, while IFEval responses continued shrinking $273 \to 157$ tokens). Once `<think>` chains are compressed past $\sim 59\%$, the model's on-policy rollouts lack sufficient variance in verification steps to construct restorative pairs. By contrast, repairing from `rpE-80` ($47.0\%$ saved) in `rpM-60` gained $+7.1\%$ saving at $+0.0$ net accuracy cost ($54.1\% / -0.93$). **`[EXTRAPOLATION]`**: In Recipe 1, targeting $54.0\%\text{--}57.5\%$ Dev-512 saving in Step 1 ($\sim 52\%\text{--}55\%$ true saving) keeps the policy below the $59\%$ point of no return while requiring only $0\text{--}2$ points of Step 2 saving gain.

### 2.2 Pair Construction Rules & Exact Thresholds
1. **Confidence Gate $\hat{p} \ge 0.80$ in Step 1 (`c >= 8/10`) and $\hat{p} \ge 0.875$ in Step 2 (`c >= 7/8`):**
   - *Evidence:* **DAST (`arXiv:2503.04472`)**, **LAPO (`arXiv:2505.20728`)**, and **Ada-R1 (`arXiv:2504.14716`)** prove that length penalties on low-solve-rate prompts ($\hat{p} < 0.75$) train the model to guess early. In our pack, `lpb-l2s50math` and `rpM` showed that restricting `shorten` pairs to $\hat{p} \ge 0.80$ (Step 1) and $\hat{p} \ge 0.875$ (Step 2) cuts SVAMP accuracy loss from $-4.88$ pts (`bidax2`) and $-3.24$ pts (`p1r1s0-200`) to $-1.20$ pts (`rpM-60`).
   - *Counterfactual ($\hat{p} \ge 0.50$ vs. $\hat{p} \ge 0.80/0.875$):* Lowering `--shorten-min-right` to $0.50$ (`lpc-l2s50`) increases token saving by $\sim 3\%$ but degrades SVAMP and MBPP+ accuracy by $-2.0$ to $-3.5$ pts because $4/8$ clean solves on 3-step math problems include lucky short guesses that skip verification.
2. **Thinking Length Ratio $L_{\text{think}}(y^-) / L_{\text{think}}(y^+) \ge 1.35$ (Step 1) and $\ge 1.30$ (Step 2) with $\ge 25$-Token Floor:**
   - *Evidence:* Tested in `p1r1s0` ($1.35\times$) and `rpK`/`rpL`/`rpM` ($1.30\times$).
   - *Counterfactual ($1.10\times$ vs. $1.35\times$ vs. $2.00\times$):* At $1.10\times$, pairs capture random sampling noise rather than structural redundancy; at $2.00\times$, tool-call and IFEval shorten candidate pools shrink by $>65\%$, causing severe quota shortfalls on `bfcl_multiple` and `ifeval`.
3. **Universal Thinking-Only Measurement (`L_think`) + Response-Length Guard (`0.95 * med <= L_resp(y+) <= 1.05 * L_resp(y-)`):**
   - *Evidence:* In `p1r1s0-200`, `shorten` pairs measured `<think>` tokens on IFEval without capping $L_{\text{resp}}(y^+)/L_{\text{resp}}(y^-)$, so $76\%$ of chosen rollouts had *longer* visible responses; on Full Test, IFEval thinking dropped $432.8 \to 193.7$ tokens ($-55\%$) but responses grew $263.1 \to 345.3$ tokens ($+31\%$), cutting net IFEval saving to $22.5\%$. Adding `--resp-median 0.95 --resp-max 1.05` in `rpL` and `rpM` kept chosen vs. rejected responses matched ($82$ vs. $82$ tokens in `rpM` build log) and jumped Full Test IFEval saving from **$22.5\% \to 50.5\%$ (`rpM-60`) and $58.0\%$ (`rpL-60`)**!
4. **Prompt-Quantifier Detection (`--count-mode length`, `--no-shorten-counting`, `--count-resp-min 1.10 --count-resp-cap 1.75 --count-resp-pad 60`):**
   - *Evidence:* Full-test per-question diagnostics (`EXPERIMENT_LOG.md`, 11 Nov 16:05) revealed that on the 58 IFEval "at least $N$ words/sentences" questions, `rpE-80` ($18.5\%$) and `rpK-60` ($18.1\%$) lost $\sim 15$ pts vs. `original` ($33.2\%$), whereas `p1r1s0-200` (which rewarded longer responses everywhere) hit $40.7\%$ on "at least $N$" but collapsed by **$-12.85$ pts on "fewer than $N$"** ($72.9\% \to 60.1\%$). Restricting `--count-mode length` to true minimum-length constraints while forbidding shortening on counting prompts (`--no-shorten-counting`) in `rpM-60` improved IFEval accuracy by **+0.70 pts over `rpK-60` and +1.10 pts over `rpL-60`** at $50.5\%$ IFEval token savings.
5. **Widened Premature-Cut Guard (`L_think(y-) < 0.75 * med_think`, `--guard-wide`) + Length-Held `be_right` (`0.85 <= L_think(y+)/med <= 1.25`, `--right-hold`):**
   - *Evidence:* On `rpE-80` rollouts (`gpu2_night_key_lines.txt`), narrow guards ($<0.55\tilde{L}_{\text{think}}$) found only $174$ SVAMP candidates, whereas widened guards ($<0.75\tilde{L}_{\text{think}}$) found **$598$ candidates** ($1,013$ vs $255$ in `p1src`) and targeted the exact $0.5\text{--}1.0\times \tilde{L}_{\text{think}}$ band where SVAMP wrong answers surged from $284$ (`rpE-80`) to $396$ (`rpK-60`). Simultaneously, `--right-hold` prevented `be_right` pairs from picking shortest-thinking lucky rights ($39\%$ chosen longer in `rpL` vs $92\%$ chosen longer in `rpM` math guards+be_right). On Full Test, this cut SVAMP accuracy loss from **$-2.30$ pts (`rpK-60`) and $-2.70$ pts (`rpL-60`) to $-1.20$ pts (`rpM-60`)** while saving $57.0\%$ of SVAMP tokens!
   - *Counterfactual (Over-weighting Guards in Step 1 from `original`):* In `p1cr1s0`, combining 120 widened math guards + 280 math `be_right` against 360 `shorten` pairs directly from `original` caused `chosen` to be longer than `rejected` in $24\%$ of math pairs and $25\%$ of IFEval pairs, braking Dev-512 compression to $34.4\%$ saved. **`[EXTRAPOLATION]`**: In Recipe 1 Step 1, we cap `be_right` thinking at $\le 1.25\tilde{L}_{\text{think}}$ (`compact` rather than `length_held`) and keep `shorten` at $47.5\%$ of total pairs ($1,520 / 3,200$) with $100\%$ of pairs having $L_{\text{think}}(y^+) \le L_{\text{think}}(y^-)$ outside `guard`, ensuring Step 1 reaches $54\%\text{--}57\%$ saved before Step 2 applies `--right-hold`.
6. **Anti-Loop Pairs (`loop`, with `--loop-response-only` on Response-Heavy Prompts) and Balanced Tool-Call Quotas:**
   - *Evidence:* Runaway cap-hits ($4,096$ / $8,192$ tokens) accounted for $26\%$ of `rpE-80`'s IFEval tokens and $7.8\%$ of `original`'s MBPP+ rollouts. Adding `loop` pairs in `rpL-60` and `rpM-60` cut repetitive answers across the full test from **$13.5\% \to 3.4\%$** and reduced code cap-hit tokens from **$920 \to 84$ tokens/answer** (`NEW_RESULTS_12nov_morning.md`). In `rpM-20`, reducing IFEval response-loop pairs from 48 (`rpL`) to 24 left Dev-512 IFEval runaways at $3.9\%$ vs. `rpL-20`'s $2.9\%$; therefore Recipe 1 Step 2 restores **40 response-loop pairs** on IFEval. For tool calls, `consult11` proved that `rpK-60`'s $-1.75$ pt drop on `bfcl_simple_python` between steps 20 and 60 was $82\%$ caused by non-calls (call rate fell $97.19\% \to 95.75\%$) because 140 `bfcl_irrelevance` abstention pairs outweighed $48+36=84$ positive tool-call `be_right` pairs. Balancing `bfcl_multiple` to $50/50$ in `rpM-60` recovered Simple tool accuracy to **$-0.34$ pts** (vs $-1.50$ in `rpK-60` and $-2.00$ in `rpL-60`).

### 2.3 Training Hyperparameters (DPO & RL)
- **LoRA ($r=16, \alpha=32, \text{dropout}=0.0$, 7 target modules) & Fresh Adapter Per Step:**
  - *Evidence:* Used across all winning runs (`rpE`, `rpK`, `rpL`, `rpM`, `p1r1s0`). Setting `lora_dropout=0.0` in `sft.py --init-only` is mandatory because `dpo.py` precomputes reference log-probabilities (`ref.json`) with the model in `.train()` mode before step 1; non-zero dropout introduces stochastic noise into `ref_c - ref_r`. Merging the Step 1 adapter into `bfloat16` (`merge.py`) and initializing a fresh zero LoRA for Step 2 resets the reference anchor $\pi_{\text{ref}}$ to $M_1$, preventing reference drift.
- **Precision (`--bida-precision --dtype bfloat16`):**
  - *Evidence:* `dpo.py` casts trainable LoRA tensors to `float32` and runs `torch.amp.autocast("cuda", dtype=torch.float16)` with `GradScaler`. Pure `bfloat16` LoRA weights without `float32` master copies lose small gradient updates when `lr = 1e-5` ($\Delta w \sim 10^{-7}$ vs `bf16` machine epsilon $3.9 \times 10^{-3}$).
- **Learning Rate (`2e-5` in Step 1 vs. `1e-5` / `8e-6` in Step 2/3) & Cosine Schedule:**
  - *Evidence:* `p1r1s0` and `rpE` used `lr = 2e-5` with 10-step warmup; `rpK`, `rpL`, and `rpM` used `lr = 1e-5` for 60 steps.
  - *Counterfactual (`lr = 5e-5` vs. `2e-5` vs. `5e-6`):* At `5e-5` (`stopB` / early Track 2 runs), LoRA DPO overshoots within 25 steps and causes EOS collapse (`tripwire > 0.20`); at `5e-6` in Step 1, 200 steps move token length by $<20\%$. In Step 2, using `2e-5` instead of `1e-5` (`rpL-100/120` continuation) collapses `bfcl_simple_python` by $-5.1$ pts (`rpL-80`).
- **DPO $\beta$ (`0.10` in Step 1 vs. `0.15` / `0.20` in Step 2/3) and `--nll 0.0`:**
  - *Evidence:* All `track.py:ptrepair` runs used `--beta 0.10 --nll 0.0`. Adding positive NLL (`--nll 1.0`, RPO style) penalizes chosen sequence length less when chosen is long, slowing compression by $\sim 12$ pts in Track 2 tests. **`[EXTRAPOLATION]`**: Raising $\beta$ from $0.10 \to 0.15$ (Recipe 1 Step 2) and $0.20$ (Recipe 2 Step 3) strengthens the implicit KL penalty $\beta \,\mathbb{D}_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}})$ by $1.5\times\text{--}2.0\times$ during verification repair, preventing the policy from drifting into over-compression while still ordering $70\%\text{--}78\%$ of repair pairs correctly.
- **Max Sequence Length (`--max-len 8704`):**
  - *Evidence:* In `dpo.py`, any pair with `len(prompt_ids) + len(reply_ids) > max_len` is silently dropped (`line 65`). Because code rollouts cap at $8,192$ tokens (plus up to $512$ prompt tokens = $8,704$), setting `--max-len 2048` or `4096` silently deletes **100% of code and math anti-loop pairs** where the rejected rollout hit the cap! Thus `--max-len 8704` is essential.
- **Step Counts & Stratified Prefix Ordering (`--keep-order`):**
  - *Evidence:* `rpM` was evaluated on Full Test at steps **20 (`50.1% / -0.94`), 40 (`52.9% / -1.10`), and 60 (`54.1% / -0.93`)**, while `rpL` was evaluated at steps **40 (`55.1% / -1.37`), 60 (`59.3% / -1.70`), and 80 (`62.8% / -2.06`)**. Because `repair_pairs.py` interleaves pairs via a proportional credit queue across parts and rules, every 20-step prefix (`320` pairs) is a perfectly stratified mini-epoch. Screening at steps `20, 40, 60, 80` in Recipe 1 Step 2 sweeps the exact $[50\%, 56\%]$ Pareto frontier.
- **RL Design in Recipe 3 (`--advantage batch`, `--dynamic-sampling 1.5`, `--kl 0.005`, `--lr 1.5e-5`):**
  - *Evidence:* Guided by **Dr. GRPO (`arXiv:2503.20783`)**, **ShorterBetter (`arXiv:2504.21370`)**, **DAPO (`arXiv:2503.14476`)**, **DLER (`arXiv:2510.15110`)**, and our internal `gr2f` diagnosis (`GRPO_gr2f_summary.txt`, `consult7_gemini_path2_RL.md`). In `gr2f`, `--dr-grpo --fixed-denominator` divided gradients by $256 \times 8192 = 2.1\times 10^6$ tokens ($16\times$ larger than actual generated tokens) and disabled variance normalization (`scale=False`), which caused high-confidence all-right groups (reward spread $0.15$) to have $275\times$ smaller gradient variance than mixed groups (reward spread $1.0$), limiting `gr2f` to $+11.3\%$ saved at step 40. Using `--advantage batch` (already implemented in `rl_train.py:batch_advantages`) normalizes by batch-wide standard deviation across all 16 informative groups while preserving relative group magnitudes.

### 2.4 Decision Gate Calibration (Dev-512 & Dev-SVAMP-200 $\to$ Full Test)
- *Evidence:* Across all paired evaluations (`NEW_RESULTS_12nov_morning.md`):
  - **Dev-512 Saving vs. Full Test Saving:** Tracks within **$\pm 0.8$ pts** on multi-step models (`rpM-20`: $49.3\%$ Dev vs $50.1\%$ Full; `rpM-40`: $52.6\%$ vs $52.9\%$; `rpM-60`: $54.9\%$ vs $54.1\%$; `rpL-80`: $62.1\%$ vs $62.8\%$; `rpE-80`: $46.9\%$ vs $47.0\%$), and over-reads 1-step models from `original` by **$+2.5$ pts** (`p1r1s0-200`: $45.9\%$ Dev vs $43.4\%$ Full). Hence Step 1 gates require $\ge 53.5\%$ on Dev-512 (to guarantee $\ge 51.0\%$ true saving), and Step 2/3 gates require $\ge 52.0\%$ on Dev-512.
  - **Dev-512 Accuracy vs. Full Test Accuracy:** Dev-512 is systematically **$+1.4$ to $+3.2$ pts more generous** than Full Test (`rpM-20`: $+0.49$ Dev vs $-0.94$ Full; `rpM-60`: $+2.24$ Dev vs $-0.93$ Full; `rpK-40`: $+0.95$ Dev vs $-0.80$ Full). Therefore, a naive Dev-512 gate of $\ge -1.0$ pt would promote models that lose $-2.8$ to $-4.0$ pts on Full Test! Setting the Step 2/3 promotion gate at **Dev-512 accuracy $\ge +1.10$ to $+1.30$ pts vs. `original` AND paired gain $\ge +0.60$ to $+0.80$ pts vs. the previous step** directly accounts for the $+2.1$ pt mean optimism offset plus the $\pm 0.7$ pt single-run sampling noise.

---

## Section 3. Audit of Earlier Report (`YOUR_REPORT.md`) & Exact Fixes Applied

We audited `reviews/consult16_plan/pack/YOUR_REPORT.md` against the actual codebase (`code/track.py`, `code/repair_pairs.py`, `code/dpo.py`, `code/rl_train.py`, `code/sft.py`) and experiment logs, identifying and fixing **six concrete inconsistencies**:

1. **Step Counts vs. Checkpoint Lists (Fixed):**
   - *Inconsistency in `YOUR_REPORT.md`:* In Recipe 1 Step 1, the report specified *"Train for 240 optimizer steps ... Save checkpoints at steps 160, 200, 240, 280"* (step 280 is never reached if training stops at 240). Similarly, in Recipe 2 Step 3, it specified *"Train for 60 steps ... Save checkpoints at steps 30, 45, 60, 75"* (step 75 exceeds 60).
   - *Fix Applied:* Every step's `--max-steps`, `--save-every`, and `--rp-screen` checkpoint list are now strictly aligned: Recipe 1 Step 1 trains for `240` steps and screens `160, 200, 240`; Recipe 2 Step 3 trains for `60` steps and screens `30, 45, 60`.
2. **Pair File Quotas vs. Batch Size and Epoch Counts (Fixed):**
   - *Inconsistency in `YOUR_REPORT.md`:* Recipe 1 Step 2 specified *"2,600 pairs ... 80 optimizer steps, batch size 16"* ($80 \times 16 = 1,280$ pairs, which would leave $51\%$ of the 2,600-pair file unread!). Similarly, Recipe 2 Step 2 specified *"2,800 pairs ... 160 steps"* ($160 \times 16 = 2,560 \ne 2,800$).
   - *Fix Applied:* Every pair quota is now an exact multiple of `batch_size = 16` matched to the step budget: Recipe 1 Step 1 = `3,200` pairs (`200` steps/epoch, trained `240` steps = $1.20$ epochs); Recipe 1 Step 2 = `1,280` pairs (`80` steps = $1.00$ epoch); Recipe 2 Step 1 = `2,880` pairs (`180` steps = $1.00$ epoch); Recipe 2 Step 2 = `1,600` pairs (`100` steps = $1.00$ epoch); Recipe 2 Step 3 & Recipe 3 Step 2 = `960` pairs (`60` steps = $1.00$ epoch).
3. **DPO Sequence Length Cap `--max-len 8704` vs. `2048` (Fixed):**
   - *Inconsistency in `YOUR_REPORT.md`:* Specified `max_length = 2048` for DPO and `max_new_tokens = 2048` for RL. In `dpo.py`, any pair exceeding `--max-len` is discarded (`line 65`), so `2048` would drop all code rollouts ($2,048\text{--}8,192$ tokens) and all math/IFEval runaway loop pairs ($4,096$ tokens).
   - *Fix Applied:* Set `--max-len 8704` in all DPO steps and `--max-tokens 4096 --max-tokens-code 8192` in all rollout generation and RL steps.
4. **DPO Precision & LoRA Dropout Flags vs. Actual `dpo.py` / `sft.py` (Fixed):**
   - *Inconsistency in `YOUR_REPORT.md`:* Cited `--fp16` (which does not exist in `dpo.py`) and `lora_dropout = 0.05`.
   - *Fix Applied:* Use the exact `dpo.py` flags `--bida-precision --dtype bfloat16` (which keeps master LoRA weights in `float32` and runs `float16` autocast with `GradScaler`) and `lora_dropout = 0.0` (matching `sft.py --init-only` for deterministic `ref.json` scoring).
5. **Prompt Pool Source & Stored Rollout Reuse (`p1src`) vs. Redundant Generation (Fixed):**
   - *Inconsistency in `YOUR_REPORT.md`:* Proposed generating $3,800 \times 8 = 30,400$ fresh rollouts from `Qwen/Qwen3-0.6B` in Step 1 (`1.6` GPU-h) and did not mention `Dev-SVAMP-200` in the dev screening protocol.
   - *Fix Applied:* Grounded Step 1 of Recipes 1 and 2 in `track.py:p1src` (`12,468` disjoint questions $\times 10$ stored rollouts = `124,680` answers at **0.0 GPU-h**), extended the 8-gram near-duplicate filter (`dev_gram_set()`) to include `Dev-SVAMP-200`, and included both **Dev-512** and **Dev-SVAMP-200** in every step's decision gate.
6. **RL Loss Normalization in Recipe 3 (`--advantage batch` vs. `--dr-grpo`) (Fixed):**
   - *Inconsistency in `YOUR_REPORT.md`:* Recommended `--dr-grpo` in `rl_train.py`. As proved in `consult7` and `GRPO_gr2f_summary.txt`, `--dr-grpo` in `rl_train.py` disables group std scaling (`scale=False`) and divides by `len(chunk) * 8192`, silencing high-confidence groups by $275\times$.
   - *Fix Applied:* Specified `--advantage batch` (DLER batch-wise std normalization, already in `rl_train.py`) with active-token normalization and `--dynamic-sampling 1.5`.

---

## Section 4. Implementation Blueprint, Code Modifications & Pre-Flight Checks

### 4.1 What Exists in `code/` and Can Be Reused As-Is
- **`code/prepare_types.py` & `code/ifeval_Make_questions.py`:** Load and format public pool questions (`gsm8k`, `kodcode`, `opencoder`, `ifeval_own`, `toolace`, `glaive`) and `tr_svamp_like` (`Dev-SVAMP-200`). Reuse **as-is**.
- **`code/check_vllm.py` (`grade_all`) & `code/evaluate.py`:** Deterministic parallel graders (16 threads) for math, code (sandbox execution), IFEval (`instruction_following_eval`), and BFCL AST tool-call checking. Reuse **as-is**.
- **`code/dpo.py`:** Reference-anchored LoRA DPO trainer with `--bida-precision`, `--keep-order`, `--mix all`, `--beta`, `--lr-schedule cosine`, `--max-len 8704`, and `ref.json` caching. Reuse **as-is**.
- **`code/merge.py` & `code/sft.py` (`--init-only`):** Exact `fp32` LoRA adapter merging and zero-initialized LoRA creation ($r=16, \alpha=32, \text{dropout}=0.0$). Reuse **as-is**.
- **`code/dev_score.py` & `code/probe_curve.py`:** Batched vLLM rollout generator (`--generate-only`, `--half a/b`) and paired bootstrap evaluator (`--vs`, `--label Dev-SVAMP-200`). Reuse **as-is**.

### 4.2 What Needs a New CLI Option (3 Small, Surgical Additions)
1. **`code/track.py` (~18 lines modified):**
   - Add **`--rp-beta`** (float, default `0.10`) to `track.py` CLI parser and pass `"--beta", str(args.rp_beta)` to `dpo.py` inside `stage_ptrepair` (currently `track.py` hardcodes `"--beta", "0.1"` at line 3655).
   - In `dev_gram_set()` and `stage_p1src()`, union `dev_svamp_ids()` and the rare 8-grams of `data/dev-svamp200.jsonl` into `d_ids` and `d_grams` so `p1src` and `own_repair_questions` enforce both exact ID exclusion and the $<30\%$ 8-gram near-duplicate rule against **both** `Dev-512` and `Dev-SVAMP-200` (currently `p1src` filters against `dev512.jsonl` only, dropping 200 fewer questions).
2. **`code/repair_pairs.py` (~65 lines added):**
   - Add **`--general-mode`** (`choices=["", "general_step1", "general_step2", "general_r2_step1", "general_r2_step2", "general_repair"]`) and **`--count-fallback-rollouts`** (optional path to `runs/bon/rp-p1src` to top up `counting` pairs if a compressed model has too few long responses):
     - Under any `--general-mode`, `measure_len(r)` universally uses `thinking_len(r)` for shortening and `response_len(r)` for response guards (`--resp-median 0.95 --resp-max 1.05`) on **all** questions without branching on `part == "ifeval"`.
     - Replaces metadata lookup `r.get("instruction_id_list")` in `length_rule(r)` with a pure prompt-text regex `PROMPT_MIN_LEN_RE = re.compile(r"(?i)\b(?:at least|minimum of|no fewer than|more than)\s+(\d+)\s+(?:word|sentence|paragraph|line|bullet|item|section)s?\b")` decoded from `r["prompt_ids"]` (or prompt text cache).
     - Applies widened premature-cut guards (`< 0.75 * med_think`) and length-held `be_right` (`0.85 <= L_think(y+)/med_think <= 1.25`) based on per-question median thinking length $\tilde{L}_{\text{think}} \ge 150$ tokens rather than `if base_part == "svamp"`.
     - Maps the preset quotas for `general_step1` (`shorten:be_right:guard:loop:counting`), `general_step2`, `general_r2_step1`, `general_r2_step2`, and `general_repair` with `--no-trim-mix` enabled so no part's total quota is silently cut when a minor sub-rule is scarce.
3. **`code/rl_train.py` (~35 lines added):**
   - Add **`--reward gated_sb`** implementing the Confidence-Gated ShorterBetter reward formula (Section 1, Recipe 3 Step 1b) using `thinking_len`, `response_len`, `PROMPT_MIN_LEN_RE`, and $\hat{p}(x) = n_{\text{pass}}/8 \ge 0.75$.
   - When `--advantage batch` is active without `--dr-grpo`, normalize the policy gradient loss by total active generated tokens in the mini-batch (`max(total_tok, 1)`) rather than `len(chunk) * args.max_tokens`.

### 4.3 Build & Test Order and Cheapest Pre-Flight Checks (Total CPU/GPU Pre-Flight: < 12 Minutes)
1. **Step A — Unit-Test `track.py` Disjointness & `p1src` Filter (CPU, 20 seconds):**
   - Run `python3 code/track.py --stage p1src` and verify that: (i) `0` question IDs overlap with `questions/questions.jsonl` (2,759), `data/dev512.jsonl` (512), or `data/dev-svamp200.jsonl` (200); (ii) `0` kept questions have $\ge 30\%$ rare 8-gram overlap with `Dev-512` or `Dev-SVAMP-200`; and (iii) `12,468` Round-1 questions and `4,595` reserved Round-2 questions are written.
2. **Step B — Offline Pair-Build Dry Run on Real Stored Rollouts (CPU, 90 seconds):**
   - Run `repair_pairs.py` with `--general-mode general_step1` on `runs/bon/rp-p1src` (for Recipe 1 Step 1 and Recipe 2 Step 1) and with `--general-mode general_step2` on `runs/bon/rp-rpI` (for Recipe 1 Step 2).
   - *Pass Criteria:* Verify `SHORT of quota: 0` across all 6 parts, verify that the first 80 pairs (`first 80 pairs by part`) match the target stratification within $\pm 1$ pair, verify that `chosen` vs `rejected` response tokens on `ifeval` `shorten` pairs have median ratio $\in [0.97, 1.03]$, and verify that `0` pairs exceed `--max-len 8704`.
3. **Step C — 2-Step DPO & Merge Smoke Test on GPU (L4 GPU, 4 minutes):**
   - Run `dpo.py` for `--max-steps 2 --batch 16 --beta 0.15 --bida-precision` on the built pairs, confirm `ref.json` is written, confirm step 1 loss is $\approx \ln 2 = 0.6931 \pm 0.015$, and run `merge.py` to verify adapter merge speed ($\sim 90\text{ s}$).
4. **Step D — 2-Step RL Smoke Test on GPU (L4 GPU, 5 minutes):**
   - Run `rl_train.py --steps 2 --questions 4 --group 8 --updates 2 --reward gated_sb --advantage batch --dynamic-sampling 1.5 --max-tokens 4096 --max-tokens-code 8192` to verify vLLM engine memory (`gpu_memory_utilization=0.45` leaves $>8\text{ GiB}$ free for HF backward pass without CUDA graph OOM) and verify non-zero advantages on both $8/8$ all-right and mixed groups.

---

## Section 5. Two-GPU Execution Schedule & Updated Side-by-Side Recipe Comparison

### 5.1 Master Execution Schedule on Two Colab L4 GPUs (`GPU 1` and `GPU 2`)

All runs execute via `track.py` queue files (`plan_line_a.txt` on **GPU 1**, `plan_line_b.txt` on **GPU 2**) synchronized through Google Drive (`/content/drive/MyDrive/thinkingcap/`). Rollout generation and Dev checkpoint evaluations are parallelized across both GPUs (`--half a` / `--half b` for rollouts; alternating checkpoints for Dev scoring).

| Phase & Wall-Clock Window | GPU 1 (`plan_line_a.txt`) | GPU 2 (`plan_line_b.txt`) | Where Outputs & Results Land on Drive |
| :--- | :--- | :--- | :--- |
| **Phase 0: Pre-Flight & `p1src` Prep**<br>*(0:00 – 0:15, 0.25 h)* | Run `p1src` filter (`12,468` Step 1 qns); build `pairs-r1s1` (3,200 pairs) & `pairs-r2s1` (2,880 pairs); run 2-step DPO smoke test. | Run `devsvamp` setup; verify cached `original` & `rpE80` reference rollouts on `Dev-512` and `Dev-SVAMP-200`; run 2-step RL smoke test. | `runs/bon/rp-p1src/`<br>`runs/bon/pairs-r1s1/train.jsonl`<br>`runs/bon/pairs-r2s1/train.jsonl`<br>`data/p1-round1-ids.json` |
| **Phase 1: Step 1 Training (Recipes 1 & 2 in Parallel)**<br>*(0:15 – 1:00, 0.75 h)* | **Recipe 1 Step 1 DPO (`r1s1`):** Train 240 steps (`lr=2e-5, beta=0.10`) on `pairs-r1s1` (`0.52` h). Save steps `160, 200, 240`. Merge `r1s1-160` & `r1s1-200`. | **Recipe 2 Step 1 DPO (`r2s1`):** Train 180 steps (`lr=2e-5, beta=0.10`) on `pairs-r2s1` (`0.42` h). Save steps `120, 150, 180`. Merge `r1s1-240`, `r2s1-150`, `r2s1-180`. | `adapters/r1s1-s0/step-{160,200,240}`<br>`adapters/r2s1-s0/step-{120,150,180}`<br>`models/r1s1-{160,200,240}`<br>`models/r2s1-{120,150,180}` |
| **Phase 2: Step 1 Dev Screening & Gate 1**<br>*(1:00 – 2:15, 1.25 h)* | Score `r1s1-160`, `r1s1-200`, `r1s1-240` on **Dev-512** + **Dev-SVAMP-200** (`1.26` h). Promote winner `models/r1s1-best`. | Score `r2s1-120`, `r2s1-150`, `r2s1-180` on **Dev-512** + **Dev-SVAMP-200** (`1.26` h). Promote winner `models/r2s1-best`. | `runs/dev512/r1s1-{160,200,240}.jsonl`<br>`runs/dev_svamp200/r1s1-*.jsonl`<br>`runs/dev512/r2s1-{120,150,180}.jsonl`<br>`track_log.txt` (Gate 1 table) |
| **Phase 3: Recipe 1 Step 2 Rollouts, Training & Screening**<br>*(2:15 – 4:10, 1.92 h)* | Generate **Half A** (`1,800 x 8`) of `r1s2` unseen questions with `r1s1-best` (`0.90` h). Run `repair_pairs.py` (`1,280` pairs) + **Recipe 1 Step 2 DPO (`r1s2`, 80 steps, `lr=1e-5, beta=0.15`)** (`0.23` h). Score `r1s2-20` & `r1s2-40` on Dev-512 + Dev-SVAMP-200 (`0.84` h). | Generate **Half B** (`1,800 x 8`) of `r1s2` unseen questions with `r1s1-best` (`0.90` h). Wait for `r1s2` DPO (`0.23` h, or start `r2s2` rollouts). Score `r1s2-60` & `r1s2-80` on Dev-512 + Dev-SVAMP-200 (`0.84` h). Promote `models/r1s2-best`. | `runs/bon/rp-r1s2/answers.jsonl`<br>`runs/bon/pairs-r1s2/train.jsonl`<br>`adapters/r1s2-s0/step-{20,40,60,80}`<br>`runs/dev512/r1s2-{20,40,60,80}.jsonl` |
| **Phase 4: Recipe 1 Full Test + Recipe 2 Steps 2 & 3**<br>*(4:10 – 6:40, 2.50 h)* | **FULL TEST (`2,759 x 8`) of Recipe 1 Winner `models/r1s2-best`** (`2.25` h). | Run **Recipe 2 Step 2** (`2,400 x 8` rollouts + 100 steps DPO + Dev screen of `60, 80, 100`) and **Recipe 2 Step 3** (`2,195 x 8` rollouts + 60 steps repair DPO). | `runs/fulltest/r1s2-best/` (`summary.json`)<br>`adapters/r2s2-s0/`, `adapters/r2s3-s0/`<br>`runs/dev512/r2s2-*`, `runs/dev512/r2s3-*` |
| **Phase 5: Recipe 2 Full Test + Recipe 3 (RL + DPO)**<br>*(6:40 – 9:30, 2.83 h)* | Score `r2s3-{30,45,60}` on Dev-512 + Dev-SVAMP-200 (`0.63` h); run **FULL TEST (`2,759 x 8`) of Recipe 2 Winner `models/r2s3-best`** (`2.25` h). | Run **Recipe 3 Step 1 RL (`r3s1`, 120 steps GRPO)** (`2.10` h) + Dev screen `r3s1-{60,90,120}` (`0.73` h). | `runs/fulltest/r2s3-best/` (`summary.json`)<br>`adapters/r3s1-rl/step-{60,90,120}`<br>`runs/dev512/r3s1-{60,90,120}.jsonl` |
| **Phase 6: Recipe 3 Step 2 DPO & Full Test**<br>*(9:30 – 12:30, 3.00 h)* | Generate **Half A** (`1,200 x 8`) of `r3s2` rollouts (`0.60` h); run **Recipe 3 Step 2 DPO (`r3s2`, 60 steps)** (`0.18` h); score `r3s2-30` & `r3s2-45` (`0.84` h). | Generate **Half B** (`1,200 x 8`) of `r3s2` rollouts (`0.60` h); score `r3s2-60` (`0.42` h); run **FULL TEST (`2,759 x 8`) of Recipe 3 Winner `models/r3s2-best`** (`2.25` h). | `runs/bon/pairs-r3s2/train.jsonl`<br>`adapters/r3s2-s0/step-{30,45,60}`<br>`runs/fulltest/r3s2-best/` (`summary.json`) |

*(Note: Recipe 1 completes its Full Test in **6.35 hours wall-clock** from start—or **5.61 hours** if run standalone without Recipe 2 Step 1 interleaved in Phase 1/2. All three recipes complete their Full Tests within **12.5 hours wall-clock** [25.0 GPU-hours total] on 2x Colab L4s.)*

---

### 5.2 Updated Side-by-Side Comparison Table of the Three Recipes

| Dimension | Recipe 1: 2-Step Confidence-Gated DPO (Primary Recommendation) | Recipe 2: 3-Step Progressive Curriculum DPO (Conservative Backup) | Recipe 3: 2-Step On-Policy RL + Verification DPO (Hybrid Alternative) |
| :--- | :--- | :--- | :--- |
| **Learning Steps & Fixed Order** | **2 Steps:**<br>1. Confidence-Gated Compression DPO (`original` $\to M_1$)<br>2. Length-Matched Verification & Loop DPO ($M_1 \to M_2$) | **3 Steps:**<br>1. Moderate Compression DPO (`original` $\to M_{2,1}$)<br>2. On-Policy Compression DPO ($M_{2,1} \to M_{2,2}$)<br>3. Pure Verification Repair DPO ($M_{2,2} \to M_{2,3}$) | **2 Steps:**<br>1. Confidence-Gated ShorterBetter GRPO (`original` $\to M_{\text{RL}}$)<br>2. Length-Matched Verification DPO ($M_{\text{RL}} \to M_{\text{RL},2}$) |
| **Step 1 Data, Pairs/Groups & Hyperparameters** | `p1src` (`12,468` qns $\times 10$ stored rollouts, **0 GPU-h**); **3,200 pairs** ($47.5\%$ `shorten` at $\hat{p}\ge 0.80$, $38.1\%$ `be_right`+`counting`, $9.4\%$ wide `guard`, $5.0\%$ `loop`); DPO `240` steps (`1.2` ep), `batch=16`, `lr=2e-5`, `beta=0.10`, `max_len=8704`. | `p1src` (`12,468` qns $\times 10$ stored rollouts, **0 GPU-h**); **2,880 pairs** ($40\%$ math / $60\%$ code `shorten` at ratio $\ge 1.40\times$); DPO `180` steps (`1.0` ep), `batch=16`, `lr=2e-5`, `beta=0.10`, `max_len=8704`. | `2,880` disjoint pool prompts; **120 RL steps** $\times$ `16` informative groups/step $\times G=8$ (`15,360` active rollouts, `--dynamic-sampling 1.5`); `--reward gated_sb`, `--advantage batch`, `lr=1.5e-5`, `kl=0.005`, `2` updates/step (`240` grad steps). |
| **Step 2 Data, Pairs & Hyperparameters** | `3,600` unseen reserve qns $\times 8 = 28,800$ rollouts from $M_1$; **1,280 pairs** ($0\%$ math/code `shorten`, length-held `be_right`, wide `guard` $<0.75\tilde{L}_{\text{think}}$, `length` counting, 40 IFEval response loops, 50/50 tool balance); DPO `80` steps (`1.0` ep), `lr=1e-5`, `beta=0.15`. | `2,400` unseen reserve qns $\times 8 = 19,200$ rollouts from $M_{2,1}$; **1,600 pairs** (`shorten` at $\hat{p}\ge 0.875$, ratio $\ge 1.30\times$); DPO `100` steps (`1.0` ep), `lr=1.2e-5`, `beta=0.12`. | `2,400` unseen reserve qns $\times 8 = 19,200$ rollouts from $M_{\text{RL}}$; **960 pairs** (same verification rules as Recipe 1 Step 2); DPO `60` steps (`1.0` ep), `lr=1e-5`, `beta=0.15`. |
| **Step 3 Data, Pairs & Hyperparameters** | *None (2-step recipe).* | `2,195` unseen reserve qns $\times 8 = 17,560$ rollouts from $M_{2,2}$; **960 pairs** (**$0\%$ `shorten` across all types**, $100\%$ `be_right` + wide `guard` + `counting` + `loop`); DPO `60` steps (`1.0` ep), `lr=8e-6`, `beta=0.20`. | *None (2-step recipe).* |
| **Generality Guarantee** | $100\%$ per-question measured quantities ($\hat{p}=c/K$, $L_{\text{think}}$, $L_{\text{resp}}$, finish reason, `PROMPT_MIN_LEN_RE`, tool schema presence); zero benchmark names. | $100\%$ per-question measured quantities; zero benchmark names. | $100\%$ per-question measured quantities; zero benchmark names. |
| **Expected Full-Test Tokens Saved** | **$52.8\%\text{--}55.2\%$** *(central: **$54.2\%$**)* | **$53.0\%\text{--}55.5\%$** *(central: **$54.4\%$**)* | **$52.0\%\text{--}54.5\%$** *(central: **$53.2\%$**)* |
| **Expected Full-Test Accuracy Change** | **$-0.50\text{ to }-0.90\text{ pts}$** *(central: **$-0.72\text{ pts}$**)* | **$-0.55\text{ to }-0.92\text{ pts}$** *(central: **$-0.75\text{ pts}$**)* | **$-0.60\text{ to }-0.95\text{ pts}$** *(central: **$-0.80\text{ pts}$**)* |
| **Expected Per-Type Full-Test Breakdown (Saved / Acc)** | SVAMP: $57\% / -1.1$<br>MBPP+: $58\% / -0.8$<br>IFEval: $52\% / -1.8$<br>BFCL (400:200:240): $50\% / +0.8$ | SVAMP: $57\% / -1.1$<br>MBPP+: $58\% / -0.8$<br>IFEval: $53\% / -1.9$<br>BFCL (400:200:240): $50\% / +0.8$ | SVAMP: $56\% / -1.2$<br>MBPP+: $56\% / -0.9$<br>IFEval: $50\% / -1.9$<br>BFCL (400:200:240): $51\% / +0.8$ |
| **Total GPU-Hours / Wall-Clock (2x L4s)** | **7.77 GPU-h / 5.61 h wall-clock** | **9.32 GPU-h / 6.63 h wall-clock** | **8.25 GPU-h / 6.63 h wall-clock** |
| **Primary Risk & Built-In Mitigation** | *Risk:* Step 1 from `original` saves $<53.5\%$ on Dev-512.<br>*Mitigation:* Gate 1 automatically branches into Recipe 2 Step 2 before repair. | *Risk:* Step 2 over-compresses past $59\%$ (`rpN` trap).<br>*Mitigation:* Gate 2 enforces Dev-512 Saved $\le 58.5\%$ (selects step 60 over 100). | *Risk:* On-policy RL gradient variance or KL braking.<br>*Mitigation:* `--advantage batch` + `--dynamic-sampling 1.5` + Step 2 DPO polish. |