# Public ThinkingCap Test entries: how the best ones got their scores

> **Reference study only. The team does not copy any of these recipes; this is for insight into what works and which settings matter.**

Written 11 Nov 2026 from the public repos, read-only (gh api raw reads and Hugging Face raw files). Every fact has its
source next to it. "UNVERIFIED" means I could not find it in the source. Anything marked **Inference** is my reading,
not something the author states.

Source roots (paths below are relative to these):
- **[BIDA]** https://github.com/michal-bida/ThinkingCap-Test/tree/thinking-compression
- **[HERZ]** https://github.com/Herz1c/WhereTheThinkingStopsMattering (main); **[HERZ17]** https://github.com/Herz1c/DoesLearningToStopScale
- **[MAT]** https://github.com/matousek-martin/qwen3-self-replay (main)
- **[BASTA]** https://github.com/basta/lightning-mcqwen3/tree/recipe (blog: https://blog.basta.one/blog/thinkingcap/, not read)
- **[MRT]** https://huggingface.co/mrtineu/thinkingcap-test-submission
- **[KRCAL]** https://github.com/marek-krcal/MakeThinkingAffordableAgain/tree/experiments/shortest-preferences

All scores are the official full test (2,759 questions x 8 samples, BFCL pooled, tokens saved = 1 - geometric mean of
token ratios, accuracy = mean over 4 benchmarks) unless marked "2-sample".

---

## 1. One-page summary

| Entry | Method family | Full-test score | Key settings | What seems to matter (their evidence) |
|---|---|---|---|---|
| **michal-bida** (arm Db) | One round of offline DPO on the model's own graded samples | **+47.0% / -0.78** [CI tokens 45.8-48.1, acc -1.49..-0.05] | 11,327 prompts matched to benchmarks (length deciles, 8-gram decontam); 4 samples each; chosen = shortest right; rejected = longest wrong/truncated AND longest right if >= 1.5x; long pairs capped per domain at wrong-pair count (5,862 pairs); DPO beta 0.1, LoRA r32/a64 all linear, lr 2e-5 cosine 5% warmup, 16 pairs/step, 1 epoch (367 steps), no NLL, ref = base, final adapter | Negative samples are what shorten (SFT on shortest = +4.5%). Long pairs alone collapse (-13 pp); wrong pairs alone are safe (+23% / +0.81). Per-domain balance cut SVAMP loss 3.9 -> 2.4 (indicative). Offline GRPO fails |
| **Herz1c** | Stop-point SFT (own reasoning cut at earliest verified-right paragraph) then short online GRPO | **+43.4% / -0.97** (mean of 2 full runs: 43.3/-1.17, 43.5/-0.77) | SFT: LoRA r16/a32, drop 0.05, lr 1e-4, 2 epochs, 16 seq/step. GRPO: reward 1[correct]*(1 - 0.2*L/mean_L_domain), group 8, 24 prompts/step, 50 steps, lr 1e-5, no KL, advantage = r - group mean, one on-policy update | Stop only after the code is drafted; "give up with a best guess" at 2,048 thinking tokens; verifiable non-IFEval constraint data; RL is what shortens SVAMP without more loss (14% -> 31%). Best MBPP saving of all (65%) |
| **matousek-martin** (Self-Replay) | Online Dr.GRPO with correctness-gated length reward + SFT replay of own shortest 4/4-correct answers | ckpt 256: **+31.2% / +0.21**; 850: **+41.6% / -1.17**; 1280: +46.6% / -2.35 | Group 4; r = c*(1 - 0.1*min(L/R_domain,1)); replay buffer 128 questions, NLL weight 0.1; LoRA r16/a32, lr 1e-5, clip 1.0, gen cap 2,048, no KL; 2xA100 | Replay does nearly all the shortening (no replay at 128: 27.1% -> 1.85%); the length reward adds a little (no length at 256: 31.2 -> 27.5). Length/accuracy trade is almost linear with training time |
| **basta** (Lightning McQwen3) | Prompt distillation SFT -> truncation-oracle DPO -> NLL-anchored right>wrong DPO -> (test-time probe gate) | **+43.2% / -0.85** weights only; +47.3% / -0.88 with gate (gate = test-time, not weights) | LoRA r16/a32 all linear, chained. SFT lr 1e-4, 1 epoch, 11,247 targets. Oracle DPO 9,034 pairs (40 shorten / 40 guard / 20 hard), beta 0.1, lr 2e-5, batch 16, STOPPED at step 100/561. Correctness DPO 732 math pairs (+-25% length), NLL weight 1, 42 steps | Guard pairs make DPO safe (shortest>longest alone: 90% saved, -27 pp in 50 steps). On-policy oracle (on the distilled model) beats base-model oracle. NLL anchor needed for right>wrong pairs. Used NO code / tool / instruction training data, yet MBPP 39%, IFEval 32% |
| **mrtineu** (Martin Senkyr) | SFT on own short answers written with Qwen3-8B "skip this" hints; trained via a zero-init MLP inserted after layer 13 | **+28.7% / -1.71** | Extra layer = copy of layer 13, attention off, 64 MLP channels trained (198k params), lr 3e-4, half epoch best | Hints matter: control (shortest of own plain answers, 0.65x length) gave 0-2%; coached targets (0.44x) gave 22.5% (LoRA) / 28.7% (MLP). Smaller adapter compressed more. Best always at half an epoch |
| **marek-krcal** | Pairwise SimPO / DPO on own GSM8K answers | No public full score found (UNVERIFIED; a "29.36%" threshold in the plan suggests an earlier SimPO point near 29%) | SimPO beta 2, gamma 1; DPO beta 0.1 (and 0.05 control); 982 updates; 8 answers/question; 2,287 GSM8K groups | Too early to tell |

**Not run by anyone (as far as public):** a run above 50% saved at better than -1 on the full test. Closest: bida 47.0 /
-0.78 (weights only), basta 47.3 / -0.88 (with a test-time gate). bida's unbalanced arm D read +50.2 / -0.41 but on a
2-sample eval only. Herz1c's 1.7B variant got 47.4 / -0.14, but that is a different model.

---

## 2. michal-bida, branch thinking-compression (arm Db, 47.0% / -0.78)

Main files: [BIDA] `IDEA.md`, `reports/FINDINGS.md`, `train_dpo_offline.py`, `prepare_domains.py`,
`behaviour_check.py`, `queue/*.sh`, `reports/compare_*.txt`, `reports/domains_samples.md`, `reports/diag_runs.txt`,
`utils/eval/recipe.py`. Branch history: one commit, 18 Oct 2026, "Shorten Qwen3-0.6B thinking with DPO on its own
graded samples" (gh api commits). The repo is a fork of the challenge, so evaluate.py/compare.py are the official ones.

### 2.1 Data (FINDINGS §16-17, IDEA.md "Method", prepare_domains.py)
| domain | sources (rows) | matched to | source quotas (prepare_domains.py `QUOTAS`) |
|---|---|---|---|
| code | KodCode-V1 Filter 1,326 / Prefill 1,235 / Leetcode 221 / Evol 207 = **2,989** | MBPP+ | Filter 0.4, Prefill 0.3, Leetcode 0.15, Evol 0.15 |
| tools | ToolACE simple 641 / multiple 1,004 / decline 365 + xLAM irrelevance 976 = **2,986** | BFCL simple / multiple / irrelevance | simple 0.4, multiple 0.3, decline 0.05, xlam irrelevance 0.25 |
| ifeval | IF_multi_constraints_upto5 (1/2/3 constraints) 963 + RLVR-IFeval 1,422 = **2,385** | IFEval | by constraint count 1/2/3 = 0.56/0.33/0.11 |
| math | GSM8K 2,152 + MetaMathQA GSM_SV 614 / GSM_FOBAR 201 = **2,967** | SVAMP | GSM8K 0.7, GSM_SV 0.15, GSM_FOBAR 0.15 |
| **total** | **11,327** prompts | | |

- Each prompt is written in its benchmark's format (MBPP+ template with one literal assert + name reminder; BFCL's own
  system prompt; SVAMP's "Final answer:" instruction). Source: FINDINGS §16.
- **Prompt-length matching:** each source group drawn evenly from the deciles of its benchmark's prompt-token
  distribution (full chat template), rows outside the benchmark's range +-10% dropped (prepare_domains.py lines 18-20,
  533-633; FINDINGS §17). Code prompts at most 700 characters, imperative "Write a function...".
- **Gradable with eval graders only:** KodCode reference solution must pass its tests in the sandbox; ToolACE gold call
  must pass BFCL's AST checker (types normalised); IFEval rows use only IFEval's checkers, RLVR constraints mapped to them,
  counts capped at IFEval maxima (FINDINGS §17).
- **Decontamination:** any prompt sharing an 8-gram with any of the 2,759 eval prompts dropped (for IFEval only the base
  request is compared) (FINDINGS §16-17).
- **Tools share:** 45% of the tools domain is "no function fits" (irrelevance 976 + decline 365 = 1,341 / 2,986).
  **Inference:** this is likely why his BFCL irrelevance went *up* +6 pp, unlike matousek's (-4 to -10 pp).
- **Behaviour check** (base model, eval sampling): accuracy train vs eval: code 53.8 vs 51.5, ifeval 63.7 vs 58.1,
  math 85.6 vs 88.5, tools 78.9 vs 81.2; median tokens 972 vs 1,019, 419 vs 462, 509 vs 390, 186 vs 170
  (FINDINGS §17, reports/domains_samples.md). Per-source accuracy, e.g. Leetcode 41.2% with 14.1% truncated, xlam
  irrelevance 92.4%, toolace multiple 68.7% (domains_samples.md).

### 2.2 Sampling and grading
- 4 samples per prompt at the eval's settings: temperature 0.6, top_p 0.95, top_k 20, cap 4,096 tokens (8,192 for
  code/MBPP), thinking on (utils/eval/recipe.py; behaviour_check.py `SamplingParams(max_tokens=cap(row), seed=..., **SAMPLING)`;
  queue/97-sample-domains.sh `--samples 4 --per-domain 0`).
- 45,308 samples, about 25 h on an RTX 2080 Ti (IDEA.md). Graded with the eval's own graders, KodCode pytest tests for
  code (queue/98-grade-domains.sh). 9,337 of 11,327 prompts have at least one right, untruncated sample. The shortest
  right sample is ~20% shorter than the mean right one in every domain (FINDINGS §17).

### 2.3 Pair rules (exact; train_dpo_offline.py `pairs()`)
- Groups = samples of one (task, qid). "good" = correct AND not truncated. Prompts with no good sample are skipped.
- chosen = the shortest good sample (min `output_tokens`).
- "wrong" pair: rejected = the **longest** sample that is wrong OR truncated (if any).
- "long" pair: rejected = the longest good sample, only if its length >= 1.5 x chosen (`--long-ratio 1.5`).
- At most 2 pairs per prompt, both sharing the same chosen sample. Pairs whose prompt + longer side > 8,704 tokens are
  skipped.
- `--balance` (arm Db): per domain, keep all wrong pairs and a random subset of long pairs of the same size
  (`rng.sample(long, len(wrong))`, seed 0).
- Token ids used exactly as sampled (no re-tokenisation).
- No matching of pairs to benchmark cells, no difficulty rule, no "similar-length right > wrong" kind.

**Pair counts:**
| arm | pairs | per domain |
|---|---|---|
| D (both kinds, unbalanced) | 7,263 | math: 1,374 long / 696 wrong (FINDINGS §19); others UNVERIFIED |
| Dw (wrong only) | 2,931 | = total wrong pairs |
| Dl (long only) | 4,332 | = total long pairs |
| **Db (balanced)** | **5,862** | code 941 / ifeval 724 / math 696 / tools 570 of EACH kind (FINDINGS §20) |

### 2.4 DPO hyperparameters (train_dpo_offline.py defaults + queue/a10-train-Db.sh `--balance`)
- Loss: standard sigmoid DPO on summed sequence log-probs: `-logsigmoid(beta*(dlogp_chosen) - beta*(dlogp_rejected))`.
  **beta 0.1.** No SFT/NLL term, no length normalisation.
- Reference: the base model = same network with the adapter disabled (`model.disable_adapter()`), computed on the fly.
- LoRA: **r 32, alpha 64 (2r), dropout 0.0**, targets q,k,v,o,gate,up,down.
- Optimiser AdamW, **lr 2e-5**, weight decay 0.0; **cosine schedule with warmup = total//20 (5%)**; grad-norm clip 1.0.
- **16 pairs per optimiser step** (gradient accumulation, one pair at a time, unpadded).
- **1 epoch**: ceil(5,862/16) = 367 steps, ~1.9 h (IDEA.md).
- Precision: base weights loaded fp32, forward in fp16 autocast with GradScaler (2080 Ti has no bf16). Gradient
  checkpointing on.
- Max length 8,704 tokens. Seed 0, ONE seed. The **final adapter** is the submission; no checkpoint picking.
- Pair shuffle each epoch with `random.Random(seed)`.

### 2.5 All arms and scores
2-sample evals unless noted (FINDINGS §9-20, reports/compare_*.txt, reports/diag_runs.txt). Base 2-sample:
irrelevance 82.7 / 206 tok, multiple 78.5 / 221, simple 81.6 / 195, IFEval 58.1 / 656, MBPP+ 51.5 / 2,150 (7.9%
truncated), SVAMP 88.6 / 692.

Early arms (training on ARC / CSQA / GSM8K, SFT, LoRA r32, 2 epochs, lr 1e-4; train_sft.py defaults):
| arm | what | tokens saved | acc |
|---|---|---|---|
| A | SFT on own shortest correct trace, uncut | -2.0% | +0.04 |
| S | SFT, own trace cut at the first "committed" (unhedged) answer sentence | +6.6% (SVAMP +35%, IFEval -14%) | -0.71 |
| T | S + rule edits (no filler, no copied question, no format talk, no articles) | **-29.6%** | **-10.17** |
| V | T with articles kept | -17.3% | -6.02 |
| MS | S + 1,409 own clean general traces (Dolly/CodeAlpaca) | +5.7% | -0.16 |
| R | Claude rewrite of traces | not run: API refused every request (FINDINGS §6) | |

Domain-prompt arms (the data of 2.1, same 4 samples):
| arm | method | pairs/rows | tokens saved | acc | per benchmark (tok / acc) |
|---|---|---|---|---|---|
| S2 | SFT on shortest right sample | 9,337 rows | +4.5% | +0.90 | SVAMP +9.1/+0.40, MBPP +3.7/+0.40, IFEval -3.6/+1.57 |
| G1 | offline GRPO (Kimi length reward, Dr.GRPO adv, clip 0.2, lr 2e-5, 1 epoch, base as old policy, no KL) | | **-13.8%** | **-9.41** | SVAMP -59.3/-14.35, MBPP -30.1/-11.24 |
| D | DPO both kinds | 7,263 | **+50.2%** | **-0.41** | irr +44.5/-0.62, mult +46.1/+1.25, simple +40.3/+2.50, IFEval +47.6/+1.57, MBPP +36.5/-0.66, SVAMP +67.5/-3.85 |
| Dw | DPO wrong only | 2,931 | +23.0% | +0.81 | SVAMP +29.3/+0.95, MBPP +13.6/+1.06, IFEval +28.2/-0.37 |
| Dl | DPO long only | 4,332 | +83.6% | -13.04 | every benchmark -7 to -19 pp; MBPP mean 2,150 -> 176 tokens |
| **Db** | DPO balanced, **8 samples (full test)** | 5,862 | **+47.0%** | **-0.78** | see below |

Db full test (reports/compare_Db-s8.txt; base-s8 vs Db-s8):
| benchmark | base acc / tokens | Db acc / tokens | tokens saved | acc change |
|---|---|---|---|---|
| BFCL irrelevance | 82.8 / 214 | 88.9 / 115 | +46.1 | **+6.04** |
| BFCL multiple | 76.6 / 224 | 75.4 / 130 | +41.9 | -1.12 |
| BFCL simple | 82.2 / 193 | 80.8 / 125 | +35.3 | -1.41 |
| IFEval | 57.8 / 694 | 57.5 / 314 | +54.7 | -0.28 |
| MBPP+ | 52.1 / 2,115 | 50.8 / 1,418 | +33.0 | -1.22 |
| SVAMP | 88.4 / 684 | 86.0 / 298 | +56.4 | **-2.41** |
| **average (4, BFCL pooled)** | | | **+47.0** | **-0.78** |

Truncation fell (MBPP+ 7.3% -> 3.8%, IFEval 3.5% -> 0.6%); median SVAMP thinking 382 -> 197; none of the 22,072
outputs thinks under 20 tokens (IDEA.md).

### 2.6 His findings and failures (FINDINGS §11-20, IDEA.md)
- **Compress by selection, not rewriting.** Any edit to the model's own text (T, V) caused loops and truncation
  (MBPP+ truncation 7.9% -> 27-45%); training loss on edited text 0.41-0.49 vs 0.23-0.26 on own text.
- **SFT on own short samples barely moves length** (A -2%, S2 +4.5%): "learning what *not* to write is what moves the
  length."
- **Long pairs carry the compression and alone collapse the model**: in Dl training 100% pair accuracy at loss 0.0001
  while both log-probs fall (DPO pushes everything down).
- **Wrong pairs are an anchor** and also shorten (wrong samples are often long / looping).
- **Why SVAMP lost in D:** math had 1,374 long vs 696 wrong pairs (base right 85% on math) -> weakest anchor. Balance
  gave SVAMP -3.85 (2-sample) -> -2.41 (8-sample); he notes the sample counts differ so it is indicative only.
- **Offline GRPO fails**: 47-49% of tokens clipped by end of training, truncation up (SVAMP 1.8% -> 10.8%); he thinks
  the negative push on long samples sends probability into looping.
- **Selection caveat** (his own): Db was designed after seeing D on the same benchmark questions.
- Not run: 2-sample eval of Db; RPO (DPO + SFT on chosen); a second DPO round on fresh samples from Db; online GRPO.

### 2.7 Compute
One RTX 2080 Ti 11 GB, Ryzen 9 3900X, 32 GB RAM. Sampling ~25 h; DPO ~2 h; full 8-sample eval ~7 h (adapter) / ~10 h
(base). API spend ~$0.01. ThinkingCap-27B run locally (llama.cpp) only for analysis, never as training data (IDEA.md).

### 2.8 Check against our EXPERIMENT_LOG
Our entry of 1 Nov ~01:30 (EXPERIMENT_LOG.md lines ~3665-3687) matches the sources on every number I checked (data
counts, pair rules, 5,862 pairs, beta, LoRA, lr, warmup 5%, 16 pairs/step, fp16, 8,704, one seed, final adapter,
per-benchmark Db numbers, ablations). Additions not in our log: LoRA dropout 0, weight decay 0, grad clip 1.0, fp32
master weights, 367 steps (log says ~366), the source quotas, and the 26 Oct note's "prompts drawn evenly from SVAMP
deciles" applies to every domain (each to its own benchmark).

---

## 3. Herz1c, WhereTheThinkingStopsMattering (43.4% / -0.97) and DoesLearningToStopScale (1.7B)

Main files: [HERZ] `README.md`, `IDEA.md`, `stopping/train_grpo.py`, `stopping/train_lora.py`,
`stopping/find_stop_points.py`, `stopping/verified_examples.py`, `stopping/give_up_examples.py`,
`stopping/constraint_tasks.py`, `stopping/generate_rollouts.py`, `stopping/train_dpo.py`. Author: Jan Herza (IDEA.md).

### 3.1 Data
- Sources: GSM8K train, KodCode-Light-RL-10K, Glaive function calling v2, databricks-dolly-15k (README, IDEA.md).
- 8-gram overlap check against every eval question; 11 Glaive rows removed. 39 KodCode tasks share a generic function
  name with MBPP+ (none shares > 54% of words) (IDEA.md).
- Instruction data: Dolly prompts + five format constraints IFEval does NOT use (no digits, every sentence ends "!",
  Markdown table, at least N emoji, end with a question); one or two per prompt; prompts sharing >= 15% of an eval
  question's 8-grams dropped; default 3,000 prompts (constraint_tasks.py).
- "Grey zone" disclosed: 2,500 Dolly prompts with *other* wrappers (`<reply>` tags, brackets, fenced block) added to the
  GRPO instruction pool only, after reading IFEval's "wrap in double quotes" loops (IDEA.md).
- Rollout defaults: 1,500 questions per domain x 4 samples, eval sampling (generate_rollouts.py; code_tasks.py /
  tool_tasks.py `--train 1500`). Exact SFT mix row counts: UNVERIFIED (not in the repo).

### 3.2 Stage 1, stop-point distillation (SFT)
- For each rollout (finished or truncated), binary search over paragraph boundaries ("\n\n") for the earliest one where
  a forced `</think>` gives a verified-right answer: 2 probe answers must both "know" it, then 4 more, >= 3 must know
  (find_stop_points.py). Training example = own reasoning up to the cut + `</think>` + the shortest of the 6 sampled
  answers that passes the strict grader. Rollouts of never-solved questions skipped.
- **Rule 1, code: stop only after the function is drafted** (`verified_examples.py --mode draft`: a `def` line or a
  closed code block before the cut; then verified with 6 answers, >= 5 must know, one must pass strict; walk forward up
  to 6 paragraphs). Plain oracle stops: drafting in reasoning 98% -> 8%, MBPP+ -5.4 pp.
- **Rule 2, give up with a best guess**: rollouts that hit the cap are cut at the first paragraph boundary at/after
  **2,048 thinking tokens** and end with the model's own forced answer (right if it knows, else a well-formed wrong
  guess, never a refusal); max 600 examples, one per question (give_up_examples.py). Capped MBPP+ responses 49 -> 6 per
  756 at no accuracy cost.
- **Rule 3, instruction stops** on the constraint data above: IFEval median reasoning 296 -> 134 tokens, no loss.
- SFT trainer (train_lora.py): LoRA **r 16, alpha 32, dropout 0.05**, all 7 projections; lr **1e-4**, cosine with 5%
  warmup; **2 epochs**; **16 sequences per step**; max length 5,120; loss on the completion only.

### 3.3 Stage 2, GRPO (train_grpo.py; README "Reproduce": `--steps 50 --prompts 24`)
- **Reward:** `R = 1[correct] * (1 - alpha * L / m_domain)`, **alpha 0.2**; L = generated tokens of the answer (thinking
  + reply); m_domain = mean length of that domain on the **first batch**, then frozen. Correct = strict grader AND
  finish_reason == "stop" (a capped answer earns 0). Wrong earns 0 whatever its length.
- **Advantage:** R minus the group mean, no division by std.
- **Update:** one on-policy update per batch (ratio = 1, plain policy gradient), loss = -A * sum logprob / total tokens
  of active samples (token-averaged over the batch); samples with |A| < 1e-6 skipped; grad clip 1.0; AdamW lr **1e-5**,
  weight decay 0; **no KL**, no reference.
- **Group 8** samples per prompt; **24 prompts per step**; **50 steps** (= 9,600 rollouts). Domain mix
  **math 0.5, code 0.2, tool 0.15, instruction 0.15**. Generation caps: math 4,096, code 6,144, tool 2,048,
  instruction 4,096. Eval sampling settings. bf16. vLLM shares the GPU (35%).
- Starts from the stage-1 adapter (mix G), LoRA r16 continued. One run, no hyperparameter search. Checkpoint: the final
  adapter at step 50 (every 10th step saved); UNVERIFIED whether an earlier step was compared.

### 3.4 Results
Final (README; run 1 = submitted results.json):
| benchmark | tokens saved | acc change |
|---|---|---|
| MBPP+ | **64.7** | -1.72 |
| IFEval | 38.9 | -0.65 |
| BFCL simple / multiple / irrelevance | 25.8 / 35.3 / 33.9 | -1.00 / -0.56 / -0.83 |
| SVAMP | 30.7 | -1.45 |
| **average** | **43.3** (run 2: 43.5) | **-1.17** (run 2: -0.77) |
Two-run means ([HERZ17] figures/scaling_report.md): BFCL +30.8/-0.94, IFEval +38.9/-0.61, MBPP +64.9/-0.98, SVAMP
+30.8/-1.33, average **+43.4 / -0.97**. Only 6,092 of 22,072 responses identical between the two runs (vLLM
non-determinism) -> run-to-run spread ~0.4 pp in accuracy.

Checkpoints along the way (full evaluations; README "Along the way"):
| checkpoint | tokens saved | acc |
|---|---|---|
| SFT on shortest correct own answer (reference, quick eval) | 2.3 | -0.53 |
| Mix C: oracle stops on math, code, tools | 37.2 | -1.80 |
| Mix E: + code drafts, give-up | 29.4 | -0.45 |
| Mix G: + instruction stops | 34.6 | -0.85 |
| **GRPO from mix G** | **43.3 / 43.5** | **-1.17 / -0.77** |

Held-out (analysis/heldout_eval.py): HumanEval 164x4 +65.0% / -1.22 [-5.03, +2.74]; GSM8K test 500x2 +44.3% and
+35 pp strict (format learning, not generalisation, by his account).

Difficulty bins (0.6B base solve rate, [HERZ17] scaling_report.md): SVAMP loses on easy questions (8/8: -2.18, saved
only 21.7%) and gains on hard (1-4/8: +7.97, 45.9% saved); MBPP loses on 5-7/8 (-4.65).

### 3.5 What failed (2-sample quick evals, README/IDEA tables)
- DPO on "later correct vs earlier wrong" stop pairs (train_dpo.py: beta 0.1, NLL weight 0.2, lr 1e-5, 8 pairs/step,
  reference = starting LoRA): the learned stop collapsed (MBPP 762 -> 1,769 tokens).
- Stops after a "check" cue: cues fired on restating the test case; no gain.
- Math stops only after the answer is computed: SVAMP -6.1 pp. Later stops by fixed or difficulty margin: less saving,
  same SVAMP loss. Math give-up at 1,024 tokens: SVAMP saving x2 at -1.5 pp. Distractor sentences in math: still -5 pp.
- **Scaling the adapter to 0.75 / 0.5: the stop disappears entirely** ("a switch, not a dial").
- Cutting looping wrapper replies: only 7 usable examples.
- Every SFT stop-data variant that shortened SVAMP by 19-48% cost 1.5-6.1 pp; ~1 pp per 8 points of saving.

### 3.6 His findings
- The metric is a ratio of sums, so the capped tail dominates: 7.9% of MBPP+ responses hit the cap and hold 30.6% of
  MBPP+ tokens; 98% of them cut off inside `<think>` (IDEA.md Motivation).
- Longer is worse: accuracy from shortest to longest fifth MBPP+ 80% -> 14%, SVAMP 96% -> 69%.
- With 60% of its GSM8K thinking the model answers right 80% of the time (87% with all).
- "Where an oracle says the answer is right is not where the model knows it."
- RL learned *when* re-checking pays: SVAMP saving 14% -> 31% at the same accuracy (-1.20 -> -1.21).
- Quick 2-sample evals differ 1-2 pp per benchmark between identical-data runs; only 8 samples ranks close checkpoints.

### 3.7 1.7B variant ([HERZ17] README)
Same recipe, no re-tuning, difficulty-matched pools (mix in MATH levels 3-5 and KodCode medium/hard so the 1.7B base
solves the same share as 0.6B did: math 70.8%, code 51.9% -> 40% hard math, 80% hard code). Result (2-run mean): +47.4% /
-0.14 (MBPP +55.7/+4.12, IFEval +50.3/-0.23, BFCL +33.7/-0.84, SVAMP +47.4/-3.60). SVAMP's loss is ~90% answer format
("Final answer: 5 + 3 = 8"), BFCL irrelevance -4.7. GRPO resumed from step 30 with a fresh optimiser after an
interruption. ~70 GPU-hours.

### 3.8 Compute
One RTX 5060 Ti 16 GB (WSL2). ~65 GPU-hours for everything; final pipeline alone ~12 GPU-hours. PyTorch 2.11, vLLM
0.23, transformers 4.57.1, peft 0.17.1 (IDEA.md).

---

## 4. matousek-martin, qwen3-self-replay (256: 31.2% / +0.21; 850: 41.6% / -1.17)

Main files: [MAT] `README.md`, `IDEA.md`, `docs/METHOD.md`, `configs/self-replay-plan.json`,
`provenance/training-data.json`, `studies/README.md`, `studies/self-replay/results/checkpoint-*/comparison.json`,
`artifacts/ablation-results/{no-replay,no-length}/comparison.json`, `studies/self-replay/training/references.json`.

### 4.1 Data (provenance/training-data.json)
- Raw sources: GSM8K train (7,473), APPS (5,000), allenai/tulu-3-sft-personas-instruction-following (29,980),
  Salesforce xlam-function-calling-60k (60,000) + Glaive function-calling v2 (no-call examples).
- After screening: train 41,857 rows = tools 30,801 / math 6,134 / coding 3,396 / instructions 1,526; development 2,589;
  confirmation 2,821. APPS difficulty in train: introductory 1,925 / interview 1,224 / competition 247.
- Instruction checks actually present: no_comma 382, lowercase 384, title 277, quotation 216, uppercase 350,
  postscript 3 (only 6 constraint types; 26,370 Tulu rows dropped as unsupported).
- **No-call (irrelevance) examples: 79 in train, 2 in development** (self-replay-plan.json `no_call_counts`).
- Screening: normalised exact + 5-token-shingle containment, threshold 0.8, against the benchmark prompts.
- Frozen plan: 3,840 questions balanced across the 4 domains (plan.json `groups`), 256 dev questions; the trajectory
  reached 1,280 questions.

### 4.2 Method (docs/METHOD.md, IDEA.md, self-replay-plan.json)
- **Reward:** `r_i = c_i * (1 - 0.1 * min(L_i / R_d, 1))`; correct >= 0.9, wrong = 0; the penalty saturates at the
  domain reference. R_d = mean generated length (incl. failures) of the pinned base on 4 answers x 8 training questions
  per domain, frozen: math 1,295, coding 2,044, tools 358, instructions 634 tokens (training/references.json).
- **Advantage:** `A_i = r_i - mean(r)` over the group of 4, NOT divided by std (Dr.GRPO, `scale_rewards: false`).
- **Policy loss:** `-1/(4*2048) * sum_i sum_t A_i * exp(l - stopgrad(l))` (fixed denominator = group size x max
  completion length), one update per group, no PPO epochs, **no KL**. Log-probs at temperature 0.6.
- **Replay buffer:** admit a question only if **all 4 answers are verified right**; store its shortest completion;
  replace only with a shorter one; capacity **128 questions**, evict oldest insertion; admission before sampling, so the
  new answer can be replayed at once; one stored completion sampled uniformly per group (separate RNG). Not re-verified
  later.
- **Replay loss:** completion-only mean NLL at temperature 1.0, weight **0.1**: `L = L_policy + 0.1 * L_replay`. All-wrong
  groups still get a replay update; step skipped only if neither signal exists.
- **Hyperparameters:** group **4**; LoRA **r 16, alpha 32**; AdamW **lr 1e-5**, weight decay 0; grad clip 1.0; seed 17;
  generation T 0.6, top-p 0.95, top-k 20, **cap 2,048 tokens** (shorter than the eval's 4,096 / 8,192); one question per
  "iteration" (checkpoint number = questions processed); saves every 32 questions. 2 x A100-SXM4-80GB for training
  (the 4 answers split across 2 GPUs); evals on RTX 4090 with one reused base run.
- Schedule shape / warmup: UNVERIFIED (not in the plan file).

### 4.3 Results (comparison.json per checkpoint; tokens saved / acc pp)
| ckpt | total | BFCL irr | BFCL mult | BFCL simple | IFEval | MBPP+ | SVAMP |
|---|---|---|---|---|---|---|---|
| 128 | 27.10 / +0.03 | 23.7 / -4.11 | 21.2 / +2.37 | 21.8 / +0.25 | 25.0 / -0.23 | 26.5 / +0.60 | 34.1 / +0.24 |
| **256** | **31.23 / +0.21** | 25.8 / -4.17 | 26.1 / +3.31 | 23.5 / -0.03 | 25.9 / +1.02 | 31.3 / +0.86 | 41.6 / -0.62 |
| 576 | 36.85 / -1.35 | 29.2 / -6.20 | 28.7 / +3.94 | 25.9 / -1.47 | 33.7 / -0.90 | 36.7 / -1.52 | 47.6 / -1.46 |
| 768 | 39.54 / -1.16 | 31.9 / -6.56 | 30.2 / +3.31 | 27.6 / -1.50 | 36.8 / +0.07 | 40.7 / -0.56 | 49.3 / -2.35 |
| **850** | **41.60 / -1.17** | 33.4 / -6.87 | 30.4 / +2.44 | 28.4 / -2.00 | 38.1 / +0.18 | 42.3 / -1.26 | 53.2 / -1.29 |
| 1024 | 43.87 / -1.77 | 34.3 / -8.70 | 32.9 / +1.75 | 29.1 / -1.53 | 40.5 / +0.14 | 44.2 / -1.85 | 56.3 / -2.55 |
| 1280 | 46.58 / -2.35 | 36.4 / -10.21 | 35.0 / +1.31 | 31.0 / -2.19 | 41.4 / -0.07 | 49.4 / -2.45 | 58.6 / -3.24 |
Base MBPP+ truncation 8.6% -> 3.7% at 256, 2.5% at 1280; SVAMP 1.7% -> 0.2%.

**Ablations** (one seed, same start, compared with the combined run at the same iteration; README):
| arm | total | per benchmark |
|---|---|---|
| no replay, 128 | **1.85 / +0.13** (combined: 27.10 / +0.03) | all benchmarks 0-3.5% saved |
| no length reward, 256 | 27.52 / +0.37 (combined: 31.23 / +0.21) | BFCL irr -3.39, SVAMP 34.7 / +0.60, MBPP 25.6 / +1.26 |

### 4.4 Other experiments he ran (studies/README.md; mostly different protocols)
- **ShorterBetter** (arXiv 2504.21370, SOL reward = alpha*correct - 0.001*|L - shortest correct in group|; full weights,
  lr 1e-6, KL 0.001, 8 generations/prompt, DeepScaleR math data per protocol.json): full evals at iterations 5-30
  for alpha 2 / alpha 5: e.g. iter 10: 24.1 / -0.89; iter 15: 35.9 / -1.58; iter 20: 46.2 / -2.64; iter 30: 63.7 /
  -4.70 (alpha 2). Very fast compression, accuracy falls in step. (Whether the run matched protocol.json exactly:
  UNVERIFIED.)
- Replay selection (more permissive admission, 3 seeds): outputs got longer (-6.35% "savings"). Adaptive replay:
  -4.07%, +0.59 pp. On-policy distillation: longer outputs. Teacher replay pilot: 47.8% saved, -12.4 pp on its
  validation. Coding A/B/C and curriculum branches: no gain over parent.

### 4.5 His findings / caveats
- Replay drives most of the shortening; the length reward alone does very little in his setup (Inference from the
  no-replay arm; he says the no-length arm still replays the shortest answer, so length preference is not fully removed).
- **BFCL no-call is the hidden regression**: -4.17 at 256, -6.88 at 850 (only 79 no-call training rows, 2 in dev).
- The 4/4 gate may starve hard coding questions; the 2,048 training cap may make it worse (hypotheses, not tested).
- One seed; checkpoint selection used the benchmarks; CIs do not include seed variance.

---

## 5. basta, Lightning McQwen3 (43.2% / -0.85 weights only; 47.3% / -0.88 with test-time gate)

Main files: [BASTA] `README.md`, `IDEA.md`, `scripts/reproduce.sh`, `mcqwen/{distill,sft,oracle,dpo,correctness,gate}.py`,
`utils/train/prepare_data.py`. Commits 14 Oct 2026.

### 5.1 Data
Only the challenge's suggested `data/train.jsonl` (prepare_data.py): **GSM8K, Skywork-OR1 math (integer answers,
DeepSeek-R1-Distill-1.5B difficulty <= 8), CommonsenseQA, ARC-Easy**. No code, tool or instruction training data in
any weight-training step. The gate alone uses 1,500 ToolACE prompts. (Our log said "math = GSM8K + Skywork"; correct,
and the oracle pairs also use CSQA and ARC.)

### 5.2 Stages (each a LoRA r16 / alpha 32 / dropout 0, all linear, continued from the previous; README, sft.py)
1. **Prompt distillation SFT.** Chain-of-Draft system prompt ("only keep a minimum draft for each thinking step, with 5
   words at most" + two worked examples; found by an overnight prompt search) answers 5,000 questions per source x 2.
   One random correct trace per question, prompt removed, traces mentioning the examples dropped, math answers must
   parse under SVAMP's grader -> **11,247 SFT targets**. SFT from base: **lr 1e-4, 1 epoch, batch 16, cosine with 5%
   warmup**, loss per target token. Score: **26.5% / -1.27**.
2. **On-policy truncation oracle.** The distilled model answers 2,500 *other* questions per source x 4. Every correct
   trace cut at the sentence ends nearest 10%, 20% ... 90% of its thinking, closed with `</think>`, re-answered 4 times
   (197k cuts; re-answer cap 1,024). Cut "safe" = all 4 re-answers right; "stable run" = safe cuts that reach the end.
3. **Oracle DPO.** Pairs (oracle.py), at most one per question and kind, both sides continue the same trace:
   - *shorten*: stable run's **second** cut + its shortest right re-answer > the full trace (second, not first, as a
     margin against lucky cuts);
   - *guard*: the same chosen > the unsafe cut just before the run + a wrong re-answer;
   - *hard* (no stable run): full trace > its unsafe cut nearest half the thinking + a wrong re-answer.
   10,000 drawn, **40 / 40 / 20** by kind, evenly across sources -> **9,034 pairs**. DPO from the distilled adapter
   (reference = starting adapter, precomputed): **beta 0.1, lr 2e-5, batch 16, cosine with 10% warmup, 1 epoch = 561
   steps planned, STOPPED at step 100** (`--stop-at 100`, LR schedule of the full run). Score: **42.5% / -1.45**.
4. **Anchored correctness DPO.** From step 2's traces: own right math trace (GSM8K, Skywork) > own wrong one, same
   question, closed and untruncated, lengths within **+-25%**, up to 2 pairs per question on disjoint traces ->
   **732 pairs**. DPO + **NLL on the chosen side (mean token NLL, weight 1)**, 1 epoch = **42 steps**, beta 0.1,
   lr 2e-5. Score: **43.2% / -0.85** (weights-only result).
5. **Think gate (test-time, in evaluate.py).** L2 logistic probe on the layer-21 hidden state at the end of the prompt
   + "<think>\n"; labels = whether 4 sampled calls with thinking and 4 without agree; 1,500 ToolACE prompts; threshold
   fires on half of ToolACE; when it fires on a tools prompt the reply starts with an empty think block (forced tokens
   counted). Only changes BFCL: 25.3% -> 44.6% saved. Headline 47.3 / -0.88 combines ungated SVAMP/MBPP/IFEval runs with
   a gated BFCL run; a full gated rerun read 46.6 / -1.33. **Note:** this is a test-time change to evaluate.py, not in
   the weights.

DPO tripwire: scores 64 held-out pairs every 25 steps and warns when P(`<|endoftext|>` right after `</think>`) rises
(the signature of the collapse below) (README, dpo.py).

### 5.3 Per-benchmark (README, with gate on BFCL)
SVAMP **66.3 / -3.8**, MBPP+ 39.1 / -1.9, IFEval 32.0 / +1.4, BFCL 44.6 / +0.8 (gated; ungated BFCL 25.3% saved,
accuracy UNVERIFIED). **Inference:** without the gate the BFCL saving (25%) is the weak spot; SVAMP carries the average.

### 5.4 What did not work (IDEA.md)
| idea | result |
|---|---|
| Truncation oracle as a measurement on base traces (hindsight) | 34.5% / +1.08 (BFCL 55%) |
| SFT on the oracle's cuts | 11.4% / -0.34: "safe but timid"; more epochs collapse accuracy |
| Shortest-over-longest DPO pairs alone | 90% saved, -27 pp within 50 steps |
| DPO on oracle pairs from the BASE model, to the end | collapsed ~step 200 (`<|endoftext|>` right after `</think>`); with anchor: 47.8% / -4.15 |
| Right>wrong pairs at matched length without NLL | -5.2 pp (with NLL: +0.6 pp at the same length) |
| SFT on the full trace when no cut was stable | 5% saved |
| More SFT data (15x) or a second cutting round | no further shortening |
| Fine-tuning the oracle-DPO model on distilled math | 28.6 / -1.09 (2-sample) |
| Merging the two LoRAs | 51.0 / -3.93 (2-sample) |
| Gating math/code too | skipping easiest 5% of SVAMP: -1.1 pp |
| Gate trained into the weights with DPO | -3.3 pp (called tools on irrelevance) |
| A larger model choosing cut points | ~40% / -0.8 (pilot) |
2-sample evals ran 0.5-1 pp optimistic against 8 (his note). Side effects: more sycophancy, invents missing context.

### 5.5 Compute
One RTX 4070 SUPER 12 GB for all DPO (~5.5 s/step oracle DPO, 11-14 s correctness DPO), gate, most evals (~1.5 h per
8-sample eval); sampling and oracle on MetaCentrum (8 shards on 40 GB GPUs, a few hours each); SFT on a 24 GB GPU.

---

## 6. Brief: mrtineu (28.7% / -1.71) and marek-krcal

### 6.1 mrtineu / Martin Senkyr ([MRT] README.md, IDEA.md, data.md, code/thinkingcap/config.json)
- **Targets:** student answers a question; Qwen3-8B sees one correct student trace (never the gold) and writes hints on
  what to skip (hints containing the answer are rejected and re-asked, 3 attempts); student answers 4 more times with
  hints in the prompt; shortest correct answer = target; trained on the plain question. Baseline 4 samples, keep
  questions with >= 2 right; 13-gram overlap filter. Training data source: UNVERIFIED (seed config only; probably the
  challenge's train.jsonl, since hint checks mention option letters).
- **Architecture:** a copy of layer 13 inserted after it with attention off, zero-init output, only 64 MLP channels
  trained (197,632 params); 29-layer plain Qwen3 checkpoint. lr 3e-4; batch 32 sequences; warmup 3%; 2 epochs planned;
  best at step 170 = half an epoch.
- **Results (data.md, full test):** extra-mlp-64 @170: **28.7 / -1.71** (SVAMP 40.4/-2.26, MBPP 15.7/-1.98, IFEval
  31.0/-1.06, BFCL mult 28.2/-4.31). LoRA r8/a16 on q,v (1.1M params, lr 2e-4) @170: 22.5 / -1.58, falling to 13.9 at 1
  epoch. Control (no hints, shortest own answer = 0.65x length) LoRA: 0.1-2.3% saved. Full inserted block (15.7M):
  25.1 / -2.16; full-width MLP (9.4M): 25.8 / -1.92; same MLP after layer 20: 19.5 / -1.11; after layer 17: 21.9 / -1.64.
  Forced first token "To" (decoding trick, not submittable): 53.6 / -7.68; "First": 29.2 / -2.74.
- **Findings:** hint-coached targets are learnable, plain shortest-of-4 targets are not; smaller adapters compressed
  more; best always at half an epoch, later checkpoints compress less. "Wait" after restarts 72% -> 62% (layer probe).
- Compute: Modal H100, ~35-38 GPU-hours, ~$150; training < 1.5 h total.

### 6.2 marek-krcal ([KRCAL] docs/experiments/preferences.md, tc/objectives.py, tc/preferences.py)
- Objectives: SimPO (length-normalised log-prob margin, **beta 2, gamma 1**, no reference) and DPO (summed log-probs,
  frozen adapter-disabled reference, **beta 0.1**; lower-KL control **beta 0.05**).
- Data: 2,287 GSM8K groups x 8 answers ("original" pairs); planned external mixture of 2,000 audited questions (575
  code: 158 MBPP + 417 KodCode; 718 IF_multi_constraints; 707 ToolACE), 1,000 groups selected and added.
- Pairings: `correct-longer` = shortest correct positive vs strictly longer wrong and >= 1.5x longer correct;
  `all-longer` = every negative >= 1.5x.
- Recipe: 982 optimiser updates, 7,852 four-response views; checkpoints 737 / 860 / 982 screened; Colab A100 for
  generation, L4 for training.
- Numbers: none published. The plan's success bar ("saves at least 29.36% ... loses at most 2 pp ... not dominated by
  Qwen/737/982") suggests an earlier SimPO point near 29% saved: **UNVERIFIED**.

---

## 7. Across entries

### 7.1 Facts that recur (with sources)
1. **Train only on the model's own text.** bida: rule-edited own traces failed (-6 to -10 pp, loops) while selected own
   traces did not (FINDINGS §15). basta: "Fine-tuning on another model's thinking did poorly"; a larger model choosing
   the cuts reached only ~40 / -0.8 (IDEA.md). Herz1c: every example is own reasoning, only the stop point moves.
   matousek: all replay targets are student answers; teacher replay -12.4 pp. mrtineu: the teacher only gives hints,
   the student writes the target.
2. **Imitating your own shortest right answer barely shortens.** bida S2 +4.5%, A -2%; Herz1c +2.3%; mrtineu control
   0-2%; basta SFT on oracle cuts 11.4%. Every entry above 40% has a contrastive or RL signal (DPO rejected side,
   advantage) on top.
3. **Pure "shorter beats longer" collapses accuracy fast.** bida Dl: 83.6% / -13.0. basta shortest>longest: 90% / -27 pp
   in 50 steps. ShorterBetter (matousek): 63.7% / -4.7 at iteration 30.
4. **Every working recipe has an explicit accuracy anchor next to the length push:** wrong-sample pairs (bida, balanced
   1:1 per domain); guard + hard pairs (basta, 40/40/20); reward 0 for any wrong or capped answer (Herz1c, matousek);
   NLL on right traces (basta correctness DPO, weight 1; matousek replay NLL, weight 0.1). Unanchored right>wrong DPO lost
   5.2 pp (basta); DPO with both sides sharing an answer scaffold collapsed into empty answers (basta).
5. **Dose matters and the best point is early.** basta stopped oracle DPO at step 100 of 561; matousek's best balance at
   256 of 1,280 questions, then nearly linear (256 -> 1280: +15.4 points saved for -2.56 pp, ~ -0.17 pp per point); mrtineu best at half an epoch;
   Herz1c only 50 GRPO steps x 24 prompts. bida is the exception: 1 full epoch, final adapter, but with a 1:1 anchor.
6. **Per-domain normalisation / balancing.** bida caps long pairs per domain at the wrong-pair count; Herz1c divides
   length by each domain's mean length; matousek uses a per-domain frozen reference. Herz1c's reason: the metric is a
   ratio of sums per benchmark.
7. **SVAMP is where everyone pays.** SVAMP accuracy: bida -2.41, Herz1c -1.33, basta -3.8, matousek -0.62 (256) to -1.29
   (850), mrtineu -2.26. Herz1c: the 0.6B's re-checks after the answer are partly useful; SFT stop data cost ~1 pp per 8
   points of SVAMP saving, RL broke that trade.
8. **BFCL irrelevance swings with training data on "no call".** bida (45% of tool prompts are no-fit/decline): +6.04.
   matousek (79 no-call rows of 30,801 tool rows): -4.2 to -10.2. Herz1c (Glaive): -0.83. basta's gate trained into
   weights called tools on irrelevance (-3.3 pp).
9. **The truncated tail is a large share of tokens.** MBPP+ base: 7.9% capped, 30.6% of MBPP+ tokens (Herz1c). All top
   entries cut truncation: bida MBPP+ 7.3 -> 3.8%, matousek 8.6 -> 3.7%, Herz1c capped 49 -> 6 per 756 (give-up rule).
10. **Offline policy gradient on stored samples failed** (bida G1: -13.8% / -9.4, half the tokens clipped); **online**
    GRPO with a small lr and few steps worked (Herz1c, matousek).
11. **Run-to-run noise is ~0.4 pp** on the full test with the same adapter (Herz1c 43.3/-1.17 vs 43.5/-0.77); 2-sample
    evals are 0.5-1 pp optimistic (basta) and differ 1-2 pp per benchmark (Herz1c).

### 7.2 Hyperparameters that recur
| setting | values seen |
|---|---|
| DPO beta | 0.1 everywhere (bida, basta x2, Herz1c's failed DPO, marek); marek tests 0.05 |
| DPO lr | 2e-5 (bida, basta); 1e-5 (Herz1c's failed DPO) |
| DPO batch | 16 pairs/step (bida, basta); 8 (Herz1c) |
| DPO reference | base with adapter off (bida, marek); the starting adapter (basta, Herz1c) |
| DPO NLL anchor | 0 (bida); 1.0 on the right>wrong stage only (basta); 0.2 (Herz1c, failed) |
| DPO length | 1 epoch (bida 367 steps; basta correctness 42 steps) or stopped early (basta 100/561) |
| LoRA | r32/a64 (bida); r16/a32 (Herz1c, basta, matousek); r8/a16 q,v only (mrtineu, weakest); all 7 linear layers in the top 4 |
| LoRA dropout | 0 (bida, basta); 0.05 (Herz1c SFT) |
| SFT lr / epochs | 1e-4, 1-2 epochs (bida, Herz1c, basta) |
| RL lr | 1e-5 (Herz1c, matousek); 1e-6 full weights (ShorterBetter) |
| RL group size | 8 (Herz1c), 4 (matousek) |
| RL KL | none (Herz1c, matousek) |
| RL advantage | r - group mean, no std division (Herz1c, matousek; bida's offline G1 too) |
| Length penalty size | alpha 0.2 x L/domain mean (Herz1c), 0.1 x min(L/ref,1) (matousek): small next to the correctness reward of 1 |
| Warmup | 5% (bida DPO, Herz1c SFT, basta SFT), 10% (basta DPO), cosine everywhere |
| Samples per prompt for offline data | 4 (bida, basta oracle, Herz1c rollouts, mrtineu) |
| Pair ratio thresholds | long >= 1.5x (bida, marek); right/wrong within +-25% length (basta) |
| Prompt count | ~11k (bida), ~10k oracle questions + 20k distill (basta), ~1.5k per domain (Herz1c), 1,280 processed (matousek) |
| Seeds | 1 everywhere |

### 7.3 Where each entry is strongest (fact) and what that suggests (inference)
Per-benchmark tokens saved at the submitted point: SVAMP basta 66 / bida 56; IFEval bida 55; MBPP+ Herz1c 65; BFCL
basta-with-gate 45, bida 35-46 per subset. **Inference:** the four entries each found a different benchmark's lever
(bida: per-domain DPO with matched data; Herz1c: give-up + draft-before-stop on code; basta: terse-draft SFT on math);
if the best per-benchmark ratios could coexist (0.34 / 0.45 / 0.35 / 0.55), the geometric mean would be ~59% saved (accuracy not accounted for).
Nobody has shown they do coexist.

### 7.4 What does not seem to work (facts)
- Editing or rewriting traces (bida T/V, basta "another model's thinking").
- SFT alone on own shortest answers or oracle cuts.
- Length-only preference pairs, or right>wrong pairs without an NLL anchor.
- Offline GRPO on stored samples.
- Scaling a learned-stop LoRA down (Herz1c: the stop disappears) or merging LoRAs (basta: -3.9 pp).
- Permissive replay admission / adaptive length targets (matousek: outputs got longer).
- Bigger adapters for the same data (mrtineu: smaller MLP compressed more; LoRA r8 q,v compressed least).

### 7.5 Inferences (mine, not tested by them)
- **The accuracy anchor seems to matter more than the length signal.** In matousek's runs removing the length reward
  cost only 3.7 points of saving, while removing the replay of shortest right answers removed almost all of it; in bida
  the 1:1 anchor kept accuracy at a 1-epoch dose. Both suggest the "be right, in your own short form" signal is doing a
  lot of the work.
- **One round of DPO with a balanced anchor is the cheapest strong point** (bida: ~25 h sampling + 2 h training on an
  11 GB card for 47 / -0.78); RL entries spend more compute for similar or lower points.
- **A large share of the remaining accuracy loss is in two places:** SVAMP (all entries) and BFCL irrelevance (data
  dependent). The 1.7B Herz1c result says SVAMP loss can be mostly answer *format* rather than arithmetic, which is
  worth checking in any SVAMP loss.
- **Benchmark-shaped prompts help but are not required:** basta reached MBPP 39% / IFEval 32% with only math and
  multiple-choice training data, so terse reasoning transfers across domains to a degree; bida's matched data gave
  higher IFEval (55%) and BFCL savings.

### 7.6 Gaps
- Herz1c: exact row counts of SFT mixes C/E/G and the GRPO checkpoint choice (step 50 assumed).
- matousek: LR schedule; whether ShorterBetter runs used protocol.json as written.
- basta: per-benchmark numbers for each stage and ungated BFCL accuracy (the blog may have them; not read).
- mrtineu: training question source and count.
- marek-krcal: any scored result.
- bida: per-domain pair counts of arm D other than math.
