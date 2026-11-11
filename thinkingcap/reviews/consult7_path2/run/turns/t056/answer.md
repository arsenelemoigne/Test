```python
full_report = """# PATH 2: Post-DPO Reinforcement Learning Strategy for Qwen3-0.6B

---

## 1. Forensic Diagnosis: Why the Team's Past RL Barely Moved (or Collapsed)

Every RL run in `EXPERIMENT_LOG.md` failed for specific, compounding mathematical defects in `rl_train.py` and `track_thinkprune.py`. None of those runs tested a properly normalised, budget-aware on-policy RL update on top of a DPO-repaired model.

### 1.1 Why `grA`, `grB`, and `grC` (3 Nov) Barely Moved and Drifted Longer
- **Results on Screen-720:** `grA` (step 50 / 100 / 150): `+1.5% / -0.96` $\\to$ `-2.3% / -1.10` $\\to$ `-5.8% / -0.83`. `grC` (KL = 0): `-0.3% / +0.43` $\\to$ `-3.0% / -1.02` $\\to$ `-6.7% / -0.37`. `grB` (4x penalty, $\\lambda_{\\max}=0.20$): `-2.0% / -0.50` $\\to$ `+1.2% / +0.11` $\\to$ `-0.6% / -0.72`.
- **Cause 1 — Microscopic length contrast in `--reward lenpen` (`rl_train.py` L764–770):**
  The reward was $R_i = \\mathbb{1}[\\text{correct}_i] \\cdot (1 - \\lambda_t \\min(1, L_{\\text{think}, i}/2048)) + 0.1 \\cdot \\mathbb{1}[\\text{formed}_i]$. In `grA` and `grC`, $\\lambda_t$ ramped from $10^{-3}$ to $0.05$ over 100–300 steps. Halving a 300-token tool or math thinking trace to 150 tokens increased the reward by $0.05 \\times 150 / 2048 = 0.00366$ ($+0.0018$ at step 50). That length signal was **273x smaller** than the $1.0$ right/wrong step and **27x smaller** than the $+0.1$ format bonus. Worse, wrong answers received $+0.1$ merely for closing `</think>`, rewarding long, well-formed wrong completions over truncated ones.
- **Cause 2 — `--dr-grpo` (`scale=False` + constant denominator) killed all-right groups and shrank the step size by ~8x (`rl_train.py` L343–350, L938–939):**
  With `--dr-grpo`, `group_advantages(rewards, scale=False)` returns raw $R_i - \\bar{R}_g$ without dividing by $\\text{std}(R_g)$. In an all-correct group where answers vary by 150 tokens, raw advantages are $\\pm 0.0018$; in a mixed right/wrong group, raw advantages are $\\pm 0.50$. Standard GRPO (`scale=True`) normalises every group to unit variance so all-correct groups provide a unit-scale shortening gradient. `--dr-grpo` silenced the all-correct groups by ~275x. Simultaneously, `total_tokens = len(chunk) * args.max_tokens` ($96 \\times 4096 = 393,216$) replaced the actual active token count (~50,000 tokens), shrinking the gradient by **~7.8x**.
- **Cause 3 — Removing $1/|o_i|$ length normalisation (`--dr-grpo`) caused length inflation:**
  As documented in the RL study (Dr. GRPO, 2503.20783 §2.8; Arora & Zanette, 2502.04463 §2.6), standard per-response $1/|o_i|$ normalisation gives shorter positive-advantage answers a larger per-token gradient push. Removing $1/|o_i|$ while keeping a negligible length penalty eliminated the implicit shortening pressure; on raw Qwen3-0.6B, mixed groups where longer chains occasionally solved harder prompts drove outputs 5.8–6.7% *longer*.
- **Cause 4 — Full-FT learning rate on LoRA + AdamW reset every 10 steps (`track_thinkprune.py` L117–128):**
  `lr = 2e-6` is a full-fine-tuning learning rate (~5–10x too small for LoRA r32), and before `--tp-persist` was added on 4 Nov, `track_thinkprune.py` spawned a new `rl_train.py` process every 10 steps without `--init-state`, wiping AdamW's first and second moments at steps 10, 20, ..., 140.
- **Cause 5 — Thinking-only penalty & 4,096 code cap:**
  `lenpen` only measured `r.index(close_id)` capped at 2,048 tokens, applying zero pressure to IFEval's response-side loops (which `consult1` showed account for 71% of IFEval cut tokens) or code traces between 2,048 and 4,096 tokens, while hard-capping MBPP+ training rollouts at 4,096 instead of the test's 8,192.

### 1.2 Why `gr2e` Broke and `gr2f` (6 Nov) Stalled at +11.3%
- **Results on Screen-720:** `gr2f` (40 steps $\\times 32 \\times 8$ from raw Qwen, `--reward gated --gr-form linear`, $\\text{lr}=10^{-5} \\to 10^{-6}$, $\\text{KL}=0.005$, `--dr-grpo --fixed-denominator`): step 10 `+4.0% / -0.4`, step 20 `+11.4% / -0.6`, step 30 `+6.8% / -0.4`, step 40 `+11.3% / +0.3`.
- **Cause 1 — The `gr2e` part-label bug (`rl_train.py` L641–650):**
  `--gr-norms-from` called `build_pool.part_of(r)` on raw rollout records instead of reading the pool's `part` field, misclassifying SVAMP questions and setting $B_{\\text{svamp}} = 82$ tokens. Under $R = \\max(0, 1 - 0.3 L / B)$, every right math answer above $82 / 0.3 = 273$ tokens received $R = 0$ (identical to wrong), training math to truncate. (Fixed in `gr2f` via `--tp-norm-means`.)
- **Cause 2 — `--fixed-denominator` + `--max-tokens-code 8192` diluted the gradient by 16–20x (`rl_train.py` L940–941):**
  Line 941 computes:
  `total_tokens = max(1, len(jobs) * max(args.max_tokens, args.max_tokens_code) // max(1, args.updates))`
  Because `--max-tokens-code 8192` was enabled, every single step divided the policy gradient by $256 \\times 8192 = 2,097,152$ tokens — even though 5 of the 6 benchmark parts have a 4,096 cap and the 256 rollouts actually totalled ~110,000–140,000 tokens. Combined with `--dr-grpo` (`scale=False`), where a 20% length difference inside an all-right group yields a raw advantage of $\\pm 0.03$, the effective gradient on length was reduced by **~250x** relative to normalized token-mean GRPO.
- **Cause 3 — Premature cosine LR decay over 40 steps:**
  With `--warmup 12 --total-steps 40 --lr 1e-5 --lr-end 1e-6`, the learning rate peaked at step 12 and fell below $3\\times 10^{-6}$ by step 28. All movement happened between steps 10 and 20 (`+4.0%` $\\to$ `+11.4%`), after which the tiny LR and 2.1M-token denominator froze the weights (`+6.8%` at step 30, `+11.3%` at step 40).
- **Cause 4 — No dynamic filtering of all-wrong groups, and starting from raw Qwen:**
  As `matousek-martin`'s ablation proved on Qwen3-0.6B (`no-replay` at 128 steps: `+1.85% saved / +0.13`), a mild linear length penalty ($c = 0.1$–$0.3$) from the *raw* model cannot compress reasoning by 50% in 40–128 steps without an SFT/DPO warm start or hard budget truncation.

### 1.3 Why Earlier RL Runs (`rc1-rl`, `drpo1`, `sb1/sb2`, `COPD`) Failed on Shortened Models
- **`rc1-rl` (23 Oct, `--reward arora`, $\\alpha=0.15$, from `rc1-soup-30`):** Trained on 782 questions repeated every 24 steps. Because `rc1-soup-30` was already heavily shortened (57.4% saved, -3.5 pts) and frequently erred by stopping too early, the $1.0$ correctness reward outweighed $\\alpha = 0.15$ and taught the model to *lengthen* its thinking from 200–250 to 400–500 tokens (`+55.6%` $\\to$ `+49.6%` saved).
- **`drpo1` and `sb1`/`sb2` (25–27 Oct, from `rc1-rl-40`):** Both `drpo` ($\\lambda=0.2$) and `shorterbetter` ($\\beta=0.005$ and $0.001$) reward whichever correct answer is *shortest in the current group* ($L_{\\text{SOL}} = \\min_{i \\in \\text{correct}} L_i$) without any floor or per-question safe budget. On an already-compressed model, pushing every question toward the minimum lucky rollout caused immediate early-stopping collapse (`drpo1-40`: `+73.3% / -6.9`; `sb2-10`: stopped by guard at `-3.2`).
- **`COPD` (`cp1a`, `cp1c`, `cp1d`, 9–11 Nov):** In `cp1a`, the Qwen3-4B teacher pushed *useful* question re-reads down twice as hard as wasted re-checks (`-0.20` vs `-0.10` in `copddemo`). In `cp1c`/`cp1d`, adding `--copd-outcome 1.0` ($w \\cdot (\\text{right}_i - \\bar{\\text{right}}_g)$) caused a severe early-stopping collapse (`57.1% / -4.7`) because wrong answers are naturally 1.4–2.0x longer than right answers (`copdanalysis`), turning an unnormalised outcome push into a blind length-cutting force on hard questions.

---

## 2. Path 2 Full Design: Budget-Gated DLER-GRPO from Path 1's Final Model

### 2.1 Starting Point: Why Path 2 Must Start from Path 1's Round-2 Model (Not Round 1)
Path 1's two rounds mirror the `rc1-soup-30` $\\to$ `rpE-80` trajectory:
- **Path 1 Round 1** shortens aggressively to **~53–57% saved at -2.2 to -3.2 pts accuracy**. Diagnostic breakdowns of `rc1-soup-30` and `ca1-soup-10` (9 Nov) proved that at 55%+ compression, the model's errors are **dropped question re-reads** (calculating the right math on a misread question) and **broken IFEval counting rules** (writing 150-token responses when 300 words are requested).
- **RL cannot invent missing habits; it re-weights habits already in the policy's support.** Running length-penalised RL directly on Round 1 (`rc1-rl`, `sb1`, `drpo1`) either fails to recover the misread-question errors orlengthens answers back along the same curve.
- **Path 1 Round 2** repairs those exact habits using compact re-read pairs (`REREAD` regex), IFEval counting-rule pairs (`--count-rules`), and anti-early-stop pairs (`capguard`), landing around **48.5–51.5% saved at -0.6 to -1.1 pts accuracy** (the `rpE-60` / `rpE-80` zone: `47.0% / -0.9` to `47.8% / -1.2`).
- **What remains in the Round-2 model:** Round 2 restores re-reading and counting compliance, but in doing so it relaxes length by 4–7 percentage points on **already-mastered questions** (solve rate $\\ge 6/8$), where thinking spreads of $1.3\\times$–$2.0\\times$ reappear and a residual 2–3% tail of MBPP+/IFEval rollouts still runs long.
- **Therefore:** Path 2 starts from **Path 1's Round-2 final model** (`p1r2` 2-seed weight soup, merged into base weights as `models/p1r2-soup-best`). Path 2's job is **not** global compression from scratch; it is to **shave 3.5–5.5 percentage points of residual waste off easy/medium questions and long tails while strictly locking in Round 2's repaired accuracy**.
  *(Fallback exception: Only if Path 1 Round 2 over-repairs length down to $<46\\%$ saved on Dev-512 while an earlier Round-2 checkpoint—e.g. step 40—sits at $50\\%$ / $-1.3$, start Path 2 from the 2-seed soup of that step-40 checkpoint instead. Never start from Round 1.)*

---

### 2.2 The Reward Function: Question-Specific Budget-Target + Tail-Cap (`--reward p2budget`)

#### Why Question-Specific Budgets Beat Both Truncation-Only and Smooth Linear Penalties
1. **Why not pure hard truncation ($R = \\mathbb{1}[\\text{correct}] \\cdot \\mathbb{1}[L \\le C_{\\text{part}}]$) alone?**
   In DLER and ThinkPrune, the base model is a 10,000-token over-thinker evaluated at 32k. Here, Path 1's Round-2 model *already* averages ~250 tokens on SVAMP, ~850 on MBPP+, ~300 on IFEval, and ~100 on BFCL. A single part-wide cap $C_{\\text{part}}$ faces an impossible dilemma: if set high (e.g., Q85 of the part), 90% of questions are 8/8 under the cap and get zero gradient (`dropped_groups`); if set low (e.g., Q50 of the part), genuinely multi-step code or math problems get $R=0$ on every correct rollout and learn to guess early.
2. **Why not group-minimum targets (`ShorterBetter` / `arora` / `lenpen`)?**
   Rewarding the shortest rollout in every group (`ShorterBetter`) pushes already-compact 120-token math thoughts down to 60 tokens, destroying the question re-read step that Round 2 just repaired.
3. **The Synthesis (Kimi K2.5 Toggle + Kimi K3 $b_0(x)$ + Herz1c domain-normalised bonus + Repair Guard):**
   From Path 1's own stored 8 answers per training question (generated by the starting Path 1 model before RL), every question $q$ has its own empirical distribution of verified-correct lengths. We set **two per-question thresholds** directly from Path 1's own correct answers on $q$:
   - **Target length $B(q)$ (the K2.5 $\\tau$-percentile budget):** the **35th percentile (P35)** of *thinking* tokens among Path 1's correct answers on $q$ (or total tokens for non-IFEval; see IFEval protection below). Any correct answer at or below $B(q)$ is *already fast enough* and receives the **full maximum reward $1.0$** — zero pressure to cut further! This completely eliminates over-shortening collapse on questions that are already compact.
   - **Hard tail cap $C(q)$ (the K3 $\\lambda b_0(x)$ / ThinkPrune cap):** $\\max\\big(\\text{P90}(L_{\\text{correct}}(q)),\\, 1.35 \\cdot \\text{P50}(L_{\\text{correct}}(q))\\big)$, clipped to the part's 90th percentile cap $C_{\\text{part}}$. Correct answers between $B(q)$ and $C(q)$ lose a smooth bonus proportional to how far they exceed $B(q)$, normalised by the part's mean length $\\bar{L}_{\\text{part}}$ (so long MBPP+ code does not dominate short BFCL calls). Correct answers exceeding $C(q)$ (on easy/medium questions) get penalized down to a floor.
   - **Difficulty-Aware Toggle (Kimi K2.5 / DA-DLER / ALP):** Length pressure is **only applied if the current group's online solve rate $p_g = \\frac{1}{G}\\sum_{i=1}^G c_i \\ge \\tau_{\\text{acc}} = 0.50$** (at least 4 of 8 rollouts in the current step are correct). If $p_g < 0.50$ (the model is struggling on $q$), **all length pressure is switched off** ($R_i = c_i$): every correct answer gets $1.0$ regardless of length, and wrong answers get $0.0$ (or $-0.20$ if they stopped prematurely).

#### Exact Mathematical Formula
For question $q$ in benchmark part $d \\in \\{\\text{svamp}, \\text{mbpp}, \\text{ifeval}, \\text{bfcl\\_simple\\_python}, \\text{bfcl\\_multiple}, \\text{bfcl\\_irrelevance}\\}$, with group of $G=8$ rollouts $\\{y_1, \\dots, y_G\\}$:

1. **Validity & Correctness Gate ($c_i \\in \\{0, 1\\}$):**
   $$c_i = \\mathbb{1}\\big[\\text{is\\_correct}(q, y_i)\\big] \\cdot \\mathbb{1}\\big[\\text{finished}(y_i)\\big] \\cdot \\mathbb{1}\\big[\\text{formed}(y_i)\\big]$$
   where:
   - $\\text{finished}(y_i)$: `reply[-1] == eos` (not truncated at `max_tokens`).
   - $\\text{formed}(y_i)$: `close_id` (`</think>`) is in `reply`, non-blank thinking tokens $\\ge 8$, and answer tokens after `</think>` $> 2$.
   - Any wrong, truncated (cut-off), or malformed rollout has $c_i = 0$. **No format bonus ($0.0$) is ever given to wrong answers.**

2. **Measured Length $\\ell_i$ (IFEval Response Protection):**
   - Because the 10 Nov diagnostic proved that IFEval accuracy losses come from shrinking the *final response* on counting/length-constraint prompts (`length_constraints:number_words`, `keywords:frequency`, `number_sentences`, etc.), while non-thinking response tokens in SVAMP/BFCL are short and fixed:
   - For **all parts**, let $L_{\\text{think}}(y_i)$ be the token index of `</think>` and $L_{\\text{resp}}(y_i) = |y_i| - L_{\\text{think}}(y_i) - 1$.
   - For `part == "ifeval"`: $\\ell_i = L_{\\text{think}}(y_i)$ (**only thinking tokens are penalised; final response length is never penalised**). Furthermore, if `counting(q)` is True and $L_{\\text{resp}}(y_i) < 0.90 \\cdot R_{\\text{base}}(q)$ (where $R_{\\text{base}}(q)$ is the median response length of Path 1's correct answers on $q$), the rollout receives no length bonus ($b_i = 0$).
   - For all other parts (`svamp`, `mbpp`, `bfcl_*`): $\\ell_i = |y_i|$ (total generated tokens, matching the official evaluation metric).

3. **Per-Question Reference Quantities (computed once from Path 1's 8 answers on $q$ before training):**
   Let $\\mathcal{L}_{\\text{ok}}(q) = \\{\\ell(y^{(0)}_k) : c^{(0)}_k = 1\\}$ be the lengths of the starting Path 1 model's correct answers on question $q$, and let $\\bar{\\ell}_d$ be the mean of $\\ell$ across all answers of part $d$ in the training pool:
   - Target budget: $B(q) = \\text{percentile}(\\mathcal{L}_{\\text{ok}}(q),\\, 35)$ (i.e. $\\min$ of the shortest 3 out of 8 correct answers).
   - Median right thinking length: $T_{\\text{med}}(q) = \\text{median}(\\{L_{\\text{think}}(y^{(0)}_k) : c^{(0)}_k = 1\\})$.
   - Tail cap: $C(q) = \\min\\Big(C_d,\\; \\max\\big(\\text{percentile}(\\mathcal{L}_{\\text{ok}}(q), 90),\\; 1.35 \\cdot \\text{median}(\\mathcal{L}_{\\text{ok}}(q))\\big)\\Big)$, where $C_d$ is the 90th percentile of correct lengths across part $d$.

4. **Reward Assignment ($R_i$):**
   Let $p_g = \\frac{1}{G}\\sum_{j=1}^G c_j$ be the group's online solve rate in the current step.
   - **Case A — Correct rollout ($c_i = 1$):**
     - If $p_g < 0.50$ (hard / currently struggling question — **K2.5 Toggle off**):
       $$R_i = 1.0$$
     - If $p_g \\ge 0.50$ (solved $\\ge 4/8$ — **length pressure active**):
       $$R_i = \\begin{cases} 1.0 & \\text{if } \\ell_i \\le B(q) \\quad \\text{(already at/below target: full reward, no push to shrink further)} \\\\[4pt] 1.0 - \\alpha_d \\cdot \\min\\!\\left(1.0,\\; \\dfrac{\\ell_i - B(q)}{\\bar{\\ell}_d}\\right) & \\text{if } B(q) < \\ell_i \\le C(q) \\quad \\text{(smooth linear penalty above } B(q)\\text{)} \\\\[6pt] 1.0 - \\alpha_d - \\gamma_{\\text{over}} & \\text{if } \\ell_i > C(q) \\quad \\text{(over-cap penalty on easy/medium questions)} \\end{cases}$$
       where:
       - $\\alpha_d$ is the per-part length weight: $\\alpha_{\\text{svamp}} = 0.20$, $\\alpha_{\\text{bfcl\\_irrelevance}} = 0.20$, and $\\alpha_d = 0.35$ for `mbpp`, `ifeval`, `bfcl_simple_python`, `bfcl_multiple` (gentler on math and no-fit where accuracy is most fragile; stronger on code, instructions, and tool calls where waste is highest).
       - $\\gamma_{\\text{over}} = 0.25$ is the step-down penalty (LASER / ThinkPrune soft cap) for exceeding the question's tail cap $C(q)$ when shorter correct answers exist. (Thus an over-cap correct answer gets $0.40$–$0.55$, still strictly above a wrong answer's $0.0$, avoiding the DLER/DRPO trap where long correct answers get pushed down harder than wrong answers when $p_g < 1$, while strongly losing to in-budget correct answers when $p_g \\ge 0.5$.)
   - **Case B — Wrong or Cut-Off rollout ($c_i = 0$):**
     $$R_i = \\begin{cases} -\\delta_{\\text{early}} = -0.20 & \\text{if } \\text{finished}(y_i) \\text{ and } L_{\\text{think}}(y_i) < 0.50 \\cdot T_{\\text{med}}(q) \\\\[4pt] 0.0 & \\text{otherwise (normal wrong or truncated at max\\_tokens)} \\end{cases}$$
     **Why $-0.20$ for premature-stop wrong answers?** As `copdanalysis` (10 Nov) proved, wrong answers are on average longer than right answers, EXCEPT when a shortened model gives up early ($L_{\\text{think}} < 0.5 \\cdot T_{\\text{med}}(q)$). Giving $-0.20$ to premature-stop wrong answers creates an explicit RL guard against the #1 collapse mode of shortened Qwen3-0.6B (stopping before re-reading the question).

---

### 2.3 Algorithm and Every Hyperparameter

| Parameter | Exact Setting | Evidence & Justification |
|---|---|---|
| **Starting Model** | `models/p1r2-soup-best` (Path 1 Round-2 2-seed weight soup, merged into base) | Starts at ~50% saved / ~-0.9 pt with re-read and counting rules already repaired. Fresh LoRA on merged base avoids mutating Path 1's adapter directly and keeps the KL reference equal to Path 1's final policy (`model.disable_adapter()`). |
| **LoRA Config** | `r = 16`, `lora_alpha = 32`, `lora_dropout = 0.0`, all 7 linear modules (`q,k,v,o,gate,up,down`) | Matches Path 1 DPO, Herz1c, basta, and matousek-martin. Keeps parameter count tight so 40–60 steps generalize without overfitting. |
| **Precision** | Base in `bfloat16` (or `float16`), LoRA weights in `float32` (`rl_train.py` L688–689) | Standard in `rl_train.py` and vLLM (`--dtype bfloat16` on L4). |
| **Objective** | Clipped Policy Gradient (`--objective grpo`) with **Clip-Higher** (`--clip-low 0.20 --clip-high 0.28`) when `--updates > 1`, or exact 1-step on-policy (`--updates 1`) with **token-level log-ratio masking** outside $[-0.22, +0.25]$ | DLER (2510.15110), DAPO (2503.14476), and Art (2602.20945) all use asymmetric clipping $[0.80, 1.28]$ to prevent entropy collapse on low-probability transition tokens. With `--updates 1`, the sampling policy has temperature/top-k/top-p differences from `batch_logps` unless `--full-support` is on; we enable `--full-support` so sampling and training log-probs match at $T=0.6$. |
| **Advantage Normalisation** | **Batch-wise std normalisation** (`--advantage batch`, **NO `--dr-grpo`**): $A_{g,i} = \\frac{R_{g,i} - \\bar{R}_g}{\\text{std}_{\\text{batch}}(R - \\bar{R}_g) + 10^{-6}}$ | **Crucial fix** (DLER Eq. 3, 2510.15110 §2.1): Group-wise std (`--advantage group`) blows up tiny noise in nearly-identical groups, while no-std (`--dr-grpo`) makes all-correct groups $15\\times$ weaker than mixed groups. Batch-wise std normalises advantages to unit scale across the step while keeping an all-correct group with a real $0.25$ reward gap strong ($A \\approx \\pm 0.8$) and a flat group near zero! |
| **Loss Denominator** | **Actual active tokens in the chunk** (`total_tokens = sum(len(j[1]) for j in chunk)`), **NO `--dr-grpo`**, **NO `--fixed-denominator`** | Eliminates the $17.5\\times$ gradient dilution that froze `gr2f`. Dividing by the batch's actual active tokens is DAPO / DLER / K2.5's token-level loss normalisation: it avoids single-response $1/|o_i|$ blowup on 10-token collapses while keeping gradient magnitude invariant to `max_tokens_code`. |
| **Group Size ($G$)** | `G = 8` replies per question (`--group 8`) | Gives reliable online solve rate $p_g \\in \\{0, 1/8, \\dots, 1\\}$ and P35/median contrast; fits fast in vLLM on one L4. |
| **Questions per Step** | `32` kept questions per step (`--questions 32`, i.e. 256 active rollouts/step) | Matches `gr2f` and Arora & Zanette; 256 rollouts/step provides stable batch-wise advantage std across all 6 task parts. |
| **Dynamic Sampling / Filtering** | `--dynamic-sampling 1.5 --drop-flat` (sample $48$ questions/step, drop groups where $\\max(R_g) - \\min(R_g) < 10^{-4}$ — i.e., all-wrong groups AND all-right groups where every rollout is already $\\le B(q)$ — keep up to $32$ informative groups) | **DLER Key Ingredient 3:** In `rl_train.py` L753, `--dynamic-sampling` currently only checks `if any(grades[k])`, keeping all-correct groups even when all rewards are equal ($1.0$). Changing the filter to `max(rewards) - min(rewards) > 1e-4` ensures every kept group has non-zero advantage signal! |
| **Learning Rate & Schedule** | `lr = 1.5e-5`, **4-step linear warmup, then constant `1.5e-5` up to step 35, cosine decay to `5e-6` at step 50** (`--lr 1.5e-5 --lr-end 5e-6 --warmup 4 --total-steps 50`) | `grA`'s $2\\times 10^{-6}$ was too small; `gr2f`'s cosine decay to $10^{-6}$ killed progress after step 20; `COPD Run A`'s cosine floor froze at step 30. Herz1c and matousek-martin proved `1e-5` works for LoRA r16 RL on Qwen3-0.6B; `1.5e-5` with a high floor (`5e-6`) ensures steady movement across 50 steps. |
| **KL Penalty** | `--kl-coef 0.002` (Schulman k3 estimator $\\exp(\\Delta) - \\Delta - 1$ to the merged Path 1 Round-2 base model) | Anchors the RL policy to Path 1's Round-2 model (since Path 1 Round 2 is merged into `--model`, `model.disable_adapter()` IS the Path 1 Round-2 model!). `0.002` is strong enough to prevent drift on unsolvable/flat tokens (Fatemi et al.) without fighting shortening (`grA`'s `0.04` was 20x too high). |
| **Updates per Step** | `--updates 1` (strictly on-policy, 1 gradient step per rollout batch; accumulated over `token_batches` of 4,096 tokens) | Art (2602.20945) and Optimal Reasoning Length (2602.09591) show on-policy (`updates=1`) prevents entropy explosion; with `--updates 1` and `--full-support`, the policy ratio is identically $1.0$. |
| **Max Rollout Length** | `--max-tokens 4096 --max-tokens-code 8192` | Matches ThinkingCap-Test evaluation caps exactly (4,096 for math/IFEval/BFCL, 8,192 for MBPP+/KodCode). |
| **Sampling Params** | `--full-support` (`temperature = 0.6` for both vLLM sampling and `LOGIT_TEMP` in `batch_logps`) | Eliminates mismatch between vLLM's top-k=20/top-p=0.95 truncated distribution and `log_softmax` during backward pass (`consult1`). |
| **Grad Clip** | `--grad-clip 0.5` | Tighter than default `1.0` to prevent any single long MBPP+ rollout from spiking LoRA weights. |
| **Total Steps** | **50 steps** (in 10-step resumable blocks with `--tp-persist` so AdamW state carries across blocks) | 50 steps $\\times 32$ kept questions = $1,600$ informative groups ($12,800$ kept rollouts; ~$19,200$ sampled with $1.5\\times$ dynamic sampling). Both Herz1c (50 steps) and `rc1-rl` / `drpo` / `cp1a` showed all useful RL movement on Qwen3-0.6B occurs in steps 10–50. |

---

### 2.4 Question Selection: Source, Difficulty Band, and Quotas (Never Dev-512)

1. **Zero Contamination & Disjointness:**
   - Pool source: Only the team's own verified sources (`pool-dp13` minus MetaMath + own top-ups `nofit1`, `nofit2`, `tools1`, `code1` + the full-difficulty GSM8K questions in SVAMP wording from `prepare_types gsm8k_std`). No `michal-bida` sources (`math1`, `math2`, `instr2`, `code2`, `nofit4`). `prostring|stringname|stringName` corrupted rows excluded.
   - **Never Dev-512, Screen-720, or Test:** Exclude every question in `Dev-512`, `Screen-720`, and `ThinkingCap-Test`, plus any question sharing $\\ge 30\\%$ rare 8-grams with them.
   - **Reuse Path 1 Round 2's On-Policy Rollouts (Zero Extra Sampling Cost!):**
     In Path 1 Round 2 (`PATH1_FINAL.md`), the Round-1 model answers ~3,200 unseen questions $\\times 8$ (per seed). Even better: before Path 2 starts, **the merged Path 1 Round-2 model (`models/p1r2-soup-best`) answers a dedicated slice of 2,400 unseen pool questions $\\times 8$** (or reuses the unused portion of the pool answered $\\times 8$ by `p1r2-soup-best`, taking ~1.1 GPU-hours across the 2 L4s). Why must the reference rollouts come from `p1r2` (or `p1r1`) rather than raw Qwen? Because $B(q)$ (P35) and $C(q)$ (P90) must reflect the **already-shortened Path 1 model's length distribution**, not raw Qwen's $2\\times$ longer distribution! (If raw Qwen's lengths were used for $B(q)$, P35 of raw Qwen would be longer than Path 1's median, so $\\ell_i \\le B(q)$ would fire on almost every rollout and Path 2 would not shorten at all!)

2. **Difficulty Band (`solve_rate` $\\in [0.50, 1.00]$ on `p1r2`'s 8 rollouts):**
   - Filter to questions where `p1r2` achieves **at least 4/8 correct (`rate >= 0.50`)** AND where the longest correct answer is at least **$1.15\\times$** the shortest correct answer (or at least 1 rollout is wrong/long, so the question is not already a degenerate constant-length question).
   - **Why exclude $< 0.50$ solve rate?** The RL study (`Art of Efficient Reasoning`, 2602.20945 §2.13; Fatemi et al., 2504.05185 §2.5) proved on Qwen3-0.6B and R1-1.5B that including hard/unsolvable prompts in length-RL causes entropy spikes and short-answer collapse.
   - **Within `[0.50, 1.00]`, enforce a 50 / 50 interleave (`--tp-mixed-share 0.50`):**
     - **50% "High-solve / High-spread" (`rate` $\\in \\{6/8, 7/8, 8/8\\}$ and $\\max L_{\\text{ok}} / \\min L_{\\text{ok}} \\ge 1.25$):** These provide the **pure shortening signal** ($p_g \\ge 0.50$ is guaranteed, and rollouts straddle $B(q)$).
     - **50% "Medium-solve" (`rate` $\\in \\{4/8, 5/8, 6/8\\}$):** These provide the **simultaneous correctness + compactness signal** (mixed right/wrong groups where in-budget right answers beat both long right answers and wrong answers).

3. **Benchmark Mix & Question Counts (for 50 steps $\\times 48$ drawn = 2,400 questions):**
   Matched to the official evaluation weighting (25% per benchmark, with BFCL split 400 / 200 / 240, and a slight boost to code/IFEval where the most tokens remain to be saved):
   - **SVAMP / GSM8K math:** 600 questions (25.0%)
   - **MBPP+ / KodCode code:** 600 questions (25.0%)
   - **IFEval instructions:** 600 questions (25.0% — half counting-rule, half non-counting)
   - **BFCL simple (one function):** 285 questions (11.9%)
   - **BFCL multiple (several functions):** 145 questions (6.0%)
   - **BFCL irrelevance (no fit):** 170 questions (7.1%)
   - **Total:** **2,400 unique questions** (each seen **at most once** during the 50-step run — zero question repetition, preventing the `rc1-rl` over-memorization length rebound).

---

### 2.5 Checkpoints, Selection Rule on Dev-512, Expected Result, and GPU-Hours

1. **Checkpointing & Two-Seed Protocol:**
   - Run **2 independent seeds** (Seed 0 on GPU 1, Seed 1 on GPU 2) from `models/p1r2-soup-best` (different question order and sampling seeds).
   - Save LoRA adapters every **10 steps**: `step-10`, `step-20`, `step-30`, `step-40`, `step-50`.
   - For each checkpoint $s \\in \\{20, 30, 40, 50\\}$, form the **2-seed weight-averaged adapter** (`p2-soup-s` = $0.5 \\cdot \\text{seed0}(s) + 0.5 \\cdot \\text{seed1}(s)$ merged into `models/p1r2-soup-best`) and score `p2-soup-s` on **Dev-512 ($512 \\times 8$)** using `dev_score.py --vs runs/bon/dev512-p1r2-soup-best/rollouts.jsonl`.

2. **Pre-Registered Selection Rule on Calibrated Dev-512:**
   - Using Step 0's calibration offset ($\\Delta_{\\text{save}} = \\text{Full}_{\\text{save}} - \\text{Dev512}_{\\text{save}}$, $\\Delta_{\\text{acc}} = \\text{Full}_{\\text{acc}} - \\text{Dev512}_{\\text{acc}}$ measured on `rc1-soup-30`, `rpE-40/60/80`, `rpI-120`), compute each checkpoint's calibrated full-test projection $(\\hat{S}_s, \\hat{A}_s)$.
   - **Acceptance Gate vs Path 1:** A Path 2 checkpoint is eligible **only if** on paired Dev-512 bootstrap (`--vs p1r2-soup-best`):
     1. It increases tokens saved by **$\\ge +1.5$ percentage points** over `p1r2-soup-best` ($\\text{d\\_saved} \\ge +1.5$),
     2. Its calibrated accuracy change is **$\\hat{A}_s \\ge -0.85$ pts** (leaving a $0.15$ pt safety margin above the $-1.0$ hard threshold), AND
     3. Its paired accuracy drop vs `p1r2-soup-best` is at most **$-0.35$ pts** ($\\text{d\\_acc} \\ge -0.35$).
   - Among eligible checkpoints $s \\in \\{20, 30, 40, 50\\}$, pick the one that maximizes $\\hat{A}_s + 0.15 \\cdot \\min(\\hat{S}_s - 50.5,\\, 3.0)$ (i.e. once safely past $51.5\\%$ calibrated saving, prefer highest accuracy).
   - **Strength-Dial Fallback (Zero-Training Insurance):** If `step-30` or `step-40` overshoots slightly (e.g. $+5.0\\%$ extra saving at $-0.6$ pt paired accuracy loss), evaluate the half-strength adapter merge (`--adapters ckpt:0.5`, which took `drpo1-20` halfway back to its start) on Dev-512 before rejecting.
   - Run **one** final official Full Evaluation ($2,759 \\times 8$) on the single selected checkpoint.

3. **Expected Result (with Quantitative Range):**
   - **Starting point (`p1r2-soup-best`):** `49.0% to 51.5%` tokens saved, `-0.7 to -1.2` pts accuracy on Full Test.
   - **Path 2 Delta (at selected step 30–40):** **`+2.5% to +4.5%` additional tokens saved** (bringing MBPP+ truncation from ~3% to <1%, trimming P50–P90 right answers toward P35 on SVAMP/MBPP+/IFEval/BFCL) at **`-0.2 to +0.2` pts accuracy change**.
   - **Combined Path 1 + Path 2 Full-Test Forecast:** **`52.0% to 54.8%` tokens saved at `-0.6 to -0.95` pts accuracy** (lifting the model to **+2.2 to +2.8 pts above the quadratic reference curve**).

4. **Compute Budget (GPU-Hours on 2 $\\times$ Colab L4 24GB):**
   - **Step 2.0 (Reference rollouts on 2,400 questions $\\times 8$ by `p1r2-soup-best`):** 1,200 questions per GPU $\\approx$ **1.0 h wall-clock** (2.0 GPU-h total; already shorter answers so 2x faster than raw Qwen).
   - **Step 2.1 (RL training, 50 steps $\\times 48$ drawn $\\times 8$ rollouts = 19,200 rollouts/seed):** Because `p1r2` answers average ~320 tokens (not raw Qwen's 650 tokens), each step of 384 short rollouts + 256 active backward passes takes **~2.8 min on an L4** $\\implies$ 50 steps = **2.3 h per seed** (run in parallel: Seed 0 on GPU 1, Seed 1 on GPU 2 = **2.3 h wall-clock**, 4.6 GPU-h total).
   - **Step 2.2 (Dev-512 $\\times 8$ scoring of 4 souped checkpoints: steps 20, 30, 40, 50):** ~18 min per Dev-512 eval $\\times 4$ = **1.2 GPU-h** (0.6 h wall-clock across 2 GPUs).
   - **Total Path 2 Budget:** **~3.9 hours wall-clock (~7.8 GPU-hours across the two L4s)**, plus **1.5 GPU-h** for the single Full Test of the winner.

---

## 3. Exact Code Changes Needed in `rl_train.py` and `track_thinkprune.py`

### 3.1 Changes in `code/rl_train.py`

1. **Add CLI arguments in `parse_arguments()` (around L160–185):**
   - Extend `--reward` choices: `choices=["shorter", "arora", "shorterbetter", "cap", "lenpen", "smart", "gated", "p2budget"]`.
   - Add `--p2-alpha-math` (`type=float, default=0.20`, applied to `svamp` and `bfcl_irrelevance`), `--p2-alpha-other` (`type=float, default=0.35`, applied to `mbpp`, `ifeval`, `bfcl_simple_python`, `bfcl_multiple`).
   - Add `--p2-overcap-pen` (`type=float, default=0.25`, subtracted when a correct rollout on a $p_g \\ge \\tau_{\\text{acc}}$ question exceeds `question["cap"]`).
   - Add `--p2-early-pen` (`type=float, default=0.20`, negative reward $-0.20$ for a finished wrong rollout with `thinking_len < 0.5 * question["base_think"]`).
   - Add `--p2-toggle-gate` (`type=float, default=0.50`, minimum group solve rate $p_g$ required to turn on length pressure).
   - Add `--drop-flat` (`action="store_true"`, when used with `--dynamic-sampling`, drops any group where `max(rewards) - min(rewards) <= 1e-4` instead of only dropping all-wrong groups).
   - Add `--clip-low` (`type=float, default=0.20`) and `--clip-high` (`type=float, default=0.28`) for asymmetric ratio / log-ratio clipping (DLER / DAPO / K2.5).

2. **Implement `--reward p2budget` in `main()` (inside `for question, prompt, group, correct in zip(...)`, around L775):**
   ```python
   elif args.reward == "p2budget":
       from repair_pairs import counting
       norms = dict(kv.split("=") for kv in args.gr_norms.split(","))
       norms.update(GR_MEANS)
       part = question.get("part", "")
       b_mean = float(norms.get(part, 512))
       b_target = float(question.get("target_len") or b_mean * 0.7)
       q_cap = float(question.get("cap") or args.max_tokens)
       base_think = float(question.get("base_think") or 0)
       base_resp = float(question.get("base_resp") or 0)
       alpha = args.p2_alpha_math if part in ("svamp", "bfcl_irrelevance") else args.p2_alpha_other
       solve_rate = sum(correct) / max(1, len(correct))
       is_cr = (part == "ifeval") and counting(question)
       rewards = []
       for r, ok in zip(group, correct):
           think_len = r.index(close_id) if close_id in r else len(r)
           resp_len = len(r) - think_len - 1 if close_id in r else 0
           blank = sum(1 for x in r[:think_len] if tokenizer.decode([x]).strip())
           formed = close_id in r and blank >= args.gr_min_think and resp_len > 2
           finished = bool(r) and r[-1] == eos
           if ok and formed and finished:
               if solve_rate < args.p2_toggle_gate:
                   rewards.append(1.0)
               elif is_cr and base_resp > 0 and resp_len < 0.90 * base_resp:
                   rewards.append(1.0 - alpha)  # counting rule response shrunk: no length bonus
               else:
                   meas_len = think_len if part == "ifeval" else len(r)
                   if meas_len <= b_target:
                       rewards.append(1.0)
                   elif meas_len <= q_cap:
                       pen = alpha * min(1.0, (meas_len - b_target) / max(1.0, b_mean))
                       rewards.append(max(0.1, 1.0 - pen))
                   else:
                       rewards.append(max(0.1, 1.0 - alpha - args.p2_overcap_pen))
           else:
               if finished and not ok and base_think > 0 and think_len < 0.50 * base_think:
                   rewards.append(-args.p2_early_pen)
               else:
                   rewards.append(0.0)
   ```

3. **Upgrade Dynamic Sampling to Drop Flat Groups (`rl_train.py` L751–818):**
   - Currently, `if args.dynamic_sampling:` at L752 filters `keep = [k for k in range(len(questions)) if any(grades[k])][:args.questions]` *before* rewards are computed, so an 8/8 correct group where all 8 replies have `meas_len <= b_target` (all rewards $= 1.0$, zero advantage) still consumes one of the 32 slots!
   - **Change:** Compute `rewards` for all `drawn` groups first. When `--dynamic-sampling > 0` and `--drop-flat` is set, filter to groups with `max(rewards) - min(rewards) > 1e-4` (keeping up to `args.questions` groups), and record `dropped = drawn - len(keep)` before computing batch-wise advantages `centered` and `spread`.

4. **Fix Active-Job Filtering and Add Asymmetric Clip-Higher (`rl_train.py` L935–1048):**
   - At L935: `active = [j for j in jobs if j[1] and abs(j[3]) > 1e-8]` (replace `not (j[3] == 0 and not j[2])`, because under `p2budget` or `batch` advantages, any reply with $A_{g,i} = 0$ contributes zero gradient and wastes forward/backward compute and `total_tokens` denominator!).
   - Before the update loop (when `args.updates > 1` or `--clip-high` is active), cache `old_logps` from the policy at step start (already implemented for `drpo` at L894–900) and apply asymmetric PPO clipping with `eps_low = args.clip_low` ($0.20$) and `eps_high = args.clip_high` ($0.28$):
     ```python
     ratio = torch.exp(logps - old_lp)
     surr1 = ratio * adv_t
     surr2 = torch.clamp(ratio, 1.0 - args.clip_low, 1.0 + args.clip_high) * adv_t
     per_token = -torch.min(surr1, surr2)
     ```
     (Or with `--updates 1`, mask token gradients where `abs(logps - old_lp) > args.clip_high` as in Kimi K2.5 Eq. 1.)

5. **Add Per-Step Diagnostic Logging in `entry` (`rl_train.py` L1056–1065):**
   - Log `by_type[kind]` with **both** `thinking` AND `response` mean lengths (`len(r) - thinking_len`), plus `early_wrong_rate` (share of wrong answers with `thinking_len < 0.5 * base_think`) and `trunc_rate` (share of replies hitting `max_tokens`), so all 5 collapse modes are visible in `rl_log.jsonl` every single step.

---

### 3.2 Changes in `code/track_thinkprune.py`

1. **Allow Rollout File & Question Pool from Path 1's Model (`track_thinkprune.py` L13–45):**
   - Add `--tp-rollouts PATH` (optional path to the 8-sample rollout `.jsonl` generated by `models/p1r2-soup-best` on the unseen pool questions) and `--tp-exclude-dev PATH` (path to `dev512` + `screen720` + Path 1 Round-1/Round-2 used question IDs, plus 8-gram filter).
   - For each question `qid`, compute and store in the `data/tp-{name}.jsonl` row:
     - `target_len`: 35th percentile of `thinking_len` (for `ifeval`) or `output_tokens` (for other parts) among `p1r2`'s correct rollouts on `qid`.
     - `cap`: $\\min(C_{\\text{part}},\\, \\max(\\text{P90}(L_{\\text{ok}}(q)),\\, 1.35 \\cdot \\text{P50}(L_{\\text{ok}}(q))))$.
     - `base_think`: median `thinking_len` of `p1r2`'s correct rollouts on `qid`.
     - `base_resp`: median `response_len` of `p1r2`'s correct rollouts on `qid`.
   - In `--tp-norm-means` (L88–99), compute `means` over **correct answers from `--tp-rollouts`** (Path 1's model, not raw Qwen's 10-answer file!).

2. **Pass Correct Optimizer Flags to `rl_train.py` (`track_thinkprune.py` L117–128):**
   - Replace hardcoded `"--advantage", "group"` with `"--advantage", args.tp_advantage` (default `"batch"` for Path 2).
   - When `args.tp_reward == "p2budget"`, pass `"--max-tokens", 4096, "--max-tokens-code", 8192, "--full-support", "--drop-flat", "--dynamic-sampling", 1.5, "--grad-clip", 0.5`.
   - Score checkpoints on **Dev-512 $\\times 8$** (`dev_score.py`) instead of `SCREEN` ($720 \\times 4$).

---

## 4. Cheap First Experiment (Smoke + Gate): Confirm or Kill Path 2 in 2.2 GPU-Hours

Before waiting for Path 1's full 2-seed run or spending 8 GPU-hours on a 2-seed Path 2 run, run a **single-GPU, 2.2-GPU-hour Gate Experiment (`p2-gate`)** on **one L4 GPU** starting from **`models/rpE-80`** (which is already on Drive, already scored on Full Test at `47.0% / -0.9`, already scored on Dev-512 in Step 0 calibration, and already has 5,700 fresh questions $\\times 8$ answered by `rpE-80` itself in `rpI`'s rollout file!).

### 4.1 Exact Setup of `p2-gate` (Zero New Pre-Sampling Needed!)
- **Start model:** `models/rpE-80` (merged).
- **Pool & reference rollouts:** Reuse `rpI`'s existing 5,700-question $\\times 8$ rollout file generated by `models/rpE-80` (10 Nov log, L5172), excluding any overlap with `Dev-512`. This costs **0.0 GPU-hours** of pre-sampling!
- **Training:** Run **30 steps** of Path 2 (`--reward p2budget --advantage batch --dynamic-sampling 1.5 --drop-flat --lr 1.5e-5 --lr-end 5e-6 --warmup 4 --total-steps 50 --kl-coef 0.002 --full-support --max-tokens 4096 --max-tokens-code 8192 --grad-clip 0.5`, 1 seed, saving at steps `15` and `30`).
  - Training time: 30 steps $\\times 2.8$ min = **1.4 GPU-hours**.
- **Evaluation:** Score `p2-gate-15` and `p2-gate-30` on **Dev-512 $\\times 8$** with `dev_score.py --vs runs/bon/dev512-rpE-80/rollouts.jsonl` (paired bootstrap against `rpE-80`).
  - Evaluation time: 2 checkpoints $\\times 20$ min = **0.7 GPU-hours**.
- **Total cost:** **2.1 GPU-hours** on a single L4.

### 4.2 Quantitative Decision Rule (Confirm, Tune, or Kill)
1. **CONFIRM (Green Light for Full Path 2 after Path 1 finishes):**
   At either step 15 or step 30, on paired Dev-512 (`--vs rpE-80`):
   - **Tokens saved increases by $\\ge +2.0$ percentage points** over `rpE-80` (with 95% paired bootstrap lower bound $> +0.8\\%$), AND
   - **Accuracy change vs `rpE-80` is $\\ge -0.30$ pts** (leaving overall calibrated accuracy better than $-1.0$ pt vs raw Qwen), with no single part dropping $> 2.5$ pts.
2. **TUNE ONCE (If compression is too strong or too timid):**
   - If $\\Delta\\text{saved} > +4.5\\%$ and $\\Delta\\text{acc} \\in [-0.8, -0.35]$: evaluate the **0.5-strength LoRA merge** (`merge.py --adapters p2-gate-30:0.5`, 20 min on Dev-512). If the 0.5-strength merge passes the Green Light rule, lower `--lr` from `1.5e-5` to `1.0e-5` for the real Path 2 run.
   - If $\\Delta\\text{saved} \\in [+0.8\\%, +1.9\\%]$ and $\\Delta\\text{acc} \\ge 0.0$: increase `--p2-alpha-other` from `0.35` to `0.45` and run to step 45.
3. **KILL (Abort Path 2 and submit Path 1's best checkpoint):**
   - If at both steps 15 and 30 (and 0.5-strength merge), either $\\Delta\\text{saved} < +0.8\\%$ OR the paired accuracy-per-saving slope is worse than $-0.25$ pts of accuracy per $+1.0\\%$ saved (i.e., Path 2 merely slides back down the existing DPO curve toward `rpE-40`), **kill Path 2 immediately**.

---

## 5. Risks, Collapse Modes, Early Detection Signals, and Automatic Guards

Every failure mode observed in `rl_length_methods_study.md` and `EXPERIMENT_LOG.md` has a specific per-step metric to log in `rl_log.jsonl` and a quantitative tripwire:

| Collapse Mode | Mechanism & Past Precedent | Monitoring Signal to Log Every Step in `rl_log.jsonl` | Early Detection Threshold (Tripwire) & Automatic Remedy |
|---|---|---|---|
| **1. Collapse to Short Wrong Answers ("Early-Stop Trap")** | Length penalty pushes the model to emit `</think>` after 20–50 tokens before re-reading the prompt (`COPD Run C`: 54/73 wrong SVAMP answers had thinking $< 0.5 \\cdot T_{\\text{med}}$; `sb1`/`drpo1`). | `early_wrong_share`: fraction of wrong replies with `thinking_len < 0.5 * base_think`, logged overall and per `by_type`. | **Tripwire:** `early_wrong_share` rises above **0.35** (over 5-step rolling mean) or increases by **+0.15** above steps 1–5. <br>**Remedy:** `--p2-early-pen` ($-0.20$, or increase to $-0.35$) directly punishes these trajectories; `--guard-drop 0.05` auto-stops training and keeps the previous 10-step checkpoint. |
| **2. Entropy Collapse / Explosion** | Symmetric PPO clipping clips low-prob transition tokens, collapsing token entropy (DLER §2.1); conversely, off-policy staleness or hard-only prompts spike entropy before length collapses (Art §2.13). | `entropy`: mean per-token policy entropy $H(\\pi_\\theta)$ across active reply tokens (already logged in `rl_train.py` L1059). | **Tripwire:** Starting from `~0.28–0.35` on shortened Qwen3-0.6B, if rolling 5-step `entropy` drops below **0.16** (collapse) or spikes above **0.55** (explosion). <br>**Remedy:** `--clip-high 0.28`, `--updates 1` (zero staleness), and `--kl-coef 0.002` prevent both; if entropy $< 0.16$, enable `--entropy-coef 0.001`. |
| **3. Over-Shortening of Math (`svamp`)** | SVAMP answers build up late in thinking (3 Nov cut study: 0.34 accuracy at 20–40% kept vs 0.73 full); uniform length pressure crushes SVAMP first (`D10`: SVAMP +68% / -5.4). | `by_type["svamp"]["thinking"]` and `by_type["svamp"]["accuracy"]` (rolling 5-step mean). | **Tripwire:** `by_type["svamp"]["thinking"]` falls below **185 tokens** (vs ~250 in `rpE-80` and 685 in raw Qwen) OR `svamp` batch accuracy drops by **$> 0.04$** vs steps 1–5. <br>**Remedy:** `--p2-alpha-math 0.20` (half the weight of code/IFEval) and $B(q) = \\text{P35}$ floor; if tripwire fires, pass `--protect-sources gsm8k,svamp` for the remaining steps. |
| **4. IFEval Response Shrinking (Counting Rules)** | Penalising total tokens `len(r)` on IFEval causes the model to shorten the *final response* after `</think>`, breaking `number_words`, `number_sentences`, and `keyword_frequency` constraints (10 Nov `ifeval_rules.json` diagnostic: -11 to -13 pts on word count rules). | `by_type["ifeval"]["response"]` (mean tokens *after* `</think>`) and `counting_short_rate` (share of counting-rule replies with `resp_len < 0.9 * base_resp`). | **Tripwire:** `by_type["ifeval"]["response"]` drops by **$> 8\\%$** from steps 1–5 (should stay $\\ge 220$ tokens) or `counting_short_rate > 0.20`. <br>**Remedy:** Built into `p2budget`: for `part == "ifeval"`, $\\ell_i = L_{\\text{think}}(y_i)$ (**response tokens are never penalised**), and any counting-rule reply with `resp_len < 0.90 * base_resp` forfeits its length bonus. |
| **5. BFCL No-Fit (`bfcl_irrelevance`) Refusal / Over-Calling Drift** | Training heavily on tool-calling prompts without balanced no-fit examples makes the model call tools when none fit (`lpt6mix-C`: no-fit `-9.4` pts; `matousek-martin`: `-6.88` pts); conversely, over-penalising length on tool calls makes the model refuse valid calls (`lpt6-nofit`: simple `-20.6` pts) because a 1-sentence refusal is shorter than a JSON tool call! | `by_type["toolace/irrelevance"]["accuracy"]`, `by_type["toolace/simple"]["accuracy"]`, and `by_type["toolace/multiple"]["accuracy"]` (logged separately via `type_of(question)` in `rl_train.py` L373). | **Tripwire:** Either `irrelevance` accuracy or `simple`/`multiple` accuracy drops by **$> 0.05$** over a 5-step window, or `simple` mean total tokens falls below **75 tokens** (indicating tool-call refusals replacing JSON calls). <br>**Remedy:** `B(q) = P35` of *correct* answers on each specific question ensures a valid JSON tool call on `simple`/`multiple` is only ever compared against *correct* JSON tool calls on that same question (never against a short refusal), and keeping `bfcl_irrelevance` at 7.1% of the batch with $\\alpha = 0.20$ preserves no-fit calibration. |
"""

with open(f"{work_dir}/report_draft.md", "w") as f:
    f.write(full_report)

words = len(full_report.split())
print(f"Report draft saved. Word count: {words} words.")
```