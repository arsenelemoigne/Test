# Online DPO Strategy on GPU 1: Variants S and A

## 1. Diagnosis of `od2` and `od3` & Why the Per-Type Brake Is Necessary

1. **Why `od3` overshot on Math and Code despite a fixed reference and per-type floors:**
   - In `od3`, every round draws pairs from the model’s *own latest answers* with `--mix-preset od3_arena` (`75%` shorten across almost all types). Because each round’s "longer right" answer is already shortened by the previous rounds, the shortening signal compounds fastest on the types with the widest intra-question length variance (Math and Code).
   - By Round 2 (`od3-r2`), Math was already at `58%` saved / `-4.0` monitor accuracy and Code at `49%` / `-2.1`, while IFEval (`31%`), Simple Tools (`32%`), Several Tools (`46%`), and No-Fit (`42%`) were still under-shortened.
   - Because `od3` had no per-type brake, Round 3 (`od3-r3`) and Round 4 (`od3-r4`) kept applying `75%` shortening to Math and Code while waiting for IFEval and Tools to catch up. On the full test ($2,759 \times 8$), `od3-r3` landed at **`58.4%` saved / `-4.01` accuracy** (Math `74% / -8.9`, Code `64% / -4.2`, IFEval `43% / -0.9`, Simple `42% / -1.3`, Several `43% / -0.6`, No-Fit `45% / -4.6`) and `od3-r4` landed at **`60.5%` saved / `-4.79` accuracy** (Math `76% / -9.4`, Code `67% / -4.8`).

2. **Why `od3` lost `-4.6` to `-5.4` on `no-fit` (`bfcl_irrelevance`) whereas every offline DPO run (`rpK`, `rpL`, `rpM`, `r2s1b`) gained `+3.4` to `+7.7`:**
   - On `bfcl_irrelevance`, the original model’s baseline solve rate is ~`52%` (so fewer than half the questions reach the `>= 7/8` shortening bar), and wrong answers are false-positive tool calls when the model over-thinks or hallucinates a parameter fit.
   - In `od3_arena`, `bfcl_irrelevance` was given `60%` shorten (gated at `7/8`) and `40%` length-matched `right` (`--right-hold`), with `0%` guards. When the model shortened across rounds, its `right` pairs forced chosen abstentions to match the length of rejected tool calls (`--right-hold` with `--ratio 1.15`), and the `7/8` bar restricted shortening to easy abstentions while leaving medium-difficulty questions unprotected against false tool calls.
   - Removing `bfcl_irrelevance` from `--right-hold` (or allowing compact `right` / `guard` pairs when braked) and capping its shortening cleanly restores the positive abstention gain seen across the entire full-test frontier (`50–66%` saved at `+3.4` to `+7.7`).

3. **Monitor-vs-Full-Test Gap Calibration:**
   - **Token Saving:** The 780-question test-matched free monitor tracks full-test saving closely, reading **`1.2` to `2.5` points lower** than the full test (`od3-r3`: monitor `55.9%` vs full test `58.4%`; `od3-r4`: monitor `59.3%` vs full test `60.5%`). Per-type monitor savings match full-test per-type savings within `1–2` points.
   - **Accuracy:** Both the free monitor and Dev-512 **over-read (are too kind on) accuracy** relative to the full test:
     - Free monitor (780 test-matched $\times 8$): `od3-r3` read `-3.38` vs `-4.01` full (`+0.63` too kind); `od3-r4` read `-2.50` vs `-4.79` full (`+2.29` too kind; noise + abstention skew). Across rounds, the free monitor over-reads full-test accuracy by **`+1.0` to `+2.0` points**.
     - Dev-512 ($\times 8$): `od3-r4` read `62.2% / -3.57` vs `60.5% / -4.79` full (`+1.22` too kind on accuracy, `+1.7%` higher on saving); `r2s1b-s180` read `51.1% / +1.25` vs `50.3% / -1.70` full (`+2.95` too kind on accuracy, `+0.8%` higher on saving).
   - **Calibration Rule for Stopping and Gating:** To guarantee `> 50.0%` tokens saved and `> -1.0` accuracy on the full test:
     - A model must show **`>= 51.0%` saved and `>= +0.2` accuracy on the free monitor**, AND **`>= 52.0%` saved and `>= +1.2` accuracy on Dev-512** (with **Dev-SVAMP `>= -0.5`** and **Dev-512 Math `>= -2.0`**).

---

## 2. VARIANT S (SAFE, NO REPAIR): Direct Landing via Interior-Frontier Brake (`od4s`)

Variant S is designed to land **directly inside the full-test target box (`52–54%` saved, `-0.5` to `-0.9` accuracy)** in a single online loop from `Qwen/Qwen3-0.6B`, with **no separate repair stage**. To achieve this without a repair step, per-type saving caps cannot be set at the *cliff edge* (where Math loses `-2.9` and Code loses `-1.8`); they must be derived from the **interior safe frontier** of `pertype_fulltest_points.json` where weighted accuracy loss sums to better than `-1.0`.

### A. Data-Derived Per-Type Caps & Floors (from `pertype_fulltest_points.json`)
Using the official benchmark weights (`0.25` Math, `0.25` Code, `0.25` IFEval, `0.25` Tools split `400:200:240` across Simple / Several / No-Fit):
- **Math (`svamp`, weight `0.25`):** The full-test frontier shows `54% / -1.2` (`rpK60`), `60% / -1.5` (`rpL40`), `64% / -2.7` (`rpL60`), `66% / -2.9` (`rpL80`), and cliff `67% / -6.6` (`rc1`). To keep Math accuracy around `-1.2` to `-1.5` without a subsequent repair step, Math must brake as soon as its round monitor crosses **`56%`** (which yields ~`58–60%` after that round’s step update), with a thinking floor of **`0.42`** of the original’s median right thinking per question.
- **Code (`mbpp`, weight `0.25`):** The full-test frontier shows `52% / -0.1`, `58% / -0.9` (`rpL40`), `62% / -0.9` (`rpL60`), `66% / -1.8` (`rpL80`), and cliff `67% / -4.8` (`od3-r4`). Braking Code at **`58%`** lands Code at `58–61%` saved with `-0.5` to `-0.9` accuracy. Thinking floor: **`0.38`**.
- **IFEval (`ifeval`, weight `0.25`):** Full-test frontier shows `46% / -1.9`, `55% / -3.0`, `62% / -2.6`, and no-thinking `62% / -1.3` (most loss in `od3` came from response length shrinkage on counting rules, not thinking). Braking at **`52%`** with `20%` counting-rule pairs and response-length holding lands IFEval at `48–52%` saved and `-0.5` to `-1.0` accuracy. Thinking floor: **`0.30`**.
- **One Function (`bfcl_simple_python`, weight `0.119`):** Frontier shows `41% / +0.3`, `43% / -0.3` (`rpL40`), `45% / -0.3`, `47% / -2.0` (`rpL60`), and cliff `51% / -5.1` (`rpL80`). For no-repair safety, brake at **`44%`** (before the `47% / -2.0` drop), landing at `44–45%` saved and `-0.3` accuracy. Thinking floor: **`0.40`**.
- **Several Functions (`bfcl_multiple`, weight `0.060`):** Frontier shows `45% / -0.6` (`rpL40`), `48% / -1.0` (`rpM60`), `49% / -0.6` (`rpL60`), and cliff `53% / -2.9` (`rpL80`). Brake at **`47%`**, landing at `47–49%` saved and `-0.6` accuracy. Thinking floor: **`0.38`**.
- **No Function Fits (`bfcl_irrelevance`, weight `0.071`):** Frontier shows `50–66%` saved at `+3.4` to `+7.7` accuracy (`rpL40` `58% / +3.9`, `rpM60` `62% / +6.2`). Brake at **`58%`**, thinking floor **`0.30`**.
- **Expected Weighted Total at Caps:**
  - Token Saving: $0.25(58) + 0.25(60) + 0.25(49) + 0.119(45) + 0.060(48) + 0.071(54) = \mathbf{53.8\%}$ saved.
  - Full-Test Accuracy: $0.25(-1.4) + 0.25(-0.8) + 0.25(-0.8) + 0.119(-0.3) + 0.060(-0.6) + 0.071(+3.0) = \mathbf{-0.61}$ points.

### B. Loop Specification (Variant S)
- **Starting Model & Reference:** Start from `Qwen/Qwen3-0.6B`. One LoRA adapter (`rank 16`, `alpha 32`) continued across rounds with `dpo.py --ref-base` (the reference model in every round is the frozen original `Qwen/Qwen3-0.6B`).
- **Hyperparameters:** Constant learning rate **`1.2e-5`** (slightly gentler than `od3`’s `1.5e-5` so Round 2 does not jump past the `56%` Math cap in one step), `beta = 0.1`, batch size `16` (`grad_accum = 8`), `fp16` autocast on `float32` master weights.
- **Rounds & Steps per Round:** Up to **6 rounds**:
  - **Round 1:** `20` DPO steps (`320` pairs) built from the original model’s stored answers on `od3prep` Slice 1 (`1,560` test-matched questions: `svamp=440, mbpp=280, ifeval=400, bfcl_simple_python=160, bfcl_multiple=120, bfcl_irrelevance=160`).
  - **Rounds 2–6:** `15` DPO steps (`240` pairs) per round on `od3prep` Slices 2–6 (`780` new test-matched questions $\times 8$ answers per round: `svamp=220, mbpp=140, ifeval=200, bfcl_simple_python=80, bfcl_multiple=60, bfcl_irrelevance=80`).
- **Pairs per Round & Fixed-Mix Rule (`--mix-preset od3_arena` + `--no-shorten-parts` brake):**
  - Round 1 quotas (`320` pairs): `svamp=80, mbpp=56, ifeval=88, bfcl_simple_python=36, bfcl_multiple=28, bfcl_irrelevance=32`.
  - Rounds 2–6 quotas (`240` pairs): `svamp=60, mbpp=42, ifeval=66, bfcl_simple_python=27, bfcl_multiple=21, bfcl_irrelevance=24`.
  - **Active (Unbraked) Per-Type Mix:**
    - `svamp`: `75%` shorten, `20%` length-matched `right` (`--right-hold`), `5%` anti-loop (`0%` guard).
    - `mbpp`: `75%` shorten, `15%` length-matched `right`, `10%` anti-loop (`0%` guard).
    - `ifeval`: `20%` counting-rule pairs (`--count-mode length --count-resp-min 1.10 --count-resp-cap 1.75 --count-resp-pad 60 --no-shorten-counting`); of the remaining `80%`, `75%` shorten / `15%` length-matched `right` / `10%` response anti-loop (`60%` shorten / `12%` right / `8%` loop / `20%` counting overall).
    - `bfcl_simple_python` & `bfcl_multiple`: `75%` shorten, `25%` length-matched `right`.
    - `bfcl_irrelevance`: `60%` shorten, `40%` compact `right` (removed from `--right-hold`, see Code Change 1 below).
  - **Braked Per-Type Mix (how the caps act):** Before building pairs in Round $k \ge 2$, `od3_monitor` scores Round $k$’s fresh answers against the original on that round’s 780-question slice. Any part whose monitored token saving is `>=` its cap (`svamp=56, mbpp=58, ifeval=52, bfcl_simple_python=44, bfcl_multiple=47, bfcl_irrelevance=58`) OR whose monitored accuracy change is `<= --od-acc-hold -1.5` is passed to `--no-shorten-parts` for that round (and latched for subsequent rounds if still above cap). When braked, that part gets **`0%` shorten, `90%` length-held `right` (`--right-hold`), `10%` anti-loop** (plus `20%` counting rules on `ifeval`). Under the fixed-mix rule, unbraked types keep their exact quotas and never refill a braked type’s shortening slots.
- **Thresholds & Floors:**
  - Shortening only on questions solved **`>= 7/8` (`--shorten-min-right 0.875`)**, spread **`>= 1.30` (`--shorten-ratio 1.30`)**.
  - Thinking-only shortening (`--think-only all`) with response length held (`--resp-median 0.95 --resp-max 1.05 --resp-near-long 0.95 --right-resp-cap 1.15`).
  - Per-type floors (`--base-think data/od3-base-think.json`): `--floor-by-part svamp=0.42,mbpp=0.38,ifeval=0.30,bfcl_simple_python=0.40,bfcl_multiple=0.38,bfcl_irrelevance=0.30`.

### C. Stop Rule & Monitor Under-Read Correction (Variant S)
- **Free Monitor Stop Rule:** Stop at the first round $k \ge 3$ where the free monitor shows **`saved >= 51.5%` AND `acc >= -0.2`** (`--od-stop-save 51.5 --od-stop-acc -0.2`), or by convergence from Round 5 onward (`saved >= 51.0%` and gain `< 1.5%`).
- **Why `-0.2` on the monitor:** Because the 780-question monitor under-reads accuracy loss by `~0.6` to `1.2` points when Math and Code are braked below `60%`, requiring `>= -0.2` on the monitor corresponds to **`-0.7` to `-0.9` on the full test**.
- **Dev-512 + Dev-SVAMP Verification & `fullpick`:** Run `devscore` and `devsvamp` on Rounds 3, 4, and 5 (`--od-dev-rounds 3 4 5`), then pick the candidate using `fullpick` with bias correction **`--fp-save-bias 1.2 --fp-acc-bias 1.8`** (so a Dev-512 score must exceed `51.2%` saved and `+0.8` accuracy to be deemed inside the `> 50% / > -1.0` full-test box).

### D. Expected Full-Test Landing & Risks (Variant S)
- **Expected Full-Test Landing:** **`52.5% – 54.0%` tokens saved at `-0.6` to `-0.9` accuracy** (Math `58% / -1.4`, Code `60% / -0.8`, IFEval `48% / -0.8`, Simple `44% / -0.3`, Several `47% / -0.6`, No-Fit `52% / +2.5`).
- **Risks:**
  1. *Discrete round jumps:* Even at `lr = 1.2e-5`, Round 2 may take Math from `~28%` to `59%` in one 15-step increment before the Round 3 brake engages. Because `59%` is still on the safe Math plateau (`60% / -1.5`), engaging the brake in Round 3 holds Math at `58–60%` during Rounds 3–4 while IFEval and Tools climb from `~32%` to `45–50%`.
  2. *Tight accuracy margin:* Without a dedicated widened-guard repair step, any premature-stop habits acquired in Rounds 1–2 are only arrested (via `90%` length-held `right` + `10%` `loop` after braking), not actively reversed with widened guards.

---

## 3. VARIANT A (AGGRESSIVE + REPAIR): Cliff-Edge Braked Arena (`od4a`) + Pure Repair (`rpO`)

Variant A separates compression from restoration:
1. **Stage 1 (`od4a`):** An online loop with per-type caps set right at the **data-derived cliff edges** from `pertype_fulltest_points.json` (`65 / 66 / 60 / 47 / 50 / 65`), pushing overall saving to **`55–57%`** while preventing the catastrophic Math (`74–76%`) and Code (`67%`) collapse seen in `od3`.
2. **Stage 2 (`rpO`):** **One pure repair step (`0%` shortening across all 6 types)** on a fresh test-matched question slice (`data/od3-step2-questions.jsonl`) that combines length-held `right` pairs, widened premature-stop `guard` pairs (`--guard-wide`), IFEval counting-rule pairs, and anti-loop pairs to bring full-test accuracy from `~ -2.2` back to **`-0.4` to `-0.8`** while giving back only `~2–3%` saving (landing at **`52.5–54.5%` saved**).

### A. Stage 1 Loop Specification (`od4a`: Aggressive Braked Arena)
- **Starting Model & Reference:** `Qwen/Qwen3-0.6B`, LoRA rank 16, fixed reference = original (`dpo.py --ref-base`), one adapter continued across rounds.
- **Hyperparameters:** Constant learning rate **`1.5e-5`**, `beta = 0.1`, batch size `16`.
- **Rounds, Steps, and Quotas:** Up to **5 rounds** (Round 1: `20` steps / `320` pairs on stored Slice 1; Rounds 2–5: `15` steps / `240` pairs on Slices 2–5; exact same per-type quotas and `od3_arena` active mix as Variant S).
- **Data-Derived Cliff-Edge Caps (`--od-caps`):**
  - Taken directly from the cliff boundaries in `pertype_fulltest_points.json`:
    - `svamp=65` (safe frontier up to `66% / -2.9` in `rpL80`; falls off at `67% / -6.6` in `rc1` and `74% / -8.9` in `od3-r3`).
    - `mbpp=66` (safe frontier up to `66% / -1.8` in `rpL80`; falls off at `67% / -4.8` in `od3-r4`).
    - `ifeval=60` (frontier reaches `62% / -2.6` and no-thinking `62% / -1.3`).
    - `bfcl_simple_python=47` (safe to `47% / -2.0` in `rpL60`; cliff at `51% / -5.1` in `rpL80`).
    - `bfcl_multiple=50` (safe to `49% / -0.6` in `rpL60`; falls at `53% / -2.9` in `rpL80`).
    - `bfcl_irrelevance=65` (positive throughout `50–66%` at `+3.4` to `+7.7`).
  - **How the Caps Act:** In each round $k \ge 2$, `odpo_fixed` scores the round’s 780 answers first; any type with monitored saving `>=` its cap gets `--no-shorten-parts` (`0%` shorten, `90%` length-held `right`, `10%` `loop`). `--od-acc-hold` is set to `-100` (off) so Stage 1 reaches the full saving target (`55–57%`) without braking prematurely on noisy single-round monitor accuracy dips.
- **Thresholds & Floors:** Solved **`>= 7/8` (`--shorten-min-right 0.875`)**, `--shorten-ratio 1.30`, `--floor-by-part svamp=0.38,mbpp=0.35,ifeval=0.25,bfcl_simple_python=0.35,bfcl_multiple=0.35,bfcl_irrelevance=0.25`.
- **Stage 1 Stop Rule (Corrected for Monitor Under-Read):**
  - Stop when the free monitor reaches **`saved >= 54.5%` AND `acc >= -2.2`** (`--od-stop-save 54.5 --od-stop-acc -2.2`), or at Round 4/5 convergence (`--od-conv-min 53.5 --od-conv-gain 1.5`).
  - **Calibration:** `54.5–56.0%` on the free monitor corresponds to **`56.0–57.5%` saved on the full test**, and `-1.5` to `-2.2` on the monitor corresponds to **`-2.0` to `-2.8` on the full test**—giving Stage 2 (`rpO`) a `5–7%` saving buffer above `50%` to trade `~2–3%` length for `+1.5` to `+2.0` accuracy points.
  - Run `devscore` and `devsvamp` on Rounds 3 and 4 (`--od-dev-rounds 3 4`) and pick the Stage 1 winner (`models/od4-best`) that meets Dev-512 `saved in [55.0%, 58.5%]` and `acc >= -0.5` (which corresponds to full-test `~ -2.2`).

### B. Stage 2 Pure Repair Specification (`rpO`: Zero Shortening)
- **Starting Model & Reference:** Fresh LoRA (`rank 16`) initialized on and anchored to the merged Stage 1 winner **`models/od4-best`** (`ptrepair --rp-base models/od4-best`).
- **Question Pool:** `data/od3-step2-questions.jsonl` (pre-built by `od3prep`, disjoint from all arena slices and from Dev/Dev-SVAMP, difficulty-matched to the full test), answered $\times 8$ (`--rp-samples 8`) by `models/od4-best`.
- **Total Pairs & Per-Type Quotas (`1,280` pairs = `80` steps at batch `16`, 1 epoch):**
  - `svamp=400` (`31.25%` of pairs — heaviest repair weight on Math where premature stopping happens).
  - `mbpp=280` (`21.9%` — Code correctness + anti-loop).
  - `ifeval=280` (`21.9%` — counting-rule compliance + response loop cleanup + instruction adherence).
  - `bfcl_simple_python=120`, `bfcl_multiple=100`, `bfcl_irrelevance=100` (`25.0%` combined tools).
- **Per-Type Kind Mix (`--rp-kinds mix --mix-preset hold` with `--guard-wide` and `--right-hold`, `0%` shorten everywhere):**
  - **Math (`svamp=400`):** `0%` shorten, **`60%` length-held `right`** (`240` pairs, `--right-hold`, chosen/rejected within `1.15x` length, half strictly longer), **`35%` widened `guard`** (`140` pairs, `--guard-wide svamp,mbpp`: shortest right answer `>= 0.70x` original’s right median preferred over premature-stop wrong answer `< 0.55x` original’s right median, capped at `<= 1.35x` rejected + `45` tokens so it never teaches verbose rambling), **`5%` anti-loop** (`20` pairs).
  - **Code (`mbpp=280`):** `0%` shorten, **`60%` length-held `right`** (`168` pairs), **`25%` widened `guard`** (`70` pairs), **`15%` anti-loop** (`42` pairs).
  - **IFEval (`ifeval=280`):** `0%` shorten, **`20%` counting-rule pairs** (`56` pairs, `--count-mode length --count-resp-min 1.10 --count-resp-cap 1.75 --count-resp-pad 60`), **`68%` length-held `right`** (`190` pairs), **`12%` response anti-loop** (`34` pairs, `--loop-response-only ifeval`).
  - **Tools (`bfcl_simple_python=120, bfcl_multiple=100, bfcl_irrelevance=100`):** `0%` shorten, **`85%` `right`**, **`15%` `guard`** (length-held on `simple` and `multiple`; unconstrained compact `right` on `irrelevance`).
- **Hyperparameters & Checkpoints:**
  - Constant learning rate **`8e-6`**, `beta = 0.1`, batch size `16`, **`60` steps** (1 pass over `960` pairs, drawing from the `1,280`-pair stratified pool).
  - **Dev Checkpoints:** Score steps **`20, 40, 60`** (`--rp-screen 20 40 60 --rp-dev --rp-dev-svamp`).
- **Stage 2 Dev Gate (Corrected for Dev Over-Read):**
  - Pick the checkpoint $s \in \{20, 40, 60\}$ that satisfies:
    1. **Dev-512 Token Saving `>= 52.0%`** (guarantees `> 50.5%` on the full test).
    2. **Dev-512 Accuracy `>= +1.3`** (correcting for Dev-512’s `+1.8` to `+2.2` optimism, this guarantees **`-0.5` to `-0.9` on the full test**).
    3. **Dev-SVAMP Accuracy `>= -0.5`** and **Dev-512 Math `>= -2.0`**.

### C. Expected Full-Test Landing & Risks (Variant A)
- **Expected Full-Test Landing:**
  - After Stage 1 (`od4a`, Round 3 or 4): **`56.2%` saved / `-2.1` accuracy** (Math `65% / -3.0`, Code `64% / -1.9`, IFEval `46% / -1.5`, Simple `45% / -1.2`, Several `47% / -0.8`, No-Fit `50% / +1.5`).
  - After Stage 2 (`rpO`, step 40): **`53.2%` saved / `-0.6` accuracy** (Math `61% / -1.4`, Code `61% / -0.9`, IFEval `45% / -0.5`, Simple `44% / -0.3`, Several `46% / -0.5`, No-Fit `52% / +3.5`).
- **Risks:**
  1. *Guard give-back:* Widened guards (`35%` of Math, `25%` of Code) increase thinking length on previously premature-stopped questions by `~3–4%` on Math and `~2–3%` on Code (`~1.5–2.2%` overall). Starting Stage 2 from `56.2%` saved absorbs this give-back safely (`56.2% -> 53.2%`).
  2. *Compute budget:* `od4a` (4 rounds = `~3.8` GPU-h) + `rpO` generation and training + 3 Dev-512/Dev-SVAMP screens (`~2.8` GPU-h) totals **`~6.6` GPU-hours** on one L4, fitting cleanly inside the 6–8 GPU-hour budget.

---

## 4. Exact Small Code Changes Needed & Launchable `track.py` Plan Lines

### A. Small Code Changes (Both Optional/Surgical; Existing Flags Support Everything Else)
All mechanisms (`odpo --od-fixed-ref`, `--od-caps`, `--od-acc-hold`, `--floor-by-part`, `--base-think`, `--no-shorten-parts`, `--mix-preset od3_arena`, `--mix-preset hold`, `fullpick`) **already exist** in `track.py` and `repair_pairs.py`. Only **two small fixes** (6 lines total) improve reliability:
1. **In `repair_pairs.py` (preset `od3_arena`, around line 845):** Remove `bfcl_irrelevance` from `right_hold` when `--right-hold all` is passed (or pass `--right-hold svamp,mbpp,ifeval,bfcl_simple_python,bfcl_multiple` explicitly on the CLI, which works **with zero code change**!) so `bfcl_irrelevance` `right` pairs can prefer compact abstentions over long hallucinated tool-call thoughts.
2. **In `track.py` (`odpo_fixed`, around line 6485):** Latch `braked` parts across rounds (`braked_ever |= set(braked)`) so a part that hits its saving cap in Round $k$ cannot oscillate back to `75%` shortening in Round $k+1$ if random slice variance dips its monitored saving by `0.3%` below the cap. Specifically, add `braked_ever = set()` before the `for k in range(args.od_rounds):` loop and replace `if braked:` with `braked_ever.update(braked); if braked_ever:` when appending `--no-shorten-parts`.

### B. Exact `track.py` Plan Lines for VARIANT S (`od4s`: Safe, No Repair)

od3prep --od-rounds 6 --r2-drop runs/bon/Shuttle_Notation-result/eval_completions.jsonl
odpo --name od4s --od-fixed-ref --rp-base Qwen/Qwen3-0.6B --od-rounds 6 --od-r1-steps 20 --od-steps 15 --rp-samples 8 --rp-kinds mix --od-r1-quota svamp=80,mbpp=56,ifeval=88,bfcl_simple_python=36,bfcl_multiple=28,bfcl_irrelevance=32 --rp-quota svamp=60,mbpp=42,ifeval=66,bfcl_simple_python=27,bfcl_multiple=21,bfcl_irrelevance=24 --rp-pair-args=--mix-preset,od3_arena,--think-only,all,--prefer-reread,--count-rules,--count-mode,length,--count-resp-min,1.10,--count-resp-cap,1.75,--count-resp-pad,60,--no-shorten-counting,--shorten-ratio,1.30,--shorten-min-right,0.875,--resp-median,0.95,--resp-max,1.05,--resp-near-long,0.95,--right-resp-cap,1.15,--loop-response-only,ifeval,--right-hold,svamp,mbpp,ifeval,bfcl_simple_python,bfcl_multiple,--base-think,data/od3-base-think.json,--floor-by-part,svamp=0.42:mbpp=0.38:ifeval=0.30:bfcl_simple_python=0.40:bfcl_multiple=0.38:bfcl_irrelevance=0.30 --od-caps svamp=56,mbpp=58,ifeval=52,bfcl_simple_python=44,bfcl_multiple=47,bfcl_irrelevance=58 --od-acc-hold -1.5 --rp-lr 1.2e-5 --od-stop-save 51.5 --od-stop-acc -0.2 --od-conv-min 51.0 --od-conv-gain 1.5 --od-dev-rounds 3 4 5
fullpick --name od4s --fp-save-bias 1.2 --fp-acc-bias 1.8 --fp-out models/od4s-pick

### C. Exact `track.py` Plan Lines for VARIANT A (`od4a` + `rpO`: Aggressive + Pure Repair)

od3prep --od-rounds 6 --r2-drop runs/bon/Shuttle_Notation-result/eval_completions.jsonl
odpo --name od4 --od-fixed-ref --rp-base Qwen/Qwen3-0.6B --od-rounds 5 --od-r1-steps 20 --od-steps 15 --rp-samples 8 --rp-kinds mix --od-r1-quota svamp=80,mbpp=56,ifeval=88,bfcl_simple_python=36,bfcl_multiple=28,bfcl_irrelevance=32 --rp-quota svamp=60,mbpp=42,ifeval=66,bfcl_simple_python=27,bfcl_multiple=21,bfcl_irrelevance=24 --rp-pair-args=--mix-preset,od3_arena,--think-only,all,--prefer-reread,--count-rules,--count-mode,length,--count-resp-min,1.10,--count-resp-cap,1.75,--count-resp-pad,60,--no-shorten-counting,--shorten-ratio,1.30,--shorten-min-right,0.875,--resp-median,0.95,--resp-max,1.05,--resp-near-long,0.95,--right-resp-cap,1.15,--loop-response-only,ifeval,--right-hold,svamp,mbpp,ifeval,bfcl_simple_python,bfcl_multiple,--base-think,data/od3-base-think.json,--floor-by-part,svamp=0.38:mbpp=0.35:ifeval=0.25:bfcl_simple_python=0.35:bfcl_multiple=0.35:bfcl_irrelevance=0.25 --od-caps svamp=65,mbpp=66,ifeval=60,bfcl_simple_python=47,bfcl_multiple=50,bfcl_irrelevance=65 --od-acc-hold -100 --rp-lr 1.5e-5 --od-stop-save 54.5 --od-stop-acc -2.2 --od-conv-min 53.5 --od-conv-gain 1.5 --od-dev-rounds 3 4
fullpick --name od4 --fp-save-bias 1.2 --fp-acc-bias 1.8 --fp-out models/od4-best
ptrepair --name rpO --rp-base models/od4-best --rp-question-file data/od3-step2-questions.jsonl --rp-samples 8 --rp-seed 41 --rp-kinds mix --rp-pair-args=--mix-preset,hold,--think-only,all,--prefer-reread,--count-rules,--count-mode,length,--count-resp-min,1.10,--count-resp-cap,1.75,--count-resp-pad,60,--no-shorten-counting,--resp-median,0.95,--resp-max,1.05,--resp-near-long,0.95,--right-resp-cap,1.15,--loop-response-only,ifeval,--guard-wide,svamp,mbpp,--right-hold,svamp,mbpp,ifeval,bfcl_simple_python,bfcl_multiple,--base-think,data/od3-base-think.json --rp-quota svamp=400,mbpp=280,ifeval=280,bfcl_simple_python=120,bfcl_multiple=100,bfcl_irrelevance=100 --rp-lr 8e-6 --rp-steps 60 --rp-epochs 1 --rp-dpo-seed 0 --rp-screen 20 40 60 --rp-dev --rp-dev-svamp
r2pick --name rpO --pk-steps 20 40 60 --pk-base models/od4-best --pk-out rpO-best --pk-save-min 52.0 --pk-save-max 60.0 --pk-acc-min 1.3 --pk-svamp-min -0.5 --pk-math-min -2.0

---

## 5. Recommendation: Which Variant to Run First & What to Do With Queued `od4`

1. **Run VARIANT A (`od4` $\rightarrow$ `rpO`) FIRST on GPU 1.**
   - **Why:**
     1. **It directly leverages the already-queued `od4` run** as its Stage 1, and gives us **two shots at the target box inside a single 6.6-hour run**: if `od4` Round 3 or Round 4 with the per-type brake already scores `>= 52%` saved and `>= +1.3` accuracy on Dev-512, we can full-test `od4-best` directly (acting like Variant S); if `od4-best` lands at `56%` saved and `-2.0` full-test equivalent accuracy (as expected at the `65/66` cliff-edge caps), Stage 2 (`rpO`) immediately repairs it into the `> 50% / > -1.0` box using widened premature-stop guards (`--guard-wide svamp,mbpp`) that online shortening rounds cannot include without compounding length drift.
     2. **Empirical proof of repairability:** Across every full-test experiment in this project (`rc1 -> rpE -> rpK/rpL/rpM`), models compressed to `56–60%` saved had `-2.5` to `-4.5` accuracy prior to guard/right repair and recovered `+1.5` to `+2.5` accuracy points when given a pure/hold repair step.

2. **Verdict on Queued `od4`: RUN WITH THREE SMALL PARAMETER ADJUSTMENTS (Do Not Replace Its Architecture).**
   - The queued `od4` design (`od3` + per-type brake `--no-shorten-parts` + data-derived caps `svamp=65,mbpp=66,ifeval=60,bfcl_simple_python=47,bfcl_multiple=50,bfcl_irrelevance=65` + `--od-acc-hold -100`) is **architecturally sound and is the exact Stage 1 of Variant A**.
   - Before `od4` starts Round 2, make **three small adjustments** to the queued `od4` line (and append `rpO` after it):
     1. **Exclude `bfcl_irrelevance` from `--right-hold`** (`--right-hold,svamp,mbpp,ifeval,bfcl_simple_python,bfcl_multiple` instead of `all`) so `bfcl_irrelevance` recovers the `+3.4` to `+7.7` abstention accuracy gain instead of `od3`’s `-4.6` loss.
     2. **Latch braked parts across rounds** in `odpo_fixed` (`braked_ever.update(braked)`) so Math and Code stay braked once they cross `65% / 66%`.
     3. **Score both Round 3 and Round 4 on Dev-512 + Dev-SVAMP** (`--od-dev-rounds 3 4` and `--od-rounds 5` instead of `6`), then immediately run `ptrepair --name rpO` on `models/od4-best`.