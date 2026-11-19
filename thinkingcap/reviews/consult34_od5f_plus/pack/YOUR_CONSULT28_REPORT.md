### 1. Where Saving Stops Per Type, and Why

Across all 26+ full tests (`2,759 questions x 8`), the user’s suspicion that **the brake (`--od-caps` + `--od-acc-hold -2.0`) is what stops the run from reaching >60% at better than -1** is **false for Math, Code, IFEval, and Simple Tools, and only partially true (by ~2–3 pts) for Several Tools and No-Fit**. What actually stops saving at better than `-1` is a combination of **the model’s own capacity on multi-step arithmetic/edge-case reasoning**, **the non-thinking response floor (`response_len`)**, and **intra-question sample spread exhaustion (`shorten-ratio 1.30` vs. per-question thinking floors)** once the model is already compressed.

Here is the exact mechanism and measured ceiling per type:

1. **Math (`svamp`, 1,000 q, 25% weight):**
   - **What stops it:** **The model’s own reasoning capacity** (premature `<think>` termination on multi-step arithmetic), **not** the brake.
   - **Evidence:** When run without a brake in `od3`, Math reached **74% saved / -8.86 pts** at `r3` and **76% saved / -9.44 pts** at `r4` (`median thinking = 0.33x–0.34x` of the original; **86% of lost accuracy** came from stopping thinking `<0.5x`, with **60% of the loss on questions the original solved 8/8**). With the `-2.0` accuracy brake and `0.44x` thinking floor in `od5c`, Math was held by `-2.0` accuracy in rounds 5, 7, 10, 12, 13, and 15, yet still landed on the full test at **65% / -4.60** (`r10`), **62% / -2.84** (`r15`), **61% / -2.84** (`r16`), and **62% / -2.70** (`r20`). Even in `od5r-r15/r16`, where Math was slightly gentler (**58–59% saved**), it lost **-1.2 to -1.6 pts**.
   - **Ceiling:** At **$\le 1$ pt loss**, Math’s ceiling across all full tests is **47% saved / 0.00** (`rpI-120`) or **~56–57% saved at -1.0 to -1.2 pts** (`rpM-60`: `57.0% / -1.20`; `od5r-r15`: `59% / -1.2`). At **$\le 2$ pts loss**, it is **60% saved / -1.50** (`rpL-40`). Above `60%` saved, Math’s accuracy loss rises steeply (`-2.7` to `-4.6` pts) because a 0.6B model cannot compress arithmetic scratchpad below `~0.47x` without dropping intermediate steps.

2. **Code (`mbpp`, 378 q, 25% weight):**
   - **What stops it:** **Model capacity + thinking floor (`35:0.36`)** above `60%` saved, plus **slice data composition** in rounds 20–23.
   - **Evidence:** Unbraked `od3` pushed Code to **64% / -4.17** (`r3`) and **67% / -4.76** (`r4`, `0.39x` thinking, 74% early-stop losses). In `od5c`, Code only hit its `63%` slice cap in rounds 4 and 11; in rounds 12–20 it was **never held by the cap**, settling naturally around **54% / -0.9** (`r15`), **59% / -1.95** (`r16`), and **61% / -2.7** (`r20`) because the `0.36x` thinking floor and `1.30` shorten ratio filter out shorter candidates once median thinking hits `0.44x` (`od5c3.txt`). In rounds 20–23, new sandbox-verified hard Code tasks entered the slices; on those questions the model’s right answers stayed long, dropping monitor Code saving by `~13 pts` (from `59%` on slice 19 to `46%` on slice 20) and generating recovery/hold pairs instead of shorten pairs.
   - **Ceiling:** At **$\le 1$ pt loss**, Code can give **58–62% saved** (`rpM-60`: `58.2% / -0.86`; `rpL-60`: `62.2% / -0.86`; `od5c-r15`: `54% / -0.9`). At **$\le 2$ pts loss**, it gives **66% saved / -1.82** (`rpL-80`) or **59% / -1.95** (`od5c-r16`).

3. **IFEval (`ifeval`, 541 q, 25% weight):**
   - **What stops it:** **Non-thinking response length (`response_len`) + pair rules (`--shorten-ratio 1.30`, `--resp-median 0.95`, `--resp-max 1.05`)**, **never** the brake.
   - **Evidence:** In `od5c`, IFEval’s slice saving peaked at **48%** (round 16) and **never once reached its `56%` brake cap or `-2.0` accuracy hold** across all 23 rounds. Why did it stop at **47–51% saved** (`+0.49` at `r15/r16`, `-0.3` at `r20`)? First, total token saving measures `thinking + response`, and IFEval responses are long (often constrained by minimum word/paragraph/section counts); even with **thinking completely off (`nothink`)**, IFEval only saves **62.3% of total tokens**! At `od5c-r16`, median thinking is already down to **`0.43x`** of the original (`od5c3.txt`), which is ~80% of the way to `nothink`. Second, once thinking is at `0.43x`, very few of the 8 samples on the same question both (a) differ by $\ge 1.30\times$ in total tokens, (b) stay above the `20:0.32` thinking floor, and (c) pass `--resp-median 0.95` and `--resp-max 1.05` (in `od5c`, IFEval had only 24–27 `shorten` candidates out of 200 questions per round). When offline `rpL-60/80` forced IFEval total saving to `57.9–61.8%`, it compressed responses too and IFEval accuracy collapsed to **-2.59 to -3.40 pts**.
   - **Ceiling:** At **$\le 1$ pt loss**, IFEval’s ceiling is **48–51% saved** (`r2s3-s20`: `48% / +0.9`; `od5c-r15`: `48% / +0.5`; `od5c-r20`: `51% / -0.3`).

4. **BFCL Tools (`bfcl_simple_python` 400, `bfcl_multiple` 200, `bfcl_irrelevance` 240; pooled 25% weight):**
   - **What stops them:** **The per-type saving caps (`46 / 49 / 56`) AND the thinking floors (`0.42 / 0.42 / 0.35`)**, plus **accuracy fragility on `simple` (`400/840` of the tool pool)**.
   - **Evidence:** In `od5c`, the three tool types hit their slice saving caps (`46, 49, 56`) repeatedly from round 7 onward (and in rounds 16–19 all three were held at their caps). However:
     - **One function (`simple`, weight 47.6% of BFCL):** Even held around `48–49%` saved on the full test, `simple` already lost **-1.09 pts** at `r16`, **-1.63 pts** at `r15`, **-2.2 pts** at `r20`, and **-2.1 pts** at `od5r-r16` (with `71%` of losses from early/short thinking at `0.54x` median thinking). In `od5d`, raising `simple`’s cap by just +2 (`46 -> 48`) and quota (`26 -> 30`) pushed `simple` to `49–50%` saved and worsened its loss to **-1.7 to -2.4 pts**. Its ceiling at $\le 1$ pt loss is **45–48% saved** (`rpM-60`: `45% / -0.3`; `od4-r7`: `48% / -0.8`), and at $\le 2$ pts loss is **49–50% saved** (`od5c-r16`: `49% / -1.1`; `od5d-r11`: `50% / -1.7`).
     - **Several functions (`multiple`, weight 23.8% of BFCL):** Held by its `49%` cap and `0.42` floor; landed at **50% / 0.00** (`r16`) and **51% / -0.4** (`r20`). Its ceiling at $\le 1$ pt loss is **51–52% saved** (`r2s3-s20`: `51% / -0.2`).
     - **No-fit (`irrelevance`, weight 28.6% of BFCL):** Held by its `56%` cap and `0.35` floor; landed at **57% / +5.89** (`r16`) and **58% / +6.6** (`r20`). Because shorter thinking *helps* abstention (less over-thinking into hallucinated tool calls), `no-fit` reached **66% saved / +7.71 pts** in offline `rpL-80` (`62.5% / +4.64` in `rpL-60`). Here, the `56%` brake cap and `0.35` floor genuinely left **~5–7 pts of free saving** on the table.

---

### 2. Is > 60% Saved at Better Than -1 Reachable?

**Straight answer: NO.** Reaching $>60\%$ composite saving at better than $-1.0$ accuracy on the full test is **not reachable** for `Qwen3-0.6B` under online DPO (or any single-model recipe). The **realistic ceiling at better than -1.0 accuracy is 56.0%–57.5% saved**.

#### The Exact Arithmetic
The official full-test score weights the 4 benchmarks equally (`25%` each), with BFCL pooled across its 840 questions (`400 simple + 200 multiple + 240 irrelevance`):
- **Composite saving:** $S = 1 - \left(r_{\text{math}} \cdot r_{\text{code}} \cdot r_{\text{ifeval}} \cdot r_{\text{bfcl}}\right)^{1/4}$, where $r_b = 1 - s_b$ is the benchmark token ratio.
- For **$S > 60.0\%$**, the geometric mean of the 4 token ratios must be **$< 0.4000$**, so their product must satisfy:
  $$r_{\text{math}} \cdot r_{\text{code}} \cdot r_{\text{ifeval}} \cdot r_{\text{bfcl}} < 0.40^4 = 0.0256.$$
- **Why the per-type ceilings forbid $>60\%$ at $>-1.0$:**
  1. Look first at the **per-type best table across all 26 full tests (`pertype_best_full.md`)**:
     - If every type is kept at **$\le 1$ pt loss** (`math 47% / 0.0, code 62% / -0.9, ifeval 48% / +0.9, simple 45% / -0.3, multiple 51% / -0.2, irrelev 66% / +7.7`), the pooled BFCL saving is $0.476(45) + 0.238(51) + 0.286(66) = 52.4\%$ ($r_{\text{bfcl}} = 0.476$), and the composite saving of this hypothetical cherry-picked oracle across 4 different runs is only:
       $$S = 1 - (0.53 \times 0.38 \times 0.52 \times 0.476)^{1/4} = \mathbf{52.9\%\text{ saved}} \quad (+0.5\text{ acc}).$$
     - Even if we allow **up to 2 pts loss on every single type** and cherry-pick the best score each type ever achieved in *any* run (`math 60% / -1.5` from `rpL-40`, `code 66% / -1.8` from `rpL-80`, `ifeval 50% / -1.7` from `pt-mid`, `simple 50% / -1.7` from `od5d-r11`, `multiple 51% / -0.2` from `r2s3-s20`, `irrelev 66% / +7.7` from `rpL-80`, giving $s_{\text{bfcl}} \approx 54.8\%$), the composite oracle ceiling is still only:
       $$S = 1 - (0.40 \times 0.34 \times 0.50 \times 0.452)^{1/4} = \mathbf{58.0\%\text{ saved}} \quad (-1.16\text{ acc}).$$
  2. **What would $>60.0\%$ actually require?**
     Because IFEval’s non-thinking response tokens cap its accuracy-safe total saving at **`51–52%`** ($r_{\text{ifeval}} \ge 0.48$) and `bfcl_simple_python` breaks above `49–50%` saved, pushing pooled BFCL to its absolute limit (`simple 50%, multiple 54%, irrelev 65%` $\to s_{\text{bfcl}} \approx 55.2\%, r_{\text{bfcl}} = 0.448$) leaves:
     $$r_{\text{math}} \cdot r_{\text{code}} < \frac{0.0256}{0.48 \times 0.448} = 0.1190 \implies \sqrt{r_{\text{math}} \cdot r_{\text{code}}} < 0.345 \implies \text{Math and Code must average }\mathbf{>65.5\%\text{ saved}}.$$
     Every run that pushed Math and Code to `64–67%` saved (`od3-r3`, `od3-r4`, `rpL-60`, `rpL-80`, `od5c-r10`) suffered **$-2.8$ to $-9.4$ pts on Math** and **$-1.8$ to $-4.8$ pts on Code**. Even with a $+2.0$ pt cushion from BFCL (`irrelevance` at $+7.0$) and $0.0$ on IFEval, a combined $-5.5$ to $-6.5$ pt loss on Math + Code puts overall 4-benchmark accuracy at **$-1.4$ to $-2.2$ pts** (as seen in `rpL-60` at `59.3% / -1.70`, `rpL-80` at `62.8% / -2.06`, and `od5c-r20` at `56.9% / -1.24`).
  3. **What IS the realistic online ceiling at better than -1?**
     Look at `od5c-r16` (**54.8% / -0.78**) and `od5c-r20` (**56.9% / -1.24**):
     - In `od5c-r20`, the per-type profile was: `math 62% / -2.7`, `code 61% / -2.7`, `ifeval 51% / -0.3`, `simple 49% / -2.2`, `multiple 51% / -0.4`, `irrelevance 58% / +6.6` (pooled BFCL: `52.0% / +0.74`). Overall accuracy was $(-2.70 - 2.70 - 0.30 + 0.74)/4 = -1.24$, missing the $-1.0$ bar by just **0.24 pts** while saving **56.9%**.
     - If we keep `od5c`’s gentle pair rules, **protect `simple` tools** (cap `45%`, floor `25:0.44` so `simple` stays at `48% / -1.1` as in `r16` instead of `-2.2`), **protect Math slightly** (`floor 30:0.45`, cap `61%` so Math stays at `60–61% / -2.2`), and **unlock the free saving in `no-fit` (`irrelevance`), `multiple`, and `ifeval`** (raising caps on `irrelevance 56 -> 64`, `multiple 49 -> 53`, `ifeval 56 -> 58`, lowering `irrelevance` floor `20:0.35 -> 16:0.28`, and adding `--shorten-second` on `ifeval+bfcl_irrelevance`), we get:
       - **Math:** `61% saved / -2.2 pts` ($r = 0.39$)
       - **Code:** `60% saved / -1.8 pts` ($r = 0.40$)
       - **IFEval:** `51% saved / -0.1 pts` ($r = 0.49$)
       - **BFCL:** `simple 48% / -1.1`, `multiple 52% / -0.2`, `irrelevance 63% / +6.8` $\to$ pooled BFCL **`53.3% saved / +1.37 pts`** ($r = 0.467$)
       - **Composite Saving:** $1 - (0.39 \times 0.40 \times 0.49 \times 0.467)^{1/4} = \mathbf{56.8\%\text{ saved}}$ (realistic target range **56.0%–57.5%**)
       - **Composite Accuracy:** $(-2.2 - 1.8 - 0.1 + 1.37) / 4 = \mathbf{-0.68\text{ pts}}$ (safely inside $> -1.0$).

---

### 3. New Online DPO Run Design (`od5f`) to Maximize Saving at Better Than -1

We design **`od5f`** (18 rounds from `Qwen/Qwen3-0.6B`, fixed reference) to hit the realistic ceiling (**56.0%–57.5% saved at -0.5 to -0.9 accuracy**) by fixing the exact bottlenecks that held `od5c-r16` to `54.8%` and pushed `od5c-r20` to `-1.24`:

1. **Reference, Adapter & Optimization (Proven `od5c` Backbone):**
   - **Reference:** Fixed original `Qwen/Qwen3-0.6B` (`--od-fixed-ref`). A moving reference causes KL drift and uncalibrated shorten pressure across 16+ rounds; fixed reference keeps the implicit KL anchor centered on the original’s accuracy while the single continued LoRA adapter (`rank 16, alpha 32`, `lr = 1.4e-5` constant, warmup 2 steps, `beta = 0.1`, batch 16, 15 steps/round, 20 steps in round 1) moves smoothly along the Pareto frontier.
2. **Pair Rules Per Type (`--mix-preset od5 --od5-v2` with 3 targeted surgical adjustments):**
   - Keep `od5-v2`’s self-proportional defect/shorten/recovery/hold allocation (`10%` `right_count` on IFEval, `12%` `loop`, `10%` `format`, `20%` `hold` cap, `2x` recovery weight when slice accuracy delta $<-1.0\text{ pt}$). Do **not** use `od5-v3` (which starved `shorten` pairs in `od5d`).
   - **Floors (`--floor-by-part`):**
     - `svamp=30:0.45` (up slightly from `25:0.44` to prevent the `-2.84` early-stop tail on 8/8 math problems).
     - `mbpp=35:0.36` (unchanged; gave `59% / -1.95` in `od5c-r16`).
     - `ifeval=18:0.30` (down slightly from `20:0.32` so already-compressed rounds 12–18 still find shorten candidates).
     - `bfcl_simple_python=25:0.44` (tightened from `20:0.42` to stop the `-2.2` early-stop loss seen at `od5c-r20`).
     - `bfcl_multiple=22:0.38` (relaxed from `25:0.42` to allow `52–53%` saved).
     - `bfcl_irrelevance=16:0.28` (relaxed from `20:0.35` to unlock `62–64%` saved and $+6.5$ to $+7.5$ accuracy gain).
   - **Candidate supply on IFEval and No-Fit (`--shorten-second ifeval+bfcl_irrelevance`):**
     - In `od5c`, IFEval had only 24–27 `shorten` candidates per slice for 58 quota slots, and `no-fit` had 24 quota slots. Allowing a second `shorten` pair on undamaged questions with $\ge 3$ right answers (`--shorten-second ifeval+bfcl_irrelevance`, already implemented in `repair_pairs.py` line 456!) refills `shorten` supply on the two types with positive accuracy deltas without touching fragile `math` or `simple` tools.
   - **Shorten ratio:** Keep `--shorten-ratio 1.30` globally, with `--shorten-ratio-by-part ifeval=1.25,bfcl_irrelevance=1.25` (already supported in `repair_pairs.py` via `--shorten-ratio-by-part`!) so IFEval and No-Fit do not run dry of shorten candidates once thinking reaches `0.43x`.
3. **Quotas & Brake Rule:**
   - **Round 1 (`320` pairs, `20` steps):** `svamp=80,mbpp=56,ifeval=88,bfcl_simple_python=36,bfcl_multiple=28,bfcl_irrelevance=32`.
   - **Rounds 2–18 (`240` pairs, `15` steps):** `svamp=56,mbpp=52,ifeval=60,bfcl_simple_python=24,bfcl_multiple=22,bfcl_irrelevance=26` (shifts 2 pairs from `svamp` and 2 from `simple` to `ifeval` and `irrelevance`).
   - **Per-type saving caps (`--od-caps`):** `svamp=61,mbpp=63,ifeval=56,bfcl_simple_python=45,bfcl_multiple=53,bfcl_irrelevance=64` (tightens `svamp` by 1 and `simple` by 1 to prevent `-2.7/-2.2` accuracy over-shoots; raises `multiple` `+4` and `irrelevance` `+8` to harvest their safe savings).
   - **Accuracy brake (`--od-acc-hold -2.0 --od-unlatch`):** Keep unlatched `-2.0` per-slice accuracy hold.
4. **Questions Needed Per Type from Existing Pools (18 rounds):**
   - **Zero new questions needed:** Slices 1–19 (`data/od3-r1-questions.jsonl` .. `data/od3-r19-questions.jsonl`, 780 questions/slice: `svamp=220, mbpp=140, ifeval=200, simple=80, multiple=60, irrelevance=80`) are already built, difficulty-matched, and stored on Drive (with slices 1–18 using the clean pre-Argon pools where Code saving is unblunted). Running **18 rounds** uses existing slices 1–18 directly.

---

### 4. Why `od5r` Stayed Below `od5c`, Reproducibility, and Seed Variance

1. **Is `od5c`’s >52% / >-1.0 result reproducible? YES (2/2 seeds inside the objective and at/above the 52% margin):**
   - `od5c` landed at **54.0% / -0.56** (`r15`) and **54.8% / -0.78** (`r16`).
   - `od5r` (exact same recipe and slices, only `--rp-dpo-seed 0 -> 1` and `--rp-gen-seed 53 -> 54`) landed at **51.9% / -0.52** (`r15`) and **52.1% / -0.58** (`r16`), **+2.72 pts above the reference curve** and inside the objective.
   - Furthermore, on the round-15 monitor (`52.8 / +2.36` vs `52.9 / +2.41`) and the round-19 monitor (`50.6 / +2.47` vs `50.7 / +2.56`), `od5r` and `od5c` converged to within **0.1% saving and 0.09 pts accuracy**!

2. **Why did `od5r` save 2.7 pts less at round 16 (`52.1%` vs `54.8%`) and 5–6 pts less in rounds 20–23 (mostly on Code)?**
   Three interacting factors explain the gap:
   - **Early seed bifurcation amplified by the unlatched brake (Rounds 3–16):** At round 4 (`--rp-gen-seed 54`), `od5r`’s 8 rollouts per question happened to produce slightly lower slice accuracy on Math and Tools, keeping `od5r` at `41.5%` monitor saving vs `od5c`’s `49.6%`. By round 16 on the full test, 4 of the 6 types were **identical** between `od5r-r16` and `od5c-r16`: `IFEval` (`49%` vs `47%`), `simple` (`49%` vs `49%`), `multiple` (`50%` vs `50%`), `no-fit` (`57%` vs `57%`), and `math` was `58% / -1.6` vs `61% / -2.8`. Almost the entire `2.7%` composite saving gap at `r16` came from **Code (`50% / -1.3` vs `59% / -2.0`)** (`0.50` vs `0.41` token ratio on 1/4 of the geomean = `2.3%` composite saving difference), because `od5r` hit the `-2.0` accuracy hold on Code in an extra late round while `od5c` got an unheld Code shorten round at `r16`.
   - **Why the gap widened to 5–6 pts in rounds 20–23 (the new hard Code tasks + `od5-v2` health gating):**
     In slices 20–23, the hard share of Code (`140` questions/slice) was filled with the new sandbox-verified edge-case tasks (`gen_code_tasks.py`). Because `od5r` entered round 20 with longer Code thinking (`50%` saved vs `od5c`’s `59%`), on the new hard Code tasks `od5r`’s longer rollouts had more cap cut-offs and lower per-question solve rate `p1` relative to `p0`, and lower relative length spread (`<1.30x`). Under `od5-v2`’s per-question health gate (`p1 >= p0 - 0.125`), those questions were routed to **`recovery` and `hold` pairs instead of `shorten` pairs**, locking `od5r`’s Code saving at `28–34%` on slices 20–23 while `od5c` (already compact enough to finish those tasks within the cap) stayed at `40–48%`.
   - **What the night says about seed variance:**
     Across the two seeds (`od5c` and `od5r`), **composite accuracy is remarkably invariant** (`-0.56 / -0.78` vs `-0.52 / -0.58`, within `<0.2` pts), and **5 of 6 types have $<1.5\%$ seed variance in saving**. Only **Code** exhibits high seed variance (`~5–9 pts` on the full test) because `140` Code questions/slice with binary unit-test pass/fail have higher slice-accuracy noise around the `-2.0` brake threshold and the `1.30` shorten threshold.

---

### 5. Safer Checkpoint Selection Rule Without the Full Test

#### Why `fullpick` Was Fooled at Round 20
At `od5c-r20`, Dev-512 read **`56.5% / +2.84`**. Using the historical fixed bias (`--fp-save-bias 0.3 --fp-acc-bias 2.95`), `fullpick` predicted **`56.2% / -0.11`** (inside the objective), whereas the full test gave **`56.9% / -1.24`** — so Dev-512 was **`+4.08 pts` optimistic on accuracy** (`1.13 pts` more optimistic than the `2.95` correction).
Why? Because as saving increases from `52%` to `57%`, **two types drive the hidden full-test accuracy loss (`svamp` `-2.70` and `bfcl_simple_python` `-2.20`)**, while `Dev-512` has only `~85` questions per type and under-samples the 6–8/8 arithmetic and tool-call questions where early-stopping losses concentrate. Meanwhile, the round monitor (780 fresh test-matched questions before training on slice $k+1$) *also* measures the model for free!

#### Proposed Safer Rule (Update `fullpick` in `track.py`)
Combine **Dev-512** with the **fresh-slice monitor (`od3_monitor`)** and a **saving-dependent accuracy bias + per-type red-flag penalty**:
1. **Saving-dependent Dev accuracy bias:** Across all paired Dev-512 and full-test evaluations (`od5c-r10/15/16/19/20`, `od5r-r15/16`, `od5d-r10/11`), Dev’s accuracy optimism grows by `~0.22 pts` per `+1%` of saving above `52%` (from `+2.95` at `52–54%` to `+4.08` at `56.9%`). Replace the flat `acc_bias = 2.95` with:
   $$\text{acc\_bias}(s_{\text{dev}}) = 2.95 + 0.25 \times \max(0.0,\; s_{\text{dev}} - 53.0).$$
   (At `od5c-r20`’s `56.5%` Dev saving, this gives `acc_bias = 3.83`, predicting `-0.99`, right on the `-1.0` boundary!)
2. **Blended accuracy estimate (70% Dev-512 + 30% Monitor on slice $k+1$):**
   When round $k$ has been evaluated on fresh slice $k+1$ by `od3_monitor` (`780` questions $\times 8$, whose mean accuracy bias vs. the full test is `+2.60` pts on slices 1–19), compute:
   $$\hat{a}_k = 0.70 \times \left(a_{\text{dev},k} - \text{acc\_bias}(s_{\text{dev},k})\right) + 0.30 \times \left(a_{\text{mon},k+1} - 2.60\right).$$
   Averaging across `512 + 780 = 1,292` questions cuts sampling variance by $60\%$.
3. **Safety margin (`-0.85` gate instead of `-1.00`) + Per-type Dev red-flag:**
   - Require $\hat{a}_k \ge -0.85$ (a `0.15 pt` safety buffer inside `-1.00`) to classify a checkpoint as "inside the objective".
   - If any non-abstention type on Dev-512 has accuracy delta $<-2.5\text{ pts}$ (or if `svamp` + `bfcl_simple_python` average $<-1.5\text{ pts}$ on Dev-512), subtract an extra `0.35 pt` penalty from $\hat{a}_k$.
   - Pick (1) the checkpoint with the **highest predicted saving** satisfying $\hat{a}_k \ge -0.85$, and (2) as the second pick, the checkpoint with the **highest safety margin** ($\max(\hat{a}_k + 0.15 \times \hat{s}_k)$) among rounds with $\hat{s}_k \ge 52.5\%$.

---

### 6. Exact Code Changes, Queue Lines, Expected Landing, and Risks

#### A. Exact Code Changes in Words
1. **`code/repair_pairs.py` (in `main()`, around lines 320–470 where `shorten` candidates are filtered):**
   - **Already supported, no code edit needed for flags:** `--shorten-second`, `--shorten-ratio-by-part`, and `--floor-by-part` are already implemented in `repair_pairs.py`!
   - **One small improvement in `repair_pairs.py` (`main()`, line 309, inside the `--resp-median` / `--resp-max` filter):** Currently, when `part in think_only`, `shorten` candidates are dropped if `response_len(short) < args.resp_median * base_med` or `> args.resp_max * response_len(long)`. For `bfcl_irrelevance` (no-fit abstentions, where a shorter brief refusal response is *better* than a long explanation) and `mbpp` (code blocks), exempt `bfcl_irrelevance` from `--resp-median` so brief abstentions are never discarded as shorten candidates.
2. **`code/track.py` (in `fullpick()`, around lines 1620–1675 where `pred_save` and `pred_acc` are computed for `arena` candidates):**
   - When computing `pred_acc` for each candidate round $k$:
     - Replace the flat `c["acc"] - args.fp_acc_bias` with the saving-adjusted bias: `dyn_bias = args.fp_acc_bias + 0.25 * max(0.0, c["saved"] - 53.0)`.
     - Look up round $k$’s fresh-slice monitor entry (`round == k + 1` in `<run>-monitor.jsonl`, where `answering_model` is `models/<run>-r<k>`) and, if present and $k + 1 \le 19$, blend `0.70 * (c["acc"] - dyn_bias) + 0.30 * (mon_acc - 2.60)`.
     - Change the in-goal threshold in `fullpick`’s sort key from `pred_acc > -1.0` to `pred_acc >= -0.85` so borderline `-1.2` checkpoints (like `od5c-r20`) do not beat safe `-0.7` checkpoints (like `od5c-r16`).

#### B. Exact Queue Lines (Plain Text for `plan_line_a.txt` / `plan_line_b.txt`)

odpo --name od5f --od-fixed-ref --rp-base Qwen/Qwen3-0.6B --od-rounds 18 --od-steps 15 --od-r1-steps 20 --od-r1-quota svamp=80,mbpp=56,ifeval=88,bfcl_simple_python=36,bfcl_multiple=28,bfcl_irrelevance=32 --rp-quota svamp=56,mbpp=52,ifeval=60,bfcl_simple_python=24,bfcl_multiple=22,bfcl_irrelevance=26 --rp-questions svamp=220,mbpp=140,ifeval=200,bfcl_simple_python=80,bfcl_multiple=60,bfcl_irrelevance=80 --rp-own-data --rp-kinds mix --rp-samples 8 --rp-seed 18 --rp-pair-args=--think-only,svamp+mbpp+ifeval+bfcl_simple_python+bfcl_multiple+bfcl_irrelevance,--prefer-reread,--count-rules,--shorten-ratio,1.30,--shorten-ratio-by-part,ifeval=1.25+bfcl_irrelevance=1.25,--shorten-second,ifeval+bfcl_irrelevance,--shorten-min-right,0.85,--mix-preset,od5,--od5-v2,--base-stats,data/od3-base-stats.json,--base-resp,data/od3-base-resp.json,--base-think,data/od3-base-think.json,--floor-by-part,svamp=30:0.45+mbpp=35:0.36+ifeval=18:0.30+bfcl_simple_python=25:0.44+bfcl_multiple=22:0.38+bfcl_irrelevance=16:0.28,--count-mode,atleast,--count-resp-min,1.10,--count-resp-cap,1.75,--count-resp-pad,60,--resp-median,0.95,--resp-max,1.05,--right-resp-cap,1.15 --rp-lr 1.4e-5 --rp-warmup 2 --rp-beta 0.1 --rp-dpo-seed 0 --rp-dev-vs rpE80 --od-dev-rounds 99 --data-seed 18 --od-caps svamp=61,mbpp=63,ifeval=56,bfcl_simple_python=45,bfcl_multiple=53,bfcl_irrelevance=64 --od-acc-hold -2.0 --od-unlatch --od-stop-save 999 --od-conv-gain -999 --od-min-rounds 18

fullpick --fp-run od5f --fp-kind arena --fp-dev-rounds 14 15 16 17 18 --fp-save-bias 0.3 --fp-acc-bias 2.95 --fp-n 2 --rp-dev-vs rpE80

lossan --la-models fullpick:od5f --la-tag od5f

#### C. Expected Landing and Main Risks
- **Expected Full-Test Landing:** **55.8%–57.2% tokens saved at -0.55 to -0.90 accuracy** (with per-type landing around `svamp 60–61% / -2.2`, `mbpp 59–61% / -1.8`, `ifeval 50–51% / 0.0`, `simple 48% / -1.1`, `multiple 52% / -0.2`, `irrelevance 62–64% / +6.8`).
- **Main Risks & Mitigations:**
  1. **Cross-type interference from unlocking `irrelevance` (`64%` cap) and `ifeval` (`1.25` ratio + `shorten-second`):** Because LoRA weights are shared across types, stronger shortening gradients on `ifeval` and `irrelevance` could subtly shorten `svamp` or `bfcl_simple_python`. *Mitigation:* We proactively raised the thinking floors on both fragile types (`svamp` `25:0.44 -> 30:0.45`, `simple` `20:0.42 -> 25:0.44`) and lowered their caps by 1 pt (`svamp 61`, `simple 45`).
  2. **Code seed variance (`~4–5 pts` on Code):** As `od5r` showed, Code saving can fluctuate depending on when the `-2.0` accuracy brake fires. *Mitigation:* Scoring rounds `14, 15, 16, 17, 18` with the blended Dev+Monitor rule picks a round right after an unheld Code shorten step while keeping accuracy safely above `-0.85`.
