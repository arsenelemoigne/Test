# Consult 21: Diagnosis of `od4` and Design of the Next Online-DPO Run (`od5`)

## 1. Diagnosis of `od4`: Why It Loses Its Last 1.3–2.0 Points and Why Held Rounds Lengthened

### A. Per-Kind Breakdown of `od4`’s Accuracy Loss (`od4-r8` 51.6% / -1.3 vs `od4-r7` 55.4% / -2.0)
Using the official benchmark weights (`0.25` Math, `0.25` Code, `0.25` IFEval, `0.25` Tools split `400:200:240` across Simple / Several / No-Fit) on `loss_analysis/od4.txt` (2,759 full-test questions $\times 8$ vs the original’s full-test answers):
- **`od4-r8` (`od4-best`, 51.6% saved / -1.28 pts weighted net):** loses **-5.02 pts** on questions solved less often than the original and gains **+3.74 pts** on questions solved more often (vs the 4-vs-4 sampling-noise reference: -4.34 lost + 4.75 gained = +0.41 net).
- **`od4-r7` (55.4% saved / -2.01 pts weighted net):** loses **-5.60 pts** and gains **+3.59 pts**.

Weighted across the full test, the gross lost accuracy in `od4-r8` (-5.02 pts) and `od4-r7` (-5.60 pts) decomposes across the four loss kinds as follows:

1. **Stopping Thinking Too Early (`< 0.5x` original’s median right thinking `b_think`) and Somewhat Short Thinking (`0.5–0.8x b_think`) — The Dominant Loss (`-2.81` + `-1.32` = `-4.13` pts in `od4-r8`; `-3.41` + `-1.33` = `-4.74` pts in `od4-r7`, vs `-0.16` + `-0.56` = `-0.72` pts in noise):**
   - **Math (`svamp`, net `-1.69` pts in `r8`, `-3.06` pts in `r7`):** Stopped early `< 0.5x` accounts for **51%** (`2.35` pts) of lost Math accuracy in `r8` and **62%** (`3.46` pts) in `r7`; `0.5–0.8x` accounts for another **26%** (`1.20` pts) in `r8` and **20%** (`1.14` pts) in `r7`. Crucially, **57%** (`2.61` pts in `r8`) and **54%** (`3.04` pts in `r7`) of all lost Math accuracy occurs on **questions the original solves 8/8 (`p0 = 1.0`)**.
   - **Code (`mbpp`, net `-0.93` pts in `r8`, `-2.08` pts in `r7`):** Stopped early `< 0.5x` causes **55%** (`2.76` pts) in `r8` and **65%** (`3.74` pts) in `r7`, plus **27%** (`1.34` pts) and **20%** (`1.15` pts) from `0.5–0.8x` thinking (`32%` on `orig 8/8`, `24%` on `orig 6–7/8`, `33%` on `orig 3–5/8`).
   - **IFEval (`ifeval`, net `-1.16` pts in `r8`, `-1.25` pts in `r7`):** Stopped early `< 0.5x` causes **71%** (`4.14` pts) in `r8` and **66%** (`4.04` pts) in `r7` (median thinking is `0.41x` / `0.40x`, with a `0.25` floor that allowed aggressive truncation of instruction-planning thoughts).
   - **No-Fit (`bfcl_irrelevance`, net `-2.03` pts in `r8`, `-3.28` pts in `r7`) — The Hidden 0.45-pt Macro Penalty:** In offline DPO runs (`rpK40`, `rpL60`, `rpM60`, `r2s1b`), `bfcl_irrelevance` **gains `+3.5` to `+6.2` pts** at `45–62%` saved. In `od4`, `bfcl_irrelevance` lost `-2.03` (`r8`) and `-3.28` (`r7`) — costing **`0.40` to `0.48` full macro accuracy points** (more than the entire `0.3`-pt gap between `od4-r8`’s `-1.3` and the `> -1.0` objective). Why? Because `bfcl_irrelevance`’s cap was set at `65%`, which its monitor saving (`48–51%`) never reached. Consequently, `bfcl_irrelevance` **never braked in all 8 rounds**: it received `75%` `shorten` + `25%` compact `right` (which also picks the *shortest* right answer when excluded from `--right-hold`) with a `0.25x` floor for 8 consecutive rounds. `92–93%` of its lost accuracy (`2.08 + 1.57 = 3.65` pts in `r8`; `2.85 + 1.32 = 4.17` pts in `r7`) is early/short thinking (`< 0.8x`), with **51%** on `orig 6–7/8` questions where the model stopped thinking before noticing a required tool parameter was missing.
   - **Simple & Multiple Tools (`bfcl_simple_python` `-1.19` / `-0.84`; `bfcl_multiple` `-0.88` / `-1.31`):** `87–92%` of their lost accuracy is `< 0.5x` (`37–49%`) and `0.5–0.8x` (`41–55%`) thinking, concentrated (`61–80%`) on `orig 6–7/8` and `orig 8/8` questions.
   - **Root cause of early stopping in `od4`:** (i) The `7/8` bar (`--shorten-min-right 0.85`) is blind to `p0`: when an `orig 8/8` question slips to `7/8` because one answer stopped early, `od4` **still shortens it** and emits no recovery guard. (ii) At `lr = 1.5e-5` with `75%` shorten and low floors (`0.38` Math, `0.35` Code/Tools, `0.25` IFEval/No-Fit), Math jumped from `30%` (r1) to `55% / -3.8` (r2) and `68% / -4.5` (r3, `72% / -6.4` on Dev-512) before the `65%` cap braked it.

2. **Repetition Loops (`-0.39` pts weighted in `od4-r8`, `-0.33` pts in `od4-r7`):**
   - Accounts for **12%** of lost Code accuracy (`0.62` pts in `r8`, `0.61` pts in `r7`), **8%** of lost IFEval accuracy (`0.49` pts in `r8`, `0.41` pts in `r7`), and **9%** of lost Math accuracy (`0.43` pts in `r8`, `0.31` pts in `r7`).
   - **Root cause in code:** In `repair_pairs.py` (line 210), `runaway = [r for r in q if not finished(r)]` only flags answers that hit the token cap (`truncated`) or never closed `</think>`. That eliminated `cut off at the cap` (`0.04` pts weighted in `od4-r8` vs `0.41` pts in noise), but **missed finished repetition loops** (`loop(text)` in `loss_analysis.py`: any 8-word phrase repeated $\ge 4$ times that eventually closes `</think>` and outputs a wrong answer).

3. **IFEval Responses Too Short for Length Rules (`-0.10` pts weighted; `0.41` pts [7%] on IFEval in `r8`, `0.43` pts [7%] in `r7`):**
   - **Root cause in code:** First, `--resp-median 0.95` in `repair_pairs.py` (line 218) compares `response_len(r)` to `rmed`, the **current round’s** median right response length on that question—not the original’s `b_resp`. Across 8 rounds, allowing `0.95x` per round lets responses drift down by $0.95^8 \approx 0.66\times$ (`< 0.7x b_resp`). Second, `--count-mode length` only protects word/sentence/paragraph/keyword-frequency rules (`length_rule`), leaving `at_least` rules (minimum placeholders, highlighted sections, letter frequency) unprotected when their responses shrink.

4. **Broken Answer Format / Unparseable Tool Calls (`-0.13` pts weighted in `r8`, `-0.16` pts in `r7`):**
   - Accounts for **10%** of lost accuracy on `bfcl_multiple` (`0.62` pts in `r8`, `0.64` pts in `r7`), **7–9%** on `bfcl_simple_python` (`0.29` pts in `r8`, `0.37` pts in `r7`), and **5%** on `svamp` (`0.21` pts in `r8`, `0.28` pts in `r7`).
   - **Root cause in code:** Tool-call parts have `0%` guard/loop share in `od3_arena`, and `right_cr` selects pairs by `thinking_len` without prioritizing `parse_error` wrong answers for rejection.

### B. Why the Held Rounds (`r6 -> r8`) Recovered Accuracy by Lengthening
Between `od4-r6` and `od4-r8`, Dev-512 saving dropped by **-7.8 points** (`60.1% -> 52.3%`; Math `66% -> 49%`, Code `70% -> 58%`) while Dev-512 accuracy jumped **+2.79 points** (`-1.18 -> +1.61`; Math `-6.0 -> -0.9`, Code `-1.9 -> +2.4`), and training loss in `r4–r8` started high (`0.71–0.85` at step 5, only `50–60%` pairs ordered right vs `85%` in `r2`). Three interacting mechanisms in `repair_pairs.py` and `dpo.py` caused this:
1. **Asymmetric Floor Filtering (`repair_pairs.py` lines 183–189):** `--floor-by-part` filters `rights = [r for r in all_rights if thinking_len(r) >= ftok]` *before* building `right_cr` and `loop` pairs, while `wrongs` are **unfiltered**. Once Math (`68%` saved, median thinking `0.34x`) and Code (`70%` saved) had many right answers below their `0.38x`/`0.35x` floors, all short right answers were discarded from `rights`, forcing every held pair to pick a longer right answer ($\ge f_{\text{tok}}$).
2. **`--right-hold` Is Biased Longer (`repair_pairs.py` lines 254–266):** `--right-hold` requires `0.95 <= thinking_len(r) / thinking_len(w) <= 1.25` and minimizes `abs(ratio - 1.05)` while preferring `REREAD` matches. Thus, `90%` of a braked type’s pairs (`right_hold`) have `chosen` averaging **`~1.05–1.12x` longer** than `rejected`.
3. **Fixed Original Reference (`dpo.py --ref-base`) Amplifies Gradients on Lengthening Pairs:** Under `--ref-base`, the reference $\pi_{\text{ref}}$ is the **original long model**, which assigns higher relative log-probability to longer thoughts than the shortened policy $\pi_\theta$ does. On held-type pairs where `chosen` is longer than `rejected`, $\pi_\theta(y_c) - \pi_\theta(y_r) < \pi_{\text{ref}}(y_c) - \pi_{\text{ref}}(y_r)$, making the initial DPO margin **negative** (`loss > 0.693`, ordered right `50–60%`). Because the DPO gradient scales with $1 - \sigma(\text{margin}) \approx 0.60$ (more than double the $\approx 0.25$ weight of already-ordered shortening pairs), once **4 of 6 types (`67%` of every batch)** were latched into `held` in Round 8, shared LoRA weights received a strong global lengthening push.

---

## 2. Viability of the User’s `od5` Rule and Exact Refinements Needed

### Verdict: Viable and Worth Testing — With Four Data-Proven Refinements
Simulating the `od5` rule on the 22,072 full-test answers (`fulltest_per_answer_all_models.csv.gz`) across `base`, `p1r1s0` (`45%` saved), `rpE80` (`47%`), `rpM60` (`54%`), `rpL60` (`58%`), and `rc1` (`60%`) confirms that comparing each question’s current solve rate $p_1$ against the original’s stored solve rate $p_0$ (`data/od3-base-stats.json`) directly fixes the `orig 8/8` blind spot of the `7/8` bar, **provided four refinements are made**:

1. **Handling the 8-Answer Noise & Closing the `orig 8/8 -> 7/8` Loophole:**
   - **Why raw "within 1 of 8" ($p_1 \ge p_0 - 1/8$) fails on `orig 8/8` ($p_0 = 1.0$):** On SVAMP (`70.1%` solved `8/8`), Simple Tools (`66.5%`), and No-Fit (`72.5%`), `54–60%` of lost Math accuracy is on $p_0 = 1.0$ questions. When $p_0 = 1.0$ (10 stored answers in `pool-dp13` or 8 in `tooltop`), there is **zero upward binomial noise** in $p_0$: a drop to `7/8` ($p_1 = 0.875$) is a real error. Allowing "within 1 of 8" on $p_0 = 1.0$ still shortens `7/8` questions just like `od4`.
   - **Why raw $p_1 < p_0$ is noisy on $p_0 < 1.0$:** Under pure binomial noise ($p_1 = p_0$, $N = 8$), $P(k_1 \le k_0 - 1 \text{ and } k_1 \ge 1)$ is `10.9%` on Code and `12.1%` on IFEval, whereas $P(k_1 \le k_0 - 2 \text{ and } k_1 \ge 1)$ is only **`1.5–3.5%`**.
   - **Refined 8-Answer Noise Rule:**
     - **Shorten-eligible (`undamaged & reliable`):** Require $p_1 \ge 0.75$ ($\ge 6/8$, never shortening shaky `3/8` questions) AND: if $p_0 = 1.0$, require **$p_1 = 1.0$ (`8/8`)**; if $p_0 < 1.0$, require **$p_1 \ge p_0 - 0.125$** ("within 1 of 8" absorbs noise on $0.75 \le p_0 < 1.0$).
     - **Damaged (`recovery-eligible`):** Require $1 \le k_1 \le 7$ AND: if $p_0 \ge 0.875$, any drop **$p_1 < p_0$** (`k1 <= 7` when $p_0 = 1.0$; `k1 <= 6` when $p_0 = 0.875$) is **damaged**; if $p_0 < 0.875$, require **$p_1 \le p_0 - 0.125$** (while undamaged mixed questions with $p_1 \ge p_0$ get a length-neutral `right_hold` pair centered at `1.00x`).

2. **Why "Shares Follow the Rule, No Caps" Needs Per-Question Thinking Floors (`~0.48x b_think`) to Self-Stop:**
   - Without a per-question thinking floor (or with `od4`’s `0.25–0.38x` floors), even when SVAMP accuracy drops by `-6.6%` (`rc1`), `61–82%` of SVAMP questions are *still* solved `8/8` or within 1 of $k_0$ (while the worst-damaged questions collapse to `0/8` and cannot form pairs). Without a higher per-question floor, the batch stays `75–88%` `shorten` every round.
   - **How per-question floors make dynamic shares self-regulating:** When a `shorten` pair requires its chosen answer to think **$\ge f_{\text{tok}}(q) = \max(\text{abs\_min}, r_{\text{floor}} \cdot b_{\text{think}}(q))$** ($r_{\text{floor}} = 0.48$ Math, `0.42` Code, `0.36` IFEval, `0.45` Simple/Multiple, `0.42` No-Fit) AND spread $\text{longest\_right} / \text{shortest\_right\_above\_floor} \ge 1.30$:
     - **Round 1 (`base`):** `90.5%` of SVAMP, `66.4%` of MBPP, `63.0%` of IFEval, and `74–89%` of Tools pairs are **`shorten`** (`0–1%` recovery).
     - **Rounds 2–3 (`p1r1s0` / `rpE80`, `~45–47%` saved):** Questions already compressed near $0.50 \times b_{\text{think}}$ no longer have a $1.30\times$ spread above $f_{\text{tok}}(q)$ and stop emitting `shorten` pairs. The natural share shifts to **`56–60%` shorten / `28–30%` recovery / `13–14%` hold** on SVAMP and **`43–47%` shorten / `28–36%` recovery** on MBPP.
     - **Rounds 4–6 (`rpM60` / `rpL60`, `~54–58%` saved):** Shorten supply naturally drops below recovery+hold (**`32–43%` shorten vs `57–68%` recovery+hold** on Math/Code, **`12–18%` shorten vs `82–88%` recovery+hold** on IFEval), creating a smooth per-question equilibrium.

3. **Fixing the Recovery Pair Definition on a Shortened Model’s Own Answers:**
   - **Why "chosen $\approx b_{\text{think}}$" fails after Round 2:** On lost questions in `od4-r7` (`loss_analysis/od4.txt`), the model’s *own right answers* think **`0.42x`** $b_{\text{think}}$ on Math/Code, **`0.39x`** on IFEval, and **`0.51x`** on Tools. In `fulltest_per_answer_all_models.csv.gz`, only **`16–27%`** of damaged SVAMP questions, **`10–26%`** of damaged MBPP questions, and **`0–12%`** of damaged IFEval questions have *any* right answer in $[0.75, 1.35] \times b_{\text{think}}$. Worse, pairing a $1.0\times b_{\text{think}}$ outlier over a $0.4\times b_{\text{think}}$ wrong answer creates a $2.5\times$ length jump that causes `od4-r8`-style lengthening.
   - **Bounded Recovery Pair:** On a damaged question, select **rejected ($w$)** with $\text{think}(w) < 0.85 \cdot b_{\text{think}}(q)$ (preferring premature stops $< 0.65 \cdot b_{\text{think}}(q)$) and **chosen ($r$)** thinking $\ge f_{\text{tok}}(q)$ and strictly longer than $w$: $1.08 \cdot \text{think}(w) + 10 \le \text{think}(r) \le \min(1.05 \cdot b_{\text{think}}(q),\, 1.45 \cdot \text{think}(w) + 50)$, preferring `REREAD` and closest to $\min(0.85 \cdot b_{\text{think}}(q),\, 1.25 \cdot t_{\text{med}}(q))$. This raises recovery supply on damaged questions from `16–27%` to **`68–81%`** while bounding length give-back.

4. **Dedicated Per-Question Pairs for Loops, Format, IFEval Response Length, and Neutral Hold:**
   - **Anti-Loop Pair (`loop`):** Flag any answer $w$ that is `!finished(w)` OR has $\ge 4$ occurrences of any 8-word phrase (`loss_analysis.py` rule). Pair `chosen = shortest non-looping right >= ftok` over `rejected = looping w` (up to `12%` of quota).
   - **Format Guard Pair (`format`):** Flag any wrong answer $w$ with `parse_error` or empty response. Pair `chosen = right answer with thinking in [0.85x, 1.25x] of w` over `rejected = w` (up to `10%` of quota).
   - **IFEval Response-Length Guard:** Compute $b_{\text{resp}}(q) = \max(1, \text{st}[\text{"tokens"}] - b_{\text{think}}(q))$ from `od3-base-stats.json` and `od3-base-think.json`. Anchor `--resp-median` on `shorten` pairs to $\max(\text{rmed},\, 0.92 \cdot b_{\text{resp}}(q))$ to prevent $0.95^k$ multi-round shrinkage; use `--count-mode atleast`; and include questions with $\text{resp}(w) < 0.75 \cdot b_{\text{resp}}(q)$ in `right_count` (up to `20%` of IFEval quota).
   - **Neutral `right_hold` (`1.00x`):** Change `--right-hold` ratio window from `[0.95, 1.25]` centered at `1.05` to **`[0.90, 1.12]` centered at `1.00`**.

---

## 3. Recommended Run (`od5`): Full Specification & Exact `track.py` Plan

We recommend running **`od5`** as a **6-round** clean online DPO run from `Qwen/Qwen3-0.6B`.

### A. Run Settings
- **Starting Model & Reference:** `Qwen/Qwen3-0.6B`, LoRA `rank 16` (`alpha 32`), one adapter continued across rounds, fixed reference = original base model (`dpo.py --ref-base`).
- **Hyperparameters:** Constant learning rate **`1.2e-5`** (prevents Round 2–3 overshoot), `beta = 0.1`, `nll = 0.0`, batch size `16`.
- **Rounds, Slices, Steps, and Quotas:**
  - **6 rounds** on test-matched `od3prep` Slices 1–6:
  - **Round 1:** `20` steps (`320` pairs) on stored Slice 1: `--od-r1-quota svamp=80,mbpp=56,ifeval=88,bfcl_simple_python=36,bfcl_multiple=28,bfcl_irrelevance=32`.
  - **Rounds 2–6:** `15` steps (`240` pairs) per round on Slices 2–6 (`780` questions $\times 8$): `--rp-quota svamp=60,mbpp=42,ifeval=66,bfcl_simple_python=27,bfcl_multiple=21,bfcl_irrelevance=24`.
- **Dynamic Pair Selection (`--mix-preset od5`):**
  - Within each part’s quota $n$, take dedicated defect pairs first when present: `right_count` (up to `20%` on IFEval), `loop` (up to `12%`), and `format` (up to `10%`).
  - Fill the remaining quota $n_{\text{rem}}$ **dynamically in proportion to candidate counts** $(C_{\text{sh}}, C_{\text{rec}}, C_{\text{hold}})$ across:
    - **Bounded `recovery`** on damaged questions ($p_1 < p_0$ for $p_0 \ge 0.875$, or $p_1 \le p_0 - 0.125$ for $p_0 < 0.875$), doubled in weight if the part’s slice accuracy is $< -1.0$ pts;
    - **`shorten`** on undamaged reliable questions ($p_1 = 1.0$ if $p_0 = 1.0$, else $p_1 \ge \max(0.75, p_0 - 0.125)$; spread $\ge 1.30$ above $f_{\text{tok}}(q)$);
    - **Neutral `right_hold` (`1.00x`)** on undamaged mixed questions ($1 \le k_1 \le 7$, not damaged).
- **Per-Question Thinking Floors (`--floor-by-part`):** `svamp=25:0.48,mbpp=35:0.42,ifeval=20:0.36,bfcl_simple_python=20:0.45,bfcl_multiple=25:0.45,bfcl_irrelevance=20:0.42`.
- **Unlatched Emergency Safety Caps (`--od-caps`):** `svamp=62,mbpp=63,ifeval=56,bfcl_simple_python=46,bfcl_multiple=49,bfcl_irrelevance=56` (`--od-acc-hold -2.0`). If triggered in a round, the part sets $C_{\text{sh}} = 0$ for that round only (unlatched) and fills $n_{\text{rem}}$ with bounded `recovery` + neutral `1.00x` `right_hold`.

### B. Stop Rule & Bias Calibration
- **Known Biases:** Free Monitor (`780 x 8`) under-reads full-test saving by `-0.5` to `-2.0%` and over-reads accuracy by **`+1.2` to `+1.9` pts** (`od4-r7`: monitor `-0.14` vs full `-2.01`). Dev-512 (`512 x 8`) over-reads full-test saving by **`+0.7` to `+1.7%`** and over-reads accuracy by **`+2.2` to `+2.9` pts** (`od4-r8`: Dev `52.3% / +1.61` vs full `51.6% / -1.28`).
- **Stop & Pick Rules:**
  - **Free Monitor Stop Rule:** Stop at $k \ge 4$ when monitor reaches **`saved >= 52.5%` AND `acc >= +0.50`** (`--od-stop-save 52.5 --od-stop-acc 0.50`), OR by convergence at $k \ge 5$ when **`saved >= 52.0%`, `acc >= +0.20`, and `|saved_k - saved_{k-1}| < 1.5%`**.
  - **Dev-512 / Dev-SVAMP Checks (`--od-dev-rounds 4 5 6`) & `fullpick`:** Use **`--fp-save-bias 1.0 --fp-acc-bias 2.5`** (requiring Dev-512 `saved >= 51.5%`, `acc >= +1.60`, and Dev-SVAMP `acc >= 0.0`).

### C. Expected Full-Test Landing & Risks
- **Expected Full-Test Landing (`od5` Round 5 or 6):** **`52.5% – 54.5%` tokens saved at `-0.4` to `-0.8` accuracy** (`+2.6` to `+3.0` above the curve: Math `57% / -1.1`, Code `59% / -0.7`, IFEval `50% / -0.8`, Simple `45% / -0.4`, Several `48% / -0.6`, No-Fit `53% / +2.5`).
- **Risks & Mitigations:** (1) *Shorten candidate drop at floors:* mitigated by pool-proportional allocation without scarcest-kind truncation. (2) *Lengthening give-back:* mitigated by capping recovery thinking at $\le \min(1.05 \cdot b_{\text{think}},\, 1.45 \cdot \text{think}(w) + 50)$, centering `right_hold` at `1.00x`, and removing permanent brake latching.

### D. Small Code Changes Needed & Exact `track.py` Plan Lines
1. **`main()` in `repair_pairs.py`:**
   - Add `--base-stats` (loading `data/od3-base-stats.json` for `p0 = st["acc"]` and `b_resp = max(1, st["tokens"] - base_think.get(qid, 0))`) and `"od5"` to `--mix-preset`.
   - Upgrade `runaway` to include finished 8-gram repetition loops (`max(Counter(tuple(w[i:i+8]) ...).values()) >= 4` on `decode(r)`), and add `cand[part]["format"]` rejecting `w` with `w.get("parse_error")` or empty response (`0.85 <= thinking_len(r) / max(1, thinking_len(w)) <= 1.25`).
   - Under `--mix-preset od5`, classify each question by `p1 = len(all_rights) / len(q)` vs `p0`: `can_shorten = (p1 == 1.0) if p0 == 1.0 else (p1 >= max(0.75, p0 - 0.125))` and `is_damaged = (p1 < p0) if p0 >= 0.875 else (p1 <= p0 - 0.125)`. Build `shorten` only on `can_shorten` questions (anchoring response floor to `max(args.resp_median * rmed, 0.92 * b_resp)`), bounded `recovery` in `cand[part]["capguard"]` on `is_damaged` questions (`1.08 * thinking_len(w) + 10 <= thinking_len(r) <= min(1.05 * bt, 1.45 * thinking_len(w) + 50)` over `thinking_len(w) < 0.85 * bt`), and neutral `right_hold` (`0.90 <= ratio <= 1.12`, `abs(ratio - 1.00)`) on undamaged mixed questions. Take `right_count` ($\le 20\%$), `loop` ($\le 12\%$), and `format` ($\le 10\%$), then allocate remaining quota proportionally to $(C_{\text{sh}}, C_{\text{rec}}, C_{\text{hold}})$ without scarcest-kind truncation.
2. **`odpo_fixed` in `track.py`:** Make `held` per-round (`held = {p_ for p_, (sv, ac) in per_.items() if sv >= caps.get(p_, 1e9) or ac <= args.od_acc_hold}` without permanent latching across all rounds) and require `ac >= args.od_stop_acc - 0.3` in the convergence stop rule.

**Exact `track.py` Plan Lines:**

od3prep --od-rounds 6 --r2-drop runs/bon/Shuttle_Notation-result/eval_completions.jsonl
odpo --name od5 --od-fixed-ref --rp-base Qwen/Qwen3-0.6B --od-rounds 6 --od-r1-steps 20 --od-steps 15 --rp-samples 8 --rp-kinds mix --od-r1-quota svamp=80,mbpp=56,ifeval=88,bfcl_simple_python=36,bfcl_multiple=28,bfcl_irrelevance=32 --rp-quota svamp=60,mbpp=42,ifeval=66,bfcl_simple_python=27,bfcl_multiple=21,bfcl_irrelevance=24 --rp-pair-args=--mix-preset,od5,--think-only,all,--prefer-reread,--count-rules,--count-mode,atleast,--count-resp-min,1.10,--count-resp-cap,1.75,--count-resp-pad,60,--no-shorten-counting,--shorten-ratio,1.30,--resp-median,0.95,--resp-max,1.05,--resp-near-long,0.95,--right-resp-cap,1.15,--right-hold,svamp,mbpp,ifeval,bfcl_simple_python,bfcl_multiple,--base-think,data/od3-base-think.json,--base-stats,data/od3-base-stats.json,--floor-by-part,svamp=25:0.48+mbpp=35:0.42+ifeval=20:0.36+bfcl_simple_python=20:0.45+bfcl_multiple=25:0.45+bfcl_irrelevance=20:0.42 --od-caps svamp=62,mbpp=63,ifeval=56,bfcl_simple_python=46,bfcl_multiple=49,bfcl_irrelevance=56 --od-acc-hold -2.0 --rp-lr 1.2e-5 --od-stop-save 52.5 --od-stop-acc 0.50 --od-conv-min 52.0 --od-conv-gain 1.5 --od-dev-rounds 4 5 6
fullpick --fp-run od5 --fp-kind arena --fp-save-bias 1.0 --fp-acc-bias 2.5

---

## 4. Contingency: Smallest Recovery Step If Needed (`rpP`)

If and only if `od5-best` (or existing `od4-r7` at `55.4% / -2.0`) finishes with `53.5–56.0%` saving but `0.2–0.8` pts short on accuracy, a **30-step (`480`-pair, `lr = 8e-6`) pure `od5` recovery pass (`rpP`)** on `data/od3-step2-questions.jsonl` ($\times 8$, `0%` shorten everywhere, bounded `recovery` + `loop` + `format` + `atleast` count + `1.00x` `right_hold`) gives back only `1.5–2.2%` saving while recovering `+1.0` to `+1.4` accuracy pts, landing **`od4-r7 -> rpP` at `53.2–53.8%` / `-0.7` to `-0.9`** and **`od5-best -> rpP` at `52.0–53.0%` / `-0.2` to `-0.6`**:

ptrepair --name rpP --rp-base models/od5-best --rp-own-data --rp-question-file data/od3-step2-questions.jsonl --rp-samples 8 --rp-seed 41 --rp-kinds mix --rp-pair-args=--mix-preset,od5,--no-shorten-parts,svamp+mbpp+ifeval+bfcl_simple_python+bfcl_multiple+bfcl_irrelevance,--think-only,all,--prefer-reread,--count-rules,--count-mode,atleast,--count-resp-min,1.10,--count-resp-cap,1.75,--count-resp-pad,60,--no-shorten-counting,--resp-median,0.95,--resp-max,1.05,--resp-near-long,0.95,--right-resp-cap,1.15,--right-hold,svamp,mbpp,ifeval,bfcl_simple_python,bfcl_multiple,--base-think,data/od3-base-think.json,--base-stats,data/od3-base-stats.json,--floor-by-part,svamp=25:0.48+mbpp=35:0.42+ifeval=20:0.36+bfcl_simple_python=20:0.45+bfcl_multiple=25:0.45+bfcl_irrelevance=20:0.42 --rp-quota svamp=150,mbpp=105,ifeval=105,bfcl_simple_python=45,bfcl_multiple=35,bfcl_irrelevance=40 --rp-lr 8e-6 --rp-steps 30 --rp-epochs 1 --rp-dpo-seed 0 --rp-screen 15 30 --rp-dev --rp-dev-svamp