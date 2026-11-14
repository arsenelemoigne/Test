### 1. Where the Remaining 1.27 Points Are Lost in `r2s2-s80` and Which Pairs Fix Each

On the full test ($2,759 \times 8$), **`models/r2s2-s80-on-r2s1b-s180`** reaches **`53.7%` tokens saved / `-1.27` accuracy points** (`+2.36` pts above the compression curve), improving on `r2s2-s40` (`52.9% / -1.74`) and `r2s1b-s180` (`50.3% / -1.70`). Under the official benchmark weights (`0.25` Math, `0.25` Code, `0.25` IFEval, and `0.25` Tools split `400 : 200 : 240` across Simple / Several / No-Fit), `r2s2-s80` loses **`-5.80` gross macro points** on questions solved less often than the original and gains **`+4.53` macro points** on questions solved more often (net **`-1.27` pts**—only **`0.28` pts** short of the `> -1.00` objective, with **`3.7` pts** of token-saving buffer above `50.0%`).

#### A. Loss Breakdown by Type (Net and Gross Macro Impact)
1. **Math (`svamp`, 1,000 q, weight `0.2500`):** Type net **`-3.25` pts** (lost `-5.99` on 259 q, gained `+2.74`) $\rightarrow$ **`-0.812` macro pts** (recovered `+1.25` type pts from Step 1's `-4.50`, but remains the #1 net deficit).
2. **Code (`mbpp`, 378 q, weight `0.2500`):** Type net **`-2.38` pts** (lost `-6.18` on 95 q, gained `+3.80`) $\rightarrow$ **`-0.595` macro pts** (recovered `+0.72` type pts from Step 1's `-3.10`, #2 net deficit).
3. **Simple Tool Calls (`bfcl_simple_python`, 400 q, weight `0.1190`):** Type net **`-4.69` pts** (lost `-8.19` on 94 q, gained `+3.50`) $\rightarrow$ **`-0.558` macro pts** (worsened by `-1.69` type pts from Step 1's `-3.00` because Step 2 applied `50%` `shorten` + `50%` compact `right_cr` without `--right-hold`, compressing thinking to `0.52x`).
4. **Several Tool Calls (`bfcl_multiple`, 200 q, weight `0.0595`):** Type net **`-0.69` pts** (lost `-7.31` on 40 q, gained `+6.62`) $\rightarrow$ **`-0.041` macro pts** (slipped from `+0.90` in Step 1 as `55%` `shorten` compressed thinking to `0.47x`).
5. **IFEval (`ifeval`, 541 q, weight `0.2500`):** Type net **`+1.20` pts** (lost `-5.11` on 117 q, gained `+6.31`) $\rightarrow$ **`+0.300` macro pts** (up `+0.20` accuracy over Step 1 while jumping from `38%` to `~56%` saved).
6. **No-Fit Tool Calls (`bfcl_irrelevance`, 240 q, weight `0.0714`):** Type net **`+6.15` pts** (lost `-0.94` on 9 q, gained `+7.08`) $\rightarrow$ **`+0.439` macro pts**.

Thus, **Math (`-0.812`), Code (`-0.595`), and Simple Tools (`-0.558`) account for `-1.965` macro pts (`100%` of the net deficit)**, while IFEval (`+0.300`) and No-Fit (`+0.439`) provide `+0.739` macro pts of positive margin.

#### B. Loss Breakdown by Cause (Across the `-5.80` Gross Lost Macro Points vs. 4-vs-4 Noise Reference)
1. **Stopping Thinking Early (`< 0.5x` original's median right thinking $b_{\text{think}}$) — `-3.348` macro pts (`57.7%` of gross loss; `-3.191` excess over noise's `-0.157`):**
   - **Per type:** Math `3.49` type pts (`58%` of Math loss), Code `3.52` (`57%`), Simple `3.87` (`47%`), Multiple `4.32` (`59%`), IFEval `3.29` (`64%`), No-Fit `0.77` (`82%`).
   - **Reliability concentration:** **`40.5%` (`-2.350` macro pts)** of all gross lost accuracy is on questions the original solved **`8/8`** (`57%` of Math loss = `3.41` pts; `55%` of Simple loss = `4.53` pts), and **`30.7%` (`-1.780` macro pts)** is on **`orig 6–7/8`** questions. On these lost questions, `r2s2-s80`'s *own right answers* think `0.45x` $b_{\text{think}}$ on Math, `0.47x` on Code, and `0.53x` on Simple—meaning `r2s2-s80` already solves them when thinking at its own compressed median (`tmed`), and fails only when a rollout stops prematurely below `0.5x–0.8x tmed`.
   - **Which pairs fix it:** **Widened Capped Guards (`--guard-wide svamp+mbpp+bfcl_simple_python --guard-wide-max 0.85`)**, where `chosen` is the shortest right answer in `[0.95, 1.25] * tmed` that re-reads the prompt (`--prefer-reread`) and `rejected` is the premature wrong answer below `0.85 * tmed` nearest `0.75 * tmed`. Measuring against the model's *own* compressed median `tmed` (`0.45x–0.52x` of original) fixes premature stops with a bounded `~1.35x` length ratio instead of reverting to original-length (`2.2x`) thoughts.
2. **Somewhat Short Thinking (`0.5–0.8x` $b_{\text{think}}$) — `-1.481` macro pts (`25.5%` of gross loss; `-0.924` excess over noise's `-0.557`):**
   - **Per type:** Simple `3.50` type pts (`43%` of Simple loss—skipped argument/type verification), Code `1.75` (`28%`—skipped assert check), Math `1.53` (`26%`—skipped question re-check), Multiple `1.21` (`17%`), IFEval `0.64` (`13%`).
   - **Why Simple Tools regressed in Step 2:** In `r2s2a`, `--right-hold` was only passed for `svamp+mbpp`. On `bfcl_simple_python` and `bfcl_multiple`, the `50%` `be-right` quota used compact `right_cr` (`rights[0] > wrongs[0]`), where `chosen` is shorter than `rejected` in **`58%` of pairs** (median thinking ratio `0.93x`). Thus even the "be-right" pairs on Simple Tools taught shorter thinking.
   - **Which pairs fix it:** **Length-Held `be-right` (`--right-hold svamp+mbpp+bfcl_simple_python+bfcl_multiple+bfcl_irrelevance`)** centered at **`1.00x`** (`0.90 <= think(r)/think(w) <= 1.15`, preferring `REREAD`), providing a pure correctness + verification signal with zero shortening bias and zero `1.05x` lengthening drift.
3. **Repetition Loops (`-0.425` macro pts, `7.3%`) + Cut Off at Cap (`-0.055` macro pts, `0.9%`) — `-0.480` macro pts (`8.3%` of gross loss):**
   - **Per type:** IFEval `0.66` loop + `0.09` cutoff (`0.75` pts = `15%`), Code `0.55` loop + `0.13` cutoff (`0.68` pts = `11%`), Math `0.49` loop (`8%`).
   - **Code diagnosis:** In `repair_pairs.py`'s standard `mix` branch (`L210`), `runaway = [r for r in q if not finished(r)]` only catches *unclosed/truncated* answers (`0.055` macro pts), missing **finished repetition loops** (`0.425` macro pts, any 8-word phrase repeated $\ge 4\times$, which `od5` at `L191` catches via `looping(r)`).
   - **Which pairs fix it:** Port `od5`'s `looping(r)` detector (`!finished(r)` OR 8-word phrase $\ge 4\times$) into the standard `mix` branch (`cand[part]["loop"]`), pairing a clean non-looping right answer against the looping wrong answer (`10%` of Math, `15%` of Code, `12.5%` of IFEval quotas).
4. **Unparseable / Wrong Tool-Call or Answer Format — `-0.219` macro pts (`3.8%` of gross loss):**
   - **Per type:** Multiple Tools `1.74` type pts (`24%` of Multiple loss), Simple Tools `0.74` (`9%`), Math `0.09` (`2%`), Code `0.02`.
   - **Which pairs fix it:** Prioritize wrong answers with a format defect (`bad_format(w)`: empty response, missing `(` after `</think>` on tool calls, or `final is None` on Math) as the rejected answer inside `right_hold`, so length-matched `be-right` pairs on `bfcl_simple_python`, `bfcl_multiple`, and `svamp` directly punish malformed calls and missing final answers.
5. **IFEval Response Too Short for Length Rules (`< 0.7x` $b_{\text{resp}}$) — `-0.090` macro pts (`1.6%` of gross loss; `0.36` type pts on IFEval):**
   - **Which pairs fix it:** Allocate `25%` of IFEval's Step-3 quota to counting-rule `be-right` pairs (`--count-rules --count-mode atleast --count-share 0.25 --count-resp-min 1.10 --count-resp-cap 1.75 --count-resp-pad 60`) plus `--right-resp-cap 1.15` on non-counting IFEval `be-right` pairs.

---

### 2. The Step-3 Design (`r2s3`)

#### A. Pure Repair (`0%` Shortening Across All 6 Types)
We use **no shortening pairs (`0%` `shorten`) on any type** in Step 3:
- `r2s2-s80` already sits at **`53.7%` saved** (`+3.7%` buffer above `50.0%`), and every type is already compressed (`thinking` medians: IFEval `0.40x`, Code `0.45x`, Math `0.47x`, Multiple `0.47x`, Simple `0.52x`, No-Fit `0.59x`).
- Step 3's single job is to recover **`+0.50` to `+0.85` macro accuracy points** (bringing full-test accuracy from `-1.27` to **`-0.45` to `-0.75`**) while giving back **only `1.0` to `2.0` points of saving** (landing at **`51.7%` to `52.7%` saved**).

#### B. Questions & Data Supply (`data/r2-step3-questions.jsonl` + Exact Extra Needs)
- **Ready in `data/r2-step3-questions.jsonl` (disjoint, test-matched):** `svamp=650`, `mbpp=339`, `ifeval=566`, `bfcl_simple_python=200`, `bfcl_multiple=159`, `bfcl_irrelevance=220` (`2,134` questions).
- **Why a `640`-pair quota (`40` steps at batch 16) + 1 extra test-matched slice (`s3m1`) is needed:**
  In `repair_pairs.py` (`L494–497`), each mixed-outcome question emits **at most 1 pair** across `loop + guard + right_cr` (`guarded = {p["id"] for p in g + lp}`). Based on Step 2's empirical pair yields on `r2s1b-s180` (`EXPERIMENT_LOG` 13 Nov 20:50) and our simulation on `rpM60` (`53.6%` saved), `r2-step3-questions.jsonl` alone yields:
  - `svamp` (`650` q): `~170.3` `right_hold U guard_wide` pairs (short of a `200` quota by `~30` pairs).
  - `mbpp` (`339` q): `~123.8` `right_hold U guard_wide` pairs (short of a `140` quota by `~16` pairs).
  - `ifeval` (`566` q): `~208.2` pairs (plenty for a `120` quota; **0 extra needed**).
  - `bfcl_simple_python` (`200` q): `~51.0` `right_hold U guard_wide` pairs (**short of a `100` quota by `~49` pairs**).
  - `bfcl_multiple` (`159` q): `~46.9` `right_hold` pairs (tight for a `48` quota).
  - `bfcl_irrelevance` (`220` q): `~30.2` `right_hold` pairs (tight for a `32` quota).
- **Exact Extra Test-Matched Questions Needed (`r2prep --r2-more s3m1`):**
  The unused leftovers in the `r2g-*`, `r2i-ifeval`, `r2j-simple` (Haiku simple tools), and `r2k-nofit` (458 Haiku hard no-fit) top-ups on Drive (`EXPERIMENT_LOG` 14 Nov 04:35 & 14:45: `math 723`, `code 473`, `simple 1,091`, `multiple 807`, `no-fit 2,119` unused) supply almost everything:
  - **`svamp`: `+250` questions** (from `r2g-math` leftovers $\rightarrow$ `900` q total, `~236` pairs for `200` quota).
  - **`mbpp`: `+260` questions** (from `r2g-code` leftovers $\rightarrow$ `599` q total, `~219` pairs for `140` quota).
  - **`bfcl_simple_python`: `+450` questions** (from `r2g-simple` + `r2j-simple` Haiku leftovers $\rightarrow$ `650` q total, `~165` pairs for `100` quota; if `r2prep` runs short on `<80%` solve-rate bins, generate **`+500` harder multi-argument Haiku simple-tool questions** via `gen_tool_questions.py` $\rightarrow$ `newtop r2l-simple`).
  - **`bfcl_multiple`: `+100` questions** (from `r2g-multiple` leftovers $\rightarrow$ `259` q total, `~76` pairs for `48` quota).
  - **`bfcl_irrelevance`: `+180` questions** (from `r2g-nofit` + `r2k-nofit` Haiku hard no-fit $\rightarrow$ `400` q total, `~55` pairs for `32` quota).
  - **`ifeval`: `+0` questions** (`566` q in `r2-step3-questions.jsonl` already yields `~208` pairs for `120` quota).

#### C. Pairs per Type and Kind (`640` Pairs Total = 40 Batches of 16, Fixed Shares, Preset `r2s3`)
Concentrating `68.75%` of the repair budget on Math, Code, and Simple Tools and `31.25%` on maintenance:

| Type (`part`) | Total Quota (Share) | `shorten` | `right` (`right_hold` at `1.00x` + Format Priority) | `guard` (Widened `<0.85x tmed`) | `loop` (Finished + Runaway Loops) | `right_count` (IFEval `atleast` Rules) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`svamp`** (Math) | **`200`** (`31.25%`) | `0` (`0%`) | **`120`** (`60%`) | **`60`** (`30%`) | **`20`** (`10%`) | `0` |
| **`mbpp`** (Code) | **`140`** (`21.88%`) | `0` (`0%`) | **`84`** (`60%`) | **`35`** (`25%`) | **`21`** (`15%`) | `0` |
| **`bfcl_simple_python`** | **`100`** (`15.62%`) | `0` (`0%`) | **`75`** (`75%`) | **`25`** (`25%`) | `0` (`0%`) | `0` |
| **`ifeval`** (IFEval) | **`120`** (`18.75%`) | `0` (`0%`) | **`75`** (`62.5%` = `83.3%` of non-count) | `0` (`0%`) | **`15`** (`12.5%` = `16.7%` of non-count) | **`30`** (`25%` of total) |
| **`bfcl_multiple`** | **`48`** (`7.50%`) | `0` (`0%`) | **`48`** (`100%`) | `0` (`0%`) | `0` (`0%`) | `0` |
| **`bfcl_irrelevance`** | **`32`** (`5.00%`) | `0` (`0%`) | **`32`** (`100%`) | `0` (`0%`) | `0` (`0%`) | `0` |
| **Total (`r2s3`)** | **`640`** (`100.0%`) | **`0` (`0%`)** | **`434` (`67.8%`)** | **`120` (`18.8%`)** | **`56` (`8.8%`)** | **`30` (`4.7%`)** |

#### D. Thinking & Response Rules
- **Why `r2s3` Will Not Lengthen Like `od4-r8`:** `od4-r8` gave back `-7.8%` saving because `--ref-base` anchored DPO to the *original long model* (giving lengthening pairs huge gradients), `--floor-by-part` dropped short right answers before `right_hold`, and `--right-hold` targeted `1.05x`. In `r2s3`, `--rp-base` is **`models/r2s2-s80-on-r2s1b-s180`** (so $\pi_{\text{ref}}$ is the already-shortened Step-2 model), we use **no `--floor-by-part`**, and we center `--right-hold` at **`1.00x` (`[0.90, 1.15]`)**.
- **Bounded Widened Guards (`--guard-wide svamp+mbpp+bfcl_simple_python --guard-wide-max 0.85`):** Only **`120` of `640` pairs (`18.8%`)** actively push length up (`chosen` in `[0.95, 1.25] * tmed` over `rejected < 0.85 * tmed` nearest `0.75 * tmed`, where `tmed` is `r2s2-s80`'s own compressed median).
- **All Pair Flags:** `--think-only svamp+mbpp+ifeval+bfcl_simple_python+bfcl_multiple+bfcl_irrelevance`, `--prefer-reread`, `--right-hold svamp+mbpp+bfcl_simple_python+bfcl_multiple+bfcl_irrelevance`, `--count-rules --count-mode atleast --count-share 0.25 --count-resp-min 1.10 --count-resp-cap 1.75 --count-resp-pad 60 --no-shorten-counting --right-resp-cap 1.15 --loop-response-only ifeval`.

#### E. Hyperparameters, Steps, and Checkpoints
- **Base & Reference Model:** `--rp-base models/r2s2-s80-on-r2s1b-s180` (fresh rank-16 LoRA initialized on the merged Step-2 winner; no `--rp-init-from` needed since `r2s2-s80` is already merged into `--rp-base`).
- **Optimizer & Schedule:** **Constant `--rp-lr 8e-6`** (`--rp-lr-min 8e-6`, `--rp-warmup 2`), **`--rp-beta 0.1`**, LoRA rank 16, batch size 16, `--keep-order`, `--bida-precision`, `bfloat16`.
- **Steps & Checkpoints:** **`--rp-steps 40` (`--rp-epochs 1`)**, scored on **Dev-512 + Dev-SVAMP-200** at **`--rp-save-at 10 20 30 40 --rp-screen 10 20 30 40`** (`~3.1` GPU-h total on one L4).

#### F. Calibrated Dev Gate & Expected Full-Test Landing
- **Measured Lineage Dev Bias (`Dev-512 minus Full Test`):**
  - `r2s1b-s180`: `51.1% / +1.25` vs `50.3% / -1.70` $\rightarrow$ **`+0.8%` saving / `+2.95` pts accuracy**.
  - `r2s2-s40`: `55.0% / +2.35` vs `52.9% / -1.74` $\rightarrow$ **`+2.1%` saving / `+4.09` pts accuracy**.
  - `r2s2-s80`: `55.5% / +1.95` vs `53.7% / -1.27` $\rightarrow$ **`+1.8%` saving / `+3.22` pts accuracy**.
- **Calibrated Gate (`fullpick`):** Using `r2s2-s80`'s exact measured shift (`--fp-save-bias 1.8 --fp-acc-bias 3.22`), a Step-3 checkpoint $s \in \{10, 20, 30, 40\}$ projects to:
  $$\widehat{\text{Saved}}_{\text{full}}(s) = \text{Saved}_{\text{Dev512}}(s) - 1.8\%, \qquad \widehat{\text{Acc}}_{\text{full}}(s) = \text{Acc}_{\text{Dev512}}(s) - 3.22$$
  - **Saving Window:** **`52.2% <= Saved_Dev512 <= 55.2%`** (projects **`50.4% to 53.4%` full-test saving**, giving back `0.3%–3.3%`), with **`Dev-SVAMP saved >= 50.0%`**.
  - **Accuracy Floor:** **`Acc_Dev512 >= +2.35`** ($\ge +0.40$ pts above `r2s2-s80`'s `+1.95`, projecting **`Full Test >= -0.87` pts**, or `>= +2.65` on Dev-512 under the conservative `3.5`-pt mean Step-2 bias), with **`Dev-SVAMP acc >= -0.50`** (up $\ge +0.50$ from Step 2's `-1.00`).
- **Expected Full-Test Landing (at `step-20` or `step-30`):**
  - **Per type:** Math `53.5% / -2.0` (`+0.31` macro gain), Code `57.0% / -1.3` (`+0.27` macro gain), Simple Tools `49.0% / -3.1` (`+0.19` macro gain), Multiple `52.0% / -0.2` (`+0.03` macro gain), IFEval `55% / +1.1` & No-Fit `58% / +5.8` (`-0.05` macro).
  - **Overall Full Test:** **`52.1% – 52.8%` tokens saved / `-0.52` to `-0.78` accuracy points** (`+2.9` to `+3.1` pts above the curve).

---

### 3. Exact `track.py` Plan Lines and Small Code Changes Needed

#### A. Small Code Changes in `repair_pairs.py` (Inside `main()`)
No changes are needed in `track.py` or `dpo.py`. In `reviews/consult22_step3/pack/code/repair_pairs.py`, make three small additions inside `main()`:
1. **Add preset `"r2s3"` (`L95` and `L469`):** Add `"r2s3"` to `--mix-preset` choices, and in the `if args.kinds == "mix":` preset block set:
   - `svamp`: `{"shorten": 0.0, "right": 0.60, "guard": 0.30, "loop": 0.10}`
   - `mbpp`: `{"shorten": 0.0, "right": 0.60, "guard": 0.25, "loop": 0.15}`
   - `bfcl_simple_python`: `{"shorten": 0.0, "right": 0.75, "guard": 0.25, "loop": 0.0}`
   - `ifeval`: `{"shorten": 0.0, "right": 0.8333, "guard": 0.0, "loop": 0.1667}` with `count_share = 0.25`
   - `bfcl_multiple` and `bfcl_irrelevance`: `{"shorten": 0.0, "right": 1.0, "guard": 0.0, "loop": 0.0}`.
2. **Catch Finished Repetition Loops in Standard `mix` Mode (`L191–215`):** Move `looping(r)` (`!finished(r)` or any 8-word phrase $\ge 4\times$) and `bad_format(r)` above `if args.kinds == "mix" and od5:` so the standard `mix` branch also uses them. In standard `mix` (`L210`), define `clean = [r for r in rights if not looping(r)]` (fallback `rights`), and `runaway = [r for r in q if looping(r) and not right(r)]` (filtered by `r["closed_thinking"]` if `base_part in loop_resp_only`), pairing `clean[0]` over `runaway[0]`.
3. **Center `--right-hold` at `1.00x` (`[0.90, 1.15]`) with Format-Defect Rejection Priority (`L254–265`):** In `if base_part in right_hold:`, change `0.95 <= ratio <= 1.25` and `abs(ratio - 1.05)` to **`0.90 <= ratio <= 1.15`** and **`key = (0 if bad_format(w) else 1, round(abs(ratio - 1.00), 2), 0 if decode and REREAD.search(decode(r)) else 1)`**.

#### B. Exact `track.py` Plan Lines (Plain Text)

r2prep --r2-more s3m1 --r2-tops r2a-math,r2a-code,r2a-simple,r2a-multiple,r2b-math,r2b-simple,r2b-multiple,r2c-nofit,r2c-math,r2d-ifeval,r2e-nofit,r2f-ifeval,r2g-math,r2g-code,r2g-ifeval,r2g-simple,r2g-multiple,r2g-nofit,r2i-ifeval,r2j-simple,r2k-nofit --r2-used-glob repair-r2s*.jsonl --r2-need2 svamp=250,mbpp=260,bfcl_simple_python=450,bfcl_multiple=100,bfcl_irrelevance=180,ifeval=0 --r2-need3 svamp=0,mbpp=0,bfcl_simple_python=0,bfcl_multiple=0,bfcl_irrelevance=0,ifeval=0 --data-seed 53

ptrepair --name r2s3 --rp-base models/r2s2-s80-on-r2s1b-s180 --rp-own-data --rp-question-file data/r2-step3-questions.jsonl --rp-questions svamp=650,mbpp=339,ifeval=566,bfcl_simple_python=200,bfcl_multiple=159,bfcl_irrelevance=220 --rp-more-files data/r2-step2-s3m1-questions.jsonl --rp-samples 8 --rp-seed 53 --rp-kinds mix --rp-pair-args=--think-only,svamp+mbpp+ifeval+bfcl_simple_python+bfcl_multiple+bfcl_irrelevance,--prefer-reread,--count-rules,--mix-preset,r2s3,--count-mode,atleast,--count-share,0.25,--count-resp-min,1.10,--count-resp-cap,1.75,--count-resp-pad,60,--no-shorten-counting,--right-resp-cap,1.15,--loop-response-only,ifeval,--guard-wide,svamp+mbpp+bfcl_simple_python,--guard-wide-max,0.85,--right-hold,svamp+mbpp+bfcl_simple_python+bfcl_multiple+bfcl_irrelevance --rp-quota svamp=200,mbpp=140,ifeval=120,bfcl_simple_python=100,bfcl_multiple=48,bfcl_irrelevance=32 --rp-require-full --rp-rebuild-if-short --rp-lr 8e-6 --rp-lr-min 8e-6 --rp-warmup 2 --rp-beta 0.1 --rp-steps 40 --rp-epochs 1 --rp-save-at 10 20 30 40 --rp-screen 10 20 30 40 --rp-dev --rp-dev-svamp --rp-dev-vs rpE80 --rp-dpo-seed 0

fullpick --fp-run r2s3 --fp-kind dpo --pk-steps 10 20 30 40 --pk-save-lo 52.2 --pk-save-hi 55.2 --pk-acc 2.35 --pk-sv-save 50.0 --pk-sv-acc -0.50 --fp-save-bias 1.8 --fp-acc-bias 3.22 --fp-base models/r2s2-s80-on-r2s1b-s180

lossan --la-models fullpick:r2s3 --la-tag r2s3

---

### 4. Risks and Fallback If Step 3 Overshoots the Repair (Gives Back Too Much Saving)

1. **Risk 1 — Pair Shortfall on `bfcl_simple_python` Stopping `--rp-require-full`:**
   - Because `repair_pairs.py` (`L484–502`) pools `s_br = right + guard + loop`, a shortfall in `capguard` or `loop` is automatically filled by `right_hold` without cutting total quota, **provided** total mixed questions reach `100` on `bfcl_simple_python`. If `bfcl_simple_python` is still short after `s3m1` because remaining simple-tool questions are too easy (`>85%` solve rate), generate 500 harder multi-argument questions with Haiku (`gen_tool_questions.py`), run `newtop --name r2l-simple --nt-sources haiku_tools --nt-parts bfcl_simple_python --tt-questions 600 --nt-samples 8 --data-seed 54`, pick 300 via `r2prep --r2-more s3m2`, and append `+data/r2-step2-s3m2-questions.jsonl` to `--rp-more-files`.
2. **Risk 2 — Step 3 Overshoots the Repair and Gives Back `> 3.7%` Saving (Full Test Drops Below `50.0%` Saved):**
   - **Early Checkpoint Selection (`step-10` / `step-20`):** Scoring every 10 steps (`160` pairs/checkpoint at `lr=8e-6`) lets `fullpick` catch the earliest checkpoint (`step-10` or `step-20`) that crosses `+2.35` Dev-512 accuracy while keeping Dev-512 saving at `53.5%–54.5%` (`~51.7%–52.7%` full test).
   - **Fallback A (Zero-Training Adapter Scaling via `merge.py`):** If even `step-10` or `step-20` overshoots the repair (e.g., landing at `49.3%` saved / `-0.45` acc on the full test or `<51.8%` saved / `+2.75` acc on Dev-512), **do not retrain**. Because `r2s2-s80` (`53.7% / -1.27`) and `r2s3-s20` (`~49.3% / -0.45`) share the exact same base model `models/r2s2-s80-on-r2s1b-s180`, scaling the Step-3 adapter by **`0.55`** (`merge.py --model models/r2s2-s80-on-r2s1b-s180 --out models/r2s3-s20-w55 --adapters adapters/r2s3-s0/step-20:0.55`) interpolates directly along the Pareto chord to **`~51.5%` saved / `-0.82` accuracy** in zero training time.
   - **Fallback B (8-Minute Pair Rebuild `r2s3b` on Stored Rollouts):** If widened guards push length up too strongly across all steps, reuse the already-generated Step-3 answers (`--rp-rollouts-from r2s3`) with `--guard-wide-max 0.70`, halve the guard shares (`15%` Math, `12.5%` Code, `0%` Simple; rest `right_hold` at `1.00x`), add `25%` safe shortening (`--shorten-min-right 0.875 --shorten-ratio 1.35`) on `bfcl_irrelevance` (`+6.15` acc cushion) and `ifeval` (`+1.20` acc cushion), and train 30 steps at `lr=6e-6`.