# Consult 17 (r2s1 Pre-Launch Review) — Working Notes

## Files Read So Far
- `reviews/consult17_r2s1/pack/r2s1/queued_command.txt`
- `reviews/consult17_r2s1/pack/r2s1/log_entries.md`
- `reviews/consult17_r2s1/pack/r2s1/dry_run_summary_first_version.txt`
- `reviews/consult17_r2s1/pack/r2s1/dry_run_summary_reuse_version.txt`
- `reviews/consult17_r2s1/pack/r2s1/code_changes_since_plan.diff`
- `reviews/consult17_r2s1/pack/YOUR_PLAN.md` (Sections 1, 2, 4, 5)

## Key Facts & Numbers Extracted So Far
1. **Queued Command (`r2s1/queued_command.txt`):**
   `ptrepair --name r2s1 --rp-base Qwen/Qwen3-0.6B --rp-own-data --rp-rollouts-from p1src --rp-extra-questions ifeval=1200 --rp-extra-source ifeval=ifeval_bida --rp-samples 8 --rp-seed 41 --rp-kinds mix --rp-pair-args=--think-only,svamp+mbpp+ifeval+bfcl_simple_python+bfcl_multiple+bfcl_irrelevance,--prefer-reread,--count-rules,--shorten-ratio,1.40,--shorten-min-right,0.80,--mix-preset,r2s1,--count-mode,length,--count-resp-min,1.10,--count-resp-cap,1.75,--count-resp-pad,60,--no-shorten-counting,--resp-median,0.95,--resp-max,1.05,--resp-near-long,0.95,--right-resp-cap,1.15,--loop-response-only,ifeval,--guard-wide,svamp+mbpp,--guard-wide-max,0.75 --rp-quota svamp=720,mbpp=720,ifeval=720,bfcl_simple_python=260,bfcl_multiple=220,bfcl_irrelevance=240 --rp-lr 2e-5 --rp-lr-min 5e-6 --rp-warmup 10 --rp-steps 180 --rp-epochs 2 --rp-dpo-seed 0 --rp-screen 120 150 180 --rp-dev --rp-dev-svamp --rp-dev-vs rpE80`

2. **Dry Run 1 (`dry_run_summary_first_version.txt`, on `p1src` alone without extra questions or `--shorten-second`):**
   - Total pairs: 2,668 (2,670 questions) instead of 2,880.
   - `svamp`: 720 (288 shorten [3610+297 cands], 288 be_right [1770 cands], 108 guard [947 cands at 0.95, 582 cands at 0.75], 36 loop [526 cands]).
   - `mbpp`: 720 (432 shorten [653+41 cands], 144 be_right [865 cands], 90 guard [606 cands at 0.95, 387 cands at 0.75], 54 loop [186 cands]).
   - `bfcl_simple_python`: 260 (182 shorten [621+122 cands], 78 be_right [228 cands]).
   - `bfcl_multiple`: 220 (110 shorten [333+29 cands], 110 be_right [181 cands]).
   - `bfcl_irrelevance`: 240 (144 shorten [429+156 cands], 96 be_right [292 cands]).
   - `ifeval`: **508 of 720** (SHORT by 212 because the user made per-type mix a HARD rule: scarcest kind limits the whole part!).
     - Target for `ifeval` (720 total, `count_share = 0.20`, rest `0.50 shorten / 0.4167 right / 0.0833 loop`):
       - `counting`: 144 (20% of 720) — had 175 candidates.
       - Remaining 576: `shorten` = 288 (50% of 576), `be_right` = 240 (41.67% of 576), `loop` = 48 (8.33% of 576).
     - Actual `ifeval` candidates in `p1src` (out of 2,206 ifeval questions answered, 1,309 with counting rule, so 897 non-counting questions!):
       - `shorten`: **177 clear (>=1.40x) + 26 backup (1.20-1.40x) = 203 total candidates!**
       - Because `shorten` only has 203 candidates (out of 288 needed, i.e. 203/288 = 70.49% of quota), the hard mix rule scales down `ifeval` to 508 pairs (203 shorten, 155 be_right, 102 counting, 48 loop)!

3. **Why `ifeval` `shorten` had only 203 candidates in `p1src`:**
   - `--no-shorten-counting` excludes the 1,309 questions with a counting rule (`is_cr`), leaving only `2206 - 1309 = 897` non-counting IFEval questions in `p1src`.
   - Plus `--resp-near-long 0.95` was passed in `--rp-pair-args`! Notice `--resp-near-long 0.95` in `queued_command.txt`! Let's check what `--resp-near-long 0.95` does in `repair_pairs.py` and whether `YOUR_PLAN.md` had `--resp-near-long 0.95` or only `--resp-median 0.95 --resp-max 1.05`!
   - Also, out of 897 non-counting IFEval questions in `p1src`, 203 gave a shorten candidate (177 clear + 26 soft), which is `203 / 897 = 22.63%` of non-counting questions, or `203 / 2206 = 9.20%` of all IFEval questions!

4. **Open Questions to Check Next:**
   - Read the full `r2s1/repair_pairs.py`, `r2s1/track.py`, and `code/prepare_types.py` (specifically `ifeval_bida`).
   - Check how `pt_ifeval_bida` or `ifeval_bida` works in `prepare_types.py`: what is its fraction of counting rules (`is_cr`), how many questions exist in `allenai/IF_multi_constraints_upto5`, how does `check_vllm.py` grade them, what fields do its records have (`id`, `source`, `type`, `prompt`, `instruction_id_list`, `kwargs`), and what happens when `sv-` prefix is added (`pool_id: sv-...`)!
   - Check if `repair_pairs.py` groups by `r["id"]` or `r["pool_id"]`, how `part_of(r)` identifies `ifeval` on `ifeval_bida` rows, whether `p1src` already has `p1_reserved` or if `p1src` in `r2s1` uses 2,206 IFEval questions (why 2,206 vs 2,037 in `YOUR_PLAN.md`? Because `p1src` wasn't rebuilt with `dev_svamp_gram_set()`!).
   - Check every detail of the 5 deliverables required by the prompt.

## Update from `r2s1/repair_pairs.py` (First Half)
- `think_only`, `loop_resp_only`, `guard_wide`, `right_hold`, `shorten_second` all split on `r'[,+]'`.
- `no_shorten = is_cr or (args.no_shorten_counting and part == 'ifeval' and counting(q[0]))`: because `--no-shorten-counting` is passed, ANY IFEval question matching `counting(q[0])` (7 broad prefixes in `COUNT_RULES`, which covers 1,309 of the 2,206 IFEval questions!) is barred from giving a `shorten` pair, even though `is_cr` under `--count-mode length` only uses `length_rule(q[0])`!
- In `shorten` candidate selection for `part in think_only`: `ok` requires THREE response conditions simultaneously:
  1) `response_len(r) >= args.resp_median * rmed` (`0.95 * rmed`)
  2) `response_len(r) <= args.resp_max * response_len(long)` (`1.05 * response_len(long)`)
  3) `response_len(r) >= args.resp_near_long * response_len(long)` (`0.95 * response_len(long)`)
  Wait: condition (3) `--resp-near-long 0.95` was added in `queued_command.txt`! If `long` (the longest-thinking right rollout) also happens to have a slightly longer response than `rmed` (say 1.10x `rmed`), requiring `response_len(r) >= 0.95 * response_len(long)` rejects candidates that are within `[0.95 * rmed, 1.05 * rmed]`!

## Update from `prepare_types.py`, `probe_curve.py`, and `tables/`
- `ifeval_bida` in `prepare_types.py` yields rows with `source: 'ifeval_like'`, `task: 'instructions'`, `grader: 'ifeval'`, `id: f'ifeval_bida-{origin}-{key}'`, `instruction_id_list`, `kwargs`, `original_prompt`, `prompt`.
- It draws from TWO HuggingFace datasets: `allenai/IF_multi_constraints_upto5` (`origin='if_multi'`) AND `allenai/RLVR-IFeval` (`origin='rlvr'`).
- In `probe_curve.py`, `GRADE_FIELDS` copies `('grader', ..., 'instruction_id_list', 'kwargs', 'original_prompt')` into each rollout row, and sets `id: question['id']` (which `load_questions` set to `row.get('pool_id')` = `sv-ifeval_bida-...`).
- Since `source` is `'ifeval_like'`, `part_of(r)` returns `'ifeval'`. And since `part == 'ifeval'` (not `'svamp'`), the `sv-` prefix on `id` does NOT affect `part_of` or `svamp_hard` splitting!
- Look at `path1_round1_v3_p1cr1s0_pairs_summary.txt` vs `dry_run_summary_first_version.txt`:
  - Why did `p1cr1s0` have `298 + 40 = 338` shorten candidates on 2,037 IFEval questions (816 non-counting), whereas `r2s1` (`dry_run_summary_first_version.txt`) had only `177 + 26 = 203` shorten candidates on 2,206 IFEval questions (897 non-counting)?
  - Let's compare `p1cr1s0` flags vs `r2s1` flags!

## Update from Comparing `p1cr1s0` vs `r2s1` Dry Run
- RESOLVED why `p1cr1s0` had 338 (`298+40`) IFEval `shorten` candidates while `r2s1` had 203 (`177+26`):
  - In `p1cr1s0` (`queued_lines.txt`), `--no-shorten-counting` was NOT passed (`--count-mode length` without `--no-shorten-counting`) AND `--shorten-ratio 1.35` was used. So `no_shorten = is_cr` only excluded the `length_rule(r)` subset (~25-30% of IFEval questions), allowing non-length counting rules (like `detectable_format:number_bullet_lists`, `detectable_content:number_placeholders`, `keywords:letter_frequency`) to yield `shorten` pairs with held response (`[0.95, 1.05]`).
  - In `r2s1` (`queued_command.txt`), `--no-shorten-counting` IS passed AND `--shorten-ratio 1.40` is used. `--no-shorten-counting` sets `no_shorten = is_cr or (args.no_shorten_counting and part == 'ifeval' and counting(q[0]))`, which blocks ALL 1,309 `counting(q[0])` questions (59.3% of the 2,206 IFEval questions) from yielding `shorten` pairs, leaving only `2206 - 1309 = 897` non-counting questions!
- Also verified what happens when `--think-only svamp+mbpp+ifeval+bfcl_simple_python+bfcl_multiple+bfcl_irrelevance` is passed in `r2s1`:
  - In `p1cr1s0` (`--think-only ifeval`), `bfcl_irrelevance` `shorten` pairs had median chosen vs rejected response `19 vs 33` (chosen response 38% shorter!) and `mbpp` had only 38.6% of `shorten` pairs with response ratio in `[0.95, 1.05]`.
  - In `r2s1` (`--think-only` on all 6 parts + `--resp-median 0.95 --resp-max 1.05 --resp-near-long 0.95`), `svamp` `shorten` candidates drop only slightly (`3925+172 -> 3610+297`, quota 288 easily met), `mbpp` `shorten` candidates go from `1028+9 -> 653+41 = 694` (quota 432 easily met), `bfcl_simple_python` `709+103 -> 621+122` (quota 182 met), `bfcl_multiple` `292+24 -> 333+29` (quota 110 met), `bfcl_irrelevance` `1673+146 -> 429+156` (quota 144 met).

## Checking `newtop` / `instr2` / `ifeval_bida` in `track.py`

## Update on `--rp-extra-source` and `ifeval_bida`
- In `prepare_types.py:ifeval_bida(args, rng)`:
  - Draws from TWO datasets: `allenai/IF_multi_constraints_upto5` (1-3 constraints) AND `allenai/RLVR-IFeval` (1 constraint mapped via `RLVR_TO_IFEVAL`).
  - Note that in `RLVR_TO_IFEVAL` (`prepare_types.py` L457-480), there are 19 rules: 9 are counting rules (`COUNT_RULES`: `capital_word_frequency`, `multiple_sections`, `number_words`, `number_sentences`, `number_placeholders`, `number_highlighted_sections`, `nth_paragraph_first_word`, `number_paragraphs`, `number_bullet_lists`) and 10 are non-counting rules (`english_lowercase`, `english_capital`, `end_checker`, `quotation`, `two_responses`, `json_format`, `postscript`, `no_comma`, `forbidden_words`, `existence`, `title` — actually 11 non-counting rules out of 20 total entries!).
  - In `competitor_methods_study.md` L51, michal-bida's filtered pool from those two sources had `963` rows from `IF_multi_constraints_upto5` + `1,422` rows from `RLVR-IFeval` = `2,385` prompts (before prompt-length decile subsampling, or after).
  - In `EXPERIMENT_LOG.md` L4063, `newtop instr2` drew 2,000 questions x4 from `ifeval_bida` (`mean solve 65%, always 1,016 / sometimes 528 / never 456`: i.e. 1,016/2,000 = 50.8% got 4/4 right!).
- Let's check the code risks and bugs in `track.py` and `repair_pairs.py`!

## Update on Deliverable 3 (Will 1,200 new IFEval questions x8 be enough to reach 288 shorten pairs?)
- In `p1src` (`dry_run_summary_first_version.txt`), 2,206 IFEval questions (1,309 counting = 59.34%, 897 non-counting = 40.66%) produced `177` clear (`>=1.40x`) + `26` backup (`1.20-1.40x`) = `203` `shorten` candidates (`22.63%` of non-counting questions, `9.20%` of all IFEval questions).
- Since `repair_pairs.py` uses both `shorten` (`>=1.40x`) and `shorten_soft` (`1.20-1.40x`), we already have `203` distinct-question candidates and need only **`288 - 203 = 85` more** (`7.08%` yield across 1,200 new questions) to reach 288 total `shorten` pairs, or **`288 - 177 = 111` more** (`9.25%` yield) if counting clear `>=1.40x` pairs alone.
- Across all three empirical benchmarks of `Qwen/Qwen3-0.6B` stored answers under the exact `r2s1` rules (`--shorten-ratio 1.40 --shorten-min-right 0.80 --no-shorten-counting --resp-median 0.95 --resp-max 1.05 --resp-near-long 0.95`):
  1. `p1src` (`2,206` qns, `K=10`, `40.66%` non-counting): `177/2206 = 8.02%` clear (`19.73%` of non-counting), `203/2206 = 9.20%` clear+soft (`22.63%` of non-counting). Adjusting from `K=10` (`>=8/10`) to `K=8` (`>=7/8`) reduces yield by ~4-5% relative (`~8.75%` clear+soft -> `105 +- 9.8` new pairs -> `308` total, `P(>=85) = 98.4%`).
  2. `Dev-512` (`132` qns, `K=8`, `50.0%` non-counting): `12/132 = 9.09%` clear (`18.18%` of non-counting), `15/132 = 11.36%` clear+soft (`22.73%` of non-counting) -> `136 +- 11.0` new pairs (`339` total, `P(>=85) = 99.9999%`).
  3. `Full Test` (`541` qns, `K=8`, `46.8%` non-counting): `52/541 = 9.61%` clear (`20.55%` of non-counting), `62/541 = 11.46%` clear+soft (`24.51%` of non-counting) -> `138 +- 11.0` new pairs (`341` total, `P(>=85) > 99.9999%`).
  4. `ifeval_bida` composition (`963` multi-constraint `IF_multi_constraints_upto5` + `1,422` single-constraint `RLVR-IFeval` = `2,385` usable prompts, i.e. ~60% single-constraint where non-counting share is `55%` and base solve rate is `~73%`): expected yield is even slightly higher (`~11.5-12.5%`, or `~138-150` new candidates -> `341-353` total).

## Update on Deliverable 4 & 5: Why `p1cr1s0` Plateaued at 34.4% Saved on Dev-512 (~32% Full Test) and What That Means for `r2s1`!
- Look at the empirical comparison between `p1r1s0` (45.9% Dev-512 / 43.4% Full Test) and `p1cr1s0` (34.6% -> 32.4% -> 34.4% Dev-512; 37.7% -> 28.1% -> 31.7% Dev-SVAMP-200):
  - Why did `p1cr1s0` save 11.5 percentage points LESS than `p1r1s0` on Dev-512 (and 23.0 points less on Dev-SVAMP-200 at step 200: 31.7% vs 54.7%), and why did its saving SHRINK from step 100 (34.6%) to step 150 (32.4%) and plateau at step 200 (34.4%)?
  - Let's check the exact numbers from the pair tables!

## Complete Synthesis of Findings for Deliverables 1–5
- **Deliverable 1 (Plan vs Queued Command & Code Table):**
  - Prepared a comprehensive, item-by-item comparison table covering Data, Pair Rules & Thresholds, Quotas per Type & Kind, Fallbacks, Training Settings, Checkpoints & Evaluation, and Decision Gate (`same`, `different`, `missing`).
- **Deliverable 2 (Bugs & Risks in New Code, with Exact Lines to Fix):**
  1. `track.py` L3774 (`--rp-extra-source` pool deduplication): only checks `data/pool-dp13.jsonl`, missing `runs/bon/tooltop-*/questions.jsonl` (including `tooltop-instr2` which was built from `ifeval_bida`).
  2. `track.py` L3779 (`--rp-extra-source` test 8-gram filter): does not ensure `questions/questions.jsonl` exists on a fresh runtime (`FileNotFoundError` if `benchmarks.py` hasn't run yet).
  3. `track.py` L3771 / `prepare_types.py` L499 (`ifeval_bida` cache generation): `prepare_types.py:ifeval_bida` reads `paths(args.questions)['ifeval'] / 'data/input_data.jsonl'` (`questions/ifeval/...`), which requires `utils/eval/benchmarks.py` to have been run first.
  4. `track.py` L3783 vs `p1src` L3962: `p1src` was already cached on Drive before `dev_svamp_ids()` and `dev_svamp_gram_set()` were added, so `runs/bon/rp-p1src/rollouts.jsonl` still lacks the `Dev-SVAMP-200` filter.
  5. `repair_pairs.py` L449-467 (`--kinds mix` hard rule): `loop` and `guard` are NOT individually included in the `total = min(...)` scaling ratio (only their sum with `r`, `len(g) + len(r)` over `s_br`), AND when `keep_br < len(g) + len(r)`, `r` is truncated first (`r = r[:max(0, keep_br - len(g))]`).
  6. `repair_pairs.py` L224 (and L319): `groups` is a dict keyed by `qid` in insertion order (`p1src` first, then `rp-r2s1-extra`). While `take()` shuffles candidate lists (`rng.shuffle(pool)`), consuming from `rng` in `take()` depends on the exact length of `pool` in each `part` in the loop `for part, n in quota.items():`. Because `ifeval` is the 3rd key in `quota` (`svamp, mbpp, ifeval, bfcl_simple_python, bfcl_multiple, bfcl_irrelevance`), changing `len(cand['ifeval'][...])` changes the state of `rng` AFTER `ifeval` is processed! Therefore, adding 1,200 new `ifeval` questions DOES NOT change `svamp` or `mbpp` pairs (which come before `ifeval` in `quota`), but it DOES change the random shuffle seed state for `bfcl_simple_python`, `bfcl_multiple`, and `bfcl_irrelevance` (which come after `ifeval` in `quota`), AND changes the interleaving shuffle `for part, ps in by_part.items(): for pr in sorted(ps, key=lambda p: rng.random()):`!
- **Deliverable 3 (Will 1,200 new IFEval questions x8 reach 288 `shorten` pairs?):**
  - Exact calculation from `p1src` (`2,206` qns, `K=10`), `Dev-512` (`132` qns, `K=8`), `Full Test` (`541` qns, `K=8`), and `ifeval_bida` single- vs multi-constraint composition (`1,422` RLVR 1-constraint + `963` IF_multi 1-3 constraints = `2,385` usable prompts). Need only `85` more `shorten` (`clear + backup`) candidates (or `111` `clear`-only) from `1,200` questions; expected yield is `105–142` new candidates (`308–345` total, `P(total >= 288) = 98.4%–99.99%`).
- **Deliverable 4 & 5 (Settings to Change Before Launch, Updated Expectations for Step 1, and Go/No-Go Gate for Step 2):**
  - Crucial empirical diagnosis from `p1cr1s0` (34.6% -> 32.4% -> 34.4% Dev-512; 37.7% -> 28.1% -> 31.7% Dev-SVAMP-200) vs `p1r1s0` (47.5% -> 43.7% -> 45.9% Dev-512, 43.4% Full Test): why `r2s1` as queued will likely land around `31%–36%` saved on Dev-512 (`~29%–34%` Full Test) instead of the `46.0%–51.5%` target unless adjusted, what settings to change if aiming for ~47% in Step 1, and the exact numerical Go / No-Go gate for Step 2.
