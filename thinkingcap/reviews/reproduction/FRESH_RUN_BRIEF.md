# Fresh-run brief: online DPO for shorter thinking in Qwen3-0.6B (clean reproduction)

Draft, 15 Nov 2026. Hand this file to a NEW coding-agent conversation. It assumes no other context.
Two items are still open and are marked **[PENDING]**: the seed replication (od5r) and the precision check (od5p).

---

## 0. What this run is, and what to say about it

This is a **clean reproduction of a recipe found in earlier exploratory work** (Oct to Nov 2026, many Claude Code
sessions). Say so at the start of the new conversation and in IDEA.md. The exploration (about 10 method families, most
of them negative) is summarized in `reviews/strategy_tree/STRATEGY_TREE.md`. Its unedited transcripts are attached
separately, with API keys and unrelated work redacted. The point of the new run is to do the winning recipe methodically:
written predictions first, then the run, then analysis and generalization. It is not a staged rediscovery.

**The result being reproduced (official evaluation, 2,759 questions x 8 answers, BottleCap's `evaluate.py` and
`compare.py`, unchanged):** `od5c`, one online DPO run from the original Qwen3-0.6B.

| model | tokens saved | accuracy (pts) | 95% interval (acc) | math | code | IFEval | one fn | several | no fit |
|---|---|---|---|---|---|---|---|---|---|
| od5c round 16 | **54.8%** | **-0.78** | -1.55 to -0.07 | 61 / -2.8 | 59 / -2.0 | 47 / +0.5 | 49 / -1.1 | 50 / 0.0 | 57 / +5.9 |
| od5c round 15 | 54.0% | -0.56 | -1.31 to +0.17 | 62 / -2.8 | 54 / -0.9 | 48 / +0.5 | 48 / -1.6 | 50 / -1.4 | 57 / +6.9 |

Per type: saved % / accuracy change. A rerun lands near these numbers, not on them, because vLLM sampling is not
bit-reproducible (about a third of answers are identical between two runs).

## 1. Ground rules (the user's, non-negotiable)

- Start from the original `Qwen/Qwen3-0.6B`. One LoRA adapter, rank 16, alpha 32, dropout 0, on q/k/v/o/gate/up/down.
- Constant learning rate. Train only on the model's own answers. Questions come from public TRAINING sets, never
  from the evaluation benchmarks; 8-gram decontamination against all 2,759 test prompts.
- No competitor recipe, no michal-bida data. Do not read or reuse other entrants' code or pair designs.
- Training questions are matched to the test's DIFFICULTY PROFILE (the original's solve-rate bins). This is allowed
  (no test question is used), but must be disclosed.
- No new datasets beyond those listed in section 4 without asking the user.
- API keys live in the macOS Keychain (`openrouter-api-key`, `google-ai-studio-key`, the HF token). Never paste a key
  into the chat, a file or a log.
- Never open, reconnect, restart or delete a Colab runtime without asking. Never click on a running cell.

## 2. Targets, written down BEFORE the run (pre-registration)

Write these into `PREREGISTRATION.md` and commit it before any training:
- **Goal:** > 50% tokens saved at better than -1 point (official weighting). **Margin target:** > 52%.
- **BottleCap's own bar (for the plot):** > 20% shorter, at most 0.2 points lost per 1% saved.
- **Checkpoint rule (fixed in advance):** after 16 rounds, score rounds 10, 12, 14, 15 and 16 on Dev-512 (our own
  512 held-out questions x 8). Correct Dev by its known over-reading (about -0.3 saved, -2.95 accuracy). Full-test the
  best 2 by the fullpick rule: inside the goal, the most saving; otherwise the closest. Report BOTH full tests, whatever
  they show. No other round gets a full test before the write-up.
- **Predictions per type** (write your own; for reference, od5c landed at the table in section 0): where the cut will
  come from (repetition and runaway answers on code and math, every answer on IFEval) and where accuracy will be lost
  (math and code questions the original solves 8 out of 8 times, through "stopped thinking early").

## 3. Hardware, setup and the queue runner

- **Hardware:** Google Colab, one L4 GPU (24 GB) per runtime. Two runtimes were used in parallel; one is enough
  (about 2.5 days in all).
- **Notebook:** `thinkingcap/full_eval_colab.ipynb` (branch `mvt-fading` of github.com/arsenelemoigne/Test). Cell 1
  mounts Google Drive; the user does this. Every result is copied to `MyDrive/thinkingcap-mvtf/` and restored from
  there. Cells 2-6: GPU check; clone BottleCap's official repo (github.com/BottleCapAI/ThinkingCap-Test) and install
  it; copy our scripts next to its `evaluate.py`; then the plan runner (`python track.py plan --plan-file plan_line_b.txt`).
- **The plan runner** pulls the branch before each stage and runs the first line of the plan file that has not run in
  this session. One line = one stage of `track.py`. Queue work by pushing a line. Make it restart-safe: comment out each
  finished line (with a date and result) before any restart. A line whose exact text already ran is skipped; to re-run
  it, change its text (the `--note x` option is ignored by the code and exists for this).
- **Code that matters:** `track.py` (stages `setup`, `newtop`, `od3prep`, `odpo`, `fullpick`, `fullmodel`, `lossan`,
  `devscore`, `transfer`); `repair_pairs.py` (the pair rules: `--mix-preset od5 --od5-v2`); `dpo.py` (DPO loss;
  `--ref-base` = the original as the fixed reference); `sft.py --init-only` (fresh LoRA); `merge.py`; `probe_curve.py`
  (vLLM sampling); `loss_analysis.py`. Optional and recommended for the public repo: extract a minimal package with just
  these paths, then prove it builds byte-identical pairs on stored answers (the way presets were checked on 15 Nov)
  before using it.

## 4. Phase A: understand the original model (no training)

1. **Official baseline:** `evaluate.py --out runs/base` (8 samples, defaults unchanged) and `compare.py` against
   itself. Expected for the original, per type (accuracy %, mean tokens per answer): math 88.7 / 686, code 52.4 / 2,125,
   IFEval 58.7 / 696, one function 82.2 / 195, several 76.1 / 216, no fit 83.4 / 213.
2. **Thinking off:** the same evaluation with thinking disabled. Expected: about 94% fewer tokens at about -23 points
   (heavy losses on math and code). This is the "how much each type needs its thinking" measurement.
3. **Anatomy of the original:** the "Microscope" analysis (scripts in the scratchpad copy `od16/`: an.py, geo.py,
   mass.py, keys2.py, cmp.py, build.py). It covers tokens per type; median vs mean; repetitive answers (13.5% of all
   answers, 52% of all tokens); answers cut at the token cap (2.4%); right vs wrong token shares; paragraphs; vocabulary;
   and token density by length band. Use it to state the weak spots: loops and runaways on code (41% of code answers
   repeat themselves), long re-checking on math, and IFEval thinking that does not change the answer.

## 5. Phase B: data (build it, then check it before anything trains)

- **Candidate questions** (all public training sets, answered 8 or 10 times by the ORIGINAL to measure their difficulty):
  - GSM8K / MetaMath math (`metamath_easy`, `gsm8k_*`);
  - KodCode code;
  - IFEval-style requests built by `prepare_types.py` (`ifeval_built`);
  - tool calls from ToolACE / Glaive / Hermes;
  - "no function fits" items: Glaive near-miss items (Haiku-verified) and Haiku-written hard ones. These are already in
    the repo: `nofit_haiku*.jsonl`, `tools_multiple_haiku.jsonl`, `tools_multiple_removed_verified.jsonl`. Do not
    regenerate them; regenerating costs money and changes the data.
- **Exclusions:** Dev-512 / Dev-SVAMP questions and their near-duplicates; 8-gram overlap with any test prompt;
  michal-bida's data (`--r2-drop` default: MetaMath SV/FOBAR, xLAM no-fit, nofit4 Glaive rows).
- **Slices:** 16 slices of 780 questions (math 220, code 140, IFEval 200, one fn 80, several 60, no fit 80). Each one
  follows the full test's solve-rate bins (<25 / 25-50 / 50-80 / 80-<100 / 100%), with a solve rate within 3 pts of the
  test's (no fit within 5). Slices are disjoint from each other. Slice 1 is twice the size and is answered from the
  original's stored answers. Builder: `od3prep`; it refuses to write a slice that is short or off-difficulty.
- **Two ways to get them:** (a) REUSE the frozen files on Drive (`data/od3-r1..r16-questions.jsonl`,
  `data/od3-base-stats.json`, `data/od3-base-think.json`, `data/od3-base-resp.json`, `runs/bon/rp-od3-r1-stored/`) and
  record their SHA-256 in the write-up; or (b) rebuild them by replaying the newtop / od3prep lines listed in
  `reviews/od5c/REPRODUCE_od5c.md` section 3, in order, with the same seeds. (a) is faithful; (b) is a stronger
  reproduction but lands on slightly different questions. Recommended: (a) for the reproduction, (b) only if time allows.
- **Checks to run and save (one report file):** questions per type per slice; solve rate per type vs the test; bin
  shares; zero overlap between slices; zero Dev / test overlap; share of Haiku-written questions per slice (slices 12-16
  needed them for several-function and no-fit types).

## 6. Phase C: the run

**[PENDING od5p]** If the plain-bf16 rerun (od5p) lands with od5c, use `--rp-plain-precision`. Otherwise keep od5c's
precision setting (fp32 base + fp16 autocast, first seen in michal-bida's public code) and disclose it.

The line (od5c's, from `plan_line_b.txt`; `--name` changed):
```
odpo --name <run> --od-fixed-ref --rp-base Qwen/Qwen3-0.6B --od-rounds 16 --od-steps 15 --od-r1-steps 20 --od-r1-quota svamp=80,mbpp=56,ifeval=88,bfcl_simple_python=36,bfcl_multiple=28,bfcl_irrelevance=32 --rp-quota svamp=58,mbpp=52,ifeval=58,bfcl_simple_python=26,bfcl_multiple=22,bfcl_irrelevance=24 --rp-questions svamp=220,mbpp=140,ifeval=200,bfcl_simple_python=80,bfcl_multiple=60,bfcl_irrelevance=80 --rp-own-data --rp-kinds mix --rp-samples 8 --rp-seed 18 --rp-pair-args=--think-only,svamp+mbpp+ifeval+bfcl_simple_python+bfcl_multiple+bfcl_irrelevance,--prefer-reread,--count-rules,--shorten-ratio,1.30,--shorten-min-right,0.85,--mix-preset,od5,--od5-v2,--base-stats,data/od3-base-stats.json,--base-resp,data/od3-base-resp.json,--base-think,data/od3-base-think.json,--floor-by-part,svamp=25:0.44+mbpp=35:0.36+ifeval=20:0.32+bfcl_simple_python=20:0.42+bfcl_multiple=25:0.42+bfcl_irrelevance=20:0.35,--count-mode,atleast,--count-resp-min,1.10,--count-resp-cap,1.75,--count-resp-pad,60,--resp-median,0.95,--resp-max,1.05,--right-resp-cap,1.15 --rp-lr 1.4e-5 --rp-warmup 2 --rp-beta 0.1 --rp-dpo-seed 0 --rp-dev-vs rpE80 --od-dev-rounds 99 --data-seed 18 --od-caps svamp=62,mbpp=63,ifeval=56,bfcl_simple_python=46,bfcl_multiple=49,bfcl_irrelevance=56 --od-acc-hold -2.0 --od-unlatch --od-stop-save 999 --od-conv-gain -999 --od-min-rounds 16
```
**What one round does:**
1. The current model answers its 780-question slice 8 times.
2. Its answers are compared, question by question, with the ORIGINAL's stored answers.
3. Pairs are built from the current model's own answers (rules in REPRODUCE_od5c.md section 4: "think less" where it is
   as good as the original, bounded "recovery" pairs where it lost accuracy, be-right, loop, format and IFEval-length
   pairs).
4. 15 DPO steps of 16 pairs (round 1: 20 steps, 320 pairs), learning rate 1.4e-5 constant, beta 0.1, with the original
   as the fixed reference.
5. The adapter is merged, and that model answers the next slice.

**The brake:** a type over its saving cap, or at <= -2.0 accuracy on the slice, gets no "think less" pairs that round.

**Expected free monitor per round** (round k's model on slice k+1; od5c; saved / acc): r1 24.5 / +0.41, r2 36.9 / -1.28,
r3 44.2 / -0.82, r4 49.6 / -1.19, r5 47.0 / +0.39, r6 46.9 / -0.33, r7 42.1 / +1.09, r8 52.8 / -0.02, r9 48.1 / +2.05,
r10 52.9 / +0.84, r11 46.8 / +1.30, r12 49.7 / +2.24, r13 51.2 / +1.25, r14 50.3 / +2.47, r15 52.9 / +2.41. The path is
not monotone. The monitor over-reads accuracy by about 2.5-3.0 points and under-reads saving by about 1. **[PENDING
od5r]** The seed replication's per-round numbers go here, as the spread to expect. On 15 Nov, od5r ran ~1 point below
od5c on rounds 1-3 and ~8 points below at round 4.

**Time:** 35-60 min per round on an L4, so 10-16 h for 16 rounds. Dev-512 is about 25 min per model; a full test is about 2.3 h.

## 7. Phase D: results and analysis

1. **The two pre-registered full tests,** with 95% intervals: compare.py prints them per benchmark. Compute the
   headline interval with a paired bootstrap over prompts (od5c r16: -1.55 to -0.07).
2. **Per-type table, up front.** Say plainly that math loses about 2.8 points. Also say that no-fit's gain (+5.9) lifts
   the headline by about 0.4: without it, r16 would read about -1.2.
3. **Loss analysis** (`lossan`, against the original's own resampling noise): losses by cause ("stopped thinking early
   < 0.5x" dominates) and by how reliably the original solved the question (most math losses: questions it solves 8/8).
4. **Microscope after training:** the same boards as Phase A, comparing the trained model with the original. See the
   canvas "Online DPO Under the Microscope". Repetitive answers fall from 13.5% to 2.9%; paragraphs per answer from 12
   to 4; the median answer from 354 to 162 tokens; hesitation words stay at the same rate; re-verification words fall.
5. **Seed robustness:** od5r (same recipe, different seeds). **[PENDING]**
6. **Examples:** answers that got shorter and stayed right, and answers the original got right that now fail, each with
   a one-line explanation.

## 8. Phase E: does it generalize?

Run `transfer` on the final model and the original, both with thinking on and 8 samples. The benchmarks were never
used or imitated in training: MATH-500 (whole-number answers), Skywork-OR1 easy math, ARC-Challenge, CommonsenseQA,
MMLU and HumanEval (plus AIME 2025 at a 32k cap, optional). Report tokens saved and accuracy change per set. Check
whether the same changes appear: fewer paragraphs, less repetition, the same hesitation words.

## 9. Known traps (all happened)

- **Drive not mounted:** stages then write nowhere. `track.py` refuses to run on Colab without Drive; the user mounts it
  (cell 1) after every reconnect.
- **Runtimes disconnect** (both at once on 15 Nov 16:25): keep the plan restart-safe (comment out finished lines first).
- **Clicking a running cell interrupts it** (a synthetic click killed a full test). Read the tabs only; never click.
- **A fresh runtime lacks files:** ptrepair now restores every `data/...` file named in the pair rules. A crash on
  `od3-base-think.json` cost an hour.
- **Short pair sets:** with `--rp-own-data`, a short type used to be "filled" with new, UNMATCHED pool questions.
  Use `--rp-no-fill` and `--rp-require-full` for any offline run.
- **Cheap estimates mislead:** the monitor and Dev over-read accuracy by 2-3.5 points. Never decide on a cheap estimate
  without its bias correction, and never claim a result without the full test.
- **Haiku 5.5 reasons before answering:** with a 10-token limit its YES/NO label checks come back empty. Use about 800
  tokens. The user's cap is $5 in all for Haiku.
- **LLM consultants (Argon / Gemini) over-promise and misquote data:** every prediction so far missed on the high side;
  check every claimed setting against the code before running it.
- **zsh:** `python3 - "$ARG" <<'EOF'` silently does nothing; write the script to a file and run it.

## 10. What to hand in

- The adapter (`.safetensors`) of the 2 pre-registered checkpoints, plus several points along the curve: rounds 10, 15
  and 16 of od5c, and Path 1 for comparison.
- **IDEA.md:** motivation (Phase A), the strategy tree (one page), the method, results with intervals and the per-type
  table, generalization, limitations (math, the -1 goal not statistically secure, test-profile matching, the
  Haiku-written questions, the precision setting if kept), hardware (Colab L4) and cost.
- The clean conversation, plus the exploratory transcripts (keys redacted, unrelated projects removed, both stated).
- The data files (question lists with hashes, the Haiku files) and the plan lines that ran.
