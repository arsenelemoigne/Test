# od5c: online DPO from the original Qwen3-0.6B. Full reproduction record

**Result (full test, 2,759 questions x 8, official evaluation):** `models/od5c-best` (= the model after round 16)
**+54.8% tokens saved / -0.78 accuracy points**, inside the objective (> 50% at better than -1) and above the 52% margin;
+3.09 above our length-accuracy curve (acc = -0.388 + 0.0899 s - 0.0028 s^2), the best point of any run.
Per type (saved / accuracy): math (SVAMP) 61 / -2.84, code (MBPP+) 59 / -1.95, IFEval 47 / +0.49, one function 49 / -1.09,
several functions 50 / 0.00, no function fits 57 / +5.89 (BFCL pooled 400/200/240 = +1.16; official = mean of the 4).
Second full test: `models/od5c-r10` (after round 10) +54.1% / -1.68 (math 65 / -4.58). Loss analysis of both:
reviews/loss_analysis/od5c.txt.
Third full test (15 Nov, to check round 16 was not luck): `models/od5c-r15` (after round 15) **+54.0% / -0.56**, also
inside the objective and above the 52% margin (+3.14 above the curve). Per type: math 62 / -2.79, code 54 / -0.86, IFEval
48 / +0.51, one function 48 / -1.56, several 50 / -1.38, no function fits 57 / +6.93. Loss analysis of r10 / r15 / r16:
reviews/loss_analysis/od5c3.txt (losses are mostly "stopped thinking early" on questions the original solved 6-8/8).

Everything below runs with this repository (branch mvt-fading, folder thinkingcap) on one Colab L4, through
`python track.py plan --plan-file plan_line_b.txt` (the queue runner; each line is one stage of track.py).

## 1. What the method is (one paragraph)
A sequence of online DPO rounds. In every round the CURRENT model answers a fresh slice of 780 questions 8 times; its
answers are compared, question by question, with the ORIGINAL model's stored answers on the same questions; pairs are
built from the current model's own answers only; one short DPO step (15 optimizer steps x 16 pairs) continues the same
LoRA adapter, always with the ORIGINAL model as the DPO reference; the adapter is merged and becomes the next round's
answerer. A per-type brake withholds "think less" pairs from a type that is over its saving cap or losing accuracy on the
slice. No repair stage, no test questions, no other model's answers.

## 2. Code (versions)
- track.py: `odpo` -> `odpo_fixed` (the round loop, the brake, the free monitor `od3_monitor`, merge per round);
  `ptrepair` (answers via probe_curve.py, pairs via repair_pairs.py, DPO via dpo.py); `od3prep` (the test-matched slices);
  `newtop` (new candidate questions answered x8 by the original); `fullpick` / `fullmodel` (full tests); `lossan`.
- repair_pairs.py: `--mix-preset od5 --od5-v2` (commit c9f8752, 14 Nov 17:35). Later commits (7bd8f84 od5 v3, c15d865
  r2s3b) were checked to leave `--od5-v2` output byte-identical on stored answers, so HEAD reproduces od5c.
- dpo.py (DPO loss, `--ref-base`: reference = the base model with the adapter off), sft.py `--init-only` (fresh LoRA),
  merge.py, probe_curve.py (vLLM sampling), evaluate.py (official evaluation, unchanged).
- od5c ran from 14 Nov 17:45 (scripts at 5dfb825) to 15 Nov ~11:00; the runner pulls the branch before every stage.

## 3. Training data (all questions, never the evaluation benchmarks)
- Candidate questions: our pool (data/pool-dp13.jsonl: public training sets GSM8K / MetaMath, KodCode, ToolACE / Glaive /
  Hermes tools, IFEval-style requests built by prepare_types.py) plus "top-up" sets runs/bon/tooltop-*/ (newtop), every one
  answered x8 by the ORIGINAL model; Dev-512 / Dev-SVAMP questions and their near-duplicates excluded; 8-gram
  decontamination against all 2,759 evaluation prompts (build_pool.py).
- The original's per-question statistics: data/od3-base-stats.json (solve rate p0, mean tokens), data/od3-base-think.json
  (median RIGHT thinking length), data/od3-base-resp.json (median right response length; od3prep --od3-resp).
- **Slices** data/od3-r1..r16-questions.jsonl (od3prep): each 780 questions = math 220, code 140, IFEval 200, one function
  80, several 60, no-fit 80, drawn so each type matches the FULL TEST's difficulty profile (the original's solve-rate bins
  <25 / 25-50 / 50-80 / 80-<100 / 100%, test_difficulty_data.py) and solve rate within 3 pts of the test's (no-fit within 5),
  disjoint from each other and from Path 1's step-2/3 sets. Built in steps:
  slices 1-6 `od3prep --od-rounds 6 --data-seed 18 --r2-tops ...` (13 Nov); 7-8 `--od3-more 6 --data-seed 47`;
  9 `--od3-more 8 --od3-tol bfcl_irrelevance=5 --data-seed 48`; 10-11 `--od3-more 7 --od3-tops r2k-nofit,<all Path 1
  top-ups> --od3-tol bfcl_irrelevance=5 --data-seed 49` (r2k-nofit = 458 Haiku near-miss no-fit questions, nofit_haiku.jsonl);
  12 `--od3-more 5 --od3-tops r2n-multiple --data-seed 51` (r2n-multiple = 1,168 Haiku several-function questions,
  tools_multiple_haiku.jsonl, gen_tool_questions.py --multiple; 664 after the pool checks); 13 `--od3-more 4 --od3-tops
  r2o-nofit` (hard Haiku no-fit, nofit_haiku3.jsonl, gen_nofit_questions.py --hard); 14 `r2p-nofit`; 15 `r2q-nofit`; 16
  `r2s-nofit` (nofit_haiku4.jsonl). Haiku = Claude Haiku 5.5 via OpenRouter; no-fit items kept only on two "NO" checks.
- Round 1 uses the original's STORED answers on slice 1 (runs/bon/rp-od3-r1-stored); rounds k >= 2 use the answers of
  round k-1's model on slice k (runs/bon/rp-od5c-rK/rollouts.jsonl, 8 samples, seed 53, vLLM, the evaluation's sampling).

## 4. The pair policy (repair_pairs.py, mix preset od5, --od5-v2), per question q of a slice
Let p0 = the original's solve rate on q (8 answers), p1 = the current model's (8 answers), bt = the original's median
right thinking, tokens counted on THINKING only (--think-only on all types).
- A right answer is "clean" if it is finished and does not loop (no 8-word phrase repeated >= 4 times); chosen answers are
  always clean. Floors (--floor-by-part): a right answer thinking less than max(ABS, RATIO x bt) is never chosen
  (math 25 / 0.44, code 35 / 0.36, IFEval 20 / 0.32, one function 20 / 0.42, several 25 / 0.42, no-fit 20 / 0.35).
- **Shorten** (q as good as the original): p1 >= 0.875 if p0 = 1; p1 >= max(0.50, p0 - 0.125) if p0 >= 0.5; else
  p1 >= max(0.25, p0). Chosen = the shortest clean right answer, rejected = the longest right answer, thinking ratio >= 1.30.
  IFEval only: the chosen response must be >= max(0.95 x the question's median right response, 0.92 x the original's median
  right response) and <= 1.05 x the rejected response; IFEval "at least N" length-rule questions (--count-mode atleast) are
  never shortened.
- **Recovery** (q damaged: 1 <= right <= 7 of 8 and p1 <= p0 - 0.25 if p0 >= 0.875, else p1 <= p0 - 0.125): rejected = a wrong
  answer thinking < 0.85 x bt (shortest first), chosen = a clean right answer thinking between 1.08 x the wrong one + 10 and
  min(1.05 x bt, 1.45 x the wrong one + 50), nearest min(0.85 x bt, 1.25 x the own median), preferring re-reads.
- **Be-right** (undamaged mixed q): compact pair (right thinking 0.50-1.00x the wrong one's, shortest first) for types not
  held; length-held (0.90-1.12x, centred 1.00) for held types.
- **Loop** (clean right > a looping or unfinished answer), **format** (clean right of similar length > a wrong answer with a
  broken format: empty response, math without "Final answer", tool reply without a call), **IFEval length rule**
  (right answer writing >= 1.10 x the wrong one's response + 15, at most 1.75x + 60, thinking <= 1.2x).
- **Allocation per type** (its quota n): length-rule pairs <= 10% (IFEval), loop <= 12%, format <= 10% first; of the rest,
  be-right <= 20%; shorten vs recovery in proportion to their candidate counts, recovery <= 20% unless the slice's accuracy
  for that type is <= -1.5 pts (then x2 weight). A held type gets 0 shorten pairs. At most 3 pairs per question.

## 5. Brake, quotas, hyper-parameters (the queue line, plan_line_b.txt)
odpo --name od5c --od-fixed-ref --rp-base Qwen/Qwen3-0.6B --od-rounds 16 --od-steps 15 --od-r1-steps 20 --od-r1-quota svamp=80,mbpp=56,ifeval=88,bfcl_simple_python=36,bfcl_multiple=28,bfcl_irrelevance=32 --rp-quota svamp=58,mbpp=52,ifeval=58,bfcl_simple_python=26,bfcl_multiple=22,bfcl_irrelevance=24 --rp-questions svamp=220,mbpp=140,ifeval=200,bfcl_simple_python=80,bfcl_multiple=60,bfcl_irrelevance=80 --rp-own-data --rp-kinds mix --rp-samples 8 --rp-seed 18 --rp-pair-args=--think-only,svamp+mbpp+ifeval+bfcl_simple_python+bfcl_multiple+bfcl_irrelevance,--prefer-reread,--count-rules,--shorten-ratio,1.30,--shorten-min-right,0.85,--mix-preset,od5,--od5-v2,--base-stats,data/od3-base-stats.json,--base-resp,data/od3-base-resp.json,--base-think,data/od3-base-think.json,--floor-by-part,svamp=25:0.44+mbpp=35:0.36+ifeval=20:0.32+bfcl_simple_python=20:0.42+bfcl_multiple=25:0.42+bfcl_irrelevance=20:0.35,--count-mode,atleast,--count-resp-min,1.10,--count-resp-cap,1.75,--count-resp-pad,60,--resp-median,0.95,--resp-max,1.05,--right-resp-cap,1.15 --rp-lr 1.4e-5 --rp-warmup 2 --rp-beta 0.1 --rp-dpo-seed 0 --rp-dev-vs rpE80 --od-dev-rounds 99 --data-seed 18 --od-caps svamp=62,mbpp=63,ifeval=56,bfcl_simple_python=46,bfcl_multiple=49,bfcl_irrelevance=56 --od-acc-hold -2.0 --od-unlatch --od-stop-save 999 --od-conv-gain -999
(the same line with `--od-min-rounds 10 / 12 / 13 / 14 / 16` was re-queued after each slice build; finished rounds are skipped.)
- Brake (round k, measured on round k-1's model's answers to slice k before its pairs): a type is held when its slice
  saving >= its cap (math 62, code 63, IFEval 56, one function 46, several 49, no-fit 56) or its slice accuracy <= -2.0;
  unlatched (re-measured each round); held types get no shorten pairs and length-held be-right pairs (no-fit excepted).
- LoRA rank 16, alpha 32, dropout 0, on q/k/v/o/gate/up/down (sft.py --init-only, seed 0); ONE adapter continued across all
  rounds; DPO beta 0.1, batch 16 pairs, 1 pass, lr 1.4e-5 constant (warm-up 2 steps; dpo.py cosine with lr-min = lr),
  AdamW, no weight decay, max length 8,704, reference = the original (adapter off), precision fp32 base + fp16 autocast with
  plain bf16). Round 1: 20 steps on 320 pairs; rounds 2-16: 15 steps on <= 240 pairs.
- After each round the adapter is merged onto Qwen/Qwen3-0.6B -> models/od5c-rK (answerer of round k+1); after round 16,
  models/od5c-best = models/od5c-r16.

## 6. Round by round (free monitor = round k's model answering slice k+1 vs the original on the same questions; Dev = 512 own held-out questions)
| model | monitor saved / acc | math | code | IFEval | one fn | several | no-fit | types held in the NEXT round (brake on this model's answers) |
|---|---|---|---|---|---|---|---|---|
| r1 | 24.5 / +0.41 | 31 / -0.0 | 29 / +1.6 | 18 / 0.0 | 21 / +0.7 | 17 / -1.6 | 22 / +0.5 | none (round 2: not recorded as held) |
| r2 | 36.9 / -1.28 | 46 / -1.9 | 38 / -1.4 | 31 / -0.4 | 26 / -1.7 | 41 / -1.5 | 38 / -0.8 | none |
| r3 | 44.2 / -0.82 | 57 / -1.7 | 50 / -3.8 | 36 / -0.3 | 33 / +2.8 | 34 / +4.7 | 37 / +0.4 | code |
| r4 | 49.6 / -1.19 | 62 / -5.9 | 59 / +1.7 | 36 / -0.9 | 39 / +0.2 | 40 / +3.8 | 47 / -2.2 | math, no-fit |
| r5 | 47.0 / +0.39 | 57 / -2.4 | 53 / -0.6 | 39 / +1.9 | 35 / +3.2 | 44 / +3.0 | 45 / +1.3 | math |
| r6 | 46.9 / -0.33 | 54 / -2.2 | 51 / -2.3 | 43 / +2.1 | 38 / +0.6 | 44 / -0.9 | 40 / +3.5 | math, code |
| r7 | 42.1 / +1.09 | 46 / -1.0 | 41 / -0.1 | 38 / +2.7 | 41 / +1.9 | 42 / +1.4 | 46 / +5.6 | none |
| r8 | 52.8 / -0.02 | 58 / -2.4 | 57 / -1.4 | 46 / +1.1 | 48 / +3.9 | 52 / -0.3 | 55 / +3.2 | math, one fn, several |
| r9 | 48.1 / +2.05 | 52 / -0.1 | 51 / -0.2 | 43 / +3.0 | 47 / +6.5 | 46 / +3.4 | 45 / +5.4 | (see log) |
| r10 | 52.9 / +0.84 | 61 / -2.0 | 58 / +0.8 | 44 / +2.1 | 47 / +2.5 | 50 / 0.0 | 51 / +4.5 | (see log) |
| r11 | 46.8 / +1.30 | 50 / -1.4 | 53 / -1.4 | 37 / +2.6 | 47 / +4.8 | 47 / +3.0 | 48 / +8.2 | one fn |
| r12 | 49.7 / +2.24 | 49 / -1.4 | 55 / +1.4 | 46 / +5.2 | 45 / +3.7 | 52 / -0.4 | 51 / +7.4 | several |
| r13 | 51.2 / +1.25 | 52 / -1.7 | 53 / +2.0 | 46 / +1.5 | 50 / +2.8 | 54 / -2.1 | 59 / +8.2 | one fn, several, no-fit |
| r14 | 50.3 / +2.47 | 52 / -0.0 | 54 / +1.8 | 43 / +6.5 | 49 / -2.2 | 54 / +0.2 | 55 / +9.1 | one fn, several |
| r15 | 52.9 / +2.41 | 56 / -0.8 | 58 / +1.3 | 47 / +5.0 | 44 / +3.5 | 57 / -2.3 | 58 / +10.2 | several, no-fit |
| r16 | (no slice 17) Dev-512 55.0 / +2.22 | | | | | | | |
Dev-512 of intermediate final models: r9-model 48.9 / +1.97, r11 48.6 / +2.56, r12 48.2 / +2.79 (Dev-SVAMP 44.6 / +1.81),
r13 Dev-SVAMP 48.8 / +1.25, r16 55.0 / +2.22.

Pairs per round (shorten / be-right / recovery / loop / format = total): r1 209/62/0/34/15 = 320; r2 137/50/29/21/3 = 240;
r3 138/49/32/17/4 = 240; r4 110/69/37/18/4 = 238; r5 79/49/55/17/4 = 204; r6 98/58/46/20/2 = 224; r7 62/83/48/18/2 = 213;
r8 143/53/24/20/0 = 240; r9-r11 in EXPERIMENT_LOG / Drive pairs files; r12 124/55/26/20/2 = 227; r13 126/51/31/18/4 = 230;
r14 90/51/30/21/4 = 196; r15 109/56/25/17/5 = 212; r16 97/58/31/20/5 = 211. Per type and with candidate counts:
reviews/consult24_od5d/pack/od5c_pairs_table.md (r3-r7) and reviews/od5c/raw_gpu1_log_rounds3to16.txt (r12-r16).
Training loss at steps 5 / 10 / 15 (ordered right at the end): r1 0.653 / 0.538 / 0.444 / 0.363 at 20 (94%); r2 -> 0.44
(81%); r3 0.520 / 0.560 / 0.490; r4 0.578 / 0.624 / 0.552; r5 0.626 / 0.622 / 0.523; r6 0.665 / 0.591 / 0.587; r7 0.622 /
0.616 / 0.584; r8 0.445 / 0.409 / 0.527; r10 -> 0.480 (75%); r11 -> 0.406 (88%); r12 0.549 / 0.519 / 0.583; r13 0.532 /
0.467 / 0.507; r14 0.525 / 0.438 / 0.439; r15 0.558 / 0.522 / 0.587; r16 0.505 / 0.650 / 0.565.

## 7. Biases of the cheap estimates (for choosing checkpoints)
Free monitor vs full test: r10 52.9 / +0.84 -> 54.1 / -1.68 (accuracy over-read 2.5, saving under-read 1.2; math 61 / -2.0
-> 65 / -4.6). Dev-512 vs full test: r16 55.0 / +2.22 -> 54.8 / -0.78 (over-read 3.0). fullpick's arena rule corrects by
1.0 saving / 2.5 accuracy and picked r10, then (skipping tested models) r16, then r15. Monitor vs full test r15:
52.9 / +2.41 -> 54.0 / -0.56 (accuracy over-read 3.0, saving under-read 1.1).

## 8. Where everything is (Google Drive thinkingcap-mvtf/ = the runtime's runs/, data/, models/, adapters/)
models/od5c-r1 ... od5c-r16, models/od5c-best; adapters/od5c-rK-s0/step-15 (r1: step-20); runs/bon/rp-od5c-rK/rollouts.jsonl
(all answers), runs/bon/pairs-od5c-rK/ (pairs.jsonl + summary.txt), runs/bon/od5c-monitor.jsonl (the monitor history),
runs/bon/od5c-latch.json, data/od5c-best.json (history), data/od5c-fullpick.json, data/od5c-fullpick-b.json,
runs/bon/dev512-od5c-best/, runs/mvtf/full-od5c-r10/ and full-od5c-best/ (results.json, records.jsonl of the full tests),
runs/bon/lossan-od5c.txt. In the repository: plan_line_b.txt (all lines, done ones commented with dates), EXPERIMENT_LOG.md
(14 Nov 17:45 -> 15 Nov 14:55), the Haiku question files, consult23 / consult24 reports (the policy's design history).

## 9. To reproduce from scratch
1. Colab L4, the notebook full_eval_colab.ipynb: mount Drive, clone, `python track.py setup`.
2. Rebuild the slices (or reuse data/od3-r*-questions.jsonl + data/od3-base-*.json from Drive): the od3prep / newtop
   lines of section 3 in that order (Haiku files are in the repository; regenerate with the gen_*.py scripts and the same
   seeds only if needed).
3. Run the queue line of section 5 (16 rounds, ~35-60 min each on an L4), then
   `fullpick --fp-run od5c --fp-kind arena --fp-save-bias 1.0 --fp-acc-bias 2.5` and `lossan --la-models fullpick:od5c --la-tag od5c`.
Sampling is not bit-reproducible with vLLM (about a third of answers are identical between two runs), so a rerun lands
near, not exactly on, these numbers.
