# The IFEval + tool-call repair (rpK) — why these two fail, in the data, and the run built from it (11 Nov 2026)

User, 11 Nov: "Analyze why ifeval + no function fit fail exactly in the data, be very thorough" and "independently plan and
queue the repair that you suggested on ifeval and tool calls". Independent of Path 1.

## Data used (all real, downloaded from Drive; nothing estimated)

- The full test's per-answer records (2,759 questions x 8 = 22,072 answers each) of five models: the original Qwen, the
  original with thinking off, rc1-soup-30 (57.4 / -3.5), rpE-80 (47.0 / -0.9), rpI-120 (41.9 / +0.1).
- The 2,759 test questions rebuilt locally with the challenge's own script (pinned revisions), and Google's official IFEval
  checkers run rule by rule on every answer (re-grading matches the stored verdicts: 0-1 mismatch per model).
- The training answers: the original's 10 stored answers (17,870 questions), rc1-soup-30's and rpE-80's own answers (5,700
  questions x 8 each).

Tokens = thinking + response, per answer, means.

| type | original | thinking off | rc1-soup-30 | rpE-80 | rpI-120 |
|---|---|---|---|---|---|
| math | 677 + 10, 88.7% | 0 + 13, 20.6% | 219 + 9, 82.1% | 307 + 8, 87.5% | 356 + 8, 88.7% |
| code | 2,079 + 47, 52.4% | 0 + 61, 40.8% | 732 + 46, 49.0% | 1,090 + 51, 51.4% | 1,269 + 50, 53.0% |
| IFEval | 433 + 263, 57.8% | 0 + 262, 56.6% | 141 + 201, 55.6% | 156 + 223, 55.4% | 163 + 247, 56.3% |
| one function | 171 + 24, 82.2% | 0 + 22, 71.4% | 91 + 24, 82.1% | 98 + 24, 82.2% | 99 + 24, 82.3% |
| several functions | 192 + 24, 76.1% | 0 + 20, 55.3% | 98 + 24, 75.2% | 107 + 23, 76.1% | 110 + 23, 75.2% |
| no function fits | 186 + 27, 83.4% | 0 + 8, 90.3% | 92 + 13, 79.3% | 96 + 11, 86.8% | 95 + 11, 89.2% |

## IFEval: the loss is a shortened RESPONSE on 58 questions, not missing thinking

IFEval's 541 questions fall in four classes. Contribution of each class to the change of IFEval accuracy (points):

| class | questions | original | thinking off | rc1-soup-30 | rpE-80 | rpI-120 |
|---|---|---|---|---|---|---|
| "at least N words / sentences" | 58 (11%) | 33.2% right | -0.16 | -1.66 | -1.57 | -1.43 |
| another counting rule (keyword count, placeholders, sections...) | 183 (34%) | 58.4% | -1.69 | -0.90 | -0.67 | -0.58 |
| no counting rule | 264 (49%) | 60.7% | +0.65 | +0.46 | -0.02 | +0.90 |
| "fewer than N words / sentences" | 36 (7%) | 72.9% | +0.02 | -0.09 | -0.12 | -0.37 |
| **IFEval total** | 541 | 57.8% | **-1.18** | **-2.20** | **-2.38** | **-1.48** |

1. **Two thirds of every trained model's IFEval loss sits in 58 "at least N" questions.** There the response is simply too
   short: on "at least N words" (34 questions, N median 343) the original writes 0.89 x N words (median) and passes 37.5%;
   rc1 / rpE-80 / rpI-120 write 0.57 / 0.63 / 0.65 x N and pass 10.7 / 11.4 / 16.9%. Response tokens on the class: 668
   (original) -> 443 / 489 / 560. The thinking even says it ("Need to keep it around 400 words" for "400 or more words").
   With thinking OFF the original still writes 0.83 x N and passes 34.9%: the words are lost by our training, not by the
   missing thinking. Rule by rule (pass rate vs the original): number_words -11 to -15 points, number_sentences ("at least")
   57% -> 36-41%, keyword frequency ("at least") 70% -> 52-56% (fewer words, fewer occurrences).
2. **On half of IFEval (no counting rule, 264 questions) thinking is worth nothing**: thinking off scores +1.3 points on the
   class; trained models are level with the original. Same for "fewer than N".
3. **Thinking earns its keep only on the other counting rules** (183 questions): thinking off loses 5 points there (keyword
   frequency -18.8 on the rule, placeholders -21.8, lower-case -12.2); the trained models lose 1.7-2.7.
4. **Inside one question, right and wrong answers think the same length** (trained models: difference 0-4 tokens over ~225
   mixed questions): thinking length does not decide correctness on IFEval.
5. **The trained models never produce a short think**: thinking under 40 tokens = 0.2% of rpE-80's IFEval answers (5th
   percentile 75 tokens, median 128). Pairs built from the model's own answers cannot prefer what never appears, which is why
   no run of ours pushed IFEval past 58% saved.

What an ideal IFEval looks like, in tokens (rpE-80 today: 156 + 223 = 380, 45% saved, -2.4): full-length responses on "at
least N" (+19 tokens on average), thinking ~30 tokens on the 300 questions where it is worth nothing (-72 tokens), thinking
kept on the other counting rules: ~327 tokens = **53% saved at about -0.5** instead of 45% at -2.4. With thinking also short on
the other counting rules: ~292 (58%). The thinking-off point (62% / -1.2) is the outer limit.

## No function fits: 28 near-fit questions carry everything; thinking only decides "call or not"

- The only wrong reply on this type is a parseable function call (prose, "[]" or a malformed call all count as "no call").
  Calls: original 16.6% of answers, thinking off 9.7%, rc1 20.7%, rpE-80 13.2%, rpI-120 10.8%.
- **28 questions hold almost all the errors**: the offered function nearly fits ("Find me restaurants in London" with
  `find_pois`; "latest trend in technology" with `get_social_trends`). On them: original 2% right, thinking off 59%, rc1 7%,
  rpE-80 18%, rpI-120 27%. On the other 212: original 94.2%, thinking off 94.4%, rpE-80 95.9%, rpI-120 97.4%. The whole
  +6.9 of thinking off comes from those 28: with no thinking the model declines by default; with thinking it talks itself
  into the call (the original's wrong answers think 264 tokens, its right ones about 170; inside one question the right answers
  think 64-86 tokens LESS).
- So on 212 of 240 questions thinking can be cut with nothing to lose, and on the 28 more thinking makes it worse.
- **Same floor as IFEval:** rpE-80's thinking is 66 tokens at the 5th percentile, 89 at the median, never under 40 (0.8% under
  60). Its response is already minimal (11 tokens; "[]" in 53% of answers). 50% saved today = 96 + 11 tokens; 75-85% saved
  needs thinking at 20-40 tokens.

## One function / several functions: thinking decides whether to call, not how

- With thinking off, when the model does call, the call is as good: one function 83.4% right among calls (84.7% with
  thinking), several functions 76.9% (80.1%). The whole loss of thinking off (-10.8 / -20.8) is NOT calling: prose or "[]"
  in 14% / 28% of answers (3% / 5% with thinking).
- Accuracy is flat across thinking length (every quintile of rpE-80's thinking equals the original on the same questions), and
  these two types were never pushed past 44% saved in any full test, with no loss (+0.1 / +0.1 for rpE-80).
- Same floor: 68 / 73 tokens at the 5th percentile, 93 / 101 at the median.

## What this says about the repair

- The target is narrow and clear: (1) IFEval responses must not shrink, (2) thinking can drop on IFEval questions without a
  counting rule and on all three tool types, (3) math and code must not move.
- On paper, from rpE-80 (47.0 / -0.9), official weighting: IFEval at 53% saved and about -0.5 -> 48.7 / about -0.45; plus no-fit
  thinking at ~30 tokens -> 50.9; plus one / several functions at ~60 tokens of thinking -> ~54. The tool types are the larger
  lever, IFEval the safer one. None of it is measured in a trained model yet.
- **The risk is the one the user named: keeping the rest stable.** The log's only direct evidence is against us: an adapter
  trained on ONE type's pairs from the original cut every other type almost as much (lpt6, 5 Nov: "instructions only" ->
  math +27% / -3.5, code +32% / -1.9; "think less is one shared habit"). Those runs had no pair at all on the other types. rpK
  tests the opposite arrangement: the other types are in every batch, with pairs that pull the other way.

## rpK — the run (queued on GPU 2 between Path 1's round 1 and round 2; ~1.2 GPU-hours, no new generation)

- **Start:** rpE-80 (our best balanced model; full test and Dev-512 known). Fresh LoRA on it, reference = rpE-80.
- **Answers:** rpE-80's OWN answers to 5,700 questions x 8 (made on 10 Nov for rpI; never a Dev-512 question).
- **Pairs, 1,060 (built locally on the real answers; the GPU builds the same file):**

  | type | pairs | "think less" | "be right" | "don't stop early" |
  |---|---|---|---|---|
  | IFEval | 320 | 112 (questions without a counting rule; thinking only; response never shorter than 0.9x) | 48 + 160 counting-rule pairs (the right answer writes at least as much) | 0 |
  | one function | 160 | 112 | 48 | 0 |
  | several functions | 120 | 84 | 36 | 0 |
  | no function fits | 140 | 84 | 56 | 0 |
  | math | 160 | **0** | 128 | 32 |
  | code | 160 | **0** | 128 | 32 |

  "Think less" = 37% of the file, all on IFEval and tool calls (bar 7/8 right, longest right >= 1.30x the shortest). Medians
  thinking, chosen vs rejected: IFEval 90 vs 166 (response 65 vs 58), one function 61 vs 105, several 66 vs 108, no fit 71 vs
  111. Math and code "be right" pairs are equal-length (286 vs 330, 427 vs 433 tokens); their "don't stop early" pairs prefer
  the longer right answer (654 vs 355).
- **Training:** 60 steps of 16, lr 1e-5 constant, beta 0.1 (rpE's settings).
- **Reading (Dev-512 at steps 20 / 40 / 60, each PAIRED against rpE-80, with the thinking / response split per type):**
  success = IFEval and tool-call tokens down, IFEval responses not shorter, math and code tokens within about +-5% of
  rpE-80 and paired accuracy not below about -0.5. Failure = math or code shortened too (the shared habit wins) or
  accuracy falling with it.
- **Honest expectation:** pairs from the model's own answers can only walk the thinking floor down a step per round (to about
  the current 10th-25th percentile), so one round is worth roughly +1 to +2 points of headline saving if the rest holds: enough
  to show whether the allocation works, not enough alone to reach 50 / -1. If it works, the same pairs are repeated on fresh
  answers (a second round) and folded into Path 1's round 2.

## Next step if rpK holds the rest stable: go below the floor

The model's own answers never show thinking under 40 tokens, so the larger prize (IFEval ~55-58%, tool calls ~60-70% saved)
needs answers made on purpose: take a natural answer, close the thinking after its first sentences (20 / 40 / 60 tokens), let
the model write the answer itself, grade it (the budget ladder of 10 Nov, with FIXED small budgets: its percentile rungs start
at ~90 tokens for tools and ~150 for IFEval, above the region that matters). Questions that stay right with the short think give
"think less" pairs; questions that turn wrong give "don't stop early" pairs. Rule check: a short real think, never an empty one
(the README wants a model "that thinks before it answers").

## rpK's results and rpL (11 Nov, after consult9)

rpK on Dev-512 (paired against rpE-80): step 20 +3.6 saved / +0.15 accuracy, step 40 +3.5 / -0.42, step 60 +6.8 / -0.60 (50.5 / 50.4 /
53.8% saved). Tool calls shortened as intended; math and code did not hold in tokens (code 954 -> 723 thinking tokens) but held
in accuracy; **IFEval's saving did not move**, for two reasons found in the pair file and the records:

1. **rpK's IFEval pairs rewarded writing more.** Over its 320 IFEval pairs the chosen answer thinks 46 tokens less but its
   response is 42 tokens LONGER (longer in 76% of pairs): the counting-rule pairs (half of IFEval, on every counting rule, with
   no cap on the chosen response) and the response guard. Net: -4 tokens. Path 1's round 1 had the same flaw (76%, +47 tokens).
2. **Runaway answers.** Answers that loop to the token cap (always wrong) hold 26% of rpE-80's IFEval tokens on the full test
   (2.4% of answers; the original 3.6% = 21%; code: 8% for rpE-80, 30% for the original). In rpE-80 they are loops in the
   RESPONSE (repeated quotes, placeholders, one word), on 51 questions of which 49 also have finished answers. Our pair builder
   only ever used FINISHED answers as the rejected side, so nothing said "do not loop"; on Dev-512, rpK's finished responses did
   not lengthen (median 81 tokens) while its runaways rose from 3.6% to 4.6% of answers.

**rpL** (queued on GPU 2 after rpK step 20's full test; same start, answers, tool-call pairs and training as rpK; `--mix-preset
typed2`):

| type | pairs | "think less" | "be right" | counting rule | anti-loop | "don't stop early" |
|---|---|---|---|---|---|---|
| IFEval | 320 | 160 | 48 | 64 | 48 | 0 |
| one function | 160 | 112 | 48 | | | |
| several functions | 120 | 84 | 36 | | | |
| no function fits | 140 | 84 | 56 | | | |
| math | 160 | 0 | 112 | | 16 | 32 |
| code | 160 | 0 | 114 | | 14 | 32 |

- Counting-rule pairs only on rules that ask for a MINIMUM amount of text (at least N words / sentences / keyword or letter
  occurrences, placeholders, highlighted sections), 20% of IFEval instead of 50%, the chosen response at most 1.35x the
  rejected one's + 25 tokens.
- "Think less" pairs on every other IFEval question: the chosen response is at least 0.85x the question's median right
  response and at most 1.05x the rejected one's (thinking shorter, response neither clipped nor longer).
- "Be right" pairs on IFEval: the chosen response at most 1.10x the rejected one's + 15 tokens.
- **Anti-loop pairs:** a finished right answer beats a runaway answer of the same question (IFEval 48, math 16, code 14).
  Overlap flagged to the user (michal-bida's rejected side also includes cut-off answers); approved as a generic pair.
- Built on rpE-80's real answers: IFEval pairs without the anti-loop ones: thinking 105 vs 166, response 110 vs 113 (chosen longer
  in 35%), total 214 vs 279; anti-loop pairs: 301 vs 4,014 tokens. Longest pair 8,380 tokens (cap 8,704).
- Reading: Dev-512 at steps 20 / 40 / 60 paired against rpE-80, now with the share of runaway answers per type. consult9's pass
  mark: saved >= 52.5% and accuracy >= +1.25 on Dev-512 with IFEval responses <= 245 tokens (its forecast: 53-55% / -0.4 to -0.9
  on the full test; its last forecast was 7-9 points too high).

## Full tests of rpK, rpL's Dev-512 results, and rpM (11 Nov afternoon, after consult10)

| model | full test | per type (saved / accuracy): math, code, IFEval, one function, several, no fit |
|---|---|---|
| rpE-80 (start) | 47.0 / -0.9 | 54 / -1.2, 46 / -1.0, 45 / -2.6, 37 / +0.1, 39 / +0.1, 50 / +3.4 |
| rpK step 20 | 49.9 / -1.3 | 58 / -2.6, 51 / -0.3, 45 / -3.1, 40 / +0.3, 42 / -1.5, 54 / +3.6 |
| rpK step 60 | **54.4 / -1.4** | 61 / -2.3, 55 / -1.2, 49 / -3.0, 45 / -1.5, 48 / -1.0, 62 / +6.2 |

rpK step 60 is the highest point above our curve of any full test (+2.38). It misses the box on accuracy only; IFEval (-3.0) and
math (-2.3) hold most of the loss.

rpL on Dev-512 (paired vs rpE-80): step 20 +3.4 saved / -0.07, step 40 +6.9 / -0.93, step 60 +11.9 / -0.44 (58.9% saved). IFEval's
saving rose to 57% and loops fell, but the 14 true "at least N words / sentences" questions fell from 37.5% right (rpE-80) to
19.6%: my "at least N" rule let placeholders / highlighted sections / letter frequency take two thirds of the 64 counting pairs,
the 1.35x cap rejected valid long answers, and the "shorter" pairs still clipped the response. Full tests of rpL steps 60 and 40
are queued.

Where rpK step 20 lost math (full-test records): its new wrong answers think 0.5-1.0x as long as rpE-80's median right answer
on the question (+91 answers; under 0.5x only +24): a calculation that skips its re-check, on the longer questions. Our guard
only caught wrong answers under 0.5x, and the math "be right" pair took the shortest right answer.

**rpM** (`--mix-preset typed3`; consult10's design, checked on rpE-80's real answers; 1,140 pairs planned, 60 steps, lr 1e-5):

| type | pairs | "think less" | "be right" | counting rule | anti-loop | "don't stop early" |
|---|---|---|---|---|---|---|
| math, SVAMP-like (rpE-80 answers 1,200 new questions x8) | 180 | 0 | 90 | | 0 | 90 |
| math, hard GSM8K (the answers rpK and rpL used) | 60 | 0 | 30 | | 0 | 30 |
| code | 160 | 0 | 114 | | 14 | 32 |
| IFEval | 320 | 144 | 72 | 80 | 24 | 0 |
| one function | 160 | 112 | 48 | | | |
| several functions | 120 | 60 | 60 | | | |
| no function fits | 140 | 84 | 56 | | | |

- **Math "don't stop early", widened:** the rejected wrong answer thinks under 0.95x the question's median right thinking (the
  one nearest 0.75x), the chosen right answer 0.95-1.25x (on hard math: 700 vs 487 tokens, chosen longer in 100%).
- **Math "be right", length-matched:** the right / wrong pair of the question whose thinking lengths are closest (right 0.95-1.25x
  the wrong): no "be shorter" signal (on hard math: 612 vs 585 tokens).
- **IFEval counting pairs** only on true length rules (at least N words / sentences, a keyword at least N times, N paragraphs), the
  chosen response 1.10x + 15 to 1.75x + 60 tokens of the rejected one's (on the real answers: 314 vs 217 tokens, always longer; 44
  IFEval pairs have a chosen response of 300+ tokens, against 14 in rpL).
- **IFEval "think less"** only on questions with no counting rule, the response held within 5% (thinking 99 vs 185, response 82 vs 82).
- **IFEval anti-loop** pairs only for loops in the response, 24 instead of 48.
- **Several functions** 50 / 50 (rpK lost 1.5 points there at 70 / 30).
- **New dev slice, Dev-SVAMP-200:** 200 SVAMP-like math questions never used in training; rpE-80, rpK-60 and rpL-60 are scored
  on it first (their full tests exist or are running), then rpM at steps 20 / 40 / 60 beside Dev-512, both paired against rpE-80.
- consult10's pass mark: Dev-512 saving >= +6 points over rpE-80; on Dev-SVAMP-200 math accuracy >= -0.3 and math thinking within
  -4% / +5% of rpE-80; IFEval "at least N" accuracy not below rpE-80's. Its forecast for the full test: 53.5-55.0% / -0.55 to
  -0.85 (its forecasts have read about half a point too kind on accuracy).
