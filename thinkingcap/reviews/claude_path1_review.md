# Independent review of PATH1_PLAN.md (a Claude review agent, 11 Nov 2026) — summary

**Verdict:** as written, Path 1 is unlikely to reach > 50% / better than -1. Biggest weakness: round 1 is a correctness-heavy,
moderate-dose DPO from the original model; the log's closest precedents land near 30% saved, not 45-50%: lpc-runa2 (original's own
answers, per-type shorten/guard/right mix, 4,442 pairs, 275 steps, 3 seeds) plateaued at 31-33% (L3553-3617; runb / runac 23-30%);
D6 simright (similar-length right>wrong pairs) halved the shortening (34 -> 28%, L4566-4574); fp32 alone cost ~11 points of saving at
equal pairs (bidax6p, L4662-4668). Round 1's preset has only ~39% shorten pairs. Expected round 1: ~25-35% saved. Round 2 must then
add saving AND accuracy, which nothing in the log has done: od1 (rpE-80 + online half guard / half shorten rounds) went 50.4/-0.2 ->
53.3/-1.8 on the 720, its distance above the curve FELL (+2.8 -> +1.75). The goal sits ~1.9 pts above our curve at 50%; our best
full tests are +1.45 to +1.64. Grounded estimate with the bugs fixed: ~35-45% saved at -0.5 to -1.5 (on-trend, not a breakthrough).

**rpJ does not test what Path 1 needs:** (1) pair-ordering bug: kinds sorted by name, so all shorten pairs came last (0 of 384 seen
by step 40) — FIXED 11 Nov (eb712bc: kinds interleaved by share); (2) 1,280 pairs vs 60 x 16 = 960 seen; (3) differs from rpE-80 in
five things at once; (4) wrong regime (rc1 is at 57% saved; the real round-1 model will be ~30-35%); (5) the decision rule needs a
paired model-vs-model interval that dev_score.py does not compute; Dev-512 noise ~ +-1.6 acc / +-2.7 saved. Proposal: drop rpJ, run
round 1 seed 0 directly (stored answers, no generation), Dev at 50/100/150/200, pre-registered gate: continue only if round 1 is
>= ~55% saved at or above the curve on calibrated Dev; redesign round 1 to be shorten-dominant (>= 60-65% shorten, with math /
no-fit protection), i.e. the rc1 -> rpE structure collapsed into two rounds.

**Other code issues:** seeds 0/1 are not real seeds (same LoRA init: sft.py --init-only has no --seed; same pair order; same draws
and vLLM seed); the round1 IFEval preset does not enforce 400 counting + 400 other; capped guard used whole-answer tokens and the
shortest early wrong (FIXED: thinking length, longest early wrong); silent backfill of short kinds (should fail above ~5 pts off
share); devset replays old draws instead of reading the data/repair-*.jsonl files (verify intersection); exclusion by id misses text
duplicates (gsm8k-std vs pool GSM8K ids; ~10 Dev math questions may be in round 1) — dedupe by 8-grams for every part; dpo.py's
--bida-precision and --max-len 8704 are labelled as michal-bida's settings: the user should decide whether they are generic enough.
Round-1 math supply: without MetaMath, ~661 easy GSM8K pool questions -> 60% = ~400 questions, almost all always-solved: the 800 math
pairs cannot fill properly; have the original answer ~2,000 gsm8k-std questions x8 (~3-4 GPU-h). Round 2: ~42,800 generations per
seed for 1,280 pairs (3% yield): measure the yield first, use ~2,500-3,000 questions, or one generation from the round-1 soup.

**Dev-512:** fine for coarse ranking, not for certifying "better than -1": GSM8K math is harder than SVAMP; IFEval fixed at 50%
counting rules (reweight to the test's share); hard top-ups over-represented; Path-1 models train on the same sources. Calibrate on
5-7 models with known full tests (ca1-soup-10, rc1-rl-40, rpE-40/60/80, rpI-120, rc1-soup-30), not 2. Add a paired
candidate-vs-reference bootstrap. Budget two full tests at the end.
