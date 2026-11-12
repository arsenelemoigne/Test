"""Run the MVT-cut and fading-worked-examples track end to end, one idempotent stage at a time.

Every stage skips work whose output already exists, so a cell can be rerun after a disconnect.
Run from the ThinkingCap-Test root, with this repo's scripts copied next to evaluate.py:

    python track.py setup                 # training data, benchmark questions, 240-question subset
    python track.py base                  # the raw model on the subset (the reference)
    python track.py rollouts              # the model's own replies to 450 training questions
    python track.py mvt --travel 256      # experiment 1: MVT cut -> SFT -> evaluate
    python track.py report                # where MVT cuts vs the settle rule, for many travel costs
    python track.py fade --rounds 2       # experiment 2: fading paragraphs + words, stops past -3 points
    python track.py fade --rounds 4 --word-scorer marginal   # experiment 3: words scored by our own model
    python track.py scores                # every evaluated run against the base
"""

import argparse
from collections import Counter, defaultdict
import filecmp
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("runs/mvtf")
QUESTIONS = "questions_40"
BASE_MODEL_ID = "Qwen/Qwen3-0.6B"
BASE = ROOT / "eval-base"
ROLLOUTS = ROOT / "probe" / "rollouts.jsonl"
LIMIT = -3.0  # accuracy points against the raw model


def claim_shared():
    """`track.py shared STAGE ...`: a line both runtimes' queues hold. The first runtime to reach it claims it on
    Drive (claims/), restores what the other runtime saved, and runs it; the other skips it. Handled before the
    argument parser, so a worker started before this existed still runs it (each line is a fresh track.py)."""
    if len(sys.argv) < 2 or sys.argv[1] != "shared":
        return
    del sys.argv[1]
    me = os.environ.get("TRACK_SESSION") or "runtime1"
    if Path("/content/drive/MyDrive").exists():
        claims = DRIVE / "claims"
        claims.mkdir(parents=True, exist_ok=True)
        key = claims / ("_".join(sys.argv[1:4]).replace("/", "-") + ".txt")
        if key.exists() and key.read_text().strip() != me:
            print(f"shared: '{' '.join(sys.argv[1:])}' already taken by {key.read_text().strip()}; skipped", flush=True)
            sys.exit(0)
        key.write_text(me)
        print(f"shared: '{' '.join(sys.argv[1:])}' claimed by {me}", flush=True)
        drive_sync("restore")  # the other runtime's models and runs, if this line needs them


def main():
    claim_shared()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("stage", choices=("setup", "base", "rollouts", "mvt", "report", "fade", "plan", "recheck", "mix", "full",
                                          "grid_data", "grid", "gridtable", "combine", "merge", "fullbest", "stats", "bon", "bon_dpo", "code_round", "chain", "repair", "rlcal", "calcheck", "caltable", "callot", "cutexamples", "probezoo", "targeted", "decide", "zooreport", "wronglen", "benchlot", "drpo", "teach", "nothink", "mathmatch", "balround", "stopgold", "mathfix", "budgetprompt", "promptsearch", "buildpool", "buildpool2", "poolfix", "poolmore", "lupinanswers", "lupin", "topiccheck", "refsets", "codecut", "crosscheck", "fullsoup", "fullmodel", "breakdown", "struggle", "copddemo", "ifevalrules", "copdanalysis", "screenset", "rescreen", "dial", "pertype", "tooltopup", "poolup", "newtop", "onpolicy", "typeonly", "polans", "ptrepair", "odpo", "ladder", "devset", "devscore", "devsvamp", "transfer", "p1src", "hold", "mixscreen", "thinkprune", "chirurgical", "astra", "astradata", "consultpack", "wanda", "wandaeval", "tokenskip", "dropmodels", "drivereport", "drivetidy", "probeknow", "audit", "balanced", "full2", "clean", "confirm", "fresh", "peek", "s1log", "end", "diag", "steer_read", "steer", "dpo_round", "scores"))
    parser.add_argument("--travel", type=float, default=256)
    parser.add_argument("--bd-names", nargs="*", default=[], help="breakdown: full-test runs to break down (runs/mvtf/full-NAME)")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--order", default="gain")
    parser.add_argument("--floor-travel", type=float)
    parser.add_argument("--word-step", type=float, default=0.1, help="fading: words compacted per round; 0 = paragraphs only")
    parser.add_argument("--tag", default="", help="suffix for the fading run's directories")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--adapter", help="recheck: the adapter to score; none scores the base")
    parser.add_argument("--ignore-limit", action="store_true", help="fade: keep going past the -3-point limit")
    parser.add_argument("--mc", help="mix: sft.jsonl giving the non-math rows")
    parser.add_argument("--math", help="mix: sft.jsonl giving the GSM8K rows")
    parser.add_argument("--name", help="mix: run name")
    parser.add_argument("--type", choices=("math", "code", "tool", "text", "choice"), help="grid: the kind of data trained on")
    parser.add_argument("--method", choices=("whole", "mvt", "marginal", "fade"), help="grid: how its traces are shortened")
    parser.add_argument("--unit", default="paragraph", help="grid: paragraph, sentence or tokensN")
    parser.add_argument("--per-source", type=int, default=250, help="grid_data: questions per kind")
    parser.add_argument("--start", help="dpo_round: the adapter the round starts from (policy and reference)")
    parser.add_argument("--seed", type=int, default=1, help="dpo_round: question draw (bon used seed 0)")
    parser.add_argument("--steps", type=int, default=150, help="dpo_round: DPO steps; checkpoints every 25")
    parser.add_argument("--inputs", nargs="*", default=[],
                        help="combine: sft.jsonl[:source[:source...]] files whose rows (of those sources) are pooled")
    parser.add_argument("--margin", type=float, default=-2.5,
                        help="fullbest: lowest 4-sample accuracy change (points) a candidate may have")
    parser.add_argument("--plan-file", default="plan.txt", help="plan: the queue this runtime works through")
    parser.add_argument("--wait-hours", type=float, default=0,
                        help="plan: when the queue is empty, keep polling the branch this long; a running cell "
                             "also keeps Colab from disconnecting an idle runtime and wiping its disk")
    parser.add_argument("--samples", type=int, default=4, help="recheck: samples per question")
    parser.add_argument("--no-shorten", default="", help="bon_dpo: sources with guard pairs only")
    parser.add_argument("--eval-samples", type=int, default=2, help="bon_dpo: samples per question in the sweep")
    parser.add_argument("--drop-sources", default="", help="bon_dpo: sources with no pairs at all; probezoo: sources left out")
    parser.add_argument("--dpo-epochs", type=int, default=2, help="bon_dpo: passes over the pairs")
    parser.add_argument("--eval-steps", type=int, nargs="*", default=[], help="bon_dpo: checkpoints scored (default all)")
    parser.add_argument("--kod-file", default="data/kodcode2.jsonl", help="chain: KodCode task file")
    parser.add_argument("--kod-exclude-runs", nargs="*", default=[], help="chain: runs whose KodCode tasks are skipped")
    parser.add_argument("--mvt-sources", default="", help="chain: sources that also get verified MVT short-beats-long pairs")
    parser.add_argument("--floor", type=float, default=-1.5, help="chain: lowest 4-sample accuracy change for the chosen soup")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2], help="balanced: DPO seeds averaged")
    parser.add_argument("--base-only", action="store_true", help="full2: score the raw model only")
    parser.add_argument("--code-source", default="none", help="clean: none or kodcode")
    parser.add_argument("--kod-questions", type=int, default=300, help="clean: KodCode tasks answered per round")
    parser.add_argument("--kod-exclude", default="", help="clean: a run whose KodCode tasks are not reused")
    parser.add_argument("--code-data", nargs="*", default=[], help="code_round: data files whose code questions are answered again")
    parser.add_argument("--dataset", default="mbpp", help="diag: the benchmark to look at")
    parser.add_argument("--data-seed", type=int, default=0, help="bon_dpo round: sample a fresh question draw with this seed")
    parser.add_argument("--dpo-seed", type=int, default=0, help="bon_dpo: DPO's seed (pair order, held-out split)")
    parser.add_argument("--mix", default="0.6,0.4,0", help="bon_dpo: shares of shorten, guard, hard pairs")
    parser.add_argument("--probe-from", help="bon_dpo round: reuse the answers sampled by this round name")
    parser.add_argument("--fresh-questions", action="store_true", help="bon_dpo round: questions not sampled before")
    parser.add_argument("--rollouts-from", help="bon_dpo: a new round sampled from this adapter, or auto (best checkpoint)")
    parser.add_argument("--save-every", type=int, default=25, help="bon_dpo: DPO checkpoint interval (steps)")
    parser.add_argument("--gen-questions", type=int, default=0, help="bon: stop sampling at this many questions (0: all)")
    parser.add_argument("--skip-eval", action="store_true", help="bon: train only, no test")
    parser.add_argument("--sft-lr", type=float, default=5e-5, help="bon: SFT learning rate")
    parser.add_argument("--sft-epochs", type=int, default=1, help="bon: SFT passes")
    parser.add_argument("--min-saved", type=float, default=0.0,
                        help="bon: keep questions whose shortest correct reply is this share under their mean")
    parser.add_argument("--max-per-source", type=int, default=70, help="mvt: SFT rows per dataset")
    parser.add_argument("--min-steps", type=int, default=1, help="mvt: never cut below this many paragraphs")
    parser.add_argument("--repeat", default="", help="mvt: dataset:times for SFT, e.g. gsm8k:2")
    parser.add_argument("--full-sources", default="", help="mvt: datasets whose traces stay whole, e.g. gsm8k")
    parser.add_argument("--rl-steps", type=int, default=120, help="rlcal: RL steps (32 questions x 8 replies each)")
    parser.add_argument("--rl-alpha", type=float, default=0.15, help="rlcal: length penalty weight (Arora & Zanette)")
    parser.add_argument("--rl-lr", type=float, default=1e-5, help="rlcal: RL learning rate (LoRA)")
    parser.add_argument("--rl-kl", type=float, default=1e-3, help="rlcal: KL to the round's start model")
    parser.add_argument("--rl-only", action="store_true", help="rlcal: no calibrated-cut DPO; RL straight from --start")
    parser.add_argument("--rl-fresh-seed", type=int, default=0,
                        help="rlcal: RL on a fresh, contamination-checked draw (this seed) instead of the round's questions")
    parser.add_argument("--rl-fresh-per-source", type=int, default=900, help="rlcal: questions per source in that draw")
    parser.add_argument("--judge", choices=["choose", "fresh"], default="choose",
                        help="rlcal: choose = pick on the choosing set x4, report the pick x8; fresh = score every soup and "
                             "every RL block straight on the fresh set x8 (no choosing set), keep the best fresh one")
    parser.add_argument("--dpo-steps", type=int, nargs="*", default=[10, 20, 30], help="rlcal: soup steps scored")
    parser.add_argument("--calcheck-model", help="calcheck: the model whose calibration is re-measured")
    parser.add_argument("--calcheck-from", default="rc1", help="calcheck: the run whose replies and cuts are reused")
    parser.add_argument("--probe-cal", help="probeknow: the cal_cuts folder under runs/bon whose labelled cuts are probed")
    parser.add_argument("--probe-model", help="probeknow: the model whose hidden states are read")
    parser.add_argument("--lot", default="ifeval_like=700,gsm_ic=300,gsm_plus=300,toolace=400,mc=100,kodcode=250",
                        help="callot: questions drawn per type (before the right-answer and contamination filters)")
    parser.add_argument("--paragraphs", action="store_true",
                        help="callot: cut at every paragraph end, right AND wrong replies (label_cuts.py), instead of cal_cuts")
    parser.add_argument("--decide-margin", type=int, default=10, help="decide: points of extra saving that count as 'helps'")
    parser.add_argument("--tg-save-every", type=int, default=10, help="targeted: DPO checkpoint interval")
    parser.add_argument("--tg-rule", default="strict", choices=("strict", "question"),
                        help="targeted: safe cut = >= 95%% right (strict) or as right as the question's own rate + margin")
    parser.add_argument("--drpo-lambda", type=float, default=0.2, help="drpo: length weight (smaller: shorter)")
    parser.add_argument("--drpo-solve-scale", action="store_true",
                        help="drpo: length pressure divided by each question's solve rate (ALP; LUPIN1.0)")
    parser.add_argument("--rl-dynamic", type=float, default=0.0,
                        help="drpo: DLER dynamic sampling factor (draw this many times the questions, keep those with a right reply)")
    parser.add_argument("--rl-variant", choices=("drpo", "shorterbetter"), default="drpo",
                        help="drpo stage: DRPO objective, or GRPO with the ShorterBetter reward (alpha*correct - beta*|len-SOL|)")
    parser.add_argument("--drpo-steps", type=int, default=60, help="drpo: RL steps (--drpo-questions x 8 replies each)")
    parser.add_argument("--drpo-questions", type=int, default=48, help="drpo: questions per step, each used once")
    parser.add_argument("--drpo-score", type=int, nargs="*", default=[5, 10, 15, 20, 25, 30, 40, 50, 60], help="drpo: checkpoints scored on fresh x8 (~7 min each)")
    parser.add_argument("--match-size", type=int, default=1500, help="mathmatch: matched GSM8K questions (train + test)")
    parser.add_argument("--match-test", type=int, default=300, help="mathmatch: of which held out as a math test set")
    parser.add_argument("--nothink-models", nargs="*", default=["Qwen/Qwen3-0.6B", "models/rc1-rl-40"],
                        help="nothink: models scored on fresh x8 with thinking switched off (empty think block)")
    parser.add_argument("--nt-full", action="store_true",
                        help="nothink: the FULL test x8 (2,759 questions) instead of the fresh 240, against full-base")
    parser.add_argument("--drpo-score-only", action="store_true",
                        help="drpo: never train; only score the saved checkpoints (e.g. a run stopped early)")
    parser.add_argument("--rl-protect", default="", help="drpo: sources with no length pressure (rl_train --protect-sources)")
    parser.add_argument("--rl-sb-beta", type=float, help="drpo --rl-variant shorterbetter: per-token length weight")
    parser.add_argument("--rl-guard", type=float, default=0.0, help="drpo: per-type accuracy guard (rl_train --guard-drop)")
    parser.add_argument("--drpo-weighted", nargs="*", default=[],
                        help="drpo: STEP:WEIGHT checkpoints merged at that strength and scored (0.5 = halfway back to --start)")
    parser.add_argument("--drpo-soup", type=int, nargs="*", default=[],
                        help="drpo: also average these checkpoints' adapters (equal weights) into one model and score it")
    parser.add_argument("--bal-questions", type=int, default=6000, help="balround: fresh benchmark-matched questions")
    parser.add_argument("--bal-samples", type=int, default=4, help="balround: answers per question")
    parser.add_argument("--bal-mix", default="gsm8k_all=0.25,ifeval_like=0.25,toolace=0.17,hermes_tools=0.08",
                        help="balround: pool shares per source (KodCode takes --bal-kod)")
    parser.add_argument("--bal-kod", type=float, default=0.25, help="balround: KodCode share of the pool")
    parser.add_argument("--bal-protect", default="", help="balround: sources or source/category given only right>wrong pairs")
    parser.add_argument("--teach-questions", type=int, default=3000, help="teach: fresh questions answered once each")
    parser.add_argument("--teach-variants", nargs="*", default=["rel:0.05", "rel:0.1"],
                        help="teach: stop targets, one model each (fixed:T = yes/no judge above T; rel:M = soft judge's "
                             "predicted success within M of the full thinking's)")
    parser.add_argument("--teach-own", help="teach: types taught by own confidence (teach_stop's default when unset)")
    parser.add_argument("--teach-judge", help="teach: types taught by the hidden-state judge ('' = none)")
    parser.add_argument("--teach-reuse", help="teach: reuse this earlier teach run's answers and own-confidence readings")
    parser.add_argument("--teach-lots", nargs="*", default=["plotB", "plotB2", "plotB3"],
                        help="teach: label lots of --start the judge is fitted on")
    parser.add_argument("--lot-samples", type=int, default=8, help="callot: full replies per question (20 = a reliable question rate)")
    parser.add_argument("--tg-shorten", default="kodcode,ifeval_like", help="targeted: types whose safe early cuts are taught")
    parser.add_argument("--tg-long", default="gsm_ic,gsm_plus", help="targeted: types taught to think longer (math)")
    parser.add_argument("--lot-cuts", type=int, default=10, help="callot --paragraphs: paragraph-end cuts per reply")
    parser.add_argument("--lot-checks", type=int, default=20, help="callot --paragraphs: forced answers per cut")
    parser.add_argument("--tg-min-samples", type=int, default=20, help="targeted: full replies a question needs for --tg-rule question")
    parser.add_argument("--tg-shorten-if-svamp", default="none",
                        help="targeted: types given stop pairs only if their lots' curve builds up like SVAMP's (math)")
    parser.add_argument("--pool-continues", nargs="*", default=[],
                        help="drpo: the RL runs this one continues: earlier draws may be reused, except these runs' pools")
    parser.add_argument("--pool-math", default="gsm_ic,gsm_plus", help="drpo/teach: math sources of the RL pool (34%% split)")
    parser.add_argument("--tg-drop-kinds", default="", help="targeted: pair kinds left out, e.g. guard")
    parser.add_argument("--tg-brake-share", type=float, default=1.0,
                        help="targeted: most share of 'keep thinking' (hard) pairs within each type")
    parser.add_argument("--tg-right", default="none", help="targeted: types given 'be right' pairs (own right vs own wrong, similar length)")
    parser.add_argument("--tg-quota", default="", help="targeted: most pairs kept per type, e.g. kodcode=700,toolace=400")
    parser.add_argument("--bp-budgets", default="svamp=400,mbpp=750,ifeval=200,bfcl=70",
                        help="budgetprompt: thinking budget per benchmark stated in the system prompt (measurement only)")
    parser.add_argument("--ps-lot", default="plotV2", help="promptsearch: label lot whose training questions are used")
    parser.add_argument("--ps-per-source", type=int, default=50, help="promptsearch: questions per type")
    parser.add_argument("--ps-samples", type=int, default=4, help="promptsearch: replies per question and prompt")
    parser.add_argument("--ps-only", default="", help="promptsearch: comma-separated candidates (default: all)")
    parser.add_argument("--dp-per-benchmark", type=int, default=3000, help="buildpool: training questions per benchmark")
    parser.add_argument("--dp-math-mix", default="gsm8k_matched=0.7,metamath_sv=0.3", help="buildpool: SVAMP-side sources")
    parser.add_argument("--dp-check-per-part", type=int, default=200, help="buildpool: pool questions per part answered x4")
    parser.add_argument("--dp-step", default="a", choices=("a", "b", "select"), help="buildpool2 / poolfix: answer half a / b, or select")
    parser.add_argument("--dp-oversample", type=float, default=1.6, help="buildpool2: candidates per kept question")
    parser.add_argument("--la-pool", default="dp6", help="lupinanswers: the checked pool to answer")
    parser.add_argument("--la-target", type=int, default=0, help="lupinanswers: top up to this many answers per question (0: --la-samples new each)")
    parser.add_argument("--la-force", default="", help="lupinanswers: open the gate anyway, with this reason (logged)")
    parser.add_argument("--la-samples", type=int, default=6, help="lupinanswers: new answers per question (4 reused)")
    parser.add_argument("--dp-reuse", default="dp3", help="poolfix: the answered pool whose other parts are reused")
    parser.add_argument("--dp-parts", default="svamp,bfcl_irrelevance", help="poolfix: parts given new candidates")
    parser.add_argument("--dp-cap", default="", help="poolfix: most new candidates per part, e.g. mbpp=1200")
    parser.add_argument("--dp-sources", default="gsm8k_easy,gsm8k_easy_test,metamath_easy,gsm8k_matched,toolace",
                        help="poolfix: prepare_types.py sources of the new candidates (dp5: ifeval_built)")
    parser.add_argument("--dp-ifeval", type=int, default=12000, help="buildpool2: IFEval-like prompts drawn to IFEval's mix")
    parser.add_argument("--dp-more-parts", default="bfcl_simple_python,bfcl_irrelevance,ifeval",
                        help="poolmore: parts whose candidates get --dp-more-samples more answers")
    parser.add_argument("--dp-more-samples", type=int, default=4, help="poolmore: extra answers per candidate (seed 5)")
    parser.add_argument("--dp-think-bins", type=int, default=4, help="poolmore: select_pool.py --think-bins")
    parser.add_argument("--dp-union-only", action="store_true", help="poolfix select: build the answered union only, no selection")
    parser.add_argument("--dp-more-extra", default="", help="poolmore: earlier poolmores whose answers are merged in, comma-separated (dp9: dp8; dp10: dp8,dp9)")
    parser.add_argument("--dp-more-target", type=int, default=0,
                        help="poolmore: answer only questions with fewer than this many answers, up to it (0: all, +samples)")
    parser.add_argument("--dp-chunk", type=int, default=150, help="poolmore: questions per generation chunk (dp10, code only: "
                        "600, so a few very long answers do not leave the GPU nearly idle at the end of every chunk)")
    parser.add_argument("--dp-more-cap", type=int, default=0, help="poolmore: most questions answered per half (0: all)")
    parser.add_argument("--dp-think-bins-part", default="", help="poolmore select: select_pool.py --think-bins-part")
    parser.add_argument("--dp-if-poor", default="", help="poolmore: run only if this pool's check leaves code off (dp11: dp10)")
    parser.add_argument("--cc-base", default="lpb-l2s50e", help="codecut: the screened recipe whose matched pairs it builds on")
    parser.add_argument("--cc-questions", type=int, default=600, help="codecut: code questions labelled")
    parser.add_argument("--cc-dose", type=float, default=0.2, help="codecut: share of the code pairs swapped for cut pairs")
    parser.add_argument("--cx-names", nargs="*", default=[], help="crosscheck: confirmed runs (their --seeds are averaged)")
    parser.add_argument("--cx-steps", type=int, nargs="*", default=[], help="crosscheck: soup steps")
    parser.add_argument("--cx-scale", type=float, nargs="*", default=[1.0], help="rescreen: merge strengths (the dial)")
    parser.add_argument("--rs-model", default="", help="refsets: the model answering (default: the original Qwen)")
    parser.add_argument("--rs-tag", default="base", help="refsets: run name tag (runs/mvtf/v4-<tag>-<seed>)")
    parser.add_argument("--lp-pool", default="auto", help="lupin: the checked pool (auto: dp11 if dp10 was superseded, else dp10)")
    parser.add_argument("--lp-recipe", default="long2short", help="lupin: lupin_pairs.py recipe")
    parser.add_argument("--lp-own-answers", default="", help="lupin round 2 (4 Nov, D12): pairs from a trained model's OWN "
                        "answers (polans NAME: runs/bon/onp-NAME-a/b) instead of the original's; use with --lp-init")
    parser.add_argument("--lp-min-solve", type=float, default=0.5, help="lupin: difficulty protection (0: none)")
    parser.add_argument("--lp-extra", default="", help="lupin: more lupin_pairs.py flags, e.g. '--add-right'")
    parser.add_argument("--lp-answers", type=int, default=10, help="lupin: answers per question used (newest first)")
    parser.add_argument("--lp-easy-ratio", type=float, default=0.0, help="lupin: lupin_pairs.py --easy-ratio")
    parser.add_argument("--lp-shorten-cap", type=float, default=1.0, help="lupin: lupin_pairs.py --shorten-cap")
    parser.add_argument("--lp-min-solve-part", default="", help="lupin: lupin_pairs.py --min-solve-part, e.g. svamp=0.8")
    parser.add_argument("--lp-right-parts", default="", help="lupin: lupin_pairs.py --right-parts, e.g. svamp")
    parser.add_argument("--dial-weights", type=float, nargs="*", default=[0.5, 0.7], help="dial: strengths of the trained change")
    parser.add_argument("--lp-answers-pool", default="", help="lupin: whose 10-answer files to use (dp12: dp10)")
    parser.add_argument("--lp-gate", action="store_true", help="lupin: refuse unless the pool's check is ALL OK or near misses")
    parser.add_argument("--lp-extra-q", default="", help="lupin: add tooltopup <name>'s extra tool questions to the pairs")
    parser.add_argument("--wd-sparsity", type=float, nargs="*", default=[0.1, 0.2, 0.3, 0.4, 0.5], help="wanda: levels")
    parser.add_argument("--pu-drop-hard", default="", help="poolup: part:source:min_solve, drop questions solved below it")
    parser.add_argument("--pu-drop", default="", help="poolup: part:source pairs left out, e.g. bfcl_simple_python:toolace")
    parser.add_argument("--pu-extras", default="nofit1,tools1,code1", help="poolup: top-up sets added to the candidates")
    parser.add_argument("--nt-sources", default="glaive_nofit_hard", help="newtop: prepare_types.py sources")
    parser.add_argument("--nt-parts", default="bfcl_irrelevance", help="newtop: build_pool.py parts")
    parser.add_argument("--nt-raw", action="store_true", help="newtop: no prompt-length matching (build_pool --no-match)")
    parser.add_argument("--nt-samples", type=int, default=10, help="newtop: answers per question by the original Qwen")
    parser.add_argument("--tt-parts", default="", help="tooltopup: BFCL parts (default all three)")
    parser.add_argument("--tt-hard", action="store_true", help="tooltopup: hardest candidates first")
    parser.add_argument("--tt-questions", type=int, default=1500, help="tooltopup: extra tool questions")
    parser.add_argument("--lp-part-share", default="", help="lupin: lupin_select.py --part-share (P1)")
    parser.add_argument("--lp-solve-tilt", default="", help="lupin: lupin_select.py --solve-tilt (P3)")
    parser.add_argument("--lp-think-tilt-bfcl", type=float, default=1.0, help="lupin: lupin_select.py --think-tilt-bfcl (P4)")
    parser.add_argument("--lp-init", default="", help="lupin: start adapter (two-stage), default a fresh LoRA")
    parser.add_argument("--lp-beta", type=float, default=0.0, help="lupin: dpo.py --beta (0: dpo.py default 0.1)")
    parser.add_argument("--lp-nll", type=float, default=None, help="lupin: dpo.py --nll (default 1.0)")
    parser.add_argument("--lp-lr", type=float, default=0.0, help="lupin: dpo.py --lr (0: default 2e-5)")
    parser.add_argument("--lp-epochs", type=int, default=1, help="lupin: dpo.py --epochs")
    parser.add_argument("--pt-screen", action="store_true", help="pertype: score on questions_screen720 x4")
    parser.add_argument("--pt-tag", default="", help="pertype: name of an extra merge of the same adapters (no retraining)")
    parser.add_argument("--pt-cap-like", default="", help="pertype: cut each group x kind to pairs-<this>'s counts (same steps)")
    parser.add_argument("--dt-apply", action="store_true", help="drivetidy: move the listed items to Drive's trash (default: list only)")
    parser.add_argument("--dt-keep-days", type=float, default=2, help="drivetidy: never touch anything changed in the last N days")
    parser.add_argument("--dt-show", type=int, default=40, help="drivetidy: items printed per category")
    parser.add_argument("--pt-dpo", default="", help="pertype: extra dpo.py flags, comma-joined (e.g. --bida-precision)")
    parser.add_argument("--pt-extend", action="store_true", help="pertype --pt-cap-like: train the capped pairs first, then all the remaining ones, in order, with a checkpoint at the capped run's last full step")
    parser.add_argument("--pt-step", type=int, default=0, help="pertype: score the adapters' step-N checkpoints instead of the finals")
    parser.add_argument("--pt-wait", type=int, default=0, help="pertype --pt-no-train: wait up to N hours for the adapters on Drive")
    parser.add_argument("--pt-save", type=int, default=25, help="pertype: save a checkpoint (adapters/<name>-<group>/step-N, kept on Drive) every N steps")
    parser.add_argument("--pt-check", action="store_true", help="pertype: print the (capped) pair counts per group and stop, no training")
    parser.add_argument("--pt-full", action="store_true", help="pertype: also run the official full test (2,759 x 8) of each merge")
    parser.add_argument("--pt-weights", default="svamp=0.5,mbpp=0.8,ifeval=1,bfcl=0.8", help="pertype: per-type merge strengths")
    parser.add_argument("--pt-no-train", action="store_true",
                        help="pertype (5 Nov): the adapters must already exist on Drive; fail instead of retraining (solo screens)")
    parser.add_argument("--lp-median-part", default="", help="lupin: lupin_pairs.py --median-shorten-part (Run B)")
    parser.add_argument("--to-parts", default=",".join(("svamp", "mbpp", "ifeval", "bfcl_simple_python", "bfcl_multiple", "bfcl_irrelevance")), help="typeonly: parts to train")
    parser.add_argument("--op-from", default="math1", help="onpolicy: the top-up whose questions are answered")
    parser.add_argument("--op-adapter", default="lpc-bidax4-s0", help="onpolicy: the trained adapter that answers")
    parser.add_argument("--op-band", default="470,905", help="onpolicy: original-Qwen mean thinking band, tokens")
    parser.add_argument("--op-samples", type=int, default=4, help="onpolicy: answers per question")
    parser.add_argument("--lp-band", default="", help="lupin: lupin_pairs.py --band, e.g. svamp:470:905")
    parser.add_argument("--tp-start", default="raw", help="thinkprune: 'raw' or an adapter name (merged first)")
    parser.add_argument("--tp-steps", type=int, default=300)
    parser.add_argument("--tp-questions", type=int, default=64)
    parser.add_argument("--tp-group", type=int, default=16)
    parser.add_argument("--tp-lr", type=float, default=5e-6)
    parser.add_argument("--tp-kl", type=float, default=1e-3)
    parser.add_argument("--tp-cap-math", type=float, default=0.95)
    parser.add_argument("--tp-cap-other", type=float, default=0.70)
    parser.add_argument("--tp-score-every", type=int, default=40)
    parser.add_argument("--tp-smoke", action="store_true")
    parser.add_argument("--tp-norm-means", action="store_true", help="thinkprune: gated B per part = the original's mean generated tokens (pool part labels)")
    parser.add_argument("--tp-check", action="store_true", help="thinkprune: print the selected questions per part vs the need, no training")
    parser.add_argument("--tp-answers10", action="store_true", help="thinkprune: select questions from the original's 10 answers, not 4")
    parser.add_argument("--tp-extra", default="nofit4,math1,math2", help="thinkprune: top-ups whose questions join the pool")
    parser.add_argument("--tp-rl", default="", help="thinkprune: extra rl_train.py arguments, comma-separated "
                        "(GRPO v2, 4 Nov: e.g. --fixed-denominator,--max-tokens-code,8192,--full-support,--grad-clip,0.5)")
    parser.add_argument("--rp-questions", default="svamp=2400,mbpp=900,bfcl_irrelevance=1500,ifeval=300,bfcl_simple_python=150,"
                        "bfcl_multiple=150", help="ptrepair (Track A): training questions per part answered by pt-mid x4")
    parser.add_argument("--rp-quota", default="svamp=640,mbpp=400,bfcl_irrelevance=240,ifeval=160,bfcl_simple_python=96,"
                        "bfcl_multiple=64", help="ptrepair: pairs per part (half guard / half similar-length right)")
    parser.add_argument("--rp-lr", type=float, default=5e-6, help="ptrepair: lr (2.5e-6 on step 1, then constant)")
    parser.add_argument("--rp-steps", type=int, default=40, help="ptrepair: training steps (16 pairs each)")
    parser.add_argument("--rp-screen", type=int, nargs="*", default=[5, 10, 20, 40], help="ptrepair: steps scored on the 720 x4")
    parser.add_argument("--mx-base", default="lpt6", help="mixscreen: per-type adapters adapters/<base>-<part>-s<seed>")
    parser.add_argument("--mx-weights", default="", help="mixscreen: part=strength,... (0 = left out)")
    parser.add_argument("--mx-seeds", type=int, nargs="*", default=[0, 1], help="mixscreen: seeds averaged per part")
    parser.add_argument("--rp-from", default="", help="ptrepair: reuse this run's pt-mid answers AND pairs (a new dose only)")
    parser.add_argument("--rp-smoke", action="store_true", help="ptrepair: 5 questions per part, 4 steps, no screen (GPU test)")
    parser.add_argument("--rp-base", default="", help="ptrepair (9 Nov): repair this MERGED model (e.g. models/ca1-soup-10) "
                        "instead of pt-mid: it answers the questions, is DPO's reference and the base of every screen")
    parser.add_argument("--rp-own-data", action="store_true",
                        help="ptrepair (9 Nov): questions from the user's own sources only (own_repair_questions): math = "
                             "GSM8K train+test at every difficulty in SVAMP wording; code / instructions / tools = pool-dp13 "
                             "rows minus MetaMath + the own top-ups nofit1, nofit2, tools1, code1; no michal-bida source")
    parser.add_argument("--rp-base-from", default="", help="ptrepair (10 Nov): BASE_MODEL:ADAPTER builds --rp-base when missing")
    parser.add_argument("--rp-samples", type=int, default=4, help="ptrepair (10 Nov): answers per question")
    parser.add_argument("--rp-seed", type=int, default=0, help="ptrepair (10 Nov): the own-question draw (another seed = new questions)")
    parser.add_argument("--rp-answers-model", default="", help="ptrepair (10 Nov): the merged model that answers (default: --rp-base)")
    parser.add_argument("--rp-init-from", default="", help="ptrepair (10 Nov): start DPO from this adapter (also DPO's reference)")
    parser.add_argument("--rp-epochs", type=int, default=1, help="ptrepair (10 Nov): passes over the pairs (steps past one "
                        "pass need 2; a finished run continues from its saved state)")
    parser.add_argument("--rp-pair-args", default="", help="ptrepair: extra repair_pairs.py flags, comma-separated (9 Nov: "
                        "--think-only,ifeval,--prefer-reread)")
    parser.add_argument("--hold-minutes", type=float, default=20, help="hold (11 Nov): minutes to wait")
    parser.add_argument("--hold-tag", default="", help="hold: any text (makes two hold lines differ, so both run)")
    parser.add_argument("--ds-name", default="", help="devscore (11 Nov): the name of the scored model's Dev-512 run")
    parser.add_argument("--ds-model", default="", help="devscore: a merged model path / Hub id, or MERGED_OUT=BASE:ADAPTER to build it")
    parser.add_argument("--rp-rollouts-from", default="", help="ptrepair (11 Nov): reuse this run's answers (runs/bon/rp-NAME), new pairs")
    parser.add_argument("--rp-public", action="store_true", help="ptrepair (11 Nov, user): draws may use the public datasets "
                        "(MetaMath, math1 / math2 / nofit4); math then SVAMP-like, not full GSM8K")
    parser.add_argument("--rp-gate", default="", help="ptrepair (11 Nov): RUN,REF,MARGIN: run only if Dev-512 saved of RUN >= "
                        "saved of REF + MARGIN (dev512-NAME/dev_score.json); a missing score = gate closed")
    parser.add_argument("--rp-warmup", type=int, default=2, help="ptrepair (11 Nov): DPO warm-up steps")
    parser.add_argument("--rp-lr-min", type=float, default=None, help="ptrepair (11 Nov): cosine floor (default = --rp-lr: constant)")
    parser.add_argument("--rp-dpo-seed", type=int, default=0, help="ptrepair (11 Nov): DPO seed (adapter init and pair order)")
    parser.add_argument("--rp-exclude-ids", default="", help="ptrepair (11 Nov): a JSON list of question ids never drawn (with --rp-exclude-dev)")
    parser.add_argument("--rp-dev", action="store_true", help="ptrepair (11 Nov): score the --rp-screen steps on Dev-512, not the 720")
    parser.add_argument("--rp-dev-svamp", action="store_true", help="ptrepair (11 Nov): with --rp-dev, also score each step on "
                        "Dev-SVAMP-200 (SVAMP-like math), paired against --rp-dev-vs")
    parser.add_argument("--rp-save-at", type=int, nargs="*", default=None,
                        help="ptrepair (12 Nov, user): save checkpoints only at these steps (the last step is always saved)")
    parser.add_argument("--rp-pairs-only", action="store_true",
                        help="ptrepair (12 Nov, user): stop after the answers and pairs; the same line without it trains on them")
    parser.add_argument("--rp-require-full", action="store_true",
                        help="ptrepair (12 Nov, user): never train unless the pairs file holds the whole --rp-quota (stop the stage instead)")
    parser.add_argument("--rp-drop", default="",
                        help="ptrepair (12 Nov, user): with --rp-rollouts-from, leave out stored answers: source:NAME or id:PREFIX, "
                        "list joined by commas (r2s1: source:metamath_sv,id:nofit4- = michal-bida's MetaMath SV/FOBAR and xLAM no-fit picks)")
    parser.add_argument("--rp-extra-source", default="",
                        help="ptrepair (12 Nov, recipe 2): PART=SOURCE,... the extra questions of these parts come from a prepare_types.py "
                        "source that is NOT in the pool (e.g. ifeval=ifeval_like: argilla ifeval-like-data drawn IFEval-matched, our own source; never ifeval_bida), "
                        "minus the pool's prompts, the test's rare 8-grams and Dev near-duplicates")
    parser.add_argument("--rp-extra-questions", default="", help="ptrepair (11 Nov, rpM): with --rp-rollouts-from, ALSO draw these "
                        "new own questions (part=n,..., public datasets, never a dev question) and let the base model answer "
                        "them; their ids get the prefix sv- (repair_pairs.py: SVAMP-like math 'svamp' vs the reused hard 'svamp_hard')")
    parser.add_argument("--tr-wait-base", action="store_true", help="transfer (12 Nov): do not generate the original's answers here; "
                        "answer with the listed models first, then wait for the other GPU's base answers on Drive (up to 14 h) and score")
    parser.add_argument("--tr-sets", default="", help="transfer (12 Nov): SOURCE=N,... instead of the default transfer sets "
                        "(e.g. tr_aime25=30); its own question file and output folders (transfer-<tag>-NAME)")
    parser.add_argument("--tr-max-tokens", type=int, default=4096, help="transfer: token cap for non-code questions (AIME: 32768)")
    parser.add_argument("--tr-source-caps", default="", help="transfer (12 Nov): per-source caps SOURCE=N,...; with --tr-tag the "
                        "math sources default to 32,768 (user: no cap on MATH-500 / Skywork; at 4,096 the original runs out of room on "
                        "more than half of them). Untagged runs keep 4,096 so they stay comparable with the stored original answers.")
    parser.add_argument("--tr-tag", default="", help="transfer: names the question file and output folders of a --tr-sets run")
    parser.add_argument("--tr-models", nargs="*", default=[], help="transfer (11 Nov): NAME=SPEC models to score (SPEC: merged model, "
                        "Hub id, or OUT=BASE:ADAPTER)")
    parser.add_argument("--ds-vs", default="", help="devscore: also the paired difference against this scored run")
    parser.add_argument("--ds-dev", default="", choices=("", "svamp200"), help="devscore: the dev set ('' = Dev-512)")
    parser.add_argument("--rp-dev-vs", default="", help="ptrepair (11 Nov): with --rp-dev, also the PAIRED difference against "
                        "this scored run (runs/bon/dev512-NAME), e.g. rpE80 or p1r1s0-200")
    parser.add_argument("--rp-exclude-dev", action="store_true", help="ptrepair / odpo (11 Nov): never draw the Dev/Confirm questions")
    parser.add_argument("--ld-questions", default="svamp=450,mbpp=300,ifeval=450,bfcl_simple_python=100,bfcl_multiple=100,"
                        "bfcl_irrelevance=100", help="ladder (10 Nov, user): own training questions per part")
    parser.add_argument("--ld-seed", type=int, default=11, help="ladder: the question draw's seed")
    parser.add_argument("--ld-samples", type=int, default=8, help="ladder: the original's natural answers per question")
    parser.add_argument("--ld-tolerance", type=int, default=1, help="ladder: answers a rung may lose vs the natural ones")
    parser.add_argument("--od-rounds", type=int, default=4, help="odpo (10 Nov, user): online DPO rounds")
    parser.add_argument("--od-steps", type=int, default=10, help="odpo: training steps per round (16 pairs each)")
    parser.add_argument("--od-init", default="", help="odpo: the adapter round 1 starts from (and its reference)")
    parser.add_argument("--od-answers0", default="", help="odpo: the merged model = --rp-base + --od-init (answers round 1)")
    parser.add_argument("--od-screen-rounds", type=int, nargs="*", default=[2, 4], help="odpo: rounds scored on the 720 x4")
    parser.add_argument("--rp-kinds", default="guard,right", choices=("guard,right", "guard", "guard,shorten", "mix"),
                        help="ptrepair: repair_pairs.py --kinds (guard = only shortest right > shortest finished wrong)")
    parser.add_argument("--tp-score-steps", type=int, nargs="*", default=[],
                        help="thinkprune (5 Nov, step sensitivity): score the 720 at exactly these steps (multiples of 10) "
                             "instead of every --tp-score-every")
    parser.add_argument("--tp-mixed-share", type=float, default=None, help="thinkprune: share of questions the original "
                        "solves 1-9 of 10 (the rest 10/10), in the order used (9 Nov: 0.5)")
    parser.add_argument("--tp-block", type=int, default=10, help="thinkprune: steps per block (a checkpoint at each)")
    parser.add_argument("--tp-persist", action="store_true",
                        help="thinkprune: one AdamW and one lr schedule across the 10-step blocks (--init-state, --total-steps)")
    parser.add_argument("--tp-include-unsolved", action="store_true",
                        help="thinkprune: keep questions at solve rate == --tp-min-solve (with 0: the never-solved ones too)")
    parser.add_argument("--tp-reward", default="cap", choices=("cap", "lenpen", "smart", "gated"), help="thinkprune: rl_train.py --reward")
    parser.add_argument("--tp-min-solve", type=float, default=0.5, help="thinkprune: questions the original gets right > this share")
    parser.add_argument("--tp-max-solve", type=float, default=1.0,
                        help="thinkprune: and right <= this share (below 1: drop questions it always gets right, no signal)")
    parser.add_argument("--tp-rank", type=int, default=32)
    parser.add_argument("--ch-start", default="raw", help="chirurgical: 'raw' or an adapter (merged first), e.g. v1 later")
    parser.add_argument("--ch-smoke", action="store_true", help="chirurgical: tiny sizes, every phase, no 720 screen")
    parser.add_argument("--ch-extra", default="", help="chirurgical: more chirurgical.py flags, comma-joined")
    parser.add_argument("--as-passes", type=int, default=10, help="astra: forward passes (the paper: 10 / 25 / 50)")
    parser.add_argument("--as-cap-usd", type=float, default=8.0, help="astra: HARD spending stop (user, 3 Nov: EUR 8)")
    parser.add_argument("--as-extra", default="", help="astra: more astra_harness.py flags, comma-joined")
    parser.add_argument("--as-holdout", type=int, default=512, help="astra: our own paired check of the final edit")
    parser.add_argument("--as-smoke", action="store_true", help="astra: the GPU tools only (no API call, no key needed)")
    parser.add_argument("--as-provider", default="openrouter", choices=("openrouter", "delos", "google"))
    parser.add_argument("--as-llm", default="", help="astra: the model (default openai/gpt-6-astra; google: gemini-4-argon)")
    parser.add_argument("--as-worker", default="", help="astra --as-provider delos: the worker id")
    parser.add_argument("--as-data", default="", help="astra: a folder made by `astradata` (raw + v1 answers) to give Astra")
    parser.add_argument("--as-goal", default="vs-dpo", choices=("vs-dpo", "on-top"),
                        help="astra: beat v1 from the raw weights (vs-dpo) or improve v1's own weights (on-top)")
    parser.add_argument("--as-start", default="raw", help="astra: 'raw' or the v1 adapter (merged first) for on-top")
    parser.add_argument("--ad-raw", default="", help="astradata: raw answers file(s), comma-joined (default: v1's a4 file)")
    parser.add_argument("--ad-v1", default="", help="astradata: v1's answers, comma-joined polans names (onp-NAME-a/b)")
    parser.add_argument("--ad-v1-train", default="", help="astradata: v1's DPO run (lupin name, e.g. lpc-bidax5q): its pairs, "
                        "settings and every checkpoint go into the data folder (seed --ad-v1-seed)")
    parser.add_argument("--ad-v1-seed", type=int, default=0)
    parser.add_argument("--tp-lam-max", type=float, default=0.05)
    parser.add_argument("--tp-ramp", type=int, default=300)
    parser.add_argument("--tp-gate", type=float, default=0.0)
    parser.add_argument("--tp-dr-grpo", action="store_true")
    parser.add_argument("--fm-screen-only", action="store_true", help="fullmodel: the 720 screen only (re-scoring old models)")
    parser.add_argument("--fm-merge", nargs="*", default=[], help="fullmodel (9 Nov): models/OUT=BASE:ADAPTER merges an "
                        "adapter onto a base model before testing models/OUT")
    parser.add_argument("--fm-soup", nargs="*", default=[], help="fullmodel (9 Nov): NAME:STEP rebuilds a missing "
                        "models/NAME-soup-STEP = the original Qwen + adapters/NAME-s0..2/step-STEP averaged (a first chain "
                        "round, e.g. ca1:10; the seeds started from the previous round's adapter, so they hold the lineage)")
    parser.add_argument("--lp-trained", default="", help="lupin: onpolicy/polans run names (comma-separated): lupin_pairs.py --trained")
    parser.add_argument("--lp-min-overcut", type=int, default=0, help="lupin: gate, at least this many trained overcut pairs or no training")
    parser.add_argument("--lp-overcut", default="", help="lupin: onpolicy run names, comma-separated: their answers are lupin_pairs.py --overcut")
    parser.add_argument("--lp-reset-seeds", nargs="*", type=int, default=[], help="lupin: delete these seeds' adapters (local and Drive) and train them again")
    parser.add_argument("--lp-pairs-only", action="store_true", help="lupin: build and report the pairs, no training")
    parser.add_argument("--lp-cap-like", default="", help="lupin --lp-no-match: cut the raw pairs at random to "
                        "pairs-<this>'s count per part x kind")
    parser.add_argument("--lp-no-match", action="store_true", help="lupin: train on the raw pairs, no lupin_select matching (michal-bida control)")
    parser.add_argument("--lp-loose", default="", help="lupin: lupin_select.py --loose-parts (parts short of pairs)")
    parser.add_argument("--lp-add-q", default="", help="lupin: top-up sets (in --lp-extra-q) whose questions all join the pool")
    parser.add_argument("--lp-kind-share", default="", help="lupin: lupin_select.py --kind-share-part (Run A pair mix)")
    parser.add_argument("--lp-lr-schedule", default="", help="lupin: dpo.py --lr-schedule (cosine: warm-up 10, down to 2e-6)")
    parser.add_argument("--lp-pressure", default="", help="lupin: lupin_select.py --shorten-ratio-part (per-type pressure)")
    parser.add_argument("--lp-balance", type=float, default=0.0, help="lupin: lupin_select.py --max-shorten-ratio")
    parser.add_argument("--lp-total", type=int, default=3200, help="lupin: pairs after matching (200 steps x 16)")
    parser.add_argument("--lp-save", type=int, default=50, help="lupin: DPO checkpoint interval")
    parser.add_argument("--lp-dpo", default="", help="lupin / typeonly: more dpo.py flags, e.g. '--nll 0'; comma-joined in a plan line")
    parser.add_argument("--lp-fresh", action="store_true", help="lupin: score soups on the fresh set x8 (finalists)")
    parser.add_argument("--dp-check-seed", type=int, default=9, help="poolmore: sampling seed of the held-out check")
    parser.add_argument("--dp-kod", type=int, default=40000, help="buildpool2: KodCode-V1 candidates")
    parser.add_argument("--tsk-ratios", type=float, nargs="+", default=[0.5, 0.7, 1.0],
                        help="tokenskip: share of the thinking's words kept, one model per ratio (1.0: same SFT, no compression)")
    parser.add_argument("--tsk-ratio-by", default="", help="tokenskip: source=ratio overrides, e.g. gsm8k_matched=0.9")
    parser.add_argument("--tsk-pick", default="random", choices=("random", "shortest"), help="tokenskip: right answer kept per question")
    parser.add_argument("--tsk-epochs", type=int, default=2, help="tokenskip: SFT passes (a checkpoint after each)")
    parser.add_argument("--tsk-lr", type=float, default=5e-5, help="tokenskip: SFT learning rate (the paper's)")
    parser.add_argument("--tsk-lots", nargs="*", default=["plotB", "plotB2", "plotB3", "plotB4", "plotB5", "plotV2"],
                        help="tokenskip: label lots whose original-model rollouts give the right answers")
    parser.add_argument("--drop", nargs="*", default=[],
                        help="dropmodels: MODEL=ADAPTER pairs; models/MODEL is removed from Drive and here only if adapters/ADAPTER is on Drive")
    parser.add_argument("--sg-lots", nargs="*", default=["plotB", "plotB2", "plotB3", "plotB4", "plotB5", "plotV2"],
                        help="stopgold: label lots of --start whose cut verdicts are the teacher")
    parser.add_argument("--sg-save-every", type=int, default=100, help="stopgold: optimizer steps between scored checkpoints")
    parser.add_argument("--sg-extra", default="", help="stopgold: extra stop_gold.py arguments, e.g. '--min-rate 0.5'")
    parser.add_argument("--tg-guard-mode", default="longer", choices=("longer", "shorter"),
                        help="targeted: guard pairs longest right > shortest wrong (longer) or shortest right > longest wrong")
    parser.add_argument("--tg-guard", default="", help="targeted: types given guard pairs (default: the --tg-long types)")
    parser.add_argument("--tg-hard-ratio", type=float, default=0.0, help="targeted: hard pairs <= this x shorten pairs ...")
    parser.add_argument("--tg-hard-cap", default="none", help="targeted: ... for these types (0 ratio: no cap)")
    parser.add_argument("--tg-skip-if", default="",
                        help="targeted: NAME:SAVED:LOSS, skip this stage when a soup of run NAME already saved >= SAVED "
                             "(e.g. 0.3) at an accuracy change >= -LOSS points on the fresh set (e.g. 1.0)")
    parser.add_argument("--probe-extra-from", nargs="*", help="probeknow: callot runs whose cuts are added to the development set")
    parser.add_argument("--probe-test-from", help="probeknow: a later run of the same model whose new cuts are the held-out test set")
    parser.add_argument("--rl-block", type=int, default=20, help="rlcal: RL steps per block (saved, merged, scored x2)")
    parser.add_argument("--rl-max-tokens", type=int, default=3072, help="rlcal: reply cap during RL")
    parser.add_argument("--smoke", action="store_true", help="rlcal: tiny sizes, no scoring: does every step run?")
    parser.add_argument("--word-scorer", default="lingua", help="fading: lingua (experiment 2) or marginal (experiment 3)")
    args = parser.parse_args()
    update()
    if args.stage != "plan":
        keep_saving(300)  # inside every stage too: a long stage is on Drive at most ~5 minutes behind
    globals()[args.stage](args)
    if args.stage != "plan":  # every finished stage is on Drive at once, even under an older plan worker
        drive_sync("save")


def update(restart=True):
    """On Colab, pull this branch's scripts before every stage, so a pushed fix needs no restart."""
    if os.environ.get("TRACK_UPDATED") or not Path("/content").exists():
        return
    clone = Path("/content/T")
    shutil.rmtree(clone, ignore_errors=True)
    branch = os.environ.get("TRACK_BRANCH", "mvt-fading")
    import time
    url = "https://github.com/arsenelemoigne/Test"
    for attempt in range(4):  # once a clone kept serving an old commit for hours: check it against the branch head
        shutil.rmtree(clone, ignore_errors=True)
        if subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", branch, url, str(clone)]).returncode:
            print("could not update the scripts; running the local copies", flush=True)
            return
        head = subprocess.run(["git", "ls-remote", url, f"refs/heads/{branch}"], capture_output=True, text=True).stdout.split()
        got = subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if not head or head[0] == got:
            break
        print(f"clone at {got[:7]} but the branch is at {head[0][:7]}; retrying", flush=True)
        time.sleep(30)
    changed = False
    for source in [*(clone / "thinkingcap").glob("*.py"), *(clone / "thinkingcap").glob("plan*.txt"),
                   *(clone / "thinkingcap").glob("astra_*.txt"), *(clone / "thinkingcap").glob("astra_*.md")]:  # Astra's prompts / card
        if not Path(source.name).exists() or not filecmp.cmp(source, source.name, shallow=False):
            shutil.copy(source, source.name)
            changed |= source.name == "track.py"
    print("scripts at " + subprocess.run(["git", "-C", str(clone), "log", "-1", "--format=%h %s"],
                                         capture_output=True, text=True).stdout.strip(), flush=True)
    if changed and restart:
        os.execve(sys.executable, [sys.executable, *sys.argv], os.environ | {"TRACK_UPDATED": "1"})


DRIVE = Path("/content/drive/MyDrive/thinkingcap-mvtf")


def drive_sync(direction, models=None, only=None):
    """_drive_sync, but a Drive error (a dropped mount: OSError, Errno 107) is reported and the stage goes on (9 Nov)."""
    try:
        return _drive_sync(direction, models=models, only=only)
    except OSError as error:
        print(f"drive: {direction} FAILED ({error}); the stage goes on, results stay on this runtime until Drive is "
              "remounted (run the Drive cell again)", flush=True)


def _drive_sync(direction, models=None, only=None):
    """Copy runs/, adapters/ and data/ to Google Drive ("save") or back ("restore"), when Drive is mounted.

    Colab wipes the runtime disk when it recycles a runtime; Drive keeps every result, adapter and rollout.
    """
    try:
        if not Path("/content/drive/MyDrive").exists():
            return
        DRIVE.mkdir(parents=True, exist_ok=True)
    except OSError as error:  # (9 Nov) a dropped Drive mount (Errno 107) crashed a whole queue: warn, keep working
        print(f"drive: NOT connected ({error}); {direction} skipped, results stay on this runtime until Drive is "
              "remounted (run the Drive cell again)", flush=True)
        return
    names = ["runs", "adapters", "data", "models", *[p.name for p in Path(".").glob("questions_*") if p.is_dir()]]
    if direction == "restore":
        names = ["runs", "adapters", "data", "models", *[p.name for p in DRIVE.glob("questions_*") if p.is_dir()]]
    if direction == "restore" and only is not None:  # a light restore: just these paths (globs under the Drive copy)
        for pattern in only:
            for source in DRIVE.glob(pattern):
                target = Path(source.relative_to(DRIVE))
                target.parent.mkdir(parents=True, exist_ok=True)
                if source.is_dir():
                    subprocess.run(["rsync", "-a", "--update", f"{source}/", f"{target}/"])
                else:
                    subprocess.run(["rsync", "-a", "--update", str(source), str(target)])
        print(f"drive: light restore done ({', '.join(only)})", flush=True)
        return
    for name in names:
        source, target = (Path(name), DRIVE / name) if direction == "save" else (DRIVE / name, Path(name))
        if direction == "restore" and name == "models" and models is not None:
            for m in models:  # only the merged models the queue names (all of them is ~50 min of copying)
                if (source / m).exists():
                    subprocess.run(["rsync", "-a", "--update", f"{source / m}/", f"{target / m}/"])
            continue
        if source.exists():
            # --update: never replace a newer file (a restore once overwrote a fresher local rollout file)
            subprocess.run(["rsync", "-a", "--update", f"{source}/", f"{target}/"])
    if direction == "save":  # the log, the scripts, plans, notebooks and reports as they ran here
        files = [str(p) for p in Path(".").glob("*") if p.suffix in (".py", ".txt", ".md", ".ipynb") and p.is_file()]
        files += [str(p) for p in Path("/content/T/thinkingcap").glob("*.md")] + ["/content/track_log.txt"]
        code = DRIVE / ("code-" + os.environ["TRACK_SESSION"] if os.environ.get("TRACK_SESSION") else "code")
        code.mkdir(exist_ok=True)  # a second runtime keeps its own log and scripts copy
        subprocess.run(["rsync", "-a", *[f for f in files if Path(f).exists()], str(code)])
    print(f"drive: {direction} done", flush=True)


def run(*command):
    print("$", " ".join(map(str, command)), flush=True)
    subprocess.run([str(c) for c in command], check=True)


def py(*command):
    run(sys.executable, *command)


def setup(args):
    try:  # the IFEval grader (typed.py, rl_train.py) needs langdetect, which requirements.txt lacks (drpo1 failed on it)
        import langdetect  # noqa: F401
    except ImportError:
        run(sys.executable, "-m", "pip", "install", "-q", "langdetect")
    if not Path("data/train.jsonl").exists():
        py("utils/train/prepare_data.py")
    if not Path("questions/questions.jsonl").exists():
        py("utils/eval/benchmarks.py")
    if not Path(QUESTIONS, "questions.jsonl").exists():
        py("subset_questions.py", "--per-dataset", 40, "--out", QUESTIONS)


def evaluate(out, adapter=None, samples=2, questions=QUESTIONS, model=None):
    if not (Path(out) / "results.json").exists():
        shutil.rmtree(out, ignore_errors=True)  # a run cut short (lost runtime) leaves a folder evaluate.py refuses
        py("evaluate.py", "--questions", questions, "--samples", samples, "--out", out,
           *(["--adapter", adapter] if adapter else []), *(["--model", model] if model else []))


def base(args):
    evaluate(BASE)


def rollouts(args):
    if not ROLLOUTS.exists():
        # The same training slice as SFT rounds 1-2: 150 questions per source, 2 samples, 2,048 tokens.
        py("probe_curve.py", "--data", "data/train.jsonl", "--sources", "gsm8k,arc_easy,commonsense_qa",
           "--per-source", 150, "--samples", 2, "--max-tokens", 2048, "--out", ROLLOUTS.parent,
           "--dtype", args.dtype, "--generate-only", "--chunk", 150)


def mvt(args):
    name = f"mvth{args.travel:g}" + (f"-full-{args.full_sources.replace(',', '-')}" if args.full_sources else "") \
        + (f"-min{args.min_steps}" if args.min_steps != 1 else "")
    # "h": the full look-ahead rule (the 2-step one, "mvt256", left at 0 on math)
    prune = ROOT / f"prune-{name}"
    name += (f"-cap{args.max_per_source}" if args.max_per_source != 70 else "") \
        + (f"-x{args.repeat.replace(':', '')}" if args.repeat else "")  # SFT-only changes share the pruning
    adapter, out = Path("adapters") / name, ROOT / f"eval-{name}"
    if not (prune / "sft.jsonl").exists():
        py("prune_traces.py", "--rollouts", ROLLOUTS, "--out", prune, "--rule", "mvt", "--travel", args.travel,
           "--check-samples", 4, "--dtype", args.dtype,
           "--min-steps", args.min_steps, *(["--full-sources", args.full_sources] if args.full_sources else []))
    print((prune / "summary.txt").read_text())
    if not (adapter / "adapter_config.json").exists():
        py("sft.py", "--data", prune / "sft.jsonl", "--out", adapter, "--max-per-source", args.max_per_source,
           "--dtype", args.dtype, *(["--repeat", args.repeat] if args.repeat else []))
    evaluate(out, adapter)
    print(score_line(name, out))
    print(diagnose(out))


def report(args):
    logs = sorted(ROOT.glob("prune-mvt*/log.jsonl"))  # any run: the curves do not depend on the rule
    if not logs:
        raise SystemExit("run `track.py mvt` first: the report reads its log")
    py("mvt_report.py", logs[0])


def fade(args):
    words = ("-marginal" if args.word_scorer == "marginal" else "-words") if args.word_step > 0 else ""
    tag = f"fade-{args.order}" + words + (f"-floor{args.floor_travel:g}" if args.floor_travel is not None else "") + args.tag
    previous = None
    for k in range(1, args.rounds + 1):
        data, adapter, out = ROOT / tag / f"r{k}", Path("adapters") / tag / f"r{k}", ROOT / f"eval-{tag}-r{k}"
        if not (data / "sft.jsonl").exists():
            extra = []
            if previous:
                extra = ["--adapter", previous[1], "--previous", previous[0] / "sft.jsonl",
                         "--cache", ROOT / tag / "r1" / "curves.jsonl"]
            if args.floor_travel is not None:
                extra += ["--floor-travel", args.floor_travel]
            py("fade_traces.py", "--rollouts", ROLLOUTS, "--out", data, "--round", k, "--order", args.order,
               "--word-step", args.word_step, "--word-scorer", args.word_scorer, "--dtype", args.dtype, *extra)
        print((data / "summary.txt").read_text())
        if not (adapter / "adapter_config.json").exists():
            py("sft.py", "--data", data / "sft.jsonl", "--out", adapter, "--epochs", 1, "--max-per-source", 100,
               "--dtype", args.dtype, *(["--init-adapter", previous[1]] if previous else []))
        evaluate(out, adapter)
        line, delta = score_line(f"{tag} r{k}", out, with_delta=True)
        print(line, flush=True)
        print(diagnose(out), flush=True)
        if delta < LIMIT and not args.ignore_limit:
            print(f"round {k} is past the {LIMIT:+.0f}-point limit: stop; the answer is round {k - 1}")
            return
        previous = (data, adapter)


def mix(args):
    """SFT on the non-math rows of one target file and the GSM8K rows of another, then evaluate.

    For example the MVT-cut ARC/CSQA traces with GSM8K traces compacted by experiment 3.
    """
    data, adapter, out = ROOT / f"mix-{args.name}", Path("adapters") / f"mix-{args.name}", ROOT / f"eval-mix-{args.name}"
    if not (data / "sft.jsonl").exists():
        data.mkdir(parents=True, exist_ok=True)
        rows = [l for l in Path(args.mc).read_text().splitlines() if l.strip() and json.loads(l)["source"] != "gsm8k"]
        rows += [l for l in Path(args.math).read_text().splitlines() if l.strip() and json.loads(l)["source"] == "gsm8k"]
        (data / "sft.jsonl").write_text("".join(l + "\n" for l in rows))
    math = [json.loads(l) for l in (data / "sft.jsonl").read_text().splitlines() if '"gsm8k"' in l]
    print(f"mix {args.name}: {len(math)} GSM8K rows, mean completion "
          f"{sum(len(r['completion_ids']) for r in math) / max(1, len(math)):.0f} tokens", flush=True)
    if not (adapter / "adapter_config.json").exists():
        py("sft.py", "--data", data / "sft.jsonl", "--out", adapter, "--max-per-source", args.max_per_source,
           "--dtype", args.dtype, *(["--repeat", args.repeat] if args.repeat else []))
    evaluate(out, adapter)
    print(score_line(f"mix-{args.name}", out))
    print(diagnose(out))


def full(args):
    """The full evaluation: every question (2,759) x the recipe's 8 samples, base then adapter."""
    base = ROOT / "full-base"
    evaluate(base, samples=8, questions="questions")
    if args.adapter:
        name = args.adapter.removeprefix("adapters/").replace("/", "-")
        out = ROOT / f"full-{name}"
        evaluate(out, args.adapter, samples=8, questions="questions")
        print(score_line(f"{name} (full evaluation)", out, base=base))
        print(diagnose(out, base=base))


# kind of data -> (training source(s) from prepare_types.py, the benchmark that scores that kind)
# TRACK_SET=matched uses the data matched to the evaluation: GSM8K filtered to SVAMP-like difficulty,
# ToolACE in BFCL's format and mix, and multiple choice as a fifth kind (no benchmark of its own).
DATA_SET = os.environ.get("TRACK_SET", "v1")
if DATA_SET == "matched":
    GRID = Path("runs/grid-matched")
    TYPES = {"math": ("gsm8k_all", "svamp"), "code": ("mbpp_train", "mbpp"), "tool": ("toolace", "bfcl"),
             "text": ("ifeval_like", "ifeval"), "choice": ("arc_easy,commonsense_qa", None)}
    TYPE_DATA = Path("data/matched.jsonl")
    PREPARE = ["--sources", "gsm8k_all,mbpp_train,toolace,ifeval_like,mc", "--per-source", 300]
    EASY = {"math": 700}  # keep GSM8K questions with a correct reply of at most 700 tokens
else:
    GRID = Path("runs/grid")
    TYPES = {"math": ("gsm8k", "svamp"), "code": ("mbpp_train", "mbpp"), "tool": ("hermes_tools", "bfcl"),
             "text": ("ifeval_like", "ifeval")}
    TYPE_DATA = Path("data/types.jsonl")
    PREPARE = ["--per-source", 400]
    EASY = {}


def grid_data(args):
    """Training questions of every kind, and the model's own replies to them (2 per question)."""
    wanted = [s for sources, _ in TYPES.values() for s in sources.split(",")]
    questions = TYPE_DATA
    have = {json.loads(l)["source"] for l in questions.read_text().splitlines() if l.strip()} \
        if questions.exists() else set()
    if set(wanted) - have:
        py("prepare_types.py", *PREPARE, "--out", questions)
    probe = GRID / "probe"
    rollouts = probe / "rollouts.jsonl"
    done = {json.loads(l)["source"] for l in rollouts.read_text().splitlines() if l.strip()} \
        if rollouts.exists() else set()
    missing = [source for source in wanted if source not in done]
    if missing:  # sample only what is missing and append it, so earlier replies stay as they are
        extra = GRID / "probe-extra"
        shutil.rmtree(extra, ignore_errors=True)
        py("probe_curve.py", "--data", questions, "--sources", ",".join(missing), "--per-source", args.per_source,
           "--samples", 2, "--out", extra, "--dtype", args.dtype, "--generate-only", "--chunk", 150)
        probe.mkdir(parents=True, exist_ok=True)
        with rollouts.open("a") as handle:
            handle.write((extra / "rollouts.jsonl").read_text())
    rows = [json.loads(l) for l in rollouts.read_text().splitlines() if l.strip()]
    for source in sorted({r["source"] for r in rows}):
        group = [r for r in rows if r["source"] == source]
        print(f"   {source:<14} {len(group)} replies, {sum(r['correct'] for r in group) / len(group):.0%} correct, "
              f"{sum(r['truncated'] for r in group) / len(group):.0%} capped, "
              f"mean {sum(r['output_tokens'] for r in group) / len(group):.0f} tokens", flush=True)


def grid(args):
    """One cell of the method x kind grid: shorten one kind's traces one way, train on them alone, test on all."""
    source, _ = TYPES[args.type]
    unit = args.unit if args.method != "whole" else "paragraph"
    name = f"{args.type}-{args.method}" + (f"{args.travel:g}" if args.method == "mvt" else "") \
        + (f"-{unit}" if unit != "paragraph" else "")
    common = ["--rollouts", GRID / "probe" / "rollouts.jsonl", "--sources", source, "--unit", unit,
              "--checker", "vllm", "--dtype", args.dtype,
              *(["--easy-tokens", EASY[args.type]] if args.type in EASY else [])]
    if args.method == "fade":
        previous = None
        for k in range(1, args.rounds + 1):
            data, adapter = GRID / name / f"r{k}", Path("adapters") / "grid" / name / f"r{k}"
            if not (data / "sft.jsonl").exists():
                extra = ["--adapter", previous[1], "--previous", previous[0] / "sft.jsonl",
                         "--cache", GRID / name / "r1" / "curves.jsonl"] if previous else []
                py("fade_traces.py", *common, "--out", data, "--round", k, "--step", 0.2, "--word-step", 0, *extra)
            print((data / "summary.txt").read_text())
            if not (adapter / "adapter_config.json").exists():
                py("sft.py", "--data", data / "sft.jsonl", "--out", adapter, "--epochs", 1, "--max-per-source", 1000,
                   "--dtype", args.dtype, *(["--init-adapter", previous[1]] if previous else []))
            grid_evaluate(f"{name}-r{k}", adapter, args)
            previous = (data, adapter)
        return
    data, adapter = GRID / name, Path("adapters") / "grid" / name
    if not (data / "sft.jsonl").exists():
        rule = {"whole": ["--full-sources", source], "mvt": ["--rule", "mvt", "--travel", args.travel],
                "marginal": ["--rule", "marginal"]}[args.method]
        py("prune_traces.py", *common, "--out", data, *rule)
    print((data / "summary.txt").read_text())
    if not (adapter / "adapter_config.json").exists():
        py("sft.py", "--data", data / "sft.jsonl", "--out", adapter, "--max-per-source", 1000, "--dtype", args.dtype)
    grid_evaluate(name, adapter, args)


def grid_evaluate(name, adapter, args):
    """Score on all four benchmarks (4 samples): the matching one, and the spill-over onto the others."""
    base, out = ROOT / "s4-base", GRID / f"eval-{name}"
    evaluate(base, samples=4)
    evaluate(out, adapter, samples=4)
    print(score_line(name, out, base=base), flush=True)
    result = compare(out, base)
    row = {"name": name, "type": args.type, "method": args.method, "unit": args.unit,
           "average": {"saved": result["average"]["token_reduction"], "delta": result["average"]["accuracy_delta"]},
           "benchmarks": {d: {"saved": r["token_reduction"], "delta": r["accuracy_delta"]}
                          for d, r in result["results"].items()}}
    rows = [l for l in (GRID / "grid.jsonl").read_text().splitlines() if json.loads(l)["name"] != name] \
        if (GRID / "grid.jsonl").exists() else []
    (GRID / "grid.jsonl").write_text("".join(l + "\n" for l in rows + [json.dumps(row)]))


def combine(args):
    """One model from the best method per kind: pool chosen rows of several runs, train, score with 4 samples."""
    data, adapter = GRID / f"combine-{args.name}", Path("adapters") / f"combine-{args.name}"
    if not (data / "sft.jsonl").exists():
        rows = []
        for spec in args.inputs:
            path, *sources = spec.split(":")
            for line in Path(path).read_text().splitlines():
                if line.strip() and (not sources or json.loads(line)["source"] in sources):
                    rows.append(line)
        data.mkdir(parents=True, exist_ok=True)
        (data / "sft.jsonl").write_text("".join(l + "\n" for l in rows))
    pooled = [json.loads(l) for l in (data / "sft.jsonl").read_text().splitlines() if l.strip()]
    for source in sorted({r["source"] for r in pooled}):
        group = [r for r in pooled if r["source"] == source]
        print(f"   {source:<16} {len(group)} rows, mean completion "
              f"{sum(len(r['completion_ids']) for r in group) / len(group):.0f} tokens", flush=True)
    if not (adapter / "adapter_config.json").exists():
        py("sft.py", "--data", data / "sft.jsonl", "--out", adapter, "--max-per-source", args.max_per_source,
           "--dtype", args.dtype, *(["--repeat", args.repeat] if args.repeat else []))
    base, out = ROOT / "s4-base", GRID / f"eval-combine-{args.name}"
    evaluate(base, samples=4)
    evaluate(out, adapter, samples=4)
    print(score_line(f"combine-{args.name}", out, base=base), flush=True)
    print(diagnose(out, base=base))


def merge(args):
    """Add per-kind adapters into the base weights at chosen strengths, then score with 4 samples."""
    model = Path("models") / f"merge-{args.name}"
    if not (model / "config.json").exists():
        py("merge.py", "--out", model, "--adapters", *args.inputs)
    n = args.samples  # --samples (4 by default; 8 for finalists)
    base = ROOT / f"s{n}-base"
    out = GRID / f"eval-merge-{args.name}" if n == 4 else ROOT / f"s{n}-merge-{args.name}"
    evaluate(base, samples=n)
    evaluate(out, samples=n, model=model)
    print(score_line(f"merge-{args.name}", out, base=base), flush=True)
    print(diagnose(out, base=base))


def fullbest(args):
    """Run the full evaluation on the candidate that saves most while staying within --margin of the base.

    Candidates are name=spec pairs: the 4-sample run directory's name under runs/ and either
    adapter:<path> or model:<path>. Chosen on the 4-sample subset scores, so the full evaluation
    goes to one model only.
    """
    best = None
    for candidate in args.inputs:
        run, spec = candidate.split("=", 1)
        kind, path = spec.split(":", 1)
        compared = Path("runs") / run / "compare.json"
        if not compared.exists():
            print(f"fullbest: {run} has no 4-sample comparison; skipped", flush=True)
            continue
        average = json.loads(compared.read_text())["average"]
        saved, delta = average["token_reduction"] or 0, 100 * average["accuracy_delta"]
        print(f"fullbest: {run}: {saved:+.1%} tokens, {delta:+.1f} pts", flush=True)
        if delta >= args.margin and (best is None or saved > best[0]):
            best = (saved, run, kind, path)
    if best is None:
        raise SystemExit("fullbest: no candidate within the margin")
    _, run, kind, path = best
    print(f"fullbest: full evaluation of {run} ({kind} {path})", flush=True)
    base = ROOT / "full-base"
    evaluate(base, samples=8, questions="questions")
    out = ROOT / f"full-{run.split('/')[-1].removeprefix('eval-')}"
    evaluate(out, samples=8, questions="questions",
             **({"adapter": path} if kind == "adapter" else {"model": path}))
    print(score_line(f"{run} (full evaluation)", out, base=base))
    print(diagnose(out, base=base))


def stats(args):
    """Per-question length and accuracy summaries, printed between markers so they can be read from the log.

    --inputs are evaluation run directories (records.jsonl) or rollouts.jsonl files, relative to
    runs/. One row per question: [dataset, kind, samples, mean tokens, sd tokens, min, max,
    accuracy, mean thinking tokens].
    """
    import statistics

    for name in args.inputs:
        path = Path("runs") / name
        file = path / "records.jsonl" if path.is_dir() else path
        if not file.exists():
            print(f"stats: {name} missing; skipped", flush=True)
            continue
        groups = {}
        for line in file.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            dataset = r.get("dataset") or r.get("source")
            kind = r.get("task") or dataset
            thinking = r.get("reasoning_tokens", len(r.get("thinking_ids", [])))
            groups.setdefault((dataset, kind, str(r["id"])), []).append(
                (r["output_tokens"], bool(r.get("correct")), thinking))
        rows = []
        for (dataset, kind, _), samples in groups.items():
            tokens = [t for t, _, _ in samples]
            rows.append([dataset, kind, len(samples), round(statistics.mean(tokens), 1),
                         round(statistics.pstdev(tokens), 1), min(tokens), max(tokens),
                         round(sum(c for _, c, _ in samples) / len(samples), 3),
                         round(statistics.mean(h for _, _, h in samples), 1)])
        print(f"STATS-BEGIN {name}", flush=True)
        print(json.dumps(rows, separators=(",", ":")), flush=True)
        print(f"STATS-END {name}", flush=True)


BON = Path("runs/bon")
BON_SOURCES = "gsm8k_all,mbpp_train,toolace,ifeval_like,arc_easy,commonsense_qa"


def bon(args):
    """Shortest-of-8 self-distillation: the model's own shortest correct reply per question, nothing edited.

    Data matched to the evaluation (GSM8K filtered to SVAMP-like difficulty, MBPP minus MBPP+,
    ToolACE in BFCL's format and mix, IFEval-like, a little multiple choice), 8 replies per
    question at the evaluation's token caps, a contamination check, one gentle SFT pass, then
    the 240-question test sample with 4 samples against the raw model.
    """
    name = args.name or "r1"
    data, probe, select = Path("data/bon.jsonl"), BON / "probe", BON / f"select-{name}"
    if not data.exists():
        py("prepare_types.py", "--sources", "gsm8k_all,mbpp_train,toolace,ifeval_like,mc", "--per-source",
           args.per_source, "--out", data)
    if not (probe / "rollouts.jsonl").exists():
        py("probe_curve.py", "--data", data, "--sources", BON_SOURCES, "--per-source", args.per_source,
           "--samples", 8, "--out", probe, "--dtype", args.dtype, "--generate-only", "--chunk", 150,
           "--max-questions", args.gen_questions)
    if not (select / "sft.jsonl").exists():
        py("select_shortest.py", "--rollouts", probe / "rollouts.jsonl", "--out", select,
           "--easy-tokens", "gsm8k_all=700", "--min-saved", args.min_saved)
        py("contamination.py", "--data", select / "sft.jsonl", "--rollouts", probe / "rollouts.jsonl")
    print((select / "summary.txt").read_text(), flush=True)
    adapter = Path("adapters") / f"bon-{name}"
    if not (adapter / "adapter_config.json").exists():
        py("sft.py", "--data", select / "sft.jsonl", "--out", adapter, "--epochs", args.sft_epochs, "--lr", args.sft_lr,
           "--max-per-source", args.max_per_source, "--dtype", args.dtype)
    if args.skip_eval:
        return
    base, out = ROOT / "s4-base", ROOT / f"s4-bon-{name}"
    evaluate(base, samples=4)
    evaluate(out, adapter, samples=4)
    print(score_line(f"bon-{name} (4 samples, 240 test questions)", out, base=base), flush=True)
    print(diagnose(out, base=base))


def bon_dpo(args):
    """Best-of-8 then DPO on the same replies: shortest correct over longest correct, and over a wrong one.

    Round 1 (no --rollouts-from): pairs come from the raw model's best-of-8 rollouts (bon_pairs.py),
    restricted to the questions select-r1 kept after the contamination check, starting from --start
    (default the best-of-8 SFT adapter), which is also the reference.
    Round 2+ (--rollouts-from ADAPTER, or "auto" for the best scored checkpoint within -2.5 points):
    that model answers the same questions 8 times again (its replies are already shorter), new pairs,
    DPO continues from it. Checkpoints every --save-every steps; --eval-steps are scored on the
    240-question test sample with 4 samples.
    """
    name = args.name or "bon-dpo"
    start, no_shorten, rollouts = args.start or "adapters/bon-r1", args.no_shorten, BON / "probe" / "rollouts.jsonl"
    if start == "raw":  # DPO straight from the raw model: a fresh LoRA is the policy's start and the reference
        start = "adapters/fresh-lora"
        if not Path(start, "adapter_config.json").exists():
            py("sft.py", "--init-only", "--out", start)
    if args.rollouts_from:
        start = best_checkpoint() if args.rollouts_from == "auto" else args.rollouts_from
        settings = Path(start).parent / "stage.json"
        if settings.exists():  # a round keeps the recipe of the checkpoint it starts from
            recipe = json.loads(settings.read_text())
            no_shorten = no_shorten or recipe.get("no_shorten", "")
            args.drop_sources = args.drop_sources or recipe.get("drop_sources", "")
        print(f"round from {start} (code guard-only: {bool(no_shorten)})", flush=True)
        probe = BON / f"probe-{args.probe_from or name}"  # --probe-from: reuse another round's answers
        if not (probe / "rollouts.jsonl").exists():
            data = new_questions(args.data_seed) if args.data_seed else "data/bon.jsonl"
            py("probe_curve.py", "--data", data, "--sources", BON_SOURCES, "--per-source", 300,
               "--samples", 8, "--adapter", start, "--out", probe, "--dtype", args.dtype, "--generate-only", "--chunk", 150,
               "--max-questions", args.gen_questions,
               *(["--exclude-rollouts", BON / "probe" / "rollouts.jsonl"] if args.fresh_questions else []))
        rollouts = probe / "rollouts.jsonl"
        keep = BON / f"select-{args.probe_from or name}"
        if not (keep / "sft.jsonl").exists():  # the same easy-math filter and contamination check, on these questions
            py("select_shortest.py", "--rollouts", rollouts, "--out", keep, "--easy-tokens", "gsm8k_all=700")
            py("contamination.py", "--data", keep / "sft.jsonl", "--rollouts", rollouts)
    pairs = BON / ((f"pairs-{name}" if args.rollouts_from else "pairs" + (f"-noshort-{no_shorten}" if no_shorten else ""))
                   + (f"-drop-{args.drop_sources}" if args.drop_sources else ""))
    if not (pairs / "pairs.jsonl").exists():
        py("bon_pairs.py", "--rollouts", rollouts, "--keep", (keep if args.rollouts_from else BON / "select-r1") / "sft.jsonl",
           "--out", pairs, "--no-shorten", no_shorten, "--drop-sources", args.drop_sources)
    print((pairs / "summary.txt").read_text(), flush=True)
    adapter = Path("adapters") / name
    if not (adapter / "adapter_config.json").exists():
        py("dpo.py", "--pairs", pairs / "pairs.jsonl", "--init-adapter", start, "--out", adapter,
           "--mix", args.mix, "--save-every", args.save_every, "--epochs", args.dpo_epochs, "--dtype", args.dtype,
           "--seed", args.dpo_seed)
        (adapter / "stage.json").write_text(json.dumps({"start": str(start), "no_shorten": no_shorten,
                                                        "drop_sources": args.drop_sources}))
    score_checkpoints(name, adapter, args)


def score_checkpoints(name, adapter, args):
    """A quick sweep (--eval-samples, 2 by default), then the sweep's best within the limit with 4 samples."""
    n = args.eval_samples
    base = ROOT / f"s{n}-base"
    evaluate(base, samples=n)
    sweep = []
    chosen = [c for c in sorted(adapter.glob("step-*"), key=lambda p: int(p.name.split("-")[1]))
              if not args.eval_steps or int(c.name.split("-")[1]) in args.eval_steps]
    todo = [c for c in chosen if not (ROOT / f"s{n}-{name}-{c.name}" / "results.json").exists()]
    if todo:  # every checkpoint in one engine (sweep_eval.py): about the cost of one evaluation
        for c in todo:
            shutil.rmtree(ROOT / f"s{n}-{name}-{c.name}", ignore_errors=True)
        py("sweep_eval.py", "--questions", QUESTIONS, "--samples", n,
           "--runs", *[f"{ROOT / f's{n}-{name}-{c.name}'}={c}" for c in todo])
    for checkpoint in chosen:
        out = ROOT / f"s{n}-{name}-{checkpoint.name}"
        line, delta = score_line(f"{name} {checkpoint.name} ({n} samples)", out, with_delta=True, base=base)
        print(line, flush=True)
        sweep.append((compare(out, base)["average"]["token_reduction"] or 0, delta, checkpoint))
    safe = [c for c in sweep if c[1] >= LIMIT]
    if n != 4 and safe:
        saved, _, checkpoint = max(safe, key=lambda c: c[0])
        evaluate(ROOT / "s4-base", samples=4)
        out = ROOT / f"s4-{name}-{checkpoint.name}"
        evaluate(out, checkpoint, samples=4)
        print(score_line(f"{name} {checkpoint.name} (4 samples, the sweep's best)", out, base=ROOT / "s4-base"), flush=True)


def resolve(adapter):
    """'auto:NAME' -> that run's best checkpoint within -2.5 points (4 samples); anything else unchanged."""
    return str(best_checkpoint(prefix=adapter[5:])) if adapter.startswith("auto:") else adapter


CLEAN_SOURCES = "gsm8k_all,toolace,ifeval_like,arc_easy,commonsense_qa"


def clean(args):
    """MBPP-free rounds: no MBPP anywhere in the lineage (the other rounds started from an SFT step with ~74 MBPP rows).

    --start raw: DPO from a fresh LoRA on round 1's answers (the raw model's) without their code questions.
    --start auto:NAME / adapter: that model answers --gen-questions not-yet-used questions x 8 (no code questions).
    --code-source kodcode: plus the MBPP-like KodCode slice answered x 8 by the start model, as
    right-beats-wrong pairs only (no short-beats-long for code: those pairs hurt code before).
    """
    name = args.name
    if args.start == "raw":
        start = Path("adapters") / f"fresh-{name}"
        if not (start / "adapter_config.json").exists():
            py("sft.py", "--init-only", "--out", start)
        rollouts, keep = BON / "probe" / "rollouts.jsonl", BON / "select-r1" / "sft.jsonl"
    else:
        start = resolve(args.start)
        print(f"{name}: starts from {start}", flush=True)
        probe, keep = BON / f"probe-{name}", BON / f"select-{name}"
        if not (probe / "rollouts.jsonl").exists():
            py("probe_curve.py", "--data", "data/bon.jsonl", "--sources", CLEAN_SOURCES, "--per-source", 300, "--samples", 8,
               "--adapter", start, "--out", probe, "--dtype", args.dtype, "--generate-only", "--chunk", 150,
               "--max-questions", args.gen_questions, "--exclude-rollouts", BON / "probe" / "rollouts.jsonl")
        if not (keep / "sft.jsonl").exists():
            py("select_shortest.py", "--rollouts", probe / "rollouts.jsonl", "--out", keep, "--easy-tokens", "gsm8k_all=700")
            py("contamination.py", "--data", keep / "sft.jsonl", "--rollouts", probe / "rollouts.jsonl")
        rollouts, keep = probe / "rollouts.jsonl", keep / "sft.jsonl"
    files = []
    base_pairs = BON / f"pairs-{name}-nocode"
    if not (base_pairs / "pairs.jsonl").exists():
        py("bon_pairs.py", "--rollouts", rollouts, "--keep", keep, "--out", base_pairs, "--drop-sources", "mbpp_train")
    files.append(base_pairs / "pairs.jsonl")
    if args.code_source == "kodcode":
        if not Path("data/kodcode.jsonl").exists():
            py("prepare_types.py", "--sources", "kodcode", "--per-source", 2000, "--out", "data/kodcode.jsonl")
        kprobe, kkeep = BON / f"kprobe-{name}", BON / f"kselect-{name}"
        if not (kprobe / "rollouts.jsonl").exists():
            exclude = ["--exclude-rollouts", BON / f"kprobe-{args.kod_exclude}" / "rollouts.jsonl"] if args.kod_exclude else []
            adapter = [] if args.start == "raw" else ["--adapter", start]
            py("probe_curve.py", "--data", "data/kodcode.jsonl", "--sources", "kodcode", "--per-source", 2000, "--samples", 8,
               *adapter, "--out", kprobe, "--dtype", args.dtype, "--generate-only", "--chunk", 150,
               "--max-questions", args.kod_questions, *exclude)
        if not (kkeep / "sft.jsonl").exists():
            py("select_shortest.py", "--rollouts", kprobe / "rollouts.jsonl", "--out", kkeep)
            py("contamination.py", "--data", kkeep / "sft.jsonl", "--rollouts", kprobe / "rollouts.jsonl")
        kpairs = BON / f"pairs-{name}-kod"
        if not (kpairs / "pairs.jsonl").exists():
            py("bon_pairs.py", "--rollouts", kprobe / "rollouts.jsonl", "--keep", kkeep / "sft.jsonl", "--out", kpairs,
               "--no-shorten", "kodcode")
        files.append(kpairs / "pairs.jsonl")
    for f in files:
        print((f.parent / "summary.txt").read_text(), flush=True)
    pairs = BON / f"pairs-{name}"
    pairs.mkdir(parents=True, exist_ok=True)
    (pairs / "pairs.jsonl").write_text("".join(l + "\n" for f in files for l in f.read_text().splitlines() if l.strip()))
    adapter = Path("adapters") / name
    if not (adapter / "adapter_config.json").exists():
        py("dpo.py", "--pairs", pairs / "pairs.jsonl", "--init-adapter", start, "--out", adapter, "--mix", args.mix,
           "--save-every", args.save_every, "--epochs", args.dpo_epochs, "--dtype", args.dtype, "--seed", args.dpo_seed)
        (adapter / "stage.json").write_text(json.dumps({"start": str(start), "code_source": args.code_source}))
    score_checkpoints(name, adapter, args)


def full2(args):
    """All 2,759 evaluation questions with 2 samples: --base-only scores the raw model; otherwise the --inputs
    (adapters or auto:NAME) in one engine (sweep_eval.py), against the raw model's run (waited for on Drive)."""
    base = ROOT / "full2-base"
    if args.base_only:
        evaluate(base, samples=2, questions="questions")
        return
    drive_sync("restore")  # the other runtime's models and scores (e.g. family A) come from Drive
    adapters = [resolve(a) for a in args.inputs]
    names = [a.removeprefix("adapters/").replace("/", "-") for a in adapters]
    todo = [(ROOT / f"full2-{n}", a) for n, a in zip(names, adapters) if not (ROOT / f"full2-{n}" / "results.json").exists()]
    if todo:
        for out, _ in todo:
            shutil.rmtree(out, ignore_errors=True)
        py("sweep_eval.py", "--questions", "questions", "--samples", 2, "--runs", *[f"{out}={a}" for out, a in todo])
    import time
    for _ in range(240):  # the raw model's full run comes from the other runtime
        if (base / "results.json").exists():
            break
        drive_sync("restore")
        if not (base / "results.json").exists():
            print("waiting for full2-base on Drive", flush=True)
            time.sleep(120)
    for n in names:
        print(score_line(f"{n} (all 2,759 questions x 2)", ROOT / f"full2-{n}", base=base), flush=True)


BAL_SOURCES = "toolace,ifeval_like,arc_easy,commonsense_qa,gsm_plus,gsm_ic"


def balanced(args):
    """A balanced round: savings from tools, instructions and multiple choice; math protected; seeds averaged.

    1. A fresh draw (prepare_types --seed): ToolACE, IFEval-like, ARC/CSQA, and math with SVAMP-style traps
       (GSM-Plus perturbations, GSM-IC irrelevant sentences); --gen-questions of them, 8 answers each, by --start.
    2. Pairs (bon_pairs.py): short-beats-long only where the question is answered right at least 6 times in 8;
       math gets at most 40 such pairs per source and its right-beats-wrong pairs use a median-length correct
       reply; every source keeps right-beats-wrong pairs. No code.
    3. DPO from --start with each of --seeds, then each checkpoint step of --eval-steps averaged over the seeds
       (merge.py, equal weights) and scored with 4 samples; the best within the limit with 8 samples on the
       usual and on the fresh questions.
    """
    name, start = args.name, resolve(args.start)
    data = Path(f"data/bal-seed{args.data_seed}.jsonl")
    if not data.exists():
        raw = Path(f"data/bal-seed{args.data_seed}-raw.jsonl")
        py("prepare_types.py", "--sources", "toolace,ifeval_like,mc,gsm_plus,gsm_ic", "--per-source", 300,
           "--seed", args.data_seed, "--out", raw)
        text = lambda r: r["prompt"][-1]["content"]
        used = [p for p in Path("data").glob("bon*.jsonl") if "raw" not in p.name]
        seen = {text(json.loads(l)) for p in used for l in p.read_text().splitlines() if l.strip()}
        rows = [json.loads(l) for l in raw.read_text().splitlines() if l.strip()]
        data.write_text("".join(json.dumps(r) + "\n" for r in rows if text(r) not in seen))
    probe, keep = BON / f"probe-{name}", BON / f"select-{name}"
    if not (probe / "rollouts.jsonl").exists():
        py("probe_curve.py", "--data", data, "--sources", BAL_SOURCES, "--per-source", 300, "--samples", 8,
           "--adapter", start, "--out", probe, "--dtype", args.dtype, "--generate-only", "--chunk", 150,
           "--max-questions", args.gen_questions)
    if not (keep / "sft.jsonl").exists():
        py("select_shortest.py", "--rollouts", probe / "rollouts.jsonl", "--out", keep)
        py("contamination.py", "--data", keep / "sft.jsonl", "--rollouts", probe / "rollouts.jsonl")
    pairs = BON / f"pairs-{name}"
    if not (pairs / "pairs.jsonl").exists():
        py("bon_pairs.py", "--rollouts", probe / "rollouts.jsonl", "--keep", keep / "sft.jsonl", "--out", pairs,
           "--min-correct", 6, "--guard-median", "gsm_plus,gsm_ic", "--shorten-cap", "gsm_plus=40", "gsm_ic=40")
    print((pairs / "summary.txt").read_text(), flush=True)
    rows = [json.loads(l) for l in (pairs / "pairs.jsonl").read_text().splitlines() if l.strip()]
    counts = [sum(r["kind"] == k for r in rows) for k in ("shorten", "guard", "hard")]
    for seed in args.seeds:
        adapter = Path("adapters") / f"{name}-s{seed}"
        if not (adapter / "adapter_config.json").exists():
            py("dpo.py", "--pairs", pairs / "pairs.jsonl", "--init-adapter", start, "--out", adapter,
               "--mix", ",".join(f"{c / sum(counts):.4f}" for c in counts), "--save-every", args.save_every,
               "--epochs", args.dpo_epochs, "--dtype", args.dtype, "--seed", seed)
    evaluate(ROOT / "s4-base", samples=4)
    scored = []
    for step in args.eval_steps:
        parts = [Path("adapters") / f"{name}-s{seed}" / f"step-{step}" for seed in args.seeds]
        parts = [p for p in parts if (p / "adapter_config.json").exists()]
        if not parts:
            continue
        model = Path("models") / f"{name}-soup-{step}"
        if not (model / "config.json").exists():
            py("merge.py", "--out", model, "--adapters", *[f"{p}:{1 / len(parts):.6f}" for p in parts])
        out = ROOT / f"s4-{name}-soup-{step}"
        evaluate(out, samples=4, model=model)
        line, delta = score_line(f"{name} soup of {len(parts)} seeds, step {step} (4 samples)", out, with_delta=True,
                                 base=ROOT / "s4-base")
        print(line, flush=True)
        scored.append((compare(out, ROOT / "s4-base")["average"]["token_reduction"] or 0, delta, step, model))
    safe = [c for c in scored if c[1] >= LIMIT]
    if not safe:
        return
    _, _, step, model = max(safe)
    evaluate(ROOT / "s8-base", samples=8)
    out = ROOT / f"s8-{name}-soup-{step}"
    evaluate(out, samples=8, model=model)
    print(score_line(f"{name} soup step {step} (usual 240 x 8)", out, base=ROOT / "s8-base"), flush=True)
    questions = f"questions_fresh{FRESH_SEED}"
    if not Path(questions, "questions.jsonl").exists():
        py("subset_questions.py", "--per-dataset", 40, "--seed", FRESH_SEED, "--exclude", QUESTIONS, "--out", questions)
    evaluate(ROOT / "f8-base", samples=8, questions=questions)
    out = ROOT / f"f8-{name}-soup-{step}"
    evaluate(out, samples=8, questions=questions, model=model)
    print(score_line(f"{name} soup step {step} (fresh 240 x 8)", out, base=ROOT / "f8-base"), flush=True)


def chain(args):
    """One round of the MBPP-free line with the full recipe; rounds chain through averaged (soup) models.

    --start: an adapter (first round, e.g. adapters/clean-b2/step-10) or soup:NAME (the chosen soup of an earlier
    round, a full model: this round trains a fresh LoRA on top of it). Data: a new draw (--data-seed) of ToolACE,
    IFEval-like, ARC/CSQA, GSM-Plus and GSM-IC, minus every question drawn before; with --code-source kodcode,
    also KodCode tasks never sampled before. Contamination check at 30% (stricter than the 50% elsewhere).
    Pairs: short-beats-long only at >= 6/8 correct, math capped and protected, KodCode right-beats-wrong only.
    DPO with 3 seeds, soups per checkpoint step, the soup saving most with a 4-sample accuracy change >= --floor
    is chosen (written to runs/mvtf/chain-NAME.json), then scored x8 on the usual and the fresh questions.
    """
    name = args.name
    if args.start.startswith("soup:"):
        base = json.loads((ROOT / f"chain-{args.start[5:]}.json").read_text())["model"]
        start = Path("adapters") / f"fresh-{name}"
        if not (start / "adapter_config.json").exists():
            py("sft.py", "--init-only", "--model", base, "--out", start)
        on = ["--model", base]
        sample_from = ["--model", base]
    else:
        base, start = "Qwen/Qwen3-0.6B", resolve(args.start)
        on = []
        sample_from = ["--adapter", start]
        if args.start == "raw":
            start = Path("adapters") / f"fresh-{name}"
            if not (start / "adapter_config.json").exists():
                py("sft.py", "--init-only", "--out", start)
            sample_from = []
    print(f"{name}: base {base}, start adapter {start}", flush=True)
    data = Path(f"data/chain-seed{args.data_seed}.jsonl")
    if not data.exists():
        raw = Path(f"data/chain-seed{args.data_seed}-raw.jsonl")
        py("prepare_types.py", "--sources", "toolace,ifeval_like,mc,gsm_plus,gsm_ic", "--per-source", 300,
           "--seed", args.data_seed, "--out", raw)
        text = lambda r: r["prompt"][-1]["content"]
        used = [p for p in Path("data").glob("*.jsonl")  # earlier draws (not train.jsonl, the mc pool itself)
                if p.name.startswith(("bon", "bal", "chain")) and "raw" not in p.name and p != data]
        seen = {text(json.loads(l)) for p in used for l in p.read_text().splitlines() if l.strip() and "prompt" in l}
        rows = [json.loads(l) for l in raw.read_text().splitlines() if l.strip()]
        data.write_text("".join(json.dumps(r) + "\n" for r in rows if text(r) not in seen))
    probe, keep = BON / f"probe-{name}", BON / f"select-{name}"
    if not (probe / "rollouts.jsonl").exists():
        py("probe_curve.py", "--data", data, "--sources", BAL_SOURCES, "--per-source", 300, "--samples", 8,
           *sample_from, "--out", probe, "--dtype", args.dtype, "--generate-only", "--chunk", 150,
           "--max-questions", args.gen_questions)
    if not (keep / "sft.jsonl").exists():
        py("select_shortest.py", "--rollouts", probe / "rollouts.jsonl", "--out", keep)
        py("contamination.py", "--data", keep / "sft.jsonl", "--rollouts", probe / "rollouts.jsonl", "--share", 0.3)
    pairs = BON / f"pairs-{name}"
    files = [pairs / "base" / "pairs.jsonl"]
    if not files[0].exists():
        py("bon_pairs.py", "--rollouts", probe / "rollouts.jsonl", "--keep", keep / "sft.jsonl", "--out", pairs / "base",
           "--min-correct", 6, "--guard-median", "gsm_plus,gsm_ic", "--shorten-cap", "gsm_plus=40", "gsm_ic=40")
    if args.code_source == "kodcode":
        kod_file = Path(args.kod_file)
        if not kod_file.exists():  # kodcode2: a wider MBPP-like slice (the 45/90-token one, 907 tasks, runs out)
            py("prepare_types.py", "--sources", "kodcode", "--per-source", 5000, "--out", kod_file,
               "--kod-max-task", 70, "--kod-max-solution", 130)
        kprobe, kkeep = BON / f"kprobe-{name}", BON / f"kselect-{name}"
        if not (kprobe / "rollouts.jsonl").exists():
            # tasks this line already answered are skipped (by question text, across files)
            earlier = [BON / f"kprobe-{n}" / "rollouts.jsonl" for n in args.kod_exclude_runs]
            py("probe_curve.py", "--data", kod_file, "--sources", "kodcode", "--per-source", 5000, "--samples", 8,
               *sample_from, "--out", kprobe, "--dtype", args.dtype, "--generate-only", "--chunk", 150,
               "--max-questions", args.kod_questions, "--exclude-rollouts", *earlier)
        if not (kkeep / "sft.jsonl").exists():
            py("select_shortest.py", "--rollouts", kprobe / "rollouts.jsonl", "--out", kkeep)
            py("contamination.py", "--data", kkeep / "sft.jsonl", "--rollouts", kprobe / "rollouts.jsonl", "--share", 0.3)
        if not (pairs / "kod" / "pairs.jsonl").exists():
            py("bon_pairs.py", "--rollouts", kprobe / "rollouts.jsonl", "--keep", kkeep / "sft.jsonl", "--out", pairs / "kod",
               "--no-shorten", "kodcode")
        files.append(pairs / "kod" / "pairs.jsonl")
    if args.mvt_sources:  # verified MVT cuts (4/4 re-answers right) as extra short-beats-long pairs
        mvt = pairs / "mvt"
        if not (mvt / "pairs.jsonl").exists():
            allowed = {json.loads(l)["id"] for l in (keep / "sft.jsonl").read_text().splitlines() if l.strip()}
            wanted = set(args.mvt_sources.split(","))
            mvt.mkdir(parents=True, exist_ok=True)
            subset = [l for l in (probe / "rollouts.jsonl").read_text().splitlines()
                      if l.strip() and json.loads(l)["source"] in wanted and json.loads(l)["id"] in allowed]
            (mvt / "rollouts.jsonl").write_text("\n".join(subset) + "\n")
            who = ["--model", base] if base != "Qwen/Qwen3-0.6B" else []
            if sample_from[:1] == ["--adapter"]:
                who += sample_from
            py("mvt_pairs.py", *who, "--rollouts", mvt / "rollouts.jsonl", "--out", mvt, "--travel", 1024,
               "--check-samples", 4, "--per-question", 1, "--strict-sources", "", "--dtype", args.dtype)
            kept = [l for l in (mvt / "pairs.jsonl").read_text().splitlines() if l.strip() and json.loads(l)["kind"] != "hard"]
            (mvt / "pairs.jsonl").write_text("\n".join(kept) + "\n")
        files.append(mvt / "pairs.jsonl")
    for f in files:
        print((f.parent / "summary.txt").read_text(), flush=True)
    rows = [json.loads(l) for f in files for l in f.read_text().splitlines() if l.strip()]
    (pairs / "pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    counts = [sum(r["kind"] == k for r in rows) for k in ("shorten", "guard", "hard")]
    for seed in args.seeds:
        adapter = Path("adapters") / f"{name}-s{seed}"
        if not (adapter / "adapter_config.json").exists():
            py("dpo.py", *on, "--pairs", pairs / "pairs.jsonl", "--init-adapter", start, "--out", adapter,
               "--mix", ",".join(f"{c / sum(counts):.4f}" for c in counts), "--save-every", args.save_every,
               "--epochs", args.dpo_epochs, "--dtype", args.dtype, "--seed", seed)
    evaluate(ROOT / "s4-base", samples=4)
    scored = []
    for step in args.eval_steps:
        parts = [Path("adapters") / f"{name}-s{seed}" / f"step-{step}" for seed in args.seeds]
        parts = [p for p in parts if (p / "adapter_config.json").exists()]
        if not parts:
            continue
        model = Path("models") / f"{name}-soup-{step}"
        if not (model / "config.json").exists():
            py("merge.py", "--model", base, "--out", model, "--adapters", *[f"{p}:{1 / len(parts):.6f}" for p in parts])
        out = ROOT / f"s4-{name}-soup-{step}"
        evaluate(out, samples=4, model=model)
        line, delta = score_line(f"{name} soup step {step} (4 samples)", out, with_delta=True, base=ROOT / "s4-base")
        print(line, flush=True)
        scored.append((compare(out, ROOT / "s4-base")["average"]["token_reduction"] or 0, delta, step, str(model)))
    ok = [c for c in scored if c[1] >= args.floor] or sorted(scored, key=lambda c: -c[1])[:1]
    saved, delta, step, model = max(ok)
    (ROOT / f"chain-{name}.json").write_text(json.dumps({"model": model, "step": step, "s4": [saved, delta]}))
    print(f"{name}: chosen soup step {step} ({saved:+.1%} / {delta:+.1f} at 4 samples)", flush=True)
    evaluate(ROOT / "s8-base", samples=8)
    out = ROOT / f"s8-{name}-soup-{step}"
    evaluate(out, samples=8, model=model)
    print(score_line(f"{name} soup step {step} (usual 240 x 8)", out, base=ROOT / "s8-base"), flush=True)
    questions = f"questions_fresh{FRESH_SEED}"
    if not Path(questions, "questions.jsonl").exists():
        py("subset_questions.py", "--per-dataset", 40, "--seed", FRESH_SEED, "--exclude", QUESTIONS, "--out", questions)
    evaluate(ROOT / "f8-base", samples=8, questions=questions)
    out = ROOT / f"f8-{name}-soup-{step}"
    evaluate(out, samples=8, questions=questions, model=model)
    print(score_line(f"{name} soup step {step} (fresh 240 x 8)", out, base=ROOT / "f8-base"), flush=True)


def repair(args):
    """Win back accuracy on a 60%+ model (rounds 4 and 5) without shortening it further.

    1. Blends (no training): --start mixed with round 2 (adapters/bon-dpo-r2g/step-10) at 75/25 and 50/50 in weight
       space; each blend scored x8 on the fresh questions.
    2. Length-matched right-beats-wrong: --start answers trap math (GSM-Plus, GSM-IC; a new draw) 8 times; pairs of a
       right and a wrong reply of about the same length (within 25%), so DPO learns correctness and nothing about
       length; one pass, checkpoints 10 and 20.
    4. Self-correction (RFT): SFT of --start on its own right replies to the same questions that are no longer than
       its typical reply; gentle (lr 2e-5, one pass).
    Every candidate is scored with 4 samples on the usual questions and x8 on the fresh ones.
    """
    name, start = args.name, resolve(args.start)
    questions = f"questions_fresh{FRESH_SEED}"
    evaluate(ROOT / "f8-base", samples=8, questions=questions)
    evaluate(ROOT / "s4-base", samples=4)

    def judge(label, adapter=None, model=None):
        tag = f"{name}-{label}"
        evaluate(ROOT / f"s4-{tag}", adapter, samples=4, model=model)
        print(score_line(f"{tag} (usual x4)", ROOT / f"s4-{tag}", base=ROOT / "s4-base"), flush=True)
        evaluate(ROOT / f"f8-{tag}", adapter, samples=8, questions=questions, model=model)
        print(score_line(f"{tag} (fresh x8)", ROOT / f"f8-{tag}", base=ROOT / "f8-base"), flush=True)

    for weight in (0.75, 0.5):  # 1. blends
        model = Path("models") / f"{name}-blend{int(weight * 100)}"
        if not (model / "config.json").exists():
            py("merge.py", "--out", model, "--adapters", f"{start}:{weight}", f"adapters/bon-dpo-r2g/step-10:{1 - weight}")
        judge(f"blend{int(weight * 100)}", model=model)
    data = Path(f"data/trapmath-seed{args.data_seed}.jsonl")  # trap math, not drawn before
    if not data.exists():
        raw = Path(f"data/trapmath-seed{args.data_seed}-raw.jsonl")
        py("prepare_types.py", "--sources", "gsm_plus,gsm_ic", "--per-source", 300, "--seed", args.data_seed, "--out", raw)
        text = lambda r: r["prompt"][-1]["content"]
        used = [p for p in Path("data").glob("*.jsonl") if p.name.startswith(("bal", "chain", "trapmath")) and "raw" not in p.name and p != data]
        seen = {text(json.loads(l)) for p in used for l in p.read_text().splitlines() if l.strip()}
        data.write_text("".join(l + "\n" for l in raw.read_text().splitlines() if l.strip() and text(json.loads(l)) not in seen))
    probe, keep = BON / f"probe-{name}", BON / f"select-{name}"
    if not (probe / "rollouts.jsonl").exists():
        py("probe_curve.py", "--data", data, "--sources", "gsm_plus,gsm_ic", "--per-source", 300, "--samples", 8,
           "--adapter", start, "--out", probe, "--dtype", args.dtype, "--generate-only", "--chunk", 150)
    if not (keep / "sft.jsonl").exists():
        py("select_shortest.py", "--rollouts", probe / "rollouts.jsonl", "--out", keep)
        py("contamination.py", "--data", keep / "sft.jsonl", "--rollouts", probe / "rollouts.jsonl", "--share", 0.3)
    pairs = BON / f"pairs-{name}-matched"  # 2. length-matched right-beats-wrong
    if not (pairs / "pairs.jsonl").exists():
        py("bon_pairs.py", "--rollouts", probe / "rollouts.jsonl", "--keep", keep / "sft.jsonl", "--out", pairs,
           "--length-match", 1.25)
    print((pairs / "summary.txt").read_text(), flush=True)
    adapter = Path("adapters") / f"{name}-matched"
    if not (adapter / "adapter_config.json").exists():
        py("dpo.py", "--pairs", pairs / "pairs.jsonl", "--init-adapter", start, "--out", adapter, "--mix", "0,1,0",
           "--save-every", 10, "--epochs", 1, "--dtype", args.dtype)
    for step in (10, 20):
        if (adapter / f"step-{step}" / "adapter_config.json").exists():
            judge(f"matched-step-{step}", adapter=adapter / f"step-{step}")
    rft = BON / f"rft-{name}"  # 4. self-correction: its own right replies, no longer than its typical one
    if not (rft / "sft.jsonl").exists():
        by_question = defaultdict(list)
        for l in (probe / "rollouts.jsonl").read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                by_question[r["id"]].append(r)
        allowed = {json.loads(l)["id"] for l in (keep / "sft.jsonl").read_text().splitlines() if l.strip()}
        rows = []
        for qid, group in by_question.items():
            median = sorted(r["output_tokens"] for r in group)[len(group) // 2]
            right = [r for r in group if r["correct"] and not r["truncated"] and r["output_tokens"] <= median]
            if qid in allowed and right:
                r = min(right, key=lambda r: abs(r["output_tokens"] - median))
                rows.append({"id": qid, "source": r["source"], "prompt_ids": r["prompt_ids"], "completion_ids": r["output_ids"]})
        rft.mkdir(parents=True, exist_ok=True)
        (rft / "sft.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"rft: {len(rows)} rows", flush=True)
    adapter = Path("adapters") / f"{name}-rft"
    if not (adapter / "adapter_config.json").exists():
        py("sft.py", "--data", rft / "sft.jsonl", "--init-adapter", start, "--out", adapter, "--epochs", 1, "--lr", 2e-5,
           "--dtype", args.dtype)
    judge("rft", adapter=adapter)


VAL_SEED = 20261023  # the choosing set: 40 random questions per part, in neither questions_40 nor the fresh set


def rlcal(args):
    """Calibrated cuts, then RL, on top of a chosen soup (--start soup:NAME), every choice made on a separate set.

    1. The start model answers a new draw (--data-seed: ToolACE, IFEval-like, ARC/CSQA, GSM-Plus, GSM-IC, and
       KodCode) 8 times; contamination check at 30%.
    2. Calibrated cuts (cal_cuts.py): its right replies are cut at 20-80% of the thinking, "</think>" closed there,
       re-answered 4x; per source, the earliest cut that keeps the asked accuracy (math 0.5 point, code 1, others 2)
       becomes the "good" example: cut beats full ("shorten") and cut beats an earlier failing cut ("hard"); plus
       the recipe's right-beats-wrong pairs (KodCode right-beats-wrong only). DPO, 3 seeds, soups; the soup saving
       most within --floor on the CHOOSING set (seed 20261023, x4) is kept.
    3. RL from that soup (rl_train.py): reward = right x (1 - alpha_q * sigmoid(z)), z = thinking length among the
       group's right replies, alpha_q = alpha x solve rate (hard questions keep their thinking), wrong = 0, thinking
       tokens only, batch-normalized advantages, KL to the soup; questions the start model solved >= 4/8.
       Checkpoints every 20 steps, chosen on the choosing set.
    Finalists are reported x8 on the usual and the fresh (unseen) questions.
    """
    name = args.name
    if args.start.startswith("soup:"):
        base = json.loads((ROOT / f"chain-{args.start[5:]}.json").read_text())["model"]
    else:  # "raw" (from scratch) or a full model directory, e.g. models/rep4-blend50
        base = "Qwen/Qwen3-0.6B" if args.start == "raw" else args.start
    small = args.smoke
    data = Path(f"data/rlcal-seed{args.data_seed}.jsonl")
    if not data.exists():
        raw = Path(f"data/rlcal-seed{args.data_seed}-raw.jsonl")
        py("prepare_types.py", "--sources", "toolace,ifeval_like,mc,gsm_plus,gsm_ic", "--per-source", 300,
           "--seed", args.data_seed, "--out", raw)
        text = lambda r: r["prompt"][-1]["content"]
        used = [p for p in Path("data").glob("*.jsonl")
                if p.name.startswith(("bon", "bal", "chain", "trapmath", "rlcal", "rl-")) and "raw" not in p.name and p != data]
        seen = {text(json.loads(l)) for p in used for l in p.read_text().splitlines() if l.strip() and "prompt" in l}
        data.write_text("".join(l + "\n" for l in raw.read_text().splitlines() if l.strip() and text(json.loads(l)) not in seen))
    kod_file = Path(args.kod_file)
    if not kod_file.exists():
        py("prepare_types.py", "--sources", "kodcode", "--per-source", 5000, "--out", kod_file,
           "--kod-max-task", 70, "--kod-max-solution", 130)
    probe, keep = BON / f"probe-{name}", BON / f"select-{name}"
    kprobe, kkeep = BON / f"kprobe-{name}", BON / f"kselect-{name}"
    skip_probe = args.rl_only and args.rl_fresh_seed  # RL on a fresh draw needs no answers of the round's own draw
    if not skip_probe and not (probe / "rollouts.jsonl").exists():
        py("probe_curve.py", "--data", data, "--sources", BAL_SOURCES, "--per-source", 300, "--samples", 8,
           "--model", base, "--out", probe, "--dtype", args.dtype, "--generate-only", "--chunk", 150,
           "--max-questions", 24 if small else args.gen_questions)
    if not skip_probe and not (kprobe / "rollouts.jsonl").exists():
        earlier = [p for p in BON.glob("kprobe-*/rollouts.jsonl") if p.parent != kprobe]
        py("probe_curve.py", "--data", kod_file, "--sources", "kodcode", "--per-source", 5000, "--samples", 8,
           "--model", base, "--out", kprobe, "--dtype", args.dtype, "--generate-only", "--chunk", 150,
           "--max-questions", 8 if small else args.kod_questions, "--exclude-rollouts", *earlier)
    for p, k in ((probe, keep), (kprobe, kkeep)):
        if not skip_probe and not (k / "sft.jsonl").exists():
            py("select_shortest.py", "--rollouts", p / "rollouts.jsonl", "--out", k)
            py("contamination.py", "--data", k / "sft.jsonl", "--rollouts", p / "rollouts.jsonl", "--share", 0.3)
    val = f"questions_val{VAL_SEED}"
    fresh_q = f"questions_fresh{FRESH_SEED}"
    if not Path(fresh_q, "questions.jsonl").exists():
        py("subset_questions.py", "--per-dataset", 40, "--seed", FRESH_SEED, "--exclude", QUESTIONS, "--out", fresh_q)
    if not Path(val, "questions.jsonl").exists():
        py("subset_questions.py", "--per-dataset", 40, "--seed", VAL_SEED, "--exclude", QUESTIONS, fresh_q, "--out", val)

    def choose(label, candidates, samples=4):
        """candidates: [(tag, model path)] -> the one saving most within --floor on the choosing set."""
        evaluate(ROOT / f"v{samples}-base", samples=samples, questions=val)
        scored = []
        for tag, model in candidates:
            out = ROOT / f"v{samples}-{tag}"
            evaluate(out, samples=samples, questions=val, model=model)
            line, delta = score_line(f"{tag} (choosing set x{samples})", out, with_delta=True, base=ROOT / f"v{samples}-base")
            print(line, flush=True)
            scored.append((compare(out, ROOT / f"v{samples}-base")["average"]["token_reduction"] or 0, delta, tag, str(model)))
        ok = [c for c in scored if c[1] >= args.floor] or sorted(scored, key=lambda c: -c[1])[:1]
        saved, delta, tag, model = max(ok)
        print(f"{label}: chosen {tag} ({saved:+.1%} / {delta:+.1f} on the choosing set x{samples})", flush=True)
        return tag, model

    def report(tag, model):
        evaluate(ROOT / "s8-base", samples=8)
        evaluate(ROOT / f"s8-{tag}", samples=8, model=model)
        print(score_line(f"{tag} (usual 240 x 8)", ROOT / f"s8-{tag}", base=ROOT / "s8-base"), flush=True)
        evaluate(ROOT / "f8-base", samples=8, questions=fresh_q)
        evaluate(ROOT / f"f8-{tag}", samples=8, questions=fresh_q, model=model)
        print(score_line(f"{tag} (fresh 240 x 8)", ROOT / f"f8-{tag}", base=ROOT / "f8-base"), flush=True)

    def judge_fresh(label, candidates):
        """Every candidate straight on the fresh set x8; the most saving within --floor (else the most accurate)."""
        evaluate(ROOT / "f8-base", samples=8, questions=fresh_q)
        scored = []
        for tag, model in candidates:
            out = ROOT / f"f8-{tag}"
            evaluate(out, samples=8, questions=fresh_q, model=model)
            line, delta = score_line(f"{tag} (fresh 240 x 8)", out, with_delta=True, base=ROOT / "f8-base")
            print(line, flush=True)
            scored.append((compare(out, ROOT / "f8-base")["average"]["token_reduction"] or 0, delta, tag, str(model)))
        ok = [c for c in scored if c[1] >= args.floor] or sorted(scored, key=lambda c: -c[1])[:1]
        saved, delta, tag, model = max(ok)
        print(f"{label}: best on the fresh set {tag} ({saved:+.1%} / {delta:+.1f})", flush=True)
        return tag, model

    if args.judge == "fresh":
        choose = lambda label, candidates, samples=4: judge_fresh(label, candidates)
        report = lambda tag, model: None

    if args.rl_only:
        cal_tag, cal_model = f"{name}-start", base
        if not small and args.judge != "fresh":  # the start model on the choosing set: the reference for every block
            evaluate(ROOT / "v2-base", samples=2, questions=val)
            evaluate(ROOT / f"v2-{name}-start", samples=2, questions=val, model=base)
            print(score_line(f"{name}-start = {args.start} (choosing set x2, reference)", ROOT / f"v2-{name}-start",
                             base=ROOT / "v2-base"), flush=True)
        return rl_blocks(args, name, base, cal_tag, cal_model, data, kod_file, (probe, kprobe), (keep, kkeep),
                         choose, report)
    pairs = BON / f"pairs-{name}"
    parts = {"cal": pairs / "cal" / "pairs.jsonl", "base": pairs / "base" / "pairs.jsonl", "kod": pairs / "kod" / "pairs.jsonl"}
    if not parts["cal"].exists():
        py("cal_cuts.py", "--model", base, "--rollouts", probe / "rollouts.jsonl", kprobe / "rollouts.jsonl",
           "--keep", keep / "sft.jsonl", kkeep / "sft.jsonl", "--out", pairs / "cal", "--dtype", args.dtype,
           *(["--limit", 20] if small else []))
    if not parts["base"].exists():  # the recipe's pairs, guard-heavy (shorten pairs come from the calibrated cuts)
        py("bon_pairs.py", "--rollouts", probe / "rollouts.jsonl", "--keep", keep / "sft.jsonl", "--out", pairs / "base",
           "--min-correct", 6, "--guard-median", "gsm_plus,gsm_ic", "--shorten-cap", "gsm_plus=0", "gsm_ic=0")
    if not parts["kod"].exists():
        py("bon_pairs.py", "--rollouts", kprobe / "rollouts.jsonl", "--keep", kkeep / "sft.jsonl", "--out", pairs / "kod",
           "--no-shorten", "kodcode")
    for f in parts.values():
        print((f.parent / "summary.txt").read_text(), flush=True)
    rows = [json.loads(l) for f in parts.values() for l in f.read_text().splitlines() if l.strip()]
    (pairs / "pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    counts = [sum(r["kind"] == k for r in rows) for k in ("shorten", "guard", "hard")]
    print(f"{name}: pairs shorten {counts[0]}, guard {counts[1]}, hard {counts[2]}", flush=True)
    start = Path("adapters") / f"fresh-{name}"
    if not (start / "adapter_config.json").exists():
        py("sft.py", "--init-only", "--model", base, "--out", start)
    seeds = args.seeds[:1] if small else args.seeds
    for seed in seeds:
        adapter = Path("adapters") / f"{name}-s{seed}"
        if not (adapter / "adapter_config.json").exists():
            py("dpo.py", "--model", base, "--pairs", pairs / "pairs.jsonl", "--init-adapter", start, "--out", adapter,
               "--mix", ",".join(f"{c / max(1, sum(counts)):.4f}" for c in counts), "--save-every", 2 if small else 10,
               "--epochs", 1, "--dtype", args.dtype, "--seed", seed)
    soups = []
    for step in ([2] if small else args.dpo_steps):
        members = [Path("adapters") / f"{name}-s{seed}" / f"step-{step}" for seed in seeds]
        members = [m for m in members if (m / "adapter_config.json").exists()]
        if not members and small:  # a smoke run may end before its first checkpoint: its final adapter, or the
            members = [Path("adapters") / f"{name}-s{seed}" for seed in seeds]  # untrained start when it had no pairs
            members = [m for m in members if (m / "adapter_config.json").exists()] or [start]
        if not members:
            continue
        model = Path("models") / f"{name}-soup-{step}"
        if not (model / "config.json").exists():
            py("merge.py", "--model", base, "--out", model, "--adapters", *[f"{m}:{1 / len(members):.6f}" for m in members])
        soups.append((f"{name}-soup-{step}", model))
    if small:
        print(f"{name}: smoke: cal-cut DPO and soup ran", flush=True)
        cal_tag, cal_model = soups[0]
    elif args.judge == "fresh":
        cal_tag, cal_model = choose(f"{name} cal-cut DPO", soups)
        (ROOT / f"chain-{name}-cal.json").write_text(json.dumps({"model": cal_model, "tag": cal_tag}))
    else:
        # the start model on the same choosing set: the reference every later step is compared with
        evaluate(ROOT / "v4-base", samples=4, questions=val)
        evaluate(ROOT / f"v4-{name}-start", samples=4, questions=val, model=base)
        print(score_line(f"{name}-start = {args.start} (choosing set x4, reference)", ROOT / f"v4-{name}-start",
                         base=ROOT / "v4-base"), flush=True)
        cal_tag, cal_model = choose(f"{name} cal-cut DPO", soups)
        (ROOT / f"chain-{name}-cal.json").write_text(json.dumps({"model": cal_model, "tag": cal_tag}))
        report(cal_tag, cal_model)
    if args.rl_steps == 0:  # pairs only, no RL
        return
    rl_blocks(args, name, base, cal_tag, cal_model, data, kod_file, (probe, kprobe), (keep, kkeep), choose, report)


def calcheck(args):
    """Calibration re-check: --calcheck-model re-answers (20x) from the very same cuts of the same replies as
    --calcheck-from's calibration (ca1's replies), and its own confidence is read at each cut: the confidence-vs-real
    tables of the two models compare like for like."""
    src = BON / f"pairs-{args.calcheck_from}" / "cal"
    out = BON / f"calcheck-{args.name}"
    if not (out / "summary.txt").exists():
        frm = args.calcheck_from
        py("cal_cuts.py", "--model", args.calcheck_model, "--rollouts", BON / f"probe-{frm}" / "rollouts.jsonl",
           BON / f"kprobe-{frm}" / "rollouts.jsonl", "--keep", BON / f"select-{frm}" / "sft.jsonl",
           BON / f"kselect-{frm}" / "sft.jsonl", "--out", out, "--dtype", args.dtype)
    print(f"calibration of {args.calcheck_model} on {args.calcheck_from}'s cuts:", flush=True)
    print((out / "summary.txt").read_text(), flush=True)
    print(f"for comparison, {args.calcheck_from}'s start model on the same cuts:", flush=True)
    print((src / "summary.txt").read_text(), flush=True)


def callot(args):
    """A labelled-cut lot for the stopping probe: --start answers a new draw with --lot questions per type x8; its
    right replies (contamination-checked) are cut at 1-80% of the thinking and re-answered 20x (cal_cuts.py). No
    training. Outputs as rlcal names them (probe-, kprobe-, select-, kselect-, pairs-NAME/cal)."""
    name, base = args.name, args.start
    counts = dict((k, int(v)) for k, v in (x.split("=") for x in args.lot.split(",")))
    data = Path(f"data/lot-{name}.jsonl")
    text = lambda r: r["prompt"][-1]["content"]
    if not data.exists():
        rows = []
        for i, (source, n) in enumerate(c for c in counts.items() if c[0] != "kodcode"):
            raw = Path(f"data/lot-{name}-{source}-raw.jsonl")
            py("prepare_types.py", "--sources", source, "--per-source", n, "--seed", args.data_seed * 10 + i, "--out", raw)
            rows += [l for l in raw.read_text().splitlines() if l.strip()]
        used = [p for p in Path("data").glob("*.jsonl") if p.name.startswith(("bon", "bal", "chain", "trapmath", "rlcal", "rl-", "lot-"))
                and "raw" not in p.name and p != data]
        seen = {text(r) for p in used for l in p.read_text().splitlines() if l.strip()
                for r in [json.loads(l)] if isinstance(r.get("prompt"), list) and r["prompt"]}  # skips non-chat rows
        kept = [l for l in rows if text(json.loads(l)) not in seen]
        data.write_text("".join(l + "\n" for l in kept))
        print(f"{name}: {len(kept)} new questions of {len(rows)} drawn (earlier draws removed)", flush=True)
    kod_file = Path(args.kod_file)
    if counts.get("kodcode") and not kod_file.exists():  # a wider KodCode slice than kodcode2 (longer, harder tasks)
        py("prepare_types.py", "--sources", "kodcode", "--per-source", 10000, "--out", kod_file,
           "--kod-max-task", 110, "--kod-max-solution", 200)
    probe, keep = BON / f"probe-{name}", BON / f"select-{name}"
    kprobe, kkeep = BON / f"kprobe-{name}", BON / f"kselect-{name}"
    n_data = sum(1 for l in data.read_text().splitlines() if l.strip())
    if not (probe / "rollouts.jsonl").exists():
        drawn = ",".join("arc_easy,commonsense_qa" if s == "mc" else s for s in counts if s != "kodcode")
        py("probe_curve.py", "--data", data, "--sources", drawn, "--per-source", 2000, "--samples", args.lot_samples,
           "--model", base, "--out", probe, "--dtype", args.dtype, "--generate-only", "--chunk", 150,
           "--max-questions", n_data)
    if counts.get("kodcode") and not (kprobe / "rollouts.jsonl").exists():
        earlier = [p for p in BON.glob("kprobe-*/rollouts.jsonl") if p.parent != kprobe]
        try:
            py("probe_curve.py", "--data", kod_file, "--sources", "kodcode", "--per-source", 5000, "--samples", args.lot_samples,
               "--model", base, "--out", kprobe, "--dtype", args.dtype, "--generate-only", "--chunk", 150,
               "--max-questions", counts["kodcode"], "--exclude-rollouts", *earlier)
        except subprocess.CalledProcessError:  # every KodCode task of the file was answered by an earlier run
            if args.paragraphs:  # label lots may reuse training tasks; only the probes' test set (rc2) stays out
                print(f"{name}: no unused KodCode task left; reusing earlier ones except rc2's", flush=True)
                py("probe_curve.py", "--data", kod_file, "--sources", "kodcode", "--per-source", 5000, "--samples", args.lot_samples,
                   "--model", base, "--out", kprobe, "--dtype", args.dtype, "--generate-only", "--chunk", 150,
                   "--max-questions", counts["kodcode"], "--exclude-rollouts", BON / "kprobe-rc2" / "rollouts.jsonl")
            else:
                print(f"{name}: no unused KodCode task left; this lot has no code questions", flush=True)
    has_kod = (kprobe / "rollouts.jsonl").exists() and (kprobe / "rollouts.jsonl").stat().st_size > 0
    pairs = [(probe, keep)] + ([(kprobe, kkeep)] if counts.get("kodcode") and has_kod else [])
    if args.paragraphs:  # every question, solved or not: the clean-question list is checked directly
        cal = BON / f"pairs-{name}" / "cal"
        if not (cal / "summary.txt").exists():
            clean = BON / f"{name}-clean.jsonl"
            if not clean.exists():
                rows = {}
                for p, _ in pairs:
                    for l in (p / "rollouts.jsonl").read_text().splitlines():
                        if l.strip():
                            r = json.loads(l)
                            rows.setdefault(r["id"], {"id": r["id"], "source": r["source"]})
                clean.write_text("".join(json.dumps(r) + "\n" for r in rows.values()))
                for p, _ in pairs:
                    py("contamination.py", "--data", clean, "--rollouts", p / "rollouts.jsonl", "--share", 0.3)
            py("label_cuts.py", "--model", base, "--rollouts", *[p / "rollouts.jsonl" for p, _ in pairs],
               "--keep", clean, "--out", cal, "--include-wrong", "--dtype", args.dtype,
               "--max-cuts", args.lot_cuts, "--samples", args.lot_checks)
        print((cal / "summary.txt").read_text(), flush=True)
        return
    for p, k in pairs:
        if not (k / "sft.jsonl").exists():
            py("select_shortest.py", "--rollouts", p / "rollouts.jsonl", "--out", k)
            py("contamination.py", "--data", k / "sft.jsonl", "--rollouts", p / "rollouts.jsonl", "--share", 0.3)
    cal = BON / f"pairs-{name}" / "cal"
    if not (cal / "summary.txt").exists():
        py("cal_cuts.py", "--model", base, "--rollouts", *[p / "rollouts.jsonl" for p, _ in pairs],
           "--keep", *[k / "sft.jsonl" for _, k in pairs], "--out", cal, "--dtype", args.dtype)
    print((cal / "summary.txt").read_text(), flush=True)


def cutexamples(args):
    """Readable examples of cuts, one question per data type, from rc2's cuts (the model's own replies)."""
    out = ROOT.parent / "viz" / "cut_examples.txt"
    py("cut_examples.py", BON / "pairs-rc2" / "cal", "--out", out, "--per-type", 1)


def ensure_lots(lots):
    """Restore label lots the other runtime finished after this runtime's start-up restore (a lot is usable only once its
    verdicts.jsonl exists; an earlier light restore may have copied a half-made folder)."""
    missing = [l for l in lots if not (BON / f"pairs-{l}" / "cal" / "verdicts.jsonl").exists()]
    if missing:
        drive_sync("restore", only=[pat for l in missing for pat in (f"runs/bon/pairs-{l}", f"runs/bon/probe-{l}",
                                    f"runs/bon/kprobe-{l}", f"runs/bon/{l}-clean.jsonl", f"runs/bon/select-{l}",
                                    f"runs/bon/kselect-{l}")])


def probezoo(args):
    """Every stopping classifier (layer sweep, linear, MLP, attention over the last 64 thinking tokens, baselines) on
    the same labelled cuts, compared on the same untouched test set (probe_zoo.py). Hidden states are read once per set
    (extract_feats.py) into /content/feats, not synced to Drive (several GB)."""
    frm, test_from = args.calcheck_from, args.probe_test_from
    done = BON / f"zoo-{args.name}" / "summary.txt"
    if not done.exists():
        drive_sync("restore", only=[f"runs/bon/zoo-{args.name}"])  # the other runtime may have run this zoo already
    if done.exists():
        print(done.read_text(), flush=True)
        return
    ensure_lots([args.probe_cal.split("/")[0][len("pairs-"):]] + list(args.probe_extra_from or [])
                + ([test_from] if test_from else []))
    needed = [args.probe_cal.split("/")[0]] + [f"pairs-{r}" for r in (args.probe_extra_from or [])] \
        + ([f"pairs-{test_from}"] if test_from else [])
    missing = [n for n in needed if not (BON / n).exists()]
    if missing or not Path(args.probe_model).exists() and args.probe_model.startswith("models/"):
        drive_sync("restore", only=[f"runs/bon/{n}" for n in missing] + ([args.probe_model] if args.probe_model.startswith("models/") else []))
    sets = [BON / args.probe_cal] + [BON / f"pairs-{r}" / "cal" for r in (args.probe_extra_from or [])] \
        + ([BON / f"pairs-{test_from}" / "cal"] if test_from else [])
    for cal in sets:
        py("own_conf.py", "--model", args.probe_model, "--cal", cal, "--dtype", args.dtype)  # skips when up to date
        name = cal.name if cal.name != "cal" else cal.parent.name
        py("extract_feats.py", "--model", args.probe_model, "--cal", cal, "--out", Path("/content/feats") / name,
           "--dtype", args.dtype)
    out = BON / f"zoo-{args.name}"
    if not (out / "summary.txt").exists():
        py("probe_zoo.py", "--cal", BON / args.probe_cal,
           "--rollouts", BON / f"probe-{frm}" / "rollouts.jsonl", BON / f"kprobe-{frm}" / "rollouts.jsonl",
           "--keep", BON / f"select-{frm}" / "sft.jsonl", BON / f"kselect-{frm}" / "sft.jsonl",
           *(["--extra-runs", *args.probe_extra_from] if args.probe_extra_from else []),
           *(["--test-cal", BON / f"pairs-{test_from}" / "cal",
              "--test-rollouts", BON / f"probe-{test_from}" / "rollouts.jsonl", BON / f"kprobe-{test_from}" / "rollouts.jsonl",
              "--test-keep", BON / f"select-{test_from}" / "sft.jsonl", BON / f"kselect-{test_from}" / "sft.jsonl"]
             if test_from else []),
           *(["--drop-sources", args.drop_sources] if args.drop_sources else []),
           "--feats-root", "/content/feats", "--out", out)
    print((out / "summary.txt").read_text(), flush=True)


def benchlot(args):
    """Labelled cuts on BENCHMARK questions (the 240 already used to choose models, never the fresh set), only to
    measure the stopping judge there (bench_rows.py, then label_cuts.py as a callot lot). Nothing trains on them."""
    name, base = args.name, args.start
    if base.startswith("models/") and not (Path(base) / "config.json").exists():  # queued after this runtime's restore
        drive_sync("restore", only=[base])
    data = Path(f"data/{name}.jsonl")
    if not data.exists():
        py("bench_rows.py", "--questions", QUESTIONS, "--out", data)
    probe, cal = BON / f"probe-{name}", BON / f"pairs-{name}" / "cal"
    n_data = sum(1 for l in data.read_text().splitlines() if l.strip())
    if not (probe / "rollouts.jsonl").exists():
        py("probe_curve.py", "--data", data, "--sources", "gsm_plus,kodcode,toolace,ifeval_like", "--per-source", 5000,
           "--samples", args.lot_samples, "--model", base, "--out", probe, "--dtype", args.dtype, "--generate-only",
           "--chunk", 150, "--max-questions", n_data)
    if not (cal / "summary.txt").exists():
        clean = BON / f"{name}-clean.jsonl"
        ids = {json.loads(l)["id"]: json.loads(l)["source"] for l in (probe / "rollouts.jsonl").read_text().splitlines() if l.strip()}
        clean.write_text("".join(json.dumps({"id": i, "source": s}) + "\n" for i, s in ids.items()))
        py("label_cuts.py", "--model", base, "--rollouts", probe / "rollouts.jsonl", "--keep", clean, "--out", cal,
           "--include-wrong", "--dtype", args.dtype)
    print((cal / "summary.txt").read_text(), flush=True)


def pool_mix(args):
    """The RL pool's shares with --pool-math as its math (34% split evenly); the other types as mixed_pool's default."""
    math = [m for m in args.pool_math.split(",") if m]
    if "gsm8k_matched" in math and not Path("data/gsm8k_matched-train.jsonl").exists():  # built by mathmatch, maybe elsewhere
        drive_sync("restore", only=["data/gsm8k_matched-train.jsonl"])
    return {**{m: 0.34 / len(math) for m in math}, "ifeval_like": 0.24, "toolace": 0.2, "hermes_tools": 0.1}


def mixed_pool(name, need, data_seed, kod_file, mix=None, kod_share=0.16, reuse=False, exclude=()):
    """A fresh question pool mixed like the evaluation (GSM-IC/Plus, IFEval-like, ToolACE, Hermes, MBPP-train,
    KodCode), shuffled so every batch mixes the types; earlier draws removed, contamination-checked."""
    import random
    pool = Path(f"data/rl-{name}-mixed.jsonl")
    if not pool.exists():
        raw = Path(f"data/rl-{name}-raw.jsonl")
        per = mix or {"gsm_ic": 0.17, "gsm_plus": 0.17, "ifeval_like": 0.24, "toolace": 0.2, "hermes_tools": 0.1}  # no MBPP (user)
        rows = []
        for i, (source, share) in enumerate(per.items()):
            part = Path(f"data/rl-{name}-{source}-raw.jsonl")
            py("prepare_types.py", "--sources", source, "--per-source", int(need * share * 1.8) + 50,
               "--seed", data_seed * 10 + i, "--out", part)
            rows += [json.loads(l) for l in part.read_text().splitlines() if l.strip()]
        kod_file = Path(kod_file)
        if not kod_file.exists():
            py("prepare_types.py", "--sources", "kodcode", "--per-source", 10000, "--out", kod_file,
               "--kod-max-task", 110, "--kod-max-solution", 200)
        text = lambda r: r["prompt"][-1]["content"]
        # reuse: a run from the RAW model may draw questions earlier experiments used (it never trained on them; the
        # evaluation contamination check below still applies); otherwise earlier draws are removed so continued
        # training sees new questions
        answered = set() if reuse else {json.loads(l).get("question") for p in BON.glob("kprobe-*/rollouts.jsonl")
                    for l in p.read_text().splitlines() if l.strip()}
        kod = [json.loads(l) for l in kod_file.read_text().splitlines() if l.strip()]
        random.Random(data_seed).shuffle(kod)
        rows += [r for r in kod if text(r) not in answered][:int(need * kod_share) + 50]  # code share (default incl. the former MBPP 6%)
        used = [p for p in Path("data").glob("*.jsonl") if p.name.startswith(("bon", "bal", "chain", "trapmath", "rlcal",
                "rl-", "lot-")) and "raw" not in p.name and p != pool]
        seen = {text(r) for p in used for l in p.read_text().splitlines() if l.strip()
                for r in [json.loads(l)] if isinstance(r.get("prompt"), list) and r["prompt"]}
        rows = [r for r in rows if reuse or text(r) not in seen]
        if exclude:  # a continued run: never the questions the run it continues was trained on
            trained = {text(json.loads(l)) for e in exclude for l in Path(f"data/rl-{e}-mixed.jsonl").read_text().splitlines()
                       if l.strip()}
            rows = [r for r in rows if text(r) not in trained]
        random.Random(data_seed).shuffle(rows)
        for index, row in enumerate(rows):
            row["id"] = f"{row['source']}-{name}-{index}"
        pool.write_text("".join(json.dumps(r) + "\n" for r in rows))
        questions = Path(f"data/rl-{name}-text.jsonl")
        questions.write_text("".join(json.dumps({"id": r["id"], "question": text(r)}) + "\n" for r in rows))
        py("contamination.py", "--data", pool, "--rollouts", questions, "--share", 0.3)
        kept = [l for l in pool.read_text().splitlines() if l.strip()]
        counts = defaultdict(int)
        for l in kept:
            counts[json.loads(l)["source"]] += 1
        print(f"{name}: mixed pool {len(kept)} questions (need {need}): {dict(counts)}", flush=True)
    return pool


def drpo(args):
    """DRPO RL (rl_train.py --objective drpo) from --start on a fresh pool mixed like the evaluation (math, instructions,
    tool calls, code), shuffled so every step mixes the types, each question used once (contamination-checked, earlier
    draws removed). Checkpoints every 5 steps; the --drpo-score ones are merged and scored on the fresh set x8."""
    import random
    name, base = args.name, args.start
    if base.startswith("models/") and not (Path(base) / "config.json").exists():  # queued after this runtime's restore
        drive_sync("restore", only=[base])
    need = math.ceil(args.drpo_steps * args.drpo_questions * max(1.0, args.rl_dynamic))
    if not Path(f"data/rl-{name}-mixed.jsonl").exists():  # built on the other runtime: reuse the same pool
        drive_sync("restore", only=[f"data/rl-{name}-mixed.jsonl", f"data/rl-{name}-text.jsonl"])
    pool = mixed_pool(name, need, args.data_seed, args.kod_file, mix=pool_mix(args),
                      reuse=not base.startswith("models/") or bool(args.pool_continues), exclude=args.pool_continues)
    out = Path("adapters") / f"{name}-{args.rl_variant}"
    if not (out / "adapter_config.json").exists():  # trained on the other runtime: reuse it, never retrain
        drive_sync("restore", only=[str(out)])
    variant = (["--objective", "drpo", "--drpo-lambda", args.drpo_lambda,
                *(["--drpo-solve-scale"] if args.drpo_solve_scale else [])] if args.rl_variant == "drpo"
               else ["--objective", "grpo", "--reward", "shorterbetter",
                     *(["--sb-beta", args.rl_sb_beta] if args.rl_sb_beta is not None else [])])
    if not (out / "adapter_config.json").exists() and not args.drpo_score_only:
        py("rl_train.py", "--model", base, "--data", pool, "--sources", "all", "--out", out,
           "--steps", args.drpo_steps, "--questions", args.drpo_questions, "--group", 8, "--max-tokens", 4096,
           *variant, "--updates", 2, "--grader", "typed",
           *(["--protect-sources", args.rl_protect] if args.rl_protect else []),
           *(["--guard-drop", args.rl_guard] if args.rl_guard else []), "--dynamic-sampling", args.rl_dynamic,
           "--redundant", "none", "--lr", args.rl_lr, "--save-every", 5, "--dtype", args.dtype)
    log_file = out / "rl_log.jsonl"
    if log_file.exists():
        print(log_file.read_text(), flush=True)
    fresh_q = f"questions_fresh{FRESH_SEED}"
    evaluate(ROOT / "f8-base", samples=8, questions=fresh_q)
    ran = sum(1 for l in log_file.read_text().splitlines() if l.strip()) if log_file.exists() else 0
    steps = list(args.drpo_score) + ([ran] if args.drpo_score and ran and ran not in args.drpo_score else [])  # guard stop
    for step in steps:
        ckpt = out / f"step-{step}"
        if not (ckpt / "adapter_config.json").exists():
            continue
        model = Path("models") / f"{name}-{step}"
        if not (model / "config.json").exists():
            py("merge.py", "--model", base, "--out", model, "--adapters", f"{ckpt}:1")
        res = ROOT / f"f8-{name}-{step}"
        evaluate(res, samples=8, questions=fresh_q, model=model)
        print(score_line(f"{name}-{step} (fresh 240 x 8)", res, base=ROOT / "f8-base"), flush=True)
    for spec in args.drpo_weighted:  # one checkpoint at reduced strength (user: halfway between the start and the step)
        step, weight = spec.split(":")
        ckpt = out / f"step-{step}"
        if not (ckpt / "adapter_config.json").exists():
            continue
        tag = f"{step}x{float(weight):g}"
        model = Path("models") / f"{name}-{tag}"
        if not (model / "config.json").exists():
            py("merge.py", "--model", base, "--out", model, "--adapters", f"{ckpt}:{float(weight)}")
        res = ROOT / f"f8-{name}-{tag}"
        evaluate(res, samples=8, questions=fresh_q, model=model)
        print(score_line(f"{name}-{tag} (fresh 240 x 8)", res, base=ROOT / "f8-base"), flush=True)
    members = [out / f"step-{s}" for s in args.drpo_soup if (out / f"step-{s}" / "adapter_config.json").exists()]
    if len(members) >= 2:  # checkpoint soup along the RL trajectory (user: chosen after the single checkpoints' scores)
        tag = "soup-" + "-".join(str(s) for s in args.drpo_soup)
        model = Path("models") / f"{name}-{tag}"
        if not (model / "config.json").exists():
            py("merge.py", "--model", base, "--out", model, "--adapters", *[f"{m}:{1 / len(members):.6f}" for m in members])
        res = ROOT / f"f8-{name}-{tag}"
        evaluate(res, samples=8, questions=fresh_q, model=model)
        print(score_line(f"{name}-{tag} (fresh 240 x 8)", res, base=ROOT / "f8-base"), flush=True)


def nothink(args):
    """How much thinking adds at this scale (user, measure only): each model answers the fresh set x8 with thinking OFF
    (the chat template's enable_thinking=False: an empty think block, the answer comes directly), compared with the same
    model thinking. Uses a copy of the official evaluate.py with that one switch flipped; never a submission setting."""
    src, dst = Path("evaluate.py"), Path("evaluate_nothink.py")
    text = src.read_text()
    assert text.count("enable_thinking=True") == 1, "evaluate.py changed: the thinking switch is not where expected"
    dst.write_text(text.replace("enable_thinking=True", "enable_thinking=False"))
    fresh_q = f"questions_fresh{FRESH_SEED}"
    if args.nt_full:  # the original only, full test x8 (blog: thinking-off point on every part's efficiency curve)
        base = ROOT / "full-base"
        if not (base / "results.json").exists():
            drive_sync("restore", only=[f"runs/mvtf/{base.name}"])
        out = ROOT / "full-base-nothink"
        if not (out / "results.json").exists():
            drive_sync("restore", only=[f"runs/mvtf/{out.name}"])
        if not (out / "results.json").exists():
            shutil.rmtree(out, ignore_errors=True)
            py("evaluate_nothink.py", "--questions", "questions", "--samples", 8, "--out", out)
        print(raw_diff("original with thinking OFF vs thinking ON (full test 2,759 x 8)", out, base), flush=True)
        return
    evaluate(ROOT / "f8-base", samples=8, questions=fresh_q)
    for m in args.nothink_models:
        if m.startswith("models/") and not (Path(m) / "config.json").exists():
            drive_sync("restore", only=[m])
        tag = "base" if m == BASE_MODEL_ID else Path(m).name
        think = ROOT / ("f8-base" if tag == "base" else f"f8-{tag}")
        if not (think / "results.json").exists():
            drive_sync("restore", only=[f"runs/mvtf/{think.name}"])  # scored earlier on the other runtime?
        if not (think / "results.json").exists():
            evaluate(think, samples=8, questions=fresh_q, model=None if tag == "base" else m)
        out = ROOT / f"f8-{tag}-nothink"
        if not (out / "results.json").exists():
            drive_sync("restore", only=[f"runs/mvtf/{out.name}"])  # scored on the other runtime?
        if not (out / "results.json").exists():
            shutil.rmtree(out, ignore_errors=True)
            py("evaluate_nothink.py", "--questions", fresh_q, "--samples", 8, "--out", out,
               *([] if tag == "base" else ["--model", m]))
        print(raw_diff(f"{tag} with thinking OFF vs the original thinking (fresh 240 x 8)", out, ROOT / "f8-base"),
              flush=True)
        if tag != "base":
            print(raw_diff(f"{tag} with thinking OFF vs {tag} thinking (fresh 240 x 8)", out, think), flush=True)


def raw_diff(name, out, base):
    """Per-benchmark accuracy and mean reply tokens of two scored runs, read from their records (compare.py refuses
    two runs of the same model, which is exactly the thinking ON vs OFF case)."""
    def load(d):
        acc, tok = defaultdict(list), defaultdict(list)
        for l in (Path(d) / "records.jsonl").read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                acc[r["dataset"]].append(float(bool(r.get("correct"))))
                tok[r["dataset"]].append(r["output_tokens"])
        return acc, tok
    (a1, t1), (a0, t0) = load(out), load(base)
    lines = [f"== {name}"]
    for ds in sorted(a0):
        m = lambda v: sum(v) / max(1, len(v))
        lines.append(f"   {ds:<20} accuracy {100 * m(a0[ds]):5.1f} -> {100 * m(a1.get(ds, [])):5.1f} "
                     f"({100 * (m(a1.get(ds, [])) - m(a0[ds])):+5.1f}); tokens {m(t0[ds]):6.0f} -> {m(t1.get(ds, [])):6.0f} "
                     f"({1 - m(t1.get(ds, [])) / max(1e-9, m(t0[ds])):+.0%} saved)")
    return "\n".join(lines)


def op_kind(expr):
    """Arithmetic kind of one calculation, named like SVAMP's Type field."""
    for sym, kind in (("/", "Common-Division"), ("*", "Multiplication"), ("-", "Subtraction"), ("+", "Addition")):
        if sym in expr:
            return kind
    return "other"


def mathmatch(args):
    """SVAMP-like practice math without cheating (user). (1) Profile the 40 SVAMP questions of the CHOOSING set (measure
    only): number of operations and kind (SVAMP's own Equation/Type fields), length, and the original Qwen's accuracy and
    thinking length from benchlot's 8 replies. (2) Pick a matched corpus from GSM8K (train + test, 1-2 calculations; never
    ASDiv/MAWPS, never SVAMP), stratified on the same operation count x kind and on length, in the evaluation's numeric
    prompt format; earlier draws removed; contamination-checked. (3) Hold out --match-test of it as a MATH TEST set and have
    the original Qwen answer it x8, to compare accuracy and thinking with SVAMP before using the rest for training."""
    import random
    import re
    import statistics
    from datasets import load_dataset
    from utils.train.prepare_data import row
    numeric = ("\n\nThe last line of your reply must start with 'Final answer:' followed by the number alone.")
    words = lambda t: len(t.split())
    close_id = 151668  # Qwen3's "</think>"; generate-only rollouts keep output_ids, not a thinking count

    def think_len(r):
        if "thinking_tokens" in r:
            return r["thinking_tokens"]
        ids = r.get("output_ids") or []
        return ids.index(close_id) if close_id in ids else r.get("output_tokens", len(ids))
    # (1) the choosing set's SVAMP questions and how the original model does on them
    ensure_lots(["bench40"])
    qs = [json.loads(l) for l in Path(QUESTIONS, "questions.jsonl").read_text().splitlines() if l.strip()]
    sv = [q for q in qs if q["dataset"] == "svamp"]
    raw = {f"svamp-{t['ID']}": t for t in json.loads(Path("questions/raw/svamp.json").read_text())}
    bench = [json.loads(l) for l in Path("data/bench40.jsonl").read_text().splitlines() if l.strip()]
    order = [r["id"] for r in bench]  # probe_curve numbers rows by their line in the WHOLE file (source-index)
    rolls = defaultdict(list)
    for l in (BON / "probe-bench40" / "rollouts.jsonl").read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            if r["source"] == "gsm_plus":  # SVAMP rows carry the practice name gsm_plus (bench_rows.py)
                rolls[order[int(r["id"].rsplit("-", 1)[1])]].append(r)
    prof = []
    for q in sv:
        t = raw[q["id"]]
        ops = re.findall(r"[-+*/]", t["Equation"])
        rs = rolls.get(q["id"], [])
        prof.append({"ops": min(2, max(1, len(ops))), "kind": t["Type"].replace("Divison", "Division"),
                     "words": words(q["messages"][0]["content"]), "acc": statistics.mean(r["correct"] for r in rs) if rs else None,
                     "think": statistics.mean(think_len(r) for r in rs) if rs else None})

    def table(rows, title):
        lines = [title, f"  {'ops':>3} {'kind':<16} {'n':>4} {'accuracy':>9} {'thinking':>9} {'words':>6}"]
        cells = defaultdict(list)
        for r in rows:
            cells[(r["ops"], r["kind"])].append(r)
        for key in sorted(cells) + [("all", "")]:
            rs = rows if key[0] == "all" else cells[key]
            acc = [r["acc"] for r in rs if r["acc"] is not None]
            th = [r["think"] for r in rs if r["think"] is not None]
            lines.append(f"  {key[0]:>3} {key[1]:<16} {len(rs):>4} {statistics.mean(acc) if acc else float('nan'):>9.2f} "
                         f"{statistics.mean(th) if th else float('nan'):>9.0f} {statistics.mean(r['words'] for r in rs):>6.0f}")
        return "\n".join(lines)
    print(table(prof, "SVAMP, choosing set (40 questions, original Qwen x8; measure only):"), flush=True)
    # (2) the matched GSM8K corpus
    out_train, out_test = Path("data/gsm8k_matched-train.jsonl"), Path("data/gsm8k_matched-test.jsonl")
    if not out_test.exists():
        text = lambda r: r["prompt"][-1]["content"]
        used = [p for p in Path("data").glob("*.jsonl") if p.name.startswith(("bon", "bal", "chain", "trapmath", "rlcal",
                "rl-", "lot-")) and "raw" not in p.name]
        seen = {text(r).split("\n\n")[0] for p in used for l in p.read_text().splitlines() if l.strip()
                for r in [json.loads(l)] if isinstance(r.get("prompt"), list) and r["prompt"]}
        pool = defaultdict(list)
        for split in ("train", "test"):
            for r in load_dataset("openai/gsm8k", "main", split=split):
                calcs = re.findall(r"<<([^=]*)=", r["answer"])
                if not 1 <= len(calcs) <= 2 or r["question"] in seen:
                    continue
                kind = op_kind(calcs[-1])
                if kind == "other":
                    continue
                answer = r["answer"].split("####")[-1].strip().replace(",", "")
                pool[(len(calcs), kind)].append(row(r["question"] + numeric, "", answer, "math", "gsm8k_matched")
                                                 | {"words": words(r["question"]), "ops": len(calcs), "kind": kind})
        for rows in pool.values():
            for r in rows:
                r["prompt"][0]["content"] = r["prompt"][0]["content"].rstrip()  # row() appends an empty instruction
        rng = random.Random(args.data_seed)
        target = defaultdict(int)
        for r in prof:
            target[(r["ops"], r["kind"])] += 1
        lo, hi = sorted(r["words"] for r in prof)[len(prof) // 10], sorted(r["words"] for r in prof)[-(len(prof) // 10) - 1]
        picked = []
        for key, n in target.items():
            cand = [r for r in pool.get(key, []) if lo * 0.8 <= r["words"] <= hi * 1.2] or pool.get(key, [])
            rng.shuffle(cand)
            picked += cand[:round(args.match_size * n / len(prof))]
        rest = [r for k, rs in pool.items() for r in rs if r not in picked]
        rng.shuffle(rest)
        picked += rest[:max(0, args.match_size - len(picked))]  # cells GSM8K cannot fill (few one-step problems) topped up
        rng.shuffle(picked)
        for i, r in enumerate(picked):
            r["id"] = f"gsm8k_matched-{i}"
        test, train = picked[:args.match_test], picked[args.match_test:]
        out_test.write_text("".join(json.dumps(r) + "\n" for r in test))
        out_train.write_text("".join(json.dumps(r) + "\n" for r in train))
        both = Path("data/gsm8k_matched-text.jsonl")
        both.write_text("".join(json.dumps({"id": r["id"], "question": text(r)}) + "\n" for r in picked))
        py("contamination.py", "--data", out_train, "--rollouts", both, "--share", 0.3)
        py("contamination.py", "--data", out_test, "--rollouts", both, "--share", 0.3)
    train = [json.loads(l) for l in out_train.read_text().splitlines() if l.strip()]
    test = [json.loads(l) for l in out_test.read_text().splitlines() if l.strip()]
    print(f"matched GSM8K corpus: {len(train)} train + {len(test)} held-out math test (after contamination check)",
          flush=True)
    # (3) the original model on the held-out math test, x8
    probe = BON / "probe-mathmatch"
    if not (probe / "rollouts.jsonl").exists():
        py("probe_curve.py", "--data", out_test, "--sources", "gsm8k_matched", "--per-source", 100000, "--samples", 8,
           "--model", "Qwen/Qwen3-0.6B", "--out", probe, "--dtype", args.dtype, "--generate-only", "--chunk", 150,
           "--max-questions", len(test))
    meta = {r["id"]: r for r in test}
    by_q = defaultdict(list)
    for l in (probe / "rollouts.jsonl").read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            by_q[r["id"]].append(r)
    keys = list(meta)
    rows = []
    for i, k in enumerate(keys):  # probe_curve renames rows source-index in file order
        rs = by_q.get(f"gsm8k_matched-{i}", [])
        q = meta[k]
        rows.append({"ops": q.get("ops", 0), "kind": q.get("kind", ""), "words": q["words"],
                     "acc": statistics.mean(r["correct"] for r in rs) if rs else None,
                     "think": statistics.mean(think_len(r) for r in rs) if rs else None})
    print(table(rows, "Matched GSM8K, held-out math test (original Qwen x8):"), flush=True)
    print(table([{"ops": r["ops"], "kind": r["kind"], "words": r["words"], "acc": None, "think": None} for r in train],
                "Matched GSM8K, training part (composition only):"), flush=True)


def teach(args):
    """Experiment 5, the judge as teacher (teach_stop.py): the stopping judge is fitted on --teach-lots of --start, the
    model answers a fresh mixed pool once per question, the judge scores every paragraph end of those answers, and a
    LoRA trains the model's own p("</think>") there toward the sharpened target (one model per --teach-variants).
    Each model is merged and scored on the fresh set x8."""
    name, base = args.name, args.start
    if base.startswith("models/") and not (Path(base) / "config.json").exists():
        drive_sync("restore", only=[base])
    missing = [l for l in args.teach_lots if not (BON / f"pairs-{l}" / "cal" / "verdicts.jsonl").exists()]
    if missing:
        drive_sync("restore", only=[f"runs/bon/pairs-{l}" for l in missing])
    out = Path("adapters") / f"{name}-teach"
    if args.teach_reuse:  # same answers and own-confidence readings as an earlier teach run: only the teachers differ
        probe = BON / f"probe-{args.teach_reuse}"
        if not (probe / "rollouts.jsonl").exists():
            drive_sync("restore", only=[f"runs/bon/probe-{args.teach_reuse}", f"adapters/{args.teach_reuse}-teach/own"])
        earlier = Path("adapters") / f"{args.teach_reuse}-teach" / "own"
        if (earlier / "own_confidence.jsonl").exists() and not (out / "own" / "own_confidence.jsonl").exists():
            shutil.copytree(earlier, out / "own", dirs_exist_ok=True)
    else:
        pool = mixed_pool(name, args.teach_questions, args.data_seed, args.kod_file, mix=pool_mix(args),
                          reuse=not base.startswith("models/"))
        probe = BON / f"probe-{name}"
        if not (probe / "rollouts.jsonl").exists():
            n = sum(1 for l in pool.read_text().splitlines() if l.strip())
            py("probe_curve.py", "--data", pool, "--per-source", 100000, "--samples", 1, "--model", base, "--out", probe,
               "--dtype", args.dtype, "--generate-only", "--chunk", 300, "--max-questions", min(n, args.teach_questions))
    teachers = ([*(["--own-sources", args.teach_own] if args.teach_own is not None else []),
                 *(["--judge-sources", args.teach_judge] if args.teach_judge is not None else [])])
    py("teach_stop.py", "--model", base, "--cal", *[BON / f"pairs-{l}" / "cal" for l in args.teach_lots],
       "--rollouts", probe / "rollouts.jsonl", "--out", out, "--variants", *args.teach_variants, "--dtype", args.dtype,
       *teachers)
    print((out / "teacher.json").read_text(), flush=True)
    fresh_q = f"questions_fresh{FRESH_SEED}"
    evaluate(ROOT / "f8-base", samples=8, questions=fresh_q)
    for variant in args.teach_variants:
        kind, value = variant.split(":")
        tag = f"{kind}{float(value):g}"
        adapter = out / tag
        model = Path("models") / f"{name}-{tag}"
        if not (model / "config.json").exists():
            py("merge.py", "--model", base, "--out", model, "--adapters", f"{adapter}:1")
        res = ROOT / f"f8-{name}-{tag}"
        evaluate(res, samples=8, questions=fresh_q, model=model)
        print(score_line(f"{name}-{tag} (fresh 240 x 8)", res, base=ROOT / "f8-base"), flush=True)


def budgetprompt(args):
    """Does a token budget in the system prompt shorten the ORIGINAL model's thinking, at what cost, and how often does
    the model talk about the budget instead of the problem (user, measure only; a system prompt changes the evaluation, so
    this is never a submission setting: at most a training-time prompt to distil and remove). A copy of evaluate.py adds
    one sentence per benchmark to the system message (appended to an existing one, e.g. BFCL's function list), then the
    fresh set x8 is compared with the original, and the thinking text is scanned for budget talk."""
    budgets = dict((k, int(v)) for k, v in (x.split("=") for x in args.bp_budgets.split(",")))
    src, dst = Path("evaluate.py"), Path("evaluate_budget.py")
    text = src.read_text()
    call = 'tokenizer.apply_chat_template(row["messages"], tokenize=True,'
    assert text.count(call) == 1, "evaluate.py changed: the chat-template call is not where expected"
    helper = (f"BUDGETS = {budgets!r}\n"
              "def budget_messages(row):\n"
              "    key = next((k for k in BUDGETS if row['dataset'].startswith(k)), None)\n"
              "    if key is None:\n"
              "        return row['messages']\n"
              "    note = f'Think briefly: keep your thinking under {BUDGETS[key]} tokens.'\n"
              "    msgs = [dict(m) for m in row['messages']]\n"
              "    if msgs and msgs[0]['role'] == 'system':\n"
              "        msgs[0]['content'] = msgs[0]['content'].rstrip() + '\\n\\n' + note\n"
              "    else:\n"
              "        msgs.insert(0, {'role': 'system', 'content': note})\n"
              "    return msgs\n\n\n")
    text = text.replace(call, 'tokenizer.apply_chat_template(budget_messages(row), tokenize=True,')
    text = text.replace("def build_jobs(", helper + "def build_jobs(", 1)
    dst.write_text(text)
    fresh_q = f"questions_fresh{FRESH_SEED}"
    evaluate(ROOT / "f8-base", samples=8, questions=fresh_q)
    out = ROOT / f"f8-{args.name}"
    if not (out / "results.json").exists():
        shutil.rmtree(out, ignore_errors=True)
        py("evaluate_budget.py", "--questions", fresh_q, "--samples", 8, "--out", out)
    print(raw_diff(f"{args.name}: original Qwen with a thinking budget in the system prompt vs without (fresh 240 x 8)",
                   out, ROOT / "f8-base"), flush=True)
    talk = re.compile(r"\b(tokens?|budget|limit|brief(ly)?|quick(ly)?|concise|short(er)?)\b", re.I)
    def scan(run):  # answers, answers mentioning the words, mentions, thinking tokens; per dataset
        stats = defaultdict(lambda: [0, 0, 0, 0])
        for line in (run / "records.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            thinking = r["text"][:max(0, len(r["text"]) - len(r["response"]))]
            hits = len(talk.findall(thinking))
            st = stats[r["dataset"]]
            st[0] += 1; st[1] += hits > 0; st[2] += hits; st[3] += r["reasoning_tokens"]
        return stats
    with_b, without = scan(out), scan(ROOT / "f8-base")  # the same words occur in ordinary thinking too: compare
    lines = [f"{d}: budget words in the thinking of {st[1] / st[0]:.0%} of answers ({st[2] / st[0]:.1f} per answer) vs "
             f"{without[d][1] / max(1, without[d][0]):.0%} ({without[d][2] / max(1, without[d][0]):.1f}) without the prompt; "
             f"thinking {st[3] / st[0]:.0f} vs {without[d][3] / max(1, without[d][0]):.0f} tokens; budget "
             f"{next((v for k, v in budgets.items() if d.startswith(k)), '-')}" for d, st in sorted(with_b.items())]
    (out / "budget_talk.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def dropmodels(args):
    """Free Drive space (user, 27 Oct): remove merged models that can be rebuilt from their LoRA adapters (merge.py, about
    a minute each; recipes in EXPERIMENT_LOG.md, 27 Oct "Drive cleanup"). A model goes only if its adapter folder is on
    Drive; results, adapters, lots and rollouts are never touched. Files removed through the Drive mount go to the Drive
    trash (recoverable for 30 days)."""
    freed, kept = 0, []
    for pair in args.drop:
        model, adapter = pair.split("=")
        target, backup = DRIVE / "models" / model, DRIVE / "adapters" / adapter
        if not target.exists():  # still clear a local copy: the next save would put it back (drop2's soups came back)
            shutil.rmtree(Path("models") / model, ignore_errors=True)
            print(f"dropmodels: {model}: not on Drive", flush=True)
            continue
        if not backup.exists() or not any(backup.iterdir()):
            kept.append(model)
            print(f"dropmodels: {model}: KEPT, adapters/{adapter} not found on Drive", flush=True)
            continue
        size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
        shutil.rmtree(target)
        shutil.rmtree(Path("models") / model, ignore_errors=True)  # a local copy would be re-uploaded by the next save
        freed += size
        print(f"dropmodels: {model}: removed ({size / 1e9:.2f} GB; rebuild from adapters/{adapter})", flush=True)
    print(f"dropmodels: freed {freed / 1e9:.1f} GB; kept for lack of adapters: {', '.join(kept) or 'none'}", flush=True)


def drivereport(args):
    """What fills Drive (user, 28 Oct: free ~40 GB): size of each top folder of the project copy, then every merged model
    with its size, date and whether an adapter folder of the same run is kept (so it can be rebuilt). Read only."""
    def size(path):
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    for top in sorted(p for p in DRIVE.iterdir() if p.is_dir()):
        print(f"drivereport: folder {top.name}: {size(top) / 1e9:.1f} GB", flush=True)
    adapters = sorted(p.name for p in (DRIVE / "adapters").iterdir() if p.is_dir())
    print("drivereport: adapters " + " ".join(adapters), flush=True)
    import time
    for m in sorted((DRIVE / "models").iterdir(), key=lambda p: p.stat().st_mtime):
        if m.is_dir():
            run = m.name.split("-soup-")[0].rsplit("-", 1)[0] if "-soup-" not in m.name else m.name.split("-soup-")[0]
            near = [a for a in adapters if a.startswith(run)]
            print(f"drivereport: model {m.name} {size(m) / 1e9:.2f} GB {time.strftime('%Y-%m-%d', time.localtime(m.stat().st_mtime))} "
                  f"adapters: {','.join(near) or 'NONE'}", flush=True)
    # (6 Nov, user: free ~50 GB, list first) every sub-folder of runs/bon, runs/mvtf, data, models, adapters over 100 MB,
    # with its last change, then every single file over 300 MB; read only
    for top in ("runs/bon", "runs/mvtf", "runs", "data", "models", "adapters"):
        base = DRIVE / top
        if not base.is_dir():
            continue
        subs = [(size(p) if p.is_dir() else p.stat().st_size, p) for p in base.iterdir() if p.name not in ("bon", "mvtf")]
        for sz, p in sorted(subs, key=lambda t: -t[0]):
            if sz >= 1e8:
                print(f"drivereport: sub {top}/{p.name} {sz / 1e9:.2f} GB "
                      f"{time.strftime('%Y-%m-%d', time.localtime(p.stat().st_mtime))}", flush=True)
    big = sorted(((f.stat().st_size, f) for f in DRIVE.rglob("*") if f.is_file() and f.stat().st_size >= 3e8), key=lambda t: -t[0])
    for sz, f in big:
        print(f"drivereport: file {f.relative_to(DRIVE)} {sz / 1e9:.2f} GB "
              f"{time.strftime('%Y-%m-%d', time.localtime(f.stat().st_mtime))}", flush=True)


def drivetidy(args):
    """(6 Nov, user: free ~90 GB on Drive, list first) What can go, by category, with exact paths and sizes. --dt-apply moves
    them (deleting on the mounted Drive sends files to Drive's TRASH: reversible ~30 days; the user empties it). Never
    touched: any results folder (runs/mvtf), any adapter's final weights, any checkpoint that was ever scored, anything
    written in the last --dt-keep-days days, and every path named in a plan line."""
    import re
    import time
    def size(path):
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.is_dir() else path.stat().st_size
    now = time.time()
    recent = lambda p: now - p.stat().st_mtime < args.dt_keep_days * 86400  # noqa: E731
    plans = " ".join(Path(f).read_text() for f in ("plan_line_a.txt", "plan_line_b.txt") if Path(f).exists())
    results = [p.name for p in (DRIVE / ROOT).iterdir() if p.is_dir()] if (DRIVE / ROOT).is_dir() else []
    cats = defaultdict(list)
    # A: merged full models whose adapters are kept (rebuilt by merge.py in a minute)
    for m in sorted((DRIVE / "models").iterdir()) if (DRIVE / "models").is_dir() else []:
        if m.is_dir() and not recent(m) and m.name not in plans:
            cats["A merged full models (adapters kept)"].append(m)
    # B: dead-end experiment folders
    for d in ("runs/sparse",):
        if (DRIVE / d).is_dir():
            cats["B Wanda pruning masks (direction ended 30 Oct)"].append(DRIVE / d)
    # C: adapter checkpoints never scored (finals and scored steps kept)
    for a in sorted((DRIVE / "adapters").iterdir()) if (DRIVE / "adapters").is_dir() else []:
        if not a.is_dir() or recent(a) or a.name in plans:
            continue
        base = re.sub(r"-(rl|drpo|teach|shorterbetter|matched|raw)$", "", re.sub(r"-s\d+$", "", a.name))
        scored = {int(m.group(1)) for r in results for m in [re.search(rf"{re.escape(base)}(?:-s\d+|-soup|-[a-z]+)?-(\d+)$", r)] if m}
        for ck in sorted(a.glob("step-*")):
            n = int(ck.name.split("-")[1]) if ck.name.split("-")[1].isdigit() else -1
            if n not in scored:
                cats["C adapter checkpoints never scored (final + scored steps kept)"].append(ck)
        gone = {str(p) for p in cats["C adapter checkpoints never scored (final + scored steps kept)"]}
        for f in a.rglob("resume.pt"):
            if str(f.parent) not in gone:
                cats["D optimizer resume states of finished RL runs"].append(f)
    # E: probe answers of older pool candidates and rejected pools (results logged; dp9n-all kept: tool top-ups read it)
    bon = DRIVE / "runs/bon"
    for d in sorted(bon.iterdir()) if bon.is_dir() else []:
        if recent(d) or d.name in plans:
            continue
        if re.match(r"probe-poolcand-dp([3-9]|1[01])(c|n)?(-all|-a|-b)?$", d.name) and d.name != "probe-poolcand-dp9n-all":
            cats["E answers to older pool candidates (pools dp3-dp11)"].append(d)
        elif re.match(r"poolup-dp1[245]$", d.name) or re.match(r"probe-poolmore-dp(8|9|10|11)-[ab]$", d.name):
            cats["E rejected / superseded pools (dp12, dp14, dp15) and their top-up probes"].append(d)
        elif d.is_file() and re.match(r"lupin-all-.*-a4\.jsonl$", d.name):
            cats["F combined answer files of finished runs (rebuilt automatically when needed)"].append(d)
    for f in bon.rglob("rollouts.jsonl.partial") if bon.is_dir() else []:
        if (f.parent / "rollouts.jsonl").exists() and not recent(f):
            cats["G leftover .partial copies next to a finished rollouts.jsonl"].append(f)
    total = 0
    for cat in sorted(cats):
        items = sorted(((size(p), p) for p in cats[cat]), key=lambda t: -t[0])
        sub = sum(sz for sz, _ in items)
        total += sub
        print(f"drivetidy: CATEGORY {cat}: {len(items)} items, {sub / 1e9:.1f} GB", flush=True)
        for sz, p in items[:args.dt_show]:
            print(f"drivetidy:   {p.relative_to(DRIVE)} {sz / 1e9:.2f} GB", flush=True)
        if len(items) > args.dt_show:
            print(f"drivetidy:   ... and {len(items) - args.dt_show} more", flush=True)
    print(f"drivetidy: TOTAL {total / 1e9:.1f} GB" + (" -> moving to Drive trash" if args.dt_apply else " (dry run, nothing moved)"),
          flush=True)
    if args.dt_apply:
        for cat in cats:
            for p in cats[cat]:
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                elif p.exists():
                    p.unlink()
        print("drivetidy: done", flush=True)


def stopgold(args):
    """Stop-token training with the lots' labels as teacher (stop_gold.py): at every labelled paragraph end of the lots'
    own replies, p("</think>") is trained toward 1 where the cut's forced answers are as right as the full thinking and 0
    where they are not; nothing else is trained. A checkpoint every --sg-save-every steps; each one is merged and scored on
    the fresh set x8 (the training dose is chosen by score, as the DPO stop runs all fell off a cliff past ~400 pairs)."""
    name, base = args.name, args.start
    ensure_lots(args.sg_lots)
    out = Path("adapters") / name
    if not (out / "adapter_config.json").exists():
        probes = [BON / f"{kind}-{l}" for l in args.sg_lots for kind in ("probe", "kprobe")]
        py("stop_gold.py", "--model", base, "--lots", *[BON / f"pairs-{l}" for l in args.sg_lots], "--probes", *probes,
           "--out", out, "--save-every", args.sg_save_every, "--dtype", args.dtype, *args.sg_extra.split())
    print((out / "labels.txt").read_text(), flush=True)
    fresh_q = f"questions_fresh{FRESH_SEED}"
    evaluate(ROOT / "f8-base", samples=8, questions=fresh_q)
    for ckpt in sorted(out.glob("step-*"), key=lambda p: int(p.name.split("-")[1])):
        model = Path("models") / f"{name}-{ckpt.name}"
        if not (model / "config.json").exists():
            py("merge.py", "--model", base, "--out", model, "--adapters", f"{ckpt}:1")
        res = ROOT / f"f8-{name}-{ckpt.name}"
        evaluate(res, samples=8, questions=fresh_q, model=model)
        print(score_line(f"{name}-{ckpt.name} (fresh 240 x 8)", res, base=ROOT / "f8-base"), flush=True)


def balround(args):
    """Stage 1 of the clean recipe: one balanced DPO round from the raw model on its own graded answers. A fresh pool
    matched to the four benchmarks with equal weight (GSM8K, simpler and closer to SVAMP than GSM-IC/Plus; KodCode;
    ToolACE + Hermes with no-fit cases; IFEval-like; no MBPP; contamination-checked, earlier draws removed); --start
    answers each question --bal-samples times; balanced_pairs.py: shortest right beats the longest wrong or truncated
    answer (guard) and beats a right answer >= 1.5x longer (shorten), shorten capped at guard per type; DPO with the NLL
    anchor (dpo.py default), --seeds, soups at --dpo-steps scored on the fresh set x8."""
    name, base = args.name, args.start
    mix = dict((k, float(v)) for k, v in (x.split("=") for x in args.bal_mix.split(",")))
    pool = mixed_pool(name, args.bal_questions, args.data_seed, args.kod_file, mix=mix, kod_share=args.bal_kod,
                      reuse=not base.startswith("models/"))
    probe = BON / f"probe-{name}"
    if not (probe / "rollouts.jsonl").exists():
        n = sum(1 for l in pool.read_text().splitlines() if l.strip())
        py("probe_curve.py", "--data", pool, "--per-source", 100000, "--samples", args.bal_samples, "--model", base,
           "--out", probe, "--dtype", args.dtype, "--generate-only", "--chunk", 300,
           "--max-questions", min(n, args.bal_questions))
    pairs = BON / f"pairs-{name}"
    if not (pairs / "pairs.jsonl").exists():
        py("balanced_pairs.py", "--rollouts", probe / "rollouts.jsonl", "--out", pairs,
           *(["--protect", args.bal_protect] if args.bal_protect else []))
    print((pairs / "summary.txt").read_text(), flush=True)
    rows = [json.loads(l) for l in (pairs / "pairs.jsonl").read_text().splitlines() if l.strip()]
    counts = [sum(r["kind"] == k for r in rows) for k in ("shorten", "guard", "hard", "right")]
    start = Path("adapters") / f"fresh-{name}"
    if not (start / "adapter_config.json").exists():
        py("sft.py", "--init-only", "--model", base, "--out", start)
    for seed in args.seeds:
        adapter = Path("adapters") / f"{name}-s{seed}"
        if not (adapter / "adapter_config.json").exists():
            py("dpo.py", "--model", base, "--pairs", pairs / "pairs.jsonl", "--init-adapter", start, "--out", adapter,
               "--mix", ",".join(f"{c / max(1, sum(counts)):.4f}" for c in counts), "--save-every", args.tg_save_every,
               "--epochs", 1, "--dtype", args.dtype, "--seed", seed)
    fresh_q = f"questions_fresh{FRESH_SEED}"
    evaluate(ROOT / "f8-base", samples=8, questions=fresh_q)
    for step in args.dpo_steps:
        members = [Path("adapters") / f"{name}-s{seed}" / f"step-{step}" for seed in args.seeds]
        members = [m for m in members if (m / "adapter_config.json").exists()]
        if not members:
            continue
        model = Path("models") / f"{name}-soup-{step}"
        if not (model / "config.json").exists():
            py("merge.py", "--model", base, "--out", model, "--adapters", *[f"{m}:{1 / len(members):.6f}" for m in members])
        out = ROOT / f"f8-{name}-soup-{step}"
        evaluate(out, samples=8, questions=fresh_q, model=model)
        print(score_line(f"{name}-soup-{step} (fresh 240 x 8)", out, base=ROOT / "f8-base"), flush=True)


def wronglen(args):
    """Spill-over check (CPU, seconds): thinking length of right, wrong and never-solved answers on the fresh set x8,
    every scored model vs the original, read straight from the Drive copy (wrong_lengths.py)."""
    py("wrong_lengths.py", "--root", DRIVE / "runs" / "mvtf", "--out", ROOT / "wrong_lengths.txt")


def buildpool(args):
    """A training pool that behaves like the evaluation, checked before anything trains on it (user, 27 Oct).
    1. Candidates from every imitating source (prepare_types.py, the parts' own prompt formats).
    2. build_pool.py: per evaluation part, prompt lengths matched to the CHOOSING set's bins, BFCL's sub-type sizes,
       math mix --dp-math-mix, 8-word contamination check against every evaluation question (shared templates exempt).
    3. The original model answers --dp-check-per-part pool questions per part x4; pool_check.py compares accuracy, reply
       and thinking length and capped share with its own run on the choosing set (v4-base). The fresh set is never read.
    No training here: the pool is only used once its check reads OK."""
    name = args.name
    val, fresh_q = f"questions_val{VAL_SEED}", f"questions_fresh{FRESH_SEED}"
    if not Path(fresh_q, "questions.jsonl").exists():
        py("subset_questions.py", "--per-dataset", 40, "--seed", FRESH_SEED, "--exclude", QUESTIONS, "--out", fresh_q)
    if not Path(val, "questions.jsonl").exists():
        py("subset_questions.py", "--per-dataset", 40, "--seed", VAL_SEED, "--exclude", QUESTIONS, fresh_q, "--out", val)
    evaluate(ROOT / "v4-base", samples=4, questions=val)  # the original model on the choosing set (kept if it exists)
    if not Path("data/gsm8k_matched-train.jsonl").exists():
        drive_sync("restore", only=["data/gsm8k_matched-train.jsonl"])
    cand, kod = Path(f"data/cand-{name}.jsonl"), Path(f"data/kodcand-{name}.jsonl")
    if not cand.exists():
        py("prepare_types.py", "--sources", "gsm8k_matched,metamath_sv,toolace,ifeval_like", "--per-source", 1000000,
           "--seed", args.data_seed, "--out", cand)
    if not kod.exists():
        py("prepare_types.py", "--sources", "kodcode", "--per-source", 15000, "--seed", args.data_seed, "--out", kod,
           "--kod-max-task", 110, "--kod-max-solution", 200)
    pool = Path(f"data/pool-{name}.jsonl")
    if not pool.exists():
        py("build_pool.py", "--candidates", cand, kod, "--choosing", val, "--all-questions", "questions",
           "--per-benchmark", args.dp_per_benchmark, "--math-mix", args.dp_math_mix, "--seed", args.data_seed,
           "--out", pool)
    check = Path(f"data/poolcheck-{name}.jsonl")
    if not check.exists():
        rows = [json.loads(l) for l in pool.read_text().splitlines() if l.strip()]
        rng = random.Random(args.data_seed)
        rng.shuffle(rows)
        per = Counter()
        keep = [r for r in rows if per.update([r["part"]]) or per[r["part"]] <= args.dp_check_per_part]
        check.write_text("".join(json.dumps(r) + "\n" for r in keep))
    n = sum(1 for l in check.read_text().splitlines() if l.strip())
    probe = BON / f"probe-poolcheck-{name}"
    if not (probe / "rollouts.jsonl").exists() or sum(1 for _ in (probe / "rollouts.jsonl").open()) < 4 * n:
        py("probe_curve.py", "--data", check, "--sources", "gsm8k_matched,metamath_sv,kodcode,toolace,ifeval_like",
           "--per-source", 100000, "--samples", 4, "--model", "Qwen/Qwen3-0.6B", "--out", probe, "--dtype", args.dtype,
           "--generate-only", "--chunk", 150, "--max-questions", n)
    py("pool_check.py", "--rollouts", probe / "rollouts.jsonl", "--reference", ROOT / "v4-base",
       "--out", BON / f"poolcheck-{name}")


def buildpool2(args):
    """dp2: the training pool rebuilt after dp1 failed its check (user, 27 Oct: "a big representative dataset across all
    data types first"), and chosen by how the original model BEHAVES on each question, not by prompt length alone.
    dp1's causes, read in its check answers (EXPERIMENT_LOG): ToolACE "int" parameter types fail BFCL's checker on right
    calls, ToolACE references hold values the question never states, undecodable replies were labelled no-fit;
    IFEval-like is mostly easy instruction ids; math is GSM8K-hard; KodCode-Light-10K is too small after the 8-gram check.
    Steps (--dp-step):
      a / b    candidates (prepare_types.py --tool-clean --ifeval-match, gsm8k_easy, KodCode-V1), build_pool.py at
               --dp-oversample x --dp-per-benchmark with the same seed on both runtimes (identical files), then the
               original model answers this runtime's half x4 (the answers are reused as LUPIN step 1's).
      select   waits for the other half on Drive, select_pool.py matches each part's (solve rate x thinking) cells to the
               original model's runs on the choosing set (v4-base) AND questions_40 (eval-base), then the held-out check:
               NEW answers (another sampling seed) to --dp-check-per-part selected questions per part, pool_check.py
               against both references, IFEval per instruction id. The fresh set is never read. No training here."""
    name = args.name
    val, fresh_q = f"questions_val{VAL_SEED}", f"questions_fresh{FRESH_SEED}"
    if not Path(fresh_q, "questions.jsonl").exists():
        py("subset_questions.py", "--per-dataset", 40, "--seed", FRESH_SEED, "--exclude", QUESTIONS, "--out", fresh_q)
    if not Path(val, "questions.jsonl").exists():
        py("subset_questions.py", "--per-dataset", 40, "--seed", VAL_SEED, "--exclude", QUESTIONS, fresh_q, "--out", val)
    cand, kod = Path(f"data/cand-{name}.jsonl"), Path(f"data/kodcand-{name}.jsonl")
    pool = Path(f"data/poolcand-{name}.jsonl")
    # one candidate pool for both runtimes: the first to get here builds it and saves it, the other takes it from Drive
    drive_sync("restore", only=[str(pool), str(cand), str(kod)])
    if not cand.exists() and not pool.exists():
        py("prepare_types.py", "--sources", "gsm8k_easy,gsm8k_matched,metamath_sv,toolace,ifeval_like", "--per-source",
           1000000, "--seed", args.data_seed, "--out", cand, "--tool-clean", "--ifeval-match", args.dp_ifeval)
    if not kod.exists() and not pool.exists():
        py("prepare_types.py", "--sources", "kodcode", "--per-source", args.dp_kod, "--seed", args.data_seed, "--out", kod,
           "--kod-dataset", "KodCode/KodCode-V1", "--kod-max-task", 110, "--kod-max-solution", 200)
    if not pool.exists():
        py("build_pool.py", "--candidates", cand, kod, "--choosing", val, "--all-questions", "questions",
           "--per-benchmark", round(args.dp_per_benchmark * args.dp_oversample), "--math-mix", args.dp_math_mix,
           "--seed", args.data_seed, "--out", pool)
        rows = [json.loads(l) for l in pool.read_text().splitlines() if l.strip()]
        count = Counter()
        for r in rows:  # a stable name per question, kept by probe_curve.py: both halves and the selection agree on it
            r["pool_id"] = f"{r['part']}-{count[r['part']]}"
            count[r["part"]] += 1
        pool.write_text("".join(json.dumps(r) + "\n" for r in rows))
        drive_sync("save")
    import hashlib
    digest = hashlib.sha256(pool.read_bytes()).hexdigest()[:16]
    sources = "gsm8k_easy,gsm8k_matched,metamath_sv,kodcode,toolace,ifeval_like"

    def answer(half, data, seed=None):
        probe = BON / f"probe-poolcand-{name}-{half}"
        n = sum(1 for l in Path(data).read_text().splitlines() if l.strip())
        drive_sync("restore", only=[str(probe)])  # the other runtime may have answered this half already
        if not (probe / "rollouts.jsonl").exists() or sum(1 for _ in (probe / "rollouts.jsonl").open()) < 4 * n:
            py("probe_curve.py", "--data", data, "--sources", sources, "--per-source", 1000000, "--samples", 4,
               "--model", "Qwen/Qwen3-0.6B", "--out", probe, "--dtype", args.dtype, "--generate-only", "--chunk", 150,
               "--max-questions", n, *(["--seed", seed] if seed is not None else []))
        return probe / "rollouts.jsonl"

    if args.dp_step in ("a", "b"):
        half = Path(f"data/poolcand-{name}-{args.dp_step}.jsonl")
        rows = [json.loads(l) for l in pool.read_text().splitlines() if l.strip()]
        keep = [r for r in rows if int(r["pool_id"].rsplit("-", 1)[1]) % 2 == (0 if args.dp_step == "a" else 1)]
        half.write_text("".join(json.dumps(r) + "\n" for r in keep))
        rollouts = answer(args.dp_step, half)
        (rollouts.parent / "poolcand.sha256").write_text(digest)  # select checks both halves came from one pool
        drive_sync("save")
        return
    # select: both halves' answers, then the selection, then the held-out check
    import time
    halves = []
    for h in ("a", "b"):
        target = BON / f"probe-poolcand-{name}-{h}" / "rollouts.jsonl"
        stamp = target.parent / "poolcand.sha256"
        for _ in range(96):  # the other runtime's half: up to 8 hours
            if target.exists() and stamp.exists():
                break
            drive_sync("restore", only=[f"runs/bon/probe-poolcand-{name}-{h}"])
            if not (target.exists() and stamp.exists()):
                print(f"waiting for {target} on Drive", flush=True)
                time.sleep(300)
        assert target.exists(), f"{target} never arrived"
        assert stamp.exists() and stamp.read_text() == digest, f"half {h} was answered on another candidate pool"
        halves.append(target)
    both = BON / f"probe-poolcand-{name}" / "rollouts.jsonl"
    both.parent.mkdir(parents=True, exist_ok=True)
    both.write_text("".join(h.read_text() for h in halves))
    drive_sync("restore", only=["runs/mvtf/eval-base", "runs/mvtf/v4-base"])
    evaluate(ROOT / "v4-base", samples=4, questions=val)
    evaluate(BASE)  # the original model on questions_40 (x2): a second non-fresh reference
    selected = Path(f"data/pool-{name}.jsonl")
    py("select_pool.py", "--pool", pool, "--rollouts", both, "--reference", ROOT / "v4-base", BASE,
       "--per-benchmark", args.dp_per_benchmark, "--seed", args.data_seed, "--out", selected)
    rows = [json.loads(l) for l in selected.read_text().splitlines() if l.strip()]
    random.Random(args.data_seed).shuffle(rows)
    per = Counter()
    check = Path(f"data/poolcheck-{name}.jsonl")
    check.write_text("".join(json.dumps(r) + "\n" for r in rows
                             if per.update([r["part"]]) or per[r["part"]] <= args.dp_check_per_part))
    held = answer("check", check, seed=7)
    for tag, refs in (("both", [ROOT / "v4-base", BASE]), ("choosing", [ROOT / "v4-base"]), ("q40", [BASE])):
        py("pool_check.py", "--rollouts", held, "--reference", *refs, "--out", BON / f"poolcheck-{name}-{tag}",
           "--pool", selected)
    drive_sync("save")


def poolfix(args):
    """dp4: rebuild only the parts of an answered pool that missed (user, 27 Oct night: start fixing right away). dp3's
    selection showed math thinking 500 vs 384 (too few easy, short-thinking problems) and no-fit tool questions at 97%
    vs 80% (unrelated functions are obvious to refuse). New candidates for --dp-parts only: gsm8k_easy_test, metamath_easy
    (rewordings of 1-2 calculation GSM8K problems), and "near" no-fit questions (a ToolACE question offered only the
    functions its reply did not call, none sharing a name word with the called one). They are answered x4 here, then the
    selection draws from dp3's answered candidates PLUS the new ones (every part re-selected, all parts re-checked on new
    answers against both references). --dp-reuse names the answered pool."""
    name, old = args.name, args.dp_reuse
    parts = args.dp_parts
    val = f"questions_val{VAL_SEED}"
    drive_sync("restore", only=[f"data/poolcand-{old}.jsonl", f"runs/bon/probe-poolcand-{old}", f"data/cand-{name}.jsonl",
                                f"data/poolcand-{name}.jsonl", "runs/mvtf/eval-base", "runs/mvtf/v4-base"])
    old_pool, old_answers = Path(f"data/poolcand-{old}.jsonl"), BON / f"probe-poolcand-{old}" / "rollouts.jsonl"
    import time
    for _ in range(96):  # the pool being reused may still be built by the other runtime: up to 8 hours
        if old_pool.exists() and old_answers.exists():
            break
        print(f"waiting for {old}'s answered pool on Drive", flush=True)
        time.sleep(300)
        drive_sync("restore", only=[f"data/poolcand-{old}.jsonl", f"runs/bon/probe-poolcand-{old}"])
    assert old_pool.exists() and old_answers.exists(), f"{old}'s answered pool is not here"
    cand, pool = Path(f"data/cand-{name}.jsonl"), Path(f"data/poolcand-{name}.jsonl")
    if not cand.exists() and not pool.exists():
        py("prepare_types.py", "--sources", args.dp_sources, "--per-source", 1000000, "--seed", args.data_seed,
           "--out", cand, "--tool-clean", "--tool-near", "--ifeval-match", args.dp_ifeval,
           *(["--kod-dataset", "KodCode/KodCode-V1", "--kod-max-task", 160, "--kod-max-solution", 400,
              "--kod-difficulty", "medium,hard"] if "kodcode" in args.dp_sources else []))
    if not pool.exists():
        py("build_pool.py", "--candidates", cand, "--choosing", val, "--all-questions", "questions", "--parts", parts,
           "--tool-sources", "toolace,hermes_tools,glaive_tools",
           "--per-benchmark", round(args.dp_per_benchmark * args.dp_oversample), "--math-mix", args.dp_math_mix,
           "--seed", args.data_seed, "--out", pool)
        seen = {json.dumps(json.loads(l)["prompt"]) for l in old_pool.read_text().splitlines() if l.strip()}
        rows = [json.loads(l) for l in pool.read_text().splitlines() if l.strip()]
        rows = [r for r in rows if json.dumps(r["prompt"]) not in seen]  # already a dp3 candidate
        caps = dict((k, int(v)) for k, v in (x.split("=") for x in args.dp_cap.split(",") if x))
        if caps:  # dp7c: hard code answers are long (~25 min per chunk): keep only what the missing tail needs
            rng = random.Random(args.data_seed)
            rng.shuffle(rows)
            kept = Counter()
            rows = [r for r in rows if kept.update([r["part"]]) or kept[r["part"]] <= caps.get(r["part"], 10 ** 9)]
        count = Counter()
        for r in rows:
            r["pool_id"] = f"{name}-{r['part']}-{count[r['part']]}"
            count[r["part"]] += 1
        pool.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"{name}: {len(rows)} new candidates {dict(count)}", flush=True)
        drive_sync("save")
    sources = "gsm8k_easy,gsm8k_easy_test,metamath_easy,gsm8k_matched,metamath_sv,kodcode,toolace,hermes_tools,glaive_tools,ifeval_like,ifeval_built"
    import hashlib
    import time
    digest = hashlib.sha256(pool.read_bytes()).hexdigest()[:16]
    halves = []
    for h, parity in (("a", 0), ("b", 1)):  # two runtimes answer one half each (user, 27 Oct night: use both GPUs)
        probe = BON / f"probe-poolcand-{name}-{h}"
        if args.dp_step not in (h, "select"):
            continue
        if args.dp_step == h:
            data = Path(f"data/poolcand-{name}-{h}.jsonl")
            rows = [json.loads(l) for l in pool.read_text().splitlines() if l.strip()]
            data.write_text("".join(json.dumps(r) + "\n" for r in rows
                                    if int(r["pool_id"].rsplit("-", 1)[1]) % 2 == parity))
            n = sum(1 for l in data.read_text().splitlines() if l.strip())
            drive_sync("restore", only=[str(probe)])
            if not (probe / "rollouts.jsonl").exists() or sum(1 for _ in (probe / "rollouts.jsonl").open()) < 4 * n:
                py("probe_curve.py", "--data", data, "--sources", sources, "--per-source", 1000000, "--samples", 4,
                   "--model", "Qwen/Qwen3-0.6B", "--out", probe, "--dtype", args.dtype, "--generate-only",
                   "--chunk", 150, "--max-questions", n)
            (probe / "poolcand.sha256").write_text(digest)
            drive_sync("save")
            return
        target, stamp = probe / "rollouts.jsonl", probe / "poolcand.sha256"
        for _ in range(96):  # the other runtime's half: up to 8 hours
            if target.exists() and stamp.exists():
                break
            drive_sync("restore", only=[str(probe)])
            if not (target.exists() and stamp.exists()):
                print(f"waiting for {target} on Drive", flush=True)
                time.sleep(300)
        assert target.exists() and stamp.exists() and stamp.read_text() == digest, f"{name} half {h} missing or from another pool"
        halves.append(target)
    probe = BON / f"probe-poolcand-{name}"
    probe.mkdir(parents=True, exist_ok=True)
    (probe / "rollouts.jsonl").write_text("".join(t.read_text() for t in halves))
    union, answers = Path(f"data/poolcand-{name}-all.jsonl"), BON / f"probe-poolcand-{name}-all" / "rollouts.jsonl"
    # dp3's gsm8k_easy questions carried the wrong answer instruction (25% unreadable, 11% capped): left out, and
    # their answers with them; dp4's own gsm8k_easy rows use the evaluation's SVAMP wording
    bad = {r["pool_id"] for r in map(json.loads, filter(str.strip, old_pool.read_text().splitlines()))
           if r["source"] == "gsm8k_easy" and "like: Final answer: 42" in r["prompt"][-1]["content"]}
    keep_old = [l for l in old_pool.read_text().splitlines() if l.strip() and json.loads(l)["pool_id"] not in bad]
    union.write_text("\n".join(keep_old) + "\n" + pool.read_text())
    answers.parent.mkdir(parents=True, exist_ok=True)
    old_rows = [l for l in old_answers.read_text().splitlines() if l.strip() and json.loads(l)["id"] not in bad]
    answers.write_text("\n".join(old_rows) + "\n" + (probe / "rollouts.jsonl").read_text())
    if args.dp_union_only:  # dp9: poolmore selects (after more answers), not this stage
        drive_sync("save")
        return
    evaluate(ROOT / "v4-base", samples=4, questions=val)
    evaluate(BASE)
    selected = Path(f"data/pool-{name}.jsonl")
    py("select_pool.py", "--pool", union, "--rollouts", answers, "--reference", ROOT / "v4-base", BASE,
       "--per-benchmark", args.dp_per_benchmark, "--seed", args.data_seed, "--out", selected)
    rows = [json.loads(l) for l in selected.read_text().splitlines() if l.strip()]
    random.Random(args.data_seed).shuffle(rows)
    per = Counter()
    check = Path(f"data/poolcheck-{name}.jsonl")
    check.write_text("".join(json.dumps(r) + "\n" for r in rows
                             if per.update([r["part"]]) or per[r["part"]] <= args.dp_check_per_part))
    held = BON / f"probe-poolcand-{name}-check"
    m = sum(1 for l in check.read_text().splitlines() if l.strip())
    if not (held / "rollouts.jsonl").exists() or sum(1 for _ in (held / "rollouts.jsonl").open()) < 4 * m:
        py("probe_curve.py", "--data", check, "--sources", sources, "--per-source", 1000000, "--samples", 4,
           "--model", "Qwen/Qwen3-0.6B", "--out", held, "--dtype", args.dtype, "--generate-only", "--chunk", 150,
           "--max-questions", m, "--seed", 7)
    for tag, refs in (("both", [ROOT / "v4-base", BASE]), ("choosing", [ROOT / "v4-base"]), ("q40", [BASE])):
        py("pool_check.py", "--rollouts", held / "rollouts.jsonl", "--reference", *refs,
           "--out", BON / f"poolcheck-{name}-{tag}", "--pool", selected)
    drive_sync("save")


def poolmore(args):
    """dp8 (user, 28 Oct): no new candidates; the parts that missed on dp7c's held-out check get more answers per
    candidate so the selection reads each question better. dp7c's selection answers said no-fit 84% vs 80%, the new
    answers 88%: questions picked for a lucky 4-answer draw drift back (simple went the other way, 80% vs 88%). Here
    every --dp-more-parts candidate of --dp-reuse (dp7c-all) is answered --dp-more-samples more times (seed 5), half
    per runtime, and selection runs on all 8 answers with --dp-think-bins thinking bins (quartiles: p25 and p75 matched,
    not only the median; IFEval p75 was -17%). The check uses ANOTHER seed (--dp-check-seed 9): dp7c's seed-7 check
    answers shaped this round's design. Other parts keep their 4 answers and the same selection rules."""
    import hashlib
    import time
    name, old = args.name, args.dp_reuse
    if args.dp_if_poor:  # dp11: only when the earlier pool's code is still off (user rule, see code_close_enough)
        for _ in range(96):
            drive_sync("restore", only=[str(BON / f"poolcheck-{args.dp_if_poor}-both")])
            close, why = code_close_enough(args.dp_if_poor)
            if close is not None:
                break
            print(f"{name}: waiting for {args.dp_if_poor}'s check ({why})", flush=True)
            time.sleep(300)
        print(f"{name}: {args.dp_if_poor} {why} -> {'close enough, SKIPPED' if close else 'still off, RUNNING'}", flush=True)
        marker = BON / (f"{name}-skipped" if close else f"{args.dp_if_poor}-superseded")
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(why + "\n")
        drive_sync("save")
        if close:
            return
    parts = set(args.dp_more_parts.split(","))
    val = f"questions_val{VAL_SEED}"
    union, answers = Path(f"data/poolcand-{old}.jsonl"), BON / f"probe-poolcand-{old}" / "rollouts.jsonl"
    for _ in range(96):
        drive_sync("restore", only=[str(union), str(answers.parent), "runs/mvtf/eval-base", "runs/mvtf/v4-base"])
        if union.exists() and answers.exists():
            break
        print(f"waiting for {old}'s answered pool on Drive", flush=True)
        time.sleep(300)
    assert union.exists() and answers.exists(), f"{old}'s answered pool is not here"
    sources = "gsm8k_easy,gsm8k_easy_test,metamath_easy,gsm8k_matched,metamath_sv,kodcode,toolace,hermes_tools,glaive_tools,ifeval_like,ifeval_built"
    extra = [BON / f"probe-poolmore-{e}-{h}" / "rollouts.jsonl" for e in args.dp_more_extra.split(",") if e for h in "ab"]
    for e in extra:
        drive_sync("restore", only=[str(e.parent)])
        assert e.exists(), f"{e} is not here"
    have = Counter(json.loads(l)["id"] for f in [answers, *extra] for l in f.read_text().splitlines() if l.strip())
    rows = [json.loads(l) for l in union.read_text().splitlines() if l.strip()]
    more = [r for r in rows if r["part"] in parts and (not args.dp_more_target or have[r["pool_id"]] < args.dp_more_target)]
    digest = hashlib.sha256("".join(r["pool_id"] for r in more).encode()).hexdigest()[:16]
    halves = []
    for h, parity in (("a", 0), ("b", 1)):  # one half per runtime
        probe = BON / f"probe-poolmore-{name}-{h}"
        if args.dp_step not in (h, "select"):
            continue
        if args.dp_step == h:
            data = Path(f"data/poolmore-{name}-{h}.jsonl")
            half = [r for k, r in enumerate(more) if k % 2 == parity]
            target = min(len(half), args.dp_more_cap or len(half))  # dp10 (user: "1200 questions only"): 600 per half
            data.write_text("".join(json.dumps(r) + "\n" for r in half))
            print(f"{name} half {h}: {len(half)} candidates x{args.dp_more_samples} ({dict(Counter(r['part'] for r in half))})",
                  flush=True)
            drive_sync("restore", only=[str(probe)])
            if not (probe / "rollouts.jsonl").exists() or sum(1 for _ in (probe / "rollouts.jsonl").open()) < args.dp_more_samples * target:
                py("probe_curve.py", "--data", data, "--sources", sources, "--per-source", 1000000,
                   "--samples", args.dp_more_samples, "--model", "Qwen/Qwen3-0.6B", "--out", probe, "--dtype", args.dtype,
                   "--generate-only", "--chunk", args.dp_chunk, "--max-questions", target, "--seed", 5)
            (probe / "poolcand.sha256").write_text(digest)
            drive_sync("save")
            return
        target, stamp = probe / "rollouts.jsonl", probe / "poolcand.sha256"
        for _ in range(96):  # the other runtime's half: up to 8 hours
            if target.exists() and stamp.exists():
                break
            drive_sync("restore", only=[str(probe)])
            if not (target.exists() and stamp.exists()):
                print(f"waiting for {target} on Drive", flush=True)
                time.sleep(300)
        assert target.exists() and stamp.exists() and stamp.read_text() == digest, f"{name} half {h} missing or from another pool"
        halves.append(target)
    merged = BON / f"probe-poolcand-{name}-all" / "rollouts.jsonl"
    merged.parent.mkdir(parents=True, exist_ok=True)
    merged.write_text("".join(f.read_text().rstrip("\n") + "\n" for f in [answers, *extra, *halves]))
    evaluate(ROOT / "v4-base", samples=4, questions=val)
    evaluate(BASE)
    selected = Path(f"data/pool-{name}.jsonl")
    py("select_pool.py", "--pool", union, "--rollouts", merged, "--reference", ROOT / "v4-base", BASE,
       "--per-benchmark", args.dp_per_benchmark, "--seed", args.data_seed, "--out", selected,
       "--think-bins", args.dp_think_bins, *(["--think-bins-part", args.dp_think_bins_part] if args.dp_think_bins_part else []))
    rows = [json.loads(l) for l in selected.read_text().splitlines() if l.strip()]
    random.Random(args.data_seed).shuffle(rows)
    per = Counter()
    check = Path(f"data/poolcheck-{name}.jsonl")
    check.write_text("".join(json.dumps(r) + "\n" for r in rows
                             if per.update([r["part"]]) or per[r["part"]] <= args.dp_check_per_part))
    held = BON / f"probe-poolcand-{name}-check"
    m = sum(1 for l in check.read_text().splitlines() if l.strip())
    if not (held / "rollouts.jsonl").exists() or sum(1 for _ in (held / "rollouts.jsonl").open()) < 4 * m:
        py("probe_curve.py", "--data", check, "--sources", sources, "--per-source", 1000000, "--samples", 4,
           "--model", "Qwen/Qwen3-0.6B", "--out", held, "--dtype", args.dtype, "--generate-only", "--chunk", 150,
           "--max-questions", m, "--seed", args.dp_check_seed)
    for tag, refs in (("both", [ROOT / "v4-base", BASE]), ("choosing", [ROOT / "v4-base"]), ("q40", [BASE])):
        py("pool_check.py", "--rollouts", held / "rollouts.jsonl", "--reference", *refs,
           "--out", BON / f"poolcheck-{name}-{tag}", "--pool", selected)
    drive_sync("save")


def near_misses_only(text):
    """User, 29 Oct ~01:15: "if it's close enough to the eval data, don't wait for me". A held-out check also opens the
    answers gate when every CHECK line misses its 95% interval by at most 1.5 points (accuracy, capped: values in %)
    or 3% of the bound (token counts). Returns (ok, the CHECK lines with their miss)."""
    import re as _re
    misses, ok = [], True
    for line in text.splitlines():
        if not line.rstrip().endswith("CHECK") or "verdict" in line:
            continue
        m = _re.search(r"(\w+) ([\d.]+)(%?) vs [\d.]+%? \[([\d.]+), ([\d.]+)\]", line)
        if not m:
            ok = False
            misses.append(line.strip() + " (unparsed)")
            continue
        value, pct, lo, hi = float(m.group(2)), m.group(3), float(m.group(4)), float(m.group(5))
        gap = lo - value if value < lo else value - hi
        near = gap <= 1.5 if pct else gap <= 0.03 * (lo if value < lo else hi)
        ok &= near
        misses.append(f"{line.strip()} (outside by {gap:.1f}{' pts' if pct else ' tokens'}, {'near' if near else 'FAR'})")
    return ok, misses


def code_close_enough(pool):
    """dp11's rule (user, 28 Oct night): run another 1,200 code questions per half unless the pool's held-out check
    (poolcheck-<pool>-both) has code within 5 accuracy points AND median / p75 thinking within 10%.
    Returns (close, text) or (None, reason) when the report is not there."""
    import re as _re
    report = BON / f"poolcheck-{pool}-both" / "report.txt"
    if not report.exists():
        return None, f"no {report}"
    block, inside = [], False
    for line in report.read_text().splitlines():
        if _re.match(r"\s{2}mbpp\b", line):
            inside = True
            continue
        if inside and _re.match(r"\s{2}\S", line) and not line.startswith("      "):
            break
        if inside:
            block.append(line)
    acc = next((float(m.group(1)) for l in block if (m := _re.search(r"accuracy .*\] ([+-][\d.]+)", l))), None)
    pct = {k: float(m.group(1)) for l in block for k in ("median_thinking", "p75_thinking")
           if l.strip().startswith(k) and (m := _re.search(r"\] ([+-]\d+)%", l))}
    if acc is None or len(pct) < 2:
        return None, "code lines not found: " + " | ".join(block[:4])
    close = abs(acc) < 5 and all(abs(v) < 10 for v in pct.values())
    return close, f"code accuracy {acc:+.1f} pts, median {pct['median_thinking']:+.0f}%, p75 {pct['p75_thinking']:+.0f}%"


REF_SEEDS = (20261024, 20261025)  # two more choosing-like sets (user, 29 Oct: "go to ~160 per type for v2's reference")


def refsets(args):
    """A bigger reference for matching (user, 29 Oct): two more random sets of 40 questions per part, never from the
    fresh set, questions_40 or the choosing set, answered x4 by --rs-model. With the choosing set and questions_40
    that is 160 per part. Only aggregate behaviour (accuracy, thinking length) is ever read from them, never content;
    the fresh set stays untouched for scores."""
    val, fresh_q = f"questions_val{VAL_SEED}", f"questions_fresh{FRESH_SEED}"
    sets = []
    for seed in REF_SEEDS:
        out = f"questions_ref{seed}"
        if not Path(out, "questions.jsonl").exists():
            py("subset_questions.py", "--per-dataset", 40, "--seed", seed, "--exclude", QUESTIONS, fresh_q, val, *sets,
               "--out", out)
        sets.append(out)
    for out in sets:
        run_dir = ROOT / f"v4-{args.rs_tag}-{out.split('ref')[1]}"
        evaluate(run_dir, samples=4, questions=out, model=args.rs_model or None)
        drive_sync("save")
        print(f"refsets: {run_dir} done", flush=True)


def crosscheck(args):
    """Second opinion on the confirmed soups (user, 29 Oct: "run them on another 240 questions from the eval, to cross
    check accuracy before more training"). Each soup of --cx-names at --cx-steps (the --seeds averaged, re-merged on
    local scratch) answers questions_ref<REF_SEEDS[0]> (40 per part, never fresh / choosing / questions_40) x4, scored
    against the original Qwen on the same set (refsets). Report only: the fresh set decides v1; a big disagreement
    between the two is a red flag to look at before round 2."""
    import time
    refq = f"questions_ref{REF_SEEDS[0]}"
    base_run = ROOT / f"v4-base-{REF_SEEDS[0]}"
    for _ in range(96):  # refsets (GPU 2) writes the set and the original Qwen's run
        drive_sync("restore", only=[refq, str(base_run)])
        if (base_run / "results.json").exists():
            break
        print(f"crosscheck: waiting for {base_run}", flush=True)
        time.sleep(300)
    for name in args.cx_names:
        for step in args.cx_steps:
            members = [Path("adapters") / f"{name}-s{seed}" / f"step-{step}" for seed in args.seeds]
            drive_sync("restore", only=[str(m.parent) for m in members])
            members = [m for m in members if (m / "adapter_config.json").exists()]
            if len(members) < len(args.seeds):
                print(f"crosscheck: {name} step {step}: only {len(members)} seeds, skipped", flush=True)
                continue
            out = ROOT / f"r4-{name}-soup-{step}"
            if not (out / "results.json").exists():
                model = Path("/content/lupin-models") / f"{name}-soup-{step}"
                py("merge.py", "--model", "Qwen/Qwen3-0.6B", "--out", model, "--adapters",
                   *[f"{m}:{1 / len(members):.6f}" for m in members])
                evaluate(out, samples=4, questions=refq, model=model)
                shutil.rmtree(model, ignore_errors=True)
                drive_sync("save")
            print(score_line(f"{name}-soup-{step} (cross-check: new 240 x 4)", out, base=base_run), flush=True)


SCREEN = "questions_screen720"  # 120 per part matched to the full evaluation (screen_set.py), from 30 Oct


def screenset(args):
    """The screen set matched to the full evaluation (user, 30 Oct): built from the original Qwen's full run, the fresh set
    left out; then the original Qwen on it x4 (s4-base), the reference every later screen is scored against."""
    base = ROOT / "full-base"
    drive_sync("restore", only=[str(base), SCREEN, f"questions_fresh{FRESH_SEED}"])
    if not (base / "results.json").exists():
        print("screenset: no full-base yet, skipped", flush=True)
        return
    if not Path(SCREEN, "questions.jsonl").exists():
        py("screen_set.py", "--full-base", base, "--questions", "questions", "--exclude", f"questions_fresh{FRESH_SEED}",
           "--per-part", 120, "--out", SCREEN)
    evaluate(ROOT / "s4-base", samples=4, questions=SCREEN)
    drive_sync("save")


def rescreen(args):
    """Score saved checkpoints on the matched screen set x4 (questions_screen720) against s4-base: one seed = that
    checkpoint, several seeds = their soup. For re-reading earlier screens and choosing steps on less noise."""
    drive_sync("restore", only=[SCREEN, str(ROOT / "s4-base")])
    for name, step, scale in [(n, s, x) for n in args.cx_names for s in args.cx_steps for x in args.cx_scale]:
        if True:  # (one level of the old nested loop kept, so the body below is unchanged)
            # step 0 = the FINAL adapter, as trained to the end (michal-bida scores only that one); --cx-scale < 1 = the
            # strength dial (2 Nov yardstick: one training traced as a curve, merge weight 0.5 / 0.75 / 1.0)
            members = [Path("adapters") / f"{name}-s{seed}" / (f"step-{step}" if step else "") for seed in args.seeds]
            drive_sync("restore", only=[str(Path("adapters") / f"{name}-s{seed}") for seed in args.seeds])
            members = [m for m in members if (m / "adapter_config.json").exists()]
            if len(members) < len(args.seeds):
                print(f"rescreen: {name} step {step}: only {len(members)} seeds, skipped", flush=True)
                continue
            tag = f"{name}-{'s' + str(args.seeds[0]) if len(members) == 1 else 'soup'}-{step or 'final'}" + (f"-x{scale:g}" if scale != 1 else "")
            out = ROOT / f"s4-{tag}"
            drive_sync("restore", only=[str(out)])
            if not (out / "results.json").exists():
                model = Path("/content/lupin-models") / tag
                py("merge.py", "--model", "Qwen/Qwen3-0.6B", "--out", model, "--adapters",
                   *[f"{m}:{scale / len(members):.6f}" for m in members])
                evaluate(out, samples=4, questions=SCREEN, model=model)
                shutil.rmtree(model, ignore_errors=True)
                drive_sync("save")
            print(score_line(f"{tag} (screen 720 x 4)", out, base=ROOT / "s4-base"), flush=True)


def fullmodel(args):
    """The full evaluation (2,759 x 8) and the 720 screen (x4) of saved MERGED models (user, 30 Oct: "eval rc1-rl-40, see
    how far off we were"), e.g. --nothink-models models/rc1-rl-40; then the representativeness check on every full run."""
    base = ROOT / "full-base"
    drive_sync("restore", only=[str(base), SCREEN, str(ROOT / "s4-base")])
    for soup in args.fm_soup:  # (9 Nov) rebuild a merged soup the clean-ups removed, from its kept adapters
        run_name, step = soup.split(":")
        model = Path("models") / f"{run_name}-soup-{step}"
        drive_sync("restore", only=[str(model)])
        if not (model / "config.json").exists():
            parts = [Path("adapters") / f"{run_name}-s{seed}" / f"step-{step}" for seed in (0, 1, 2)]
            drive_sync("restore", only=[str(p) for p in parts])
            parts = [p for p in parts if (p / "adapter_config.json").exists()]
            assert len(parts) == 3, f"{model}: only {len(parts)} of its 3 seed adapters are on Drive"
            py("merge.py", "--model", "Qwen/Qwen3-0.6B", "--out", model, "--adapters", *[f"{p}:{1 / 3:.6f}" for p in parts])
            drive_sync("save")
            print(f"fullmodel: rebuilt {model} from {', '.join(map(str, parts))}", flush=True)
    for spec in args.fm_merge:  # (9 Nov) models/OUT=BASE:ADAPTER, e.g. a repair checkpoint merged onto its base model
        out_model, rest = spec.split("=")
        merge_base, adapter = rest.split(":")
        drive_sync("restore", only=[out_model])
        if not (Path(out_model) / "config.json").exists():
            hub = merge_base.startswith("Qwen/")  # (11 Nov) a Hub id as the base (a round-1 adapter trained on the original)
            drive_sync("restore", only=[*([] if hub else [merge_base]), adapter])
            assert hub or (Path(merge_base) / "config.json").exists(), f"{merge_base} not found"
            assert (Path(adapter) / "adapter_config.json").exists(), f"{adapter} not found"
            py("merge.py", "--model", merge_base, "--out", out_model, "--adapters", f"{adapter}:1")
            drive_sync("save")
            print(f"fullmodel: built {out_model} = {merge_base} + {adapter}", flush=True)
    for model in args.nothink_models:
        name = Path(model).name
        drive_sync("restore", only=[model])
        assert (Path(model) / "config.json").exists(), f"{model} not found"
        out, screen = ROOT / f"full-{name}", ROOT / f"s4-{name}"
        drive_sync("restore", only=[str(out), str(screen)])
        if not args.fm_screen_only:
            evaluate(out, samples=8, questions="questions", model=model)
            drive_sync("save")
            print(score_line(f"{name} (FULL evaluation 2,759 x 8)", out, base=base), flush=True)
        evaluate(screen, samples=4, questions=SCREEN, model=model)
        drive_sync("save")
        print(score_line(f"{name} (screen 720 x 4)", screen, base=ROOT / "s4-base"), flush=True)
    sets = ["questions_val20261023", "questions_40", f"questions_ref{REF_SEEDS[0]}", f"questions_ref{REF_SEEDS[1]}",
            f"questions_fresh{FRESH_SEED}", SCREEN]
    drive_sync("restore", only=sets)
    cands = sorted(str(p) for p in ROOT.glob("full-*") if p.name != "full-base" and p.name != "full-subsets"
                   and (p / "results.json").exists())
    py("subset_check.py", "--base", base, "--cands", *cands, "--sets", *[s for s in sets if Path(s).exists()],
       "--out", ROOT / "full-subsets")
    drive_sync("save")


def breakdown(args):
    """(9 Nov, user) Where a fully tested model loses accuracy and saves tokens, per part, by the original's solve rate
    and by its answer length (difficulty_breakdown.py), then the questions it lost and why, with full answer texts
    (struggle_dump.py), for each --bd-names (runs/mvtf/full-NAME)."""
    base = ROOT / "full-base"
    for name in args.bd_names:
        cand = ROOT / f"full-{name}"
        drive_sync("restore", only=[str(base), str(cand)])
        assert (cand / "records.jsonl").exists(), f"{cand} has no records.jsonl"
        py("difficulty_breakdown.py", "--base", base, "--cand", cand, "--out", ROOT / f"bd-{name}")
        py("struggle_dump.py", "--base", base, "--cand", cand, "--questions", "questions", "--out", ROOT / f"st-{name}")
    drive_sync("save")


def struggle(args):
    """(9 Nov, user) The questions a fully tested model lost and why (struggle_dump.py), for each --bd-names."""
    base = ROOT / "full-base"
    for name in args.bd_names:
        cand = ROOT / f"full-{name}"
        drive_sync("restore", only=[str(base), str(cand)])
        py("struggle_dump.py", "--base", base, "--cand", cand, "--questions", "questions", "--out", ROOT / f"st-{name}",
           "--show", 6)
    drive_sync("save")


def copddemo(args):
    """(9 Nov, user) Do the COPD coaches' two hats differ? copd_demo.py with the 0.6B and the 4B (GPU, about 3 minutes)."""
    py("copd_demo.py", "--teachers", "Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B")


def ifevalrules(args):
    """(9 Nov, user) Which IFEval rules the shortened models break (ifeval_rules.py), per rule type; --bd-names are run
    folders under runs/mvtf: full-* against full-base, s4-* against s4-base (CPU only, a few minutes)."""
    if not Path("questions/questions.jsonl").exists():
        py("utils/eval/benchmarks.py")
    for prefix in ("full-", "s4-"):
        names = [n for n in args.bd_names if n.startswith(prefix)]
        if not names:
            continue
        runs = [ROOT / f"{prefix}base", *[ROOT / n for n in names]]
        drive_sync("restore", only=[str(r) for r in runs])
        runs = [r for r in runs if (r / "records.jsonl").exists()]
        py("ifeval_rules.py", "--questions", "questions", "--runs", *runs, "--out", ROOT / f"ir-{prefix.rstrip('-')}")
    drive_sync("save")


def copdanalysis(args):
    """(10 Nov, user) COPD Runs A and C from the data (copd_analysis.py): right vs wrong lengths on their training questions
    (the original's 10 answers), their 720 screens per part, their training logs (CPU only)."""
    answers = BON / "lupin-all-dp13-nofit1-tools1-code1-nofit2-nofit4-math1-math2.jsonl"
    steps = [10, 20, 25, 30, 35, 40]
    runs_a = [ROOT / f"s4-cp1a-{s}" for s in steps]
    runs_c = [ROOT / f"s4-cp1c2-{s}" for s in steps]
    drive_sync("restore", only=[str(answers), "data/tp-cp1a.jsonl", "data/tp-cp1c2.jsonl", str(ROOT / "s4-base"),
                                *map(str, runs_a + runs_c), "adapters/cp1a-rl", "adapters/cp1c2-rl"])
    py("copd_analysis.py", "--answers", answers, "--questions-a", "data/tp-cp1a.jsonl", "--questions-c", "data/tp-cp1c2.jsonl",
       "--base", ROOT / "s4-base", "--runs-a", *[r for r in runs_a if (r / "records.jsonl").exists()],
       "--runs-c", *[r for r in runs_c if (r / "records.jsonl").exists()], "--logs-a", "adapters/cp1a-rl",
       "--logs-c", "adapters/cp1c2-rl", "--out", ROOT / "copd-analysis")
    drive_sync("save")


def fullsoup(args):
    """The official full evaluation on confirmed soups (user, 30 Oct: "the obvious next step"): every question (2,759)
    x8, the original Qwen first (full-base, kept for every later model), then each soup of --cx-names at --cx-steps (the
    --seeds averaged). Our 240-question sets are subsets of these questions, so the per-question records also tell how
    far any 240 draw lands from the full score."""
    base = ROOT / "full-base"
    drive_sync("restore", only=[str(base)])
    evaluate(base, samples=8, questions="questions")
    drive_sync("save")
    for name in args.cx_names:
        for step in args.cx_steps:
            # step 0 = the FINAL adapter (michal-bida scores only that one), as in rescreen
            members = [Path("adapters") / f"{name}-s{seed}" / (f"step-{step}" if step else "") for seed in args.seeds]
            drive_sync("restore", only=[str(Path("adapters") / f"{name}-s{seed}") for seed in args.seeds])
            members = [m for m in members if (m / "adapter_config.json").exists()]
            if len(members) < len(args.seeds):
                print(f"fullsoup: {name} step {step}: only {len(members)} seeds, skipped", flush=True)
                continue
            tag = f"{name}-{'s' + str(args.seeds[0]) if len(members) == 1 else 'soup'}-{step}"  # one seed: its own score
            out = ROOT / f"full-{tag}"
            drive_sync("restore", only=[str(out)])
            if not (out / "results.json").exists():
                model = Path("/content/lupin-models") / tag
                py("merge.py", "--model", "Qwen/Qwen3-0.6B", "--out", model, "--adapters",
                   *[f"{m}:{1 / len(members):.6f}" for m in members])
                evaluate(out, samples=8, questions="questions", model=model)
                shutil.rmtree(model, ignore_errors=True)
                drive_sync("save")
            print(score_line(f"{tag} (FULL evaluation 2,759 x 8)", out, base=base), flush=True)
    # how representative each of our small sets is (subset_check.py), on every full run so far
    sets = ["questions_val20261023", "questions_40", f"questions_ref{REF_SEEDS[0]}", f"questions_ref{REF_SEEDS[1]}",
            f"questions_fresh{FRESH_SEED}", SCREEN]
    drive_sync("restore", only=sets)
    cands = sorted(str(p) for p in ROOT.glob("full-*-soup-*") if (p / "results.json").exists())
    py("subset_check.py", "--base", base, "--cands", *cands, "--sets", *[s for s in sets if Path(s).exists()],
       "--out", ROOT / "full-subsets")
    drive_sync("save")


def dial(args):
    """Strength dial (LUPIN spec 2.4; user 29 Oct: "build a very strong v1"): the soup of --cx-names at --cx-steps
    (--seeds averaged), its trained change scaled by each --dial-weights (merge.py adapter weights summing to w), scored
    on the FRESH set x8 against the original Qwen. No training: one trained model gives a saving-vs-accuracy curve."""
    fresh_q = f"questions_fresh{FRESH_SEED}"
    evaluate(ROOT / "f8-base", samples=8, questions=fresh_q)
    for name in args.cx_names:
        for step in args.cx_steps:
            members = [Path("adapters") / f"{name}-s{seed}" / f"step-{step}" for seed in args.seeds]
            drive_sync("restore", only=[str(m.parent) for m in members])
            members = [m for m in members if (m / "adapter_config.json").exists()]
            if not members:
                continue
            for w in args.dial_weights:
                tag = f"{name}-soup-{step}-x{round(w * 100)}"
                out = ROOT / f"f8-{tag}"
                if not (out / "results.json").exists():
                    model = Path("/content/lupin-models") / tag
                    py("merge.py", "--model", "Qwen/Qwen3-0.6B", "--out", model, "--adapters",
                       *[f"{m}:{w / len(members):.6f}" for m in members])
                    evaluate(out, samples=8, questions=fresh_q, model=model)
                    shutil.rmtree(model, ignore_errors=True)
                    drive_sync("save")
                print(score_line(f"{tag} (fresh 240 x 8, strength x{w})", out, base=ROOT / "f8-base"), flush=True)


PT_GROUPS = {"svamp": ("svamp",), "mbpp": ("mbpp",), "ifeval": ("ifeval",),
             "bfcl": ("bfcl_simple_python", "bfcl_multiple", "bfcl_irrelevance")}


def pertype(args):
    """Per-type adapters (user, 29 Oct: "build a strong v1"): one LoRA per benchmark, each trained on ITS type's pairs only
    (from --cc-base's matched pairs, the l2s50e recipe), then merged with per-type strengths (--pt-weights; e.g. math 0.5
    to protect it, instructions 1.0), plus the plain sum (all 1.0) as the control. One adapter per type means the type
    losing accuracy can be turned down without retraining the others. Scored on the choosing set x4."""
    import time
    name, base_pairs = args.name, BON / f"pairs-{args.cc_base}" / "pairs.jsonl"
    drive_sync("restore", only=[str(base_pairs.parent), "runs/mvtf/v4-base"])
    rows = [json.loads(l) for l in base_pairs.read_text().splitlines() if l.strip()]
    like = Counter()
    if args.pt_cap_like:  # (6 Nov, user) same pair count per group x kind as another set (October's), so the same steps
        like_file = BON / f"pairs-{args.pt_cap_like}" / "pairs.jsonl"
        drive_sync("restore", only=[str(like_file.parent)])
        like = Counter((group, json.loads(l)["kind"]) for l in open(like_file) if l.strip()
                       for group, parts in PT_GROUPS.items() if json.loads(l)["part"] in parts)
    finals = {}
    for group, parts in PT_GROUPS.items():
        mine = [r for r in rows if r["part"] in parts]
        if args.pt_cap_like:
            pool = mine[:]
            random.Random(0).shuffle(pool)
            got, mine = Counter(), []
            for r in pool:
                if got[r["kind"]] < like[(group, r["kind"])]:
                    got[r["kind"]] += 1
                    mine.append(r)
            if args.pt_extend:  # (6 Nov, user) the capped pairs FIRST (= the capped run up to its last full step), then the rest
                kept_ids = {id(r) for r in mine}
                mine = mine + [r for r in pool if id(r) not in kept_ids]
            short = {k: like[(group, k)] - got[k] for k in ("shorten", "guard", "hard", "right") if like[(group, k)] > got[k]}
            print(f"pertype {name}: {group} capped like {args.pt_cap_like}: {dict(got)}" + (f", SHORT {short}" if short else ""),
                  flush=True)
        if args.pt_check:  # (6 Nov, user) the pair counts only: no adapter trained, nothing scored
            have = Counter(r["kind"] for r in rows if r["part"] in parts)
            want = {k: like[(group, k)] for k in ("shorten", "guard", "hard", "right") if like[(group, k)]}
            print(f"pertype CHECK {name} {group}: available {dict(have)}; October {want}; kept {len(mine)} pairs "
                  f"= {max(0, len(mine) - 48) // 16} steps at 16 pairs/step after 48 held out", flush=True)
            continue
        pairs = BON / f"pairs-{name}-{group}"
        if not (pairs / "pairs.jsonl").exists():
            pairs.mkdir(parents=True, exist_ok=True)
            (pairs / "pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in mine))
        counts = [sum(r["kind"] == k for r in mine) for k in ("shorten", "guard", "hard", "right")]
        adapter = Path("adapters") / f"{name}-{group}"
        drive_sync("restore", only=[str(adapter)])  # (5 Nov) a fresh runtime must not retrain an adapter that exists
        if args.pt_no_train:
            for _ in range(args.pt_wait * 12):  # (6 Nov) the other GPU may still be training it: wait for Drive
                if (adapter / "adapter_config.json").exists():
                    break
                print(f"pertype {name}: waiting for {adapter} on Drive", flush=True)
                time.sleep(300)
                drive_sync("restore", only=[str(adapter)])
            assert (adapter / "adapter_config.json").exists(), f"{adapter} is not on Drive (--pt-no-train)"
        start = Path("adapters") / f"fresh-{name}-{group}"
        if not (adapter / "adapter_config.json").exists() and not (start / "adapter_config.json").exists():
            py("sft.py", "--init-only", "--model", "Qwen/Qwen3-0.6B", "--out", start)
        if not (adapter / "adapter_config.json").exists():
            capped_steps = (sum(got.values()) - 48) // 16 if args.pt_extend else 0
            if args.pt_extend:
                print(f"pertype {name}: {group} extended: {len(mine)} pairs in order, the first {sum(got.values())} = the "
                      f"capped set; checkpoint at step {capped_steps} = its last full step", flush=True)
            py("dpo.py", "--model", "Qwen/Qwen3-0.6B", "--pairs", pairs / "pairs.jsonl", "--init-adapter", start, "--out",
               adapter, "--mix", "all" if args.pt_extend else ",".join(f"{c / max(1, sum(counts)):.4f}" for c in counts),
               *(["--keep-order", "--save-at", capped_steps] if args.pt_extend else []), "--save-every", args.pt_save,
               "--epochs", 1, "--dtype", args.dtype, "--seed", 0, *args.pt_dpo.replace(",", " ").split())
            drive_sync("save")
        finals[group] = adapter / f"step-{args.pt_step}" if args.pt_step else adapter
        assert (finals[group] / "adapter_config.json").exists(), f"{finals[group]} missing"
        print(f"pertype {name}: {group} adapter on {len(mine)} pairs" + (f", scored at step {args.pt_step}" if args.pt_step else ""),
              flush=True)
    if args.pt_check:
        return
    weights = dict((k, float(v)) for k, v in (x.split("=") for x in args.pt_weights.split(",") if x))
    # --pt-screen (1 Nov): score on the 720 screen set x4 against s4-base instead of the choosing set
    val = SCREEN if args.pt_screen else f"questions_val{VAL_SEED}"
    pre, ref, label = ("s4", ROOT / "s4-base", "screen 720 x 4") if args.pt_screen else ("v4", ROOT / "v4-base", "choosing 240 x 4")
    if args.pt_screen:
        drive_sync("restore", only=[SCREEN, str(ROOT / "s4-base")])
    for tag, w in ((("sum", {g: 1.0 for g in PT_GROUPS}),) if not args.pt_tag else ()) + ((args.pt_tag or "tuned", weights),):
        out = ROOT / f"{pre}-{name}-{tag}"
        drive_sync("restore", only=[str(out)])
        full = ROOT / f"full-{name}-{tag}"  # (5 Nov, user) --pt-full: the official full test (2,759 x 8) of the merge
        if args.pt_full:
            drive_sync("restore", only=[str(full), str(ROOT / "full-base")])
        need = [(o, q, n) for o, q, n in ((out, val, 4), (full, "questions", 8))
                if (o is out or args.pt_full) and not (o / "results.json").exists()]
        if need:
            model = Path("/content/lupin-models") / f"{name}-{tag}"
            py("merge.py", "--model", "Qwen/Qwen3-0.6B", "--out", model, "--adapters",
               *[f"{finals[g]}:{w.get(g, 1.0)}" for g in PT_GROUPS if w.get(g, 1.0) > 0])  # 0 = left out (solo screens)
            for o, q, n in need:
                evaluate(o, samples=n, questions=q, model=model)
                drive_sync("save")
            shutil.rmtree(model, ignore_errors=True)
        print(score_line(f"{name}-{tag} {w} ({label})", out, base=ref), flush=True)
        if args.pt_full:
            print(score_line(f"{name}-{tag} {w} (FULL evaluation 2,759 x 8)", full, base=ROOT / "full-base"), flush=True)


def tooltopup(args):
    """More "be right" pairs for the tool parts (user, 29 Oct): tool candidates already built and contamination-checked
    (dp9n-all, not in the training pool) that the original model gets right SOMETIMES (0 < solve rate < 1 on their 4-8
    answers: only those give right > wrong pairs), --tt-questions of them in BFCL's 400 / 200 / 240 split, each topped up
    to 10 answers by the original model. lupin --lp-extra-q <name> then adds them to its pairs (same eval matching)."""
    name, pool_name = args.name, lupin_pool(args)
    out = BON / f"tooltop-{name}"
    if (out / "rollouts.jsonl").exists():
        print(f"tooltopup {name}: done", flush=True)
        return
    from build_pool import BFCL_SIZES
    union = Path("data/poolcand-dp9n-all.jsonl")
    sources = [BON / "probe-poolcand-dp9n-all" / "rollouts.jsonl",
               *[BON / f"probe-poolmore-{x}-{h}" / "rollouts.jsonl" for x in ("dp8", "dp9", "dp10", "dp11") for h in "ab"]]
    drive_sync("restore", only=[str(union), f"data/pool-{pool_name}.jsonl", *[str(f.parent) for f in sources]])
    used = {json.loads(l)["pool_id"] for l in Path(f"data/pool-{pool_name}.jsonl").read_text().splitlines() if l.strip()}
    wanted_parts = [x for x in args.tt_parts.split(",") if x] or list(BFCL_SIZES)
    rows = {r["pool_id"]: r for r in map(json.loads, filter(str.strip, union.read_text().splitlines()))
            if r["part"] in wanted_parts and r["pool_id"] not in used}
    have = defaultdict(list)
    for f in sources:
        if f.exists():
            for line in open(f):
                r = json.loads(line)
                if r["id"] in rows:
                    have[r["id"]].append(bool(r["correct"]) and r["closed_thinking"] and not r["truncated"])
    rng = random.Random(args.data_seed)
    picked = []
    total = sum(BFCL_SIZES.values())
    sizes = BFCL_SIZES | {"mbpp": sum(BFCL_SIZES.values())}  # code alone takes --tt-questions
    parts = wanted_parts
    total = sum(sizes[x] for x in parts)
    for part in parts:
        n = sizes[part]
        mixed = [i for i, v in have.items() if rows[i]["part"] == part and 0 < sum(v) < len(v)]
        rng.shuffle(mixed)
        if args.tt_hard:  # hardest first (lowest solve rate): closer to BFCL's no-fit (eval 79% vs our pool 88%)
            mixed.sort(key=lambda i: sum(have[i]) / len(have[i]))
        picked += mixed[:round(args.tt_questions * n / total)]
        print(f"tooltopup {name}: {part}: {len(mixed)} sometimes-right candidates, {min(len(mixed), round(args.tt_questions * n / total))} kept", flush=True)
    need = {i: max(0, 10 - len(have[i])) for i in picked}
    print(f"tooltopup {name}: {len(picked)} tool questions answered sometimes right "
          f"({dict(Counter(rows[i]['part'] for i in picked))}); answers to make {dict(Counter(need.values()))}", flush=True)
    made = []
    for k in sorted(v for v in set(need.values()) if v > 0):
        sub = Path(f"data/tooltop-{name}-k{k}.jsonl")
        sub.write_text("".join(json.dumps(rows[i]) + "\n" for i in picked if need[i] == k))
        probe = BON / f"tooltop-{name}-k{k}"
        drive_sync("restore", only=[str(probe)])
        if not (probe / "rollouts.jsonl").exists():
            py("probe_curve.py", "--data", sub, "--sources", "toolace,hermes_tools,glaive_tools,kodcode", "--per-source", 1000000,
               "--samples", k, "--model", "Qwen/Qwen3-0.6B", "--out", probe, "--dtype", args.dtype, "--generate-only",
               "--chunk", 300 if "mbpp" not in parts else 150, "--max-questions", sum(1 for _ in open(sub)), "--seed", 23 + k)
            drive_sync("save")
        made.append(probe / "rollouts.jsonl")
    out.mkdir(parents=True, exist_ok=True)
    ids = set(picked)
    with open(out / "rollouts.jsonl.tmp", "w") as handle:  # new answers first, then the earlier ones (lupin keeps 10)
        for f in [*made, *sources]:
            if f.exists():
                for line in open(f):
                    if line.strip() and json.loads(line)["id"] in ids:
                        handle.write(line if line.endswith("\n") else line + "\n")
    (out / "questions.jsonl").write_text("".join(json.dumps(rows[i]) + "\n" for i in picked))
    (out / "rollouts.jsonl.tmp").rename(out / "rollouts.jsonl")
    drive_sync("save")


def references():
    """The original Qwen's runs used as the evaluation reference for matching: choosing set + questions_40, plus the two
    refsets (29 Oct: 160 questions per part when they exist)."""
    refs = [ROOT / "v4-base", BASE] + [ROOT / f"v4-base-{seed}" for seed in REF_SEEDS]
    drive_sync("restore", only=[str(r) for r in refs])
    return [r for r in refs if (r / "records.jsonl").exists()]


def poolup(args):
    """Data upgrade, step 2 (user, 29 Oct: "keep iterating this data improvement until it is very representative of the
    whole eval"): the training pool re-picked from pool-<--dp-reuse> + the top-up questions (nofit1, tools1, code1: harder
    sometimes-right questions with 10 answers), matched by select_pool.py to ALL the original Qwen's evaluation runs
    (references(): choosing + questions_40 + two refsets = 160 per part), 4 thinking bins (8 for bfcl_simple), on each
    question's 10 newest answers; then the held-out check: 200 questions per part answered x4 with a NEW seed
    (--dp-check-seed), pool_check.py against the same references. lupin --lp-gate trains only on a passing pool."""
    import time
    name, old = args.name, args.dp_reuse
    extras = [BON / f"tooltop-{x}" for x in args.pu_extras.split(",") if x]
    for extra in extras:
        for _ in range(288):
            drive_sync("restore", only=[str(extra)])
            if (extra / "rollouts.jsonl").exists():
                break
            print(f"poolup {name}: waiting for {extra}", flush=True)
            time.sleep(300)
    union = Path(f"data/poolcand-{name}.jsonl")
    answers = BON / f"poolup-{name}" / "rollouts.jsonl"
    halves = [BON / f"lupin-answers-{old}-{h}" / "rollouts.jsonl" for h in "ab"]
    sources = [*halves, *[e / "rollouts.jsonl" for e in extras],
               *[BON / f"probe-poolmore-{x}-{h}" / "rollouts.jsonl" for x in ("dp11", "dp10", "dp9", "dp8") for h in "ab"],
               BON / "probe-poolcand-dp9n-all" / "rollouts.jsonl"]
    drive_sync("restore", only=[f"data/pool-{old}.jsonl", *[str(f.parent) for f in sources]])
    if not answers.exists():
        rows = [json.loads(l) for l in Path(f"data/pool-{old}.jsonl").read_text().splitlines() if l.strip()]
        for e in extras:
            rows += [json.loads(l) for l in (e / "questions.jsonl").read_text().splitlines() if l.strip()]
        drop = [tuple(x.split(":")) for x in args.pu_drop.split(",") if x]  # e.g. bfcl_simple_python:toolace
        rows = [r for r in rows if (r["part"], r["source"]) not in drop]
        seen, uniq = set(), []
        for r in rows:
            if r["pool_id"] not in seen:
                seen.add(r["pool_id"])
                uniq.append(r)
        union.write_text("".join(json.dumps(r) + "\n" for r in uniq))
        taken = Counter()
        answers.parent.mkdir(parents=True, exist_ok=True)
        with open(answers, "w") as out:  # 10 newest answers per question, as lupin uses them
            for f in sources:
                if f.exists():
                    for line in open(f):
                        if line.strip():
                            i = json.loads(line)["id"]
                            if i in seen and taken[i] < 10:
                                taken[i] += 1
                                out.write(line if line.endswith("\n") else line + "\n")
        hard = [x.split(":") for x in args.pu_drop_hard.split(",") if x]  # e.g. bfcl_simple_python:toolace:0.5
        if hard:  # drop the hardest questions of a part/source (solve rate below a threshold on their 10 answers)
            rate = defaultdict(list)
            for line in open(answers):
                r = json.loads(line)
                rate[r["id"]].append(bool(r["correct"]) and r["closed_thinking"] and not r["truncated"])
            low = {r["pool_id"] for r in uniq for part, src, thr in hard
                   if r["part"] == part and r["source"] == src and rate[r["pool_id"]]
                   and sum(rate[r["pool_id"]]) / len(rate[r["pool_id"]]) < float(thr)}
            uniq = [r for r in uniq if r["pool_id"] not in low]
            union.write_text("".join(json.dumps(r) + "\n" for r in uniq))
            print(f"poolup {name}: dropped {len(low)} hard questions ({args.pu_drop_hard})", flush=True)
        print(f"poolup {name}: {len(uniq)} candidate questions ({dict(Counter(r['part'] for r in uniq))})", flush=True)
        drive_sync("save")
    refs = references()
    print(f"poolup {name}: references {[str(r) for r in refs]}", flush=True)
    selected = Path(f"data/pool-{name}.jsonl")
    py("select_pool.py", "--pool", union, "--rollouts", answers, "--reference", *refs, "--per-benchmark", args.dp_per_benchmark,
       "--seed", args.data_seed, "--out", selected, "--think-bins", 4, "--think-bins-part", "bfcl_simple_python=8")
    rows = [json.loads(l) for l in selected.read_text().splitlines() if l.strip()]
    random.Random(args.data_seed).shuffle(rows)
    per = Counter()
    check = Path(f"data/poolcheck-{name}.jsonl")
    check.write_text("".join(json.dumps(r) + "\n" for r in rows
                             if per.update([r["part"]]) or per[r["part"]] <= args.dp_check_per_part))
    held = BON / f"probe-poolcand-{name}-check"
    m = sum(1 for l in check.read_text().splitlines() if l.strip())
    drive_sync("restore", only=[str(held)])
    if not (held / "rollouts.jsonl").exists() or sum(1 for _ in (held / "rollouts.jsonl").open()) < 4 * m:
        py("probe_curve.py", "--data", check, "--sources", "gsm8k_easy,gsm8k_easy_test,metamath_easy,gsm8k_matched,metamath_sv,"
           "kodcode,toolace,hermes_tools,glaive_tools,ifeval_like,ifeval_built", "--per-source", 1000000, "--samples", 4,
           "--model", "Qwen/Qwen3-0.6B", "--out", held, "--dtype", args.dtype, "--generate-only", "--chunk", 150,
           "--max-questions", m, "--seed", args.dp_check_seed)
    py("pool_check.py", "--rollouts", held / "rollouts.jsonl", "--reference", *refs, "--out", BON / f"poolcheck-{name}-both",
       "--pool", selected)
    drive_sync("save")


def newtop(args):
    """New top-up candidates built from scratch (30 Oct, data loop): prepare_types.py --sources --nt-sources, build_pool.py
    on --nt-parts (the choosing set's prompt-length bins, 8-word contamination check against every evaluation question),
    up to --tt-questions kept, each answered 10 times by the original Qwen (seed 41). Written like a tooltopup
    (runs/bon/tooltop-<name>/questions.jsonl + rollouts.jsonl) so poolup --pu-extras can add it."""
    import statistics
    name = args.name
    out = BON / f"tooltop-{name}"
    drive_sync("restore", only=[str(out)])
    if (out / "rollouts.jsonl").exists():
        print(f"newtop {name}: done", flush=True)
        return
    val = f"questions_val{VAL_SEED}"
    cand, pool = Path(f"data/cand-{name}.jsonl"), Path(f"data/poolcand-{name}.jsonl")
    if not cand.exists():
        py("prepare_types.py", "--sources", args.nt_sources, "--per-source", 1000000, "--seed", args.data_seed, "--out", cand,
           "--tool-clean")
    if not pool.exists():
        py("build_pool.py", "--candidates", cand, "--choosing", val, "--all-questions", "questions", "--parts", args.nt_parts,
           "--tool-sources", "toolace,hermes_tools,glaive_tools", "--per-benchmark", args.tt_questions * 4,
           "--seed", args.data_seed, "--out", pool, *(["--no-match"] if args.nt_raw else []))
        rows = [json.loads(l) for l in pool.read_text().splitlines() if l.strip()]
        # never a question already in the training pool or an earlier top-up (1 Nov: no-fit sources overlap)
        seen = set()
        for f in [*Path("data").glob("pool-dp*.jsonl"), *BON.glob("tooltop-*/questions.jsonl")]:
            for line in open(f):
                if line.strip():
                    seen.add(json.dumps(json.loads(line).get("messages") or json.loads(line).get("prompt"), sort_keys=True))
        before = len(rows)
        rows = [r for r in rows if json.dumps(r.get("messages") or r.get("prompt"), sort_keys=True) not in seen]
        print(f"newtop {name}: {before - len(rows)} candidates already in the pool or a top-up, dropped", flush=True)
        random.Random(args.data_seed).shuffle(rows)
        rows = rows[:args.tt_questions]
        for k, r in enumerate(rows):
            r["pool_id"] = f"{name}-{r['part']}-{k}"
        pool.write_text("".join(json.dumps(r) + "\n" for r in rows))
        drive_sync("save")
    n = sum(1 for _ in open(pool))
    probe = BON / f"tooltop-{name}-k10"
    drive_sync("restore", only=[str(probe)])
    if not (probe / "rollouts.jsonl").exists():
        py("probe_curve.py", "--data", pool, "--sources", "toolace,hermes_tools,glaive_tools,kodcode,gsm8k_matched,metamath_sv,ifeval_like", "--per-source", 1000000,
           "--samples", args.nt_samples, "--model", "Qwen/Qwen3-0.6B", "--out", probe, "--dtype", args.dtype,
           "--generate-only", "--chunk", 300, "--max-questions", n, "--seed", 41)
        drive_sync("save")
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in pool.read_text().splitlines() if l.strip()]
    rate = defaultdict(list)
    for line in open(probe / "rollouts.jsonl"):
        r = json.loads(line)
        rate[r["id"]].append(bool(r["correct"]))
    solve = [sum(v) / len(v) for v in rate.values()]
    print(f"newtop {name}: {n} questions x{args.nt_samples}; mean solve rate {statistics.fmean(solve):.0%}, "
          f"sometimes right {sum(0 < s < 1 for s in solve)}, always {sum(s == 1 for s in solve)}, never {sum(s == 0 for s in solve)}", flush=True)
    (out / "questions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    shutil.copy(probe / "rollouts.jsonl", out / "rollouts.jsonl")
    drive_sync("save")


def onpolicy(args):
    """A trained model's own answers on a top-up's questions in one thinking band (1 Nov, math hole): the questions of
    runs/bon/tooltop-<--op-from> whose original-Qwen answers think --op-band tokens on average (mean thinking length),
    answered --op-samples times by adapters/<--op-adapter> (seed 43) -> runs/bon/onp-<name>/rollouts.jsonl. These are
    the losers of lupin_pairs.py --overcut ("cut too short, got it wrong")."""
    import statistics
    name, src = args.name, BON / f"tooltop-{args.op_from}"
    out, adapter = BON / f"onp-{name}", Path("adapters") / args.op_adapter
    drive_sync("restore", only=[str(out), str(src), str(adapter)])
    if (out / "rollouts.jsonl").exists():
        print(f"onpolicy {name}: done", flush=True)
        return
    lo, hi = (float(x) for x in args.op_band.split(","))
    think = defaultdict(list)
    for line in open(src / "rollouts.jsonl"):
        if line.strip():
            r = json.loads(line)
            think[r["id"]].append(len(r["thinking_ids"]))
    keep = {i for i, v in think.items() if lo <= statistics.fmean(v) < hi}
    rows = [json.loads(l) for l in (src / "questions.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("pool_id", r.get("id")) in keep]
    data = Path(f"data/onp-{name}.jsonl")
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"onpolicy {name}: {len(rows)} questions of {src.name} in the {lo:.0f}-{hi:.0f} band", flush=True)
    py("probe_curve.py", "--data", data, "--sources", "toolace,hermes_tools,glaive_tools,kodcode,gsm8k_matched,metamath_sv",
       "--per-source", 1000000, "--samples", args.op_samples, "--model", "Qwen/Qwen3-0.6B", "--adapter", adapter,
       "--out", out, "--dtype", args.dtype, "--generate-only", "--chunk", 300, "--max-questions", len(rows), "--seed", 43)
    rate = defaultdict(list)
    for line in open(out / "rollouts.jsonl"):
        r = json.loads(line)
        rate[r["id"]].append((bool(r["correct"]) and not r["truncated"], len(r["thinking_ids"])))
    allv = [a for v in rate.values() for a in v]
    print(f"onpolicy {name}: {len(rate)} questions x{args.op_samples}; right {statistics.fmean(c for c, _ in allv):.1%}, "
          f"mean thinking {statistics.fmean(t for _, t in allv):.0f}; wrong answers {sum(not c for c, _ in allv)}", flush=True)
    drive_sync("save")


TYPE_PARTS = ("svamp", "mbpp", "ifeval", "bfcl_simple_python", "bfcl_multiple", "bfcl_irrelevance")


def typeonly(args):
    """One adapter per type AND subtype (user, 2 Nov: "build per type data only shorten and hold accuracy, see how much
    it spills over"): from --cc-base's pairs (runs/bon/pairs-<cc_base>), each of the 6 parts trained alone on its own
    pairs, bidax4's settings (cosine, no NLL, 16 pairs/step, 1 pass), --seeds each, adapters/<name>-<part>-s<seed>.
    Scored afterwards with rescreen --cx-names <name>-<part> (every part's score = its own effect + its spill-over)."""
    base = BON / f"pairs-{args.cc_base}"
    drive_sync("restore", only=[str(base)])
    rows = [json.loads(l) for l in (base / "pairs.jsonl").read_text().splitlines() if l.strip()]
    for part in [p for p in TYPE_PARTS if p in args.to_parts.split(",")]:
        mine = [r for r in rows if r["part"] == part]
        pairs = BON / f"pairs-{args.name}-{part}"
        drive_sync("restore", only=[str(pairs)])
        if not (pairs / "pairs.jsonl").exists():
            pairs.mkdir(parents=True, exist_ok=True)
            (pairs / "pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in mine))
        counts = [sum(r["kind"] == k for r in mine) for k in ("shorten", "guard", "hard", "right")]
        print(f"typeonly {args.name}: {part} {len(mine)} pairs (shorten {counts[0]}, guard {counts[1]})", flush=True)
        start = Path("adapters") / f"fresh-{args.name}-{part}"
        drive_sync("restore", only=[str(start)] + [str(Path("adapters") / f"{args.name}-{part}-s{s}") for s in args.seeds])
        if not (start / "adapter_config.json").exists():
            py("sft.py", "--init-only", "--model", "Qwen/Qwen3-0.6B", "--out", start)
        for seed in args.seeds:
            adapter = Path("adapters") / f"{args.name}-{part}-s{seed}"
            if not (adapter / "adapter_config.json").exists():
                py("dpo.py", "--model", "Qwen/Qwen3-0.6B", "--pairs", pairs / "pairs.jsonl", "--init-adapter", start,
                   "--out", adapter, "--mix", ",".join(f"{c / max(1, sum(counts)):.4f}" for c in counts),
                   "--save-every", 10, "--epochs", 1, "--dtype", args.dtype, "--seed", seed, "--nll", 0.0,
                   "--lr-schedule", "cosine", *args.lp_dpo.replace(",", " ").split())
                drive_sync("save")


def polans(args):
    """On-policy answers on the WHOLE training pool (2 Nov, user: the on-policy round): adapters/<--op-adapter> answers
    every question of data/pool-<--lp-pool>.jsonl plus the --lp-add-q top-ups, --op-samples times (seed 47), half the
    questions per runtime (--dp-step a / b) -> runs/bon/onp-<name>-<half>/rollouts.jsonl. lupin_pairs.py --recipe four
    takes them as --trained (lupin --lp-trained <name>-a,<name>-b)."""
    name, adapter = args.name, Path("adapters") / args.op_adapter
    out = BON / f"onp-{name}-{args.dp_step}"
    pool = Path(f"data/pool-{args.lp_pool}.jsonl")
    extras = [BON / f"tooltop-{x}" for x in args.lp_add_q.split(",") if x]
    drive_sync("restore", only=[str(out), str(adapter), str(pool), *[str(e) for e in extras]])
    if (out / "rollouts.jsonl").exists():
        print(f"polans {name}-{args.dp_step}: done", flush=True)
        return
    rows = [json.loads(l) for l in pool.read_text().splitlines() if l.strip()]
    for e in extras:
        rows += [json.loads(l) for l in (e / "questions.jsonl").read_text().splitlines() if l.strip()]
    parity = 0 if args.dp_step == "a" else 1
    data = Path(f"data/polans-{name}-{args.dp_step}.jsonl")
    data.write_text("".join(json.dumps(r) + "\n" for k, r in enumerate(rows) if k % 2 == parity))
    n = sum(1 for _ in open(data))
    print(f"polans {name}-{args.dp_step}: {n} questions x{args.op_samples} by {adapter}", flush=True)
    py("probe_curve.py", "--data", data, "--sources", "toolace,hermes_tools,glaive_tools,kodcode,gsm8k_matched,metamath_sv,"
       "gsm8k_easy,gsm8k_easy_test,metamath_easy,ifeval_like,ifeval_built", "--per-source", 1000000,
       "--samples", args.op_samples, "--model", "Qwen/Qwen3-0.6B", "--adapter", adapter, "--out", out, "--dtype", args.dtype,
       "--generate-only", "--chunk", 300, "--max-questions", n, "--seed", 47)
    drive_sync("save")


PT_MID = {"svamp": 0.30, "mbpp": 0.45, "ifeval": 0.60, "bfcl": 0.45}  # pt-mid = lpw-pertype merged at these strengths


def own_repair_questions(want, seed=0, exclude=None, public=False, avoid_grams=None):
    """(9 Nov, user: "100% not copying michal-bida") The repair's questions from the user's OWN sources only, want[part] per
    test part, drawn at random (the trained model's own 4 answers then decide which give pairs):
      math         GSM8K train + test at EVERY difficulty, in SVAMP's wording (prepare_types.py gsm8k_std; GSM8K is in the
                   data since night 1, 20 Oct): harder than the pool's SVAMP-like math, so the model is sometimes wrong
                   and math gets "be right" pairs; questions already trained on in rounds 1-4 dropped
      code, instructions, tools
                   pool-dp13 rows (KodCode, IFEval built / argilla, Glaive / ToolACE / Hermes) and the own top-ups
                   nofit1, nofit2 (Haiku-verified no-fit), tools1, code1
    Left out: MetaMath (every metamath_* source), math1, math2, instr2, code2, nofit4 (michal-bida's sources). Every
    question sharing an 8-word sequence with an evaluation question is dropped (as build_pool.py)."""
    from build_pool import ngrams, part_of, words
    own_extras = ["nofit1", "nofit2", "tools1", "code1"] + (["nofit4", "math1", "math2"] if public else [])
    # (11 Nov, user) public=True: the public DATASETS michal-bida also used (MetaMath, the math1 / math2 / nofit4 top-ups built
    # from GSM8K / MetaMath / xLAM) are allowed (recipes stay off-limits); math then comes from these SVAMP-like sources,
    # not from full-difficulty GSM8K (harder than the test's SVAMP)
    pool = Path("data/pool-dp13.jsonl")
    used_globs = [f"data/{prefix}*.jsonl" for prefix in ("bon", "bal", "chain", "trapmath", "rlcal", "rl-", "lot-")]
    drive_sync("restore", only=[str(pool), *[str(BON / f"tooltop-{x}" / "questions.jsonl") for x in own_extras], *used_globs])
    if not Path("questions/questions.jsonl").exists():
        py("utils/eval/benchmarks.py")
    gsm = Path("data/gsm8k-std-all.jsonl")
    if not gsm.exists():
        py("prepare_types.py", "--sources", "gsm8k_std", "--per-source", 1000000, "--out", gsm)

    def read(path):
        return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()] if Path(path).exists() else []
    seen = Counter()
    for q in read("questions/questions.jsonl"):
        seen.update(ngrams(words(q["messages"])))
    eval_grams = {g for g, n in seen.items() if n < 3}  # shared templates (>= 3 questions) are not evidence of copying
    used = Counter()  # math already trained on (rounds 1-4: bon, chain, trap math, rlcal draws), any wording
    for f in sorted({str(x) for pattern in used_globs for x in Path(".").glob(pattern)}):
        for r in read(f):
            if str(r.get("source", "")).startswith("gsm") and r.get("prompt"):
                used.update(ngrams(words(r["prompt"])))
    used_grams = {g for g, n in used.items() if n < 3}
    cands, texts, dropped = defaultdict(list), set(), Counter()
    gsm_rows = [r | {"id": f"gsm8k-std-{i}"} for i, r in enumerate(read(gsm))]
    other = [r | {"id": r.get("pool_id", r.get("id"))} for r in read(pool)]
    for x in own_extras:
        other += [r | {"id": r.get("pool_id", r.get("id"))} for r in read(BON / f"tooltop-{x}" / "questions.jsonl")]
    for r in gsm_rows + other:
        part = part_of(r)
        if part not in want or (str(r["source"]).startswith("metamath") and not public):
            continue
        if part == "svamp" and (r["id"].startswith("gsm8k-std-") == public):  # own: full GSM8K only; public: SVAMP-like only
            continue
        key = json.dumps(r["prompt"])
        if key in texts:
            continue
        texts.add(key)
        grams = ngrams(words(r["prompt"]))
        if grams & eval_grams:
            dropped[(part, "overlaps an evaluation question")] += 1
            continue
        if part == "svamp" and grams and len(grams & used_grams) / len(grams) > 0.3:
            dropped[(part, "already trained on")] += 1
            continue
        if avoid_grams and grams and len(grams & avoid_grams) / len(grams) >= 0.3:  # (11 Nov, consult8) the same question
            dropped[(part, "near-duplicate of a Dev-512 question")] += 1           # under another id (GSM8K rewordings)
            continue
        cands[part].append(r)
    if exclude:  # (11 Nov) the Dev-512 / Confirm-512 questions are never drawn for training (only when asked: old draws stay)
        for part in cands:
            cands[part] = [r for r in cands[part] if r["id"] not in exclude]
    rng = random.Random(seed)
    chosen = []
    for part, n in want.items():
        rng.shuffle(cands[part])
        chosen += cands[part][:n]
        print(f"own repair questions: {part} {min(n, len(cands[part]))} of {len(cands[part])} usable "
              f"(dropped: {dict((k[1], v) for k, v in dropped.items() if k[0] == part)})", flush=True)
    return chosen


def ptrepair(args):
    """Track A (GAME_PLAN.md v3, 5 Nov): repair pt-mid. (1) pt-mid (the 4 lpw-pertype adapters merged at PT_MID) answers
    --rp-questions training questions x4 (the original's mixed-outcome questions first, then all-right and all-wrong);
    (2) repair_pairs.py: "be right" pairs only (shortest right > shortest finished wrong; similar-length right > wrong,
    half with the right answer longer), --rp-quota, stratified order; (3) a fresh LoRA on merged pt-mid (reference =
    pt-mid), float32 + fp16 autocast, beta 0.1, no NLL, batch 16, lr 2.5e-6 then --rp-lr constant, --rp-steps, saved
    every 5, pairs read in order; (4) pt-mid + the repair adapter at each --rp-screen step on the 720 x4.
    9 Nov (user: repair the rc line, own data only): --rp-base repairs any merged model instead of pt-mid (it answers,
    it is DPO's reference, every screen merges onto it); --rp-own-data draws the questions from own_repair_questions()
    instead of the original's mixed-outcome dp13 questions; --rp-kinds guard keeps only the 21 Oct "be right" pair."""
    name = args.name
    smoke = args.rp_smoke
    if getattr(args, "rp_gate", ""):  # (11 Nov) path 1: round 2 only if round 1 shortened enough (pre-registered)
        run_n, ref_n, margin = args.rp_gate.split(",")
        sc = {}
        for n in (run_n, ref_n):
            f = BON / f"dev512-{n}" / "dev_score.json"
            drive_sync("restore", only=[str(f)])
            sc[n] = json.loads(f.read_text())["saved"] if f.exists() else None
        ok = None not in sc.values() and sc[run_n] >= sc[ref_n] + float(margin)
        print(f"repair {name}: GATE {'OPEN' if ok else 'CLOSED'}: Dev-512 saved {run_n} {sc[run_n]} vs {ref_n} {sc[ref_n]} "
              f"+ {margin}", flush=True)
        if not ok:
            return
    extras = ["nofit1", "tools1", "code1", "nofit2", "nofit4", "math1", "math2"]
    answers = BON / f"lupin-all-dp13-{'-'.join(extras)}-a4.jsonl"
    pt = {g: Path("adapters") / f"lpw-pertype-{g}" for g in PT_MID}
    pool = Path("data/pool-dp13.jsonl")
    if args.rp_base:  # (9 Nov) any merged model; pt-mid's adapters and the original's answers are not needed
        hub = args.rp_base.startswith("Qwen/")  # (10 Nov, odpo from the original) a Hub id: nothing to restore or build
        drive_sync("restore", only=[*([] if hub else [args.rp_base]), SCREEN, str(ROOT / "s4-base")])
        merged = Path(args.rp_base) if not hub else args.rp_base
        if args.rp_base_from and not hub and not (merged / "config.json").exists():  # (10 Nov) build it: BASE_MODEL:ADAPTER
            base_model, base_adapter = args.rp_base_from.rsplit(":", 1)
            drive_sync("restore", only=[base_adapter])
            assert (Path(base_adapter) / "adapter_config.json").exists(), f"{base_adapter} is not on Drive"
            py("merge.py", "--model", base_model, "--out", merged, "--adapters", f"{base_adapter}:1")
            drive_sync("save")
            print(f"repair {name}: built {merged} = {base_model} + {base_adapter}", flush=True)
        assert hub or (merged / "config.json").exists(), f"{merged} is not on Drive (rebuild it with merge.py)"
    else:
        drive_sync("restore", only=[*([] if args.rp_from or args.rp_own_data else
                                      [str(answers), str(pool), *[str(BON / f"tooltop-{x}") for x in extras]]),
                                    *[str(a) for a in pt.values()], SCREEN, str(ROOT / "s4-base")])
        for a in pt.values():
            assert (a / "adapter_config.json").exists(), f"{a} is not on Drive"
        merged = Path("/content/lupin-models/pt-mid")
        if not (merged / "config.json").exists():
            py("merge.py", "--model", "Qwen/Qwen3-0.6B", "--out", merged, "--adapters", *[f"{pt[g]}:{w}" for g, w in PT_MID.items()])
    if args.rp_from:  # (5 Nov) same answers and pairs as --rp-from, only the training dose changes
        pairs = BON / f"pairs-{args.rp_from}"
        drive_sync("restore", only=[str(pairs)])
        assert (pairs / "pairs.jsonl").exists(), f"{pairs} is not on Drive"
        print(f"repair {name}: pairs of {args.rp_from}", flush=True)
    reuse = getattr(args, "rp_rollouts_from", "")  # (11 Nov) another run's answers, new pairs: no draw, no generation
    own = args.rp_own_data and not args.rp_from and not reuse
    rows = {}
    for f in (() if args.rp_from or own or reuse else (pool, *[BON / f"tooltop-{x}" / "questions.jsonl" for x in extras])):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r.get("pool_id", r.get("id"))] = r
    solve = defaultdict(list)
    for line in (() if args.rp_from or own or reuse else open(answers)):
        r = json.loads(line)
        solve[r["id"]].append(bool(r["correct"]) and r["closed_thinking"] and not r["truncated"])
    from build_pool import part_of
    want = {k: int(v) for k, v in (x.split("=") for x in args.rp_questions.split(","))}
    if smoke:
        want = {k: 5 for k in want}
    rng = random.Random(0)
    by_part = defaultdict(lambda: {"mixed": [], "right": [], "wrong": []})
    for qid, s in solve.items():
        if qid in rows and part_of(rows[qid]) in want:
            p = sum(s) / len(s)
            by_part[part_of(rows[qid])]["mixed" if 0 < p < 1 else "right" if p == 1 else "wrong"].append(qid)
    chosen = []
    if own:  # (9 Nov) own sources, drawn at random per part; the trained model's own answers decide the pairs
        k, n_rounds = getattr(args, "rp_round", (0, 1))  # (10 Nov, odpo) round k of n takes the k-th slice: no question twice
        per_part = defaultdict(list)
        for r in own_repair_questions({p: n * n_rounds for p, n in want.items()}, seed=args.rp_seed,
                                      exclude=excluded_ids(args), public=getattr(args, "rp_public", False),
                                      avoid_grams=dev_gram_set() if getattr(args, "rp_exclude_dev", False) else None):
            per_part[part_of(r)].append(r)
        for part, n in want.items():
            for r in per_part[part][k * n:(k + 1) * n]:
                rows[r["id"]] = r
                chosen.append(r["id"])
    for part, n in ({} if own or reuse else want).items():
        b = by_part[part]
        for k in b:
            rng.shuffle(b[k])
        take = b["mixed"][:n]
        rest = [x for pair in zip(b["right"], b["wrong"]) for x in pair]
        take += rest[:n - len(take)]
        print(f"repair {name}: {part} {len(take)} questions ({min(n, len(b['mixed']))} mixed of {len(b['mixed'])})", flush=True)
        chosen += take
    data = Path(f"data/repair-{name}.jsonl")
    if not args.rp_from and not reuse:
        # (11 Nov) pool_id = the question's own id: probe_curve.py names an answer by pool_id, else by "source-line number",
        # and two answer files of one run (a fill batch) would then share ids for different questions
        data.write_text("".join(json.dumps(rows[q] | {"id": q, "pool_id": rows[q].get("pool_id") or q}) + "\n" for q in chosen))
    roll = BON / f"rp-{args.rp_from or reuse or name}"
    if not args.rp_from:
        drive_sync("restore", only=[str(roll)])
    if reuse:
        assert (roll / "rollouts.jsonl").exists(), f"{roll} has no rollouts on Drive"
        print(f"repair {name}: the answers of {reuse} ({roll}), new pairs", flush=True)
    answerer = Path(args.rp_answers_model) if args.rp_answers_model else merged  # (10 Nov) who answers: a new round
    if args.rp_answers_model:
        drive_sync("restore", only=[str(answerer)])
        assert (answerer / "config.json").exists(), f"{answerer} is not on Drive"
    if not args.rp_from and not reuse and not (roll / "rollouts.jsonl").exists():
        py("probe_curve.py", "--data", data, "--sources", "toolace,hermes_tools,glaive_tools,kodcode,gsm8k_matched,metamath_sv,"
           "gsm8k_easy,gsm8k_easy_test,metamath_easy,ifeval_like,ifeval_built", "--per-source", 1000000, "--samples", args.rp_samples,
           "--model", answerer, "--out", roll, "--dtype", args.dtype, "--generate-only", "--chunk", 300,
           "--max-questions", len(chosen), "--seed", 53)
        drive_sync("save")
    pairs = BON / f"pairs-{args.rp_from or name}"
    drive_sync("restore", only=[str(pairs)])
    if not (pairs / "pairs.jsonl").exists():
        quota = args.rp_quota if not smoke else ",".join(f"{k}=8" for k in want)
        rolls = [roll / "rollouts.jsonl"]
        drops = [x for x in re.split(r"[,+]", getattr(args, "rp_drop", "")) if x]
        if drops and reuse:  # (12 Nov, user: no michal-bida data) leave out the answers of these sources / id prefixes
            kept_f = BON / f"rp-{name}-kept" / "rollouts.jsonl"
            kept_f.parent.mkdir(parents=True, exist_ok=True)
            n_in = n_out = 0
            with open(roll / "rollouts.jsonl") as src_f, open(kept_f, "w") as dst_f:
                for l in src_f:
                    if not l.strip():
                        continue
                    n_in += 1
                    r = json.loads(l)
                    if any(r.get("source") == d[len("source:"):] if d.startswith("source:") else
                           str(r.get("id", "")).startswith(d[len("id:"):]) for d in drops):
                        continue
                    dst_f.write(l)
                    n_out += 1
            print(f"repair {name}: --rp-drop {drops}: kept {n_out} of {n_in} stored answers", flush=True)
            rolls = [kept_f]
        if getattr(args, "rp_extra_questions", "") and reuse:  # (11 Nov, rpM) new SVAMP-like math beside the reused answers
            want_x = {k: int(v) for k, v in (x.split("=") for x in args.rp_extra_questions.split(","))}
            roll_x = BON / f"rp-{name}-extra"
            data_x = Path(f"data/repair-{name}-extra.jsonl")
            drive_sync("restore", only=[str(roll_x), str(data_x)])
            if not (roll_x / "rollouts.jsonl").exists():
                ex = dev_ids() | dev_svamp_ids() | (excluded_ids(args) or set())
                srcs = {k: v for k, v in (x.split("=") for x in getattr(args, "rp_extra_source", "").split(",") if x)}
                rows_x = own_repair_questions({k: v for k, v in want_x.items() if k not in srcs}, seed=args.rp_seed, exclude=ex,
                                              public=True, avoid_grams=dev_gram_set()) if set(want_x) - set(srcs) else []
                for part_s, src in srcs.items():  # (12 Nov) new questions from a source outside the pool
                    from build_pool import ngrams, words
                    if not Path("questions/questions.jsonl").exists():  # (12 Nov, consult17 bug 1) a fresh runtime
                        py("utils/eval/benchmarks.py")
                    # (12 Nov, user: no michal-bida data) our IFEval sources are drawn IFEval-matched (instruction count and
                    # each instruction id's share follow IFEval's 541 prompts) with a seed of their own, so the draw is not
                    # the pool's (the pool's own IFEval rows are dropped below by exact prompt anyway)
                    prep = (["--ifeval-match", 4 * want_x[part_s], "--seed", 1000 + args.rp_seed]
                            if src in ("ifeval_like", "ifeval_built") else [])
                    cache = Path(f"data/extra-src-{src}" + (f"-m{4 * want_x[part_s]}-s{1000 + args.rp_seed}" if prep else "")
                                 + ".jsonl")
                    drive_sync("restore", only=[str(cache)])
                    if not cache.exists():
                        py("prepare_types.py", "--sources", src, "--per-source", 1000000, *prep, "--out", cache)
                    pool_texts = set()
                    drive_sync("restore", only=["runs/bon/tooltop-*/questions.jsonl"])
                    for f in [Path("data/pool-dp13.jsonl"), *BON.glob("tooltop-*/questions.jsonl")]:  # (consult17 bug 2)
                        for l in (f.read_text().splitlines() if f.exists() else []):
                            if l.strip():
                                pr = json.loads(l).get("prompt")
                                pool_texts.add(json.dumps(pr, sort_keys=True))
                    seen_t = Counter()
                    for l in Path("questions/questions.jsonl").read_text().splitlines():
                        if l.strip():
                            seen_t.update(ngrams(words(json.loads(l)["messages"])))
                    test_g = {g for g, n in seen_t.items() if n < 3}
                    dev_g = dev_gram_set() | dev_svamp_gram_set()
                    keep = []
                    for l in cache.read_text().splitlines():
                        if not l.strip():
                            continue
                        r = json.loads(l)
                        g = ngrams(words(r["prompt"]))
                        if (r["id"] in ex or json.dumps(r["prompt"], sort_keys=True) in pool_texts or g & test_g
                                or (g and len(g & dev_g) * 10 >= 3 * len(g))):
                            continue
                        keep.append(r)
                    random.Random(args.rp_seed).shuffle(keep)
                    print(f"repair {name}: {part_s} extra questions from {src}: {len(keep)} usable, taking {want_x[part_s]}", flush=True)
                    rows_x += keep[:want_x[part_s]]
                    # (12 Nov) the rest of this draw, for the fill-until-full batches below (reused answers have no other fill)
                    Path(f"data/repair-{name}-extra-left-{part_s}.jsonl").write_text(
                        "".join(json.dumps(r) + "\n" for r in keep[want_x[part_s]:]))
                data_x.write_text("".join(json.dumps(r | {"pool_id": f"sv-{r['id']}"}) + "\n" for r in rows_x))
                print(f"repair {name}: {len(rows_x)} extra questions (public datasets; ids sv-*), answered by {answerer}", flush=True)
                py("probe_curve.py", "--data", data_x, "--sources", DEV_SOURCES, "--per-source", 1000000, "--samples",
                   args.rp_samples, "--model", answerer, "--out", roll_x, "--dtype", args.dtype, "--generate-only", "--chunk", 300,
                   "--max-questions", len(rows_x), "--seed", 54)
                drive_sync("save")
            rolls.append(roll_x / "rollouts.jsonl")

        def build():
            py("repair_pairs.py", "--rollouts", *rolls, "--out", pairs, "--quota", quota, "--kinds", args.rp_kinds,
               "--seed", getattr(args, "rp_dpo_seed", 0), *[x for x in args.rp_pair_args.split(",") if x])
        build()
        # (12 Nov, user: "enough data for a full pass?") reused answers + new questions from an outside source: if a part
        # with such a source is still short, answer up to 2 more batches of that source's leftover questions and rebuild
        srcs_fill = {k for k, v in (x.split("=") for x in getattr(args, "rp_extra_source", "").split(",") if x)} \
            if reuse and getattr(args, "rp_extra_questions", "") and not smoke else set()
        for extra in range(1, 4 if srcs_fill else 1):  # up to 3 batches
            short = json.loads((pairs / "shortfall.json").read_text()) if (pairs / "shortfall.json").exists() else {}
            short = {k: v for k, v in short.items() if v > 0 and k in srcs_fill}
            if not short:
                break
            taken = {json.loads(l)["id"] for f in rolls for l in open(f) if l.strip()}
            rows_x = []
            for part_s in short:
                left_f = Path(f"data/repair-{name}-extra-left-{part_s}.jsonl")
                left = [json.loads(l) for l in (left_f.read_text().splitlines() if left_f.exists() else []) if l.strip()]
                left = [r for r in left if r["id"] not in taken and f"sv-{r['id']}" not in taken]
                n_more = min(len(left), max(400, 6 * short[part_s]))
                rows_x += left[:n_more]
                left_f.write_text("".join(json.dumps(r) + "\n" for r in left[n_more:]))
            print(f"repair {name}: outside-source fill, batch {extra}: short {short}; answering {len(rows_x)} more questions",
                  flush=True)
            if not rows_x:
                break
            data_x = Path(f"data/repair-{name}-xs{extra}.jsonl")
            roll_x = BON / f"rp-{name}-xs{extra}"
            data_x.write_text("".join(json.dumps(r | {"pool_id": f"sv-{r['id']}"}) + "\n" for r in rows_x))
            drive_sync("restore", only=[str(roll_x)])
            if not (roll_x / "rollouts.jsonl").exists():
                py("probe_curve.py", "--data", data_x, "--sources", DEV_SOURCES, "--per-source", 1000000, "--samples",
                   args.rp_samples, "--model", answerer, "--out", roll_x, "--dtype", args.dtype, "--generate-only", "--chunk", 300,
                   "--max-questions", len(rows_x), "--seed", 54 + extra)
                drive_sync("save")
            rolls.append(roll_x / "rollouts.jsonl")
            shutil.rmtree(pairs, ignore_errors=True)
            build()
        # (11 Nov, user) fill until full: with --kinds mix the proportion is fixed, so a short part means fewer pairs; on an
        # own-question draw, answer MORE new questions for the short parts (up to 2 extra batches) and rebuild
        for extra in range(1, 3 if own and args.rp_kinds == "mix" and not smoke else 1):
            short = json.loads((pairs / "shortfall.json").read_text()) if (pairs / "shortfall.json").exists() else {}
            short = {k: v for k, v in short.items() if v > 0.1 * dict((x.split("=")[0], int(x.split("=")[1]))
                                                                       for x in quota.split(",")).get(k, 0)}
            if not short:
                break
            more = {k: max(100, 3 * v) for k, v in short.items()}
            taken = {json.loads(l)["id"] for f in rolls for l in open(f) if l.strip()}
            ex = (excluded_ids(args) or set()) | taken
            rows_x = [r for r in own_repair_questions(more, seed=args.rp_seed + 100 * extra, exclude=ex,
                                                       public=getattr(args, "rp_public", False),
                                                       avoid_grams=dev_gram_set() if getattr(args, "rp_exclude_dev", False) else None)]
            print(f"repair {name}: fill until full, batch {extra}: short {short}; answering {len(rows_x)} more questions", flush=True)
            if not rows_x:
                break
            data_x = Path(f"data/repair-{name}-x{extra}.jsonl")
            data_x.write_text("".join(json.dumps(r | {"pool_id": r.get("pool_id") or r["id"]}) + "\n" for r in rows_x))
            roll_x = BON / f"rp-{name}-x{extra}"
            drive_sync("restore", only=[str(roll_x)])
            if not (roll_x / "rollouts.jsonl").exists():
                py("probe_curve.py", "--data", data_x, "--sources", DEV_SOURCES, "--per-source", 1000000, "--samples",
                   args.rp_samples, "--model", answerer, "--out", roll_x, "--dtype", args.dtype, "--generate-only", "--chunk", 300,
                   "--max-questions", len(rows_x), "--seed", 53 + extra)
                drive_sync("save")
            rolls.append(roll_x / "rollouts.jsonl")
            shutil.rmtree(pairs, ignore_errors=True)
            build()
        drive_sync("save")
    print((pairs / "summary.txt").read_text(), flush=True)
    if getattr(args, "rp_require_full", False) and not smoke:  # (12 Nov, user: "do not start until we have enough pairs")
        need = sum(int(x.split("=")[1]) for x in args.rp_quota.split(",") if x)
        have = sum(1 for l in open(pairs / "pairs.jsonl") if l.strip())
        if have < need:
            print(f"repair {name}: STOPPED BEFORE TRAINING: {have} pairs < {need} (a full pass needs all of them); "
                  f"shortfall {(pairs / 'shortfall.json').read_text() if (pairs / 'shortfall.json').exists() else '?'}", flush=True)
            drive_sync("save")
            return
        print(f"repair {name}: {have} pairs = the full {need}", flush=True)
    if getattr(args, "rp_pairs_only", False):  # (12 Nov, user: "nothing starts until we have the IFEval data ready")
        print(f"repair {name}: PAIRS ONLY: answers and pairs saved ({pairs}); no training. Rerun the line without "
              f"--rp-pairs-only to train on exactly these pairs.", flush=True)
        drive_sync("save")
        return
    adapter = Path("adapters") / f"{name}-s0"
    start = Path(args.rp_init_from) if args.rp_init_from else Path("adapters") / f"fresh-{name}"  # (10 Nov) continue an adapter
    drive_sync("restore", only=[str(adapter), str(start)])
    if args.rp_init_from:
        assert (start / "adapter_config.json").exists(), f"{start} is not on Drive"
    elif not (start / "adapter_config.json").exists():
        py("sft.py", "--init-only", "--model", merged, "--out", start, "--seed", getattr(args, "rp_dpo_seed", 0))
    steps = 4 if smoke else args.rp_steps
    if not (adapter / f"step-{steps}" / "adapter_config.json").exists():
        py("dpo.py", "--model", merged, "--pairs", pairs / "pairs.jsonl", "--init-adapter", start, "--out", adapter,
           "--mix", "all", "--keep-order", "--nll", 0.0, "--beta", 0.1, "--batch", 16, "--lr", args.rp_lr,
           "--lr-schedule", "cosine", "--warmup", getattr(args, "rp_warmup", 2),
           "--lr-min", args.rp_lr if getattr(args, "rp_lr_min", None) is None else args.rp_lr_min, "--max-steps", steps, "--epochs", args.rp_epochs,
           "--save-every",
           1 if smoke else (10 ** 6 if getattr(args, "rp_save_at", None) else 5), "--held-out", 0, "--max-len", 8704,
           "--bida-precision", "--dtype", args.dtype, "--seed", getattr(args, "rp_dpo_seed", 0),
           *(["--save-at", *args.rp_save_at] if getattr(args, "rp_save_at", None) and not smoke else []))
        drive_sync("save")
    if smoke:
        print(f"repair {name}: SMOKE done (answers, pairs, {steps} training steps)", flush=True)
        return
    if getattr(args, "rp_dev", False):  # (11 Nov) choices on Dev-512 only (consult6 protocol), never the 720
        for step in args.rp_screen:
            if args.rp_base:
                spec = f"/content/lupin-models/{name}-{step}={args.rp_base}:{adapter / f'step-{step}'}"
                print(dev_score_model(f"{name}-{step}", spec, vs=getattr(args, "rp_dev_vs", "")), flush=True)
                if getattr(args, "rp_dev_svamp", False):
                    print(dev_score_model(f"{name}-{step}", spec, vs=getattr(args, "rp_dev_vs", ""), dev="svamp200"), flush=True)
        return
    for step in args.rp_screen:
        out = ROOT / f"s4-{name}-{step}"
        drive_sync("restore", only=[str(out)])
        if not (out / "results.json").exists():
            model = Path("/content/lupin-models") / f"{name}-{step}"
            if args.rp_base:
                py("merge.py", "--model", args.rp_base, "--out", model, "--adapters", f"{adapter / f'step-{step}'}:1")
            else:
                py("merge.py", "--model", "Qwen/Qwen3-0.6B", "--out", model, "--adapters",
                   *[f"{pt[g]}:{w}" for g, w in PT_MID.items()], f"{adapter / f'step-{step}'}:1")
            evaluate(out, samples=4, questions=SCREEN, model=model)
            shutil.rmtree(model, ignore_errors=True)
            drive_sync("save")
        print(score_line(f"{name}-{step} (screen 720 x 4)", out, base=ROOT / "s4-base"), flush=True)


def odpo(args):
    """Online DPO (10 Nov, user): --od-rounds rounds of: the CURRENT model (--rp-base + the adapter so far, merged) answers a
    new slice of own questions x --rp-samples; repair_pairs.py builds BOTH kinds from those answers (--rp-kinds guard,shorten:
    shortest right > shortest wrong, and shortest right > longest right on questions it gets right >= 6 of 8 times); DPO
    --od-steps steps continuing the adapter, the round's start as the reference. Each round is a ptrepair run named
    <name>-r<k>; the answering model of the next round is merged to models/<name>-r<k>. 720 x4 at --od-screen-rounds."""
    import copy
    prev, answerer = args.od_init, args.od_answers0 or args.rp_base  # no --od-init: round 1 trains a fresh LoRA on --rp-base
    assert args.rp_base and (answerer or not prev), "odpo needs --rp-base (and --od-answers0 with --od-init)"
    for k in range(args.od_rounds):
        a = copy.copy(args)
        a.name = f"{args.name}-r{k + 1}"
        a.rp_round = (k, args.od_rounds)
        a.rp_init_from, a.rp_answers_model = prev, ("" if answerer == args.rp_base else answerer)
        a.rp_steps = args.od_steps
        a.rp_epochs = max(args.rp_epochs, 2)  # (10 Nov, reviewer) a round short of 160 pairs still reaches step --od-steps
        a.rp_screen = [args.od_steps] if k + 1 in args.od_screen_rounds else []
        print(f"odpo {args.name}: round {k + 1}/{args.od_rounds}: {answerer} answers, DPO from {prev}", flush=True)
        ptrepair(a)
        prev = f"adapters/{a.name}-s0/step-{args.od_steps}"
        if k + 1 < args.od_rounds:
            nxt = Path("models") / a.name
            drive_sync("restore", only=[str(nxt)])
            if not (nxt / "config.json").exists():
                py("merge.py", "--model", args.rp_base, "--out", nxt, "--adapters", f"{prev}:1")
                drive_sync("save")
            answerer = str(nxt)
    print(f"odpo {args.name}: done; last adapter {prev}", flush=True)


DEV_FILE = Path("data/dev512.jsonl")
DEV_SVAMP_FILE = Path("data/dev-svamp200.jsonl")  # (11 Nov) 200 SVAMP-like math questions: Dev-512's math is hard GSM8K
DEV_SETS = {"": (DEV_FILE, "dev512", "Dev-512"), "svamp200": (DEV_SVAMP_FILE, "devsv", "Dev-SVAMP-200")}
DEV_SOURCES = ("toolace,hermes_tools,glaive_tools,kodcode,gsm8k_matched,metamath_sv,gsm8k_easy,gsm8k_easy_test,metamath_easy,"
               "ifeval_like,ifeval_built")


def excluded_ids(args):
    """(11 Nov) questions a new draw must skip: Dev-512 (--rp-exclude-dev) and a JSON id list (--rp-exclude-ids)."""
    out = set()
    if getattr(args, "rp_exclude_dev", False):
        out |= dev_ids() | dev_svamp_ids()
    f = getattr(args, "rp_exclude_ids", "")
    if f:
        drive_sync("restore", only=[f])
        out |= set(json.loads(Path(f).read_text()))
    return out or None


P1_ANSWERS = "lupin-all-dp13-nofit1-tools1-code1-nofit2-nofit4-math1-math2.jsonl"  # the original Qwen's 10 answers per question
P1_IDS = Path("data/p1-round1-ids.json")
# (11 Nov) % of each type's questions kept UNSEEN for round 2 (a fixed hash of the id). Sized on real answers: round 2 needs
# ~1,000 IFEval questions for its counting-rule half (2,998 exist) and ~250 "several functions" questions (716 exist)
P1_RESERVE = {"ifeval": 33, "bfcl_multiple": 35}
P1_RESERVE_DEFAULT = 25


def p1src(args):
    """(11 Nov, user: consult6 path 1 from the ORIGINAL Qwen; revised after both reviews) Round 1's material:
    the original's 10 STORED answers per pool question (runs/bon/P1_ANSWERS), public datasets allowed (user, 11 Nov: MetaMath
    and the math1 / math2 / nofit4 top-ups; recipes stay off-limits), never a Dev-512 question nor a near-duplicate of one, and
    75% of the questions (a fixed hash of the id; the rest stay unseen for round 2: 25%, IFEval 33%, several functions 35%,
    P1_RESERVE). No new generation.
    Written as runs/bon/rp-p1src/rollouts.jsonl (+ data/p1-round1-ids.json: round 2 passes --rp-exclude-ids)."""
    import hashlib
    from build_pool import ngrams, part_of, words
    out = BON / "rp-p1src"
    drive_sync("restore", only=[str(out), str(P1_IDS)])
    if (out / "rollouts.jsonl").exists() and P1_IDS.exists():
        print(f"p1src: already built ({sum(1 for _ in open(out / 'rollouts.jsonl'))} answers)", flush=True)
        return
    dev = dev_ids()
    assert dev, "run devset first (Dev-512 must exist before round 1 is drawn)"
    seen = Counter()  # rare 8-grams of Dev-512 only: shared templates (>= 3 dev questions) are not evidence of a duplicate
    for l in DEV_FILE.read_text().splitlines():
        if l.strip():
            seen.update(ngrams(words(json.loads(l)["prompt"])))
    dev_grams = {g for g, n in seen.items() if n < 3}

    def near_dev(prompt):  # a near-duplicate: >= 30% of its 8-grams are rare Dev-512 8-grams (the project's contamination rule)
        g = ngrams(words(prompt))
        return bool(g) and len(g & dev_grams) / len(g) >= 0.3
    tops_banned = []  # (11 Nov, user) the public datasets are allowed: nothing banned
    tops_own = [BON / f"tooltop-{x}" / "questions.jsonl" for x in ("nofit1", "tools1", "code1", "nofit2", "nofit4", "math1", "math2")]
    pool = Path("data/pool-dp13.jsonl")
    drive_sync("restore", only=[str(BON / P1_ANSWERS), str(pool), *[str(t) for t in tops_banned + tops_own]])
    banned, prompt_of = set(), {}
    for t in tops_banned:
        assert t.exists(), f"{t} is needed to drop the michal-bida top-ups"
        for l in t.read_text().splitlines():
            if l.strip():
                q = json.loads(l)
                banned.add(q.get("pool_id", q.get("id")))
    for f in [pool, *tops_own]:
        for l in (f.read_text().splitlines() if f.exists() else []):
            if l.strip():
                q = json.loads(l)
                prompt_of[q.get("pool_id", q.get("id"))] = q.get("prompt")
    keep_ids, n_in, n_out, drop = set(), 0, 0, Counter()
    near = {}
    out.mkdir(parents=True, exist_ok=True)
    with open(BON / P1_ANSWERS) as src, open(out / "rollouts.part", "w") as dst:
        for line in src:
            if not line.strip():
                continue
            r = json.loads(line)
            n_in += 1
            if n_in == 1:
                missing = [k for k in ("prompt_ids", "output_ids", "thinking_ids", "correct", "closed_thinking", "truncated",
                                       "output_tokens", "source") if k not in r]
                assert not missing, f"the stored answers lack {missing}: cannot build pairs from them"
            qid = r["id"]
            if qid in dev:
                drop["Dev-512"] += 1
                continue
            if qid not in near:
                pr = prompt_of.get(qid)
                near[qid] = bool(pr is not None and near_dev(pr))
            if near[qid]:
                drop["shares an 8-gram with Dev-512"] += 1
                continue
            if int(hashlib.md5(str(qid).encode()).hexdigest(), 16) % 100 < P1_RESERVE.get(part_of(r), P1_RESERVE_DEFAULT):
                drop["kept for round 2"] += 1
                continue
            keep_ids.add(qid)
            dst.write(line if line.endswith("\n") else line + "\n")
            n_out += 1
    print(f"p1src: {n_out} of {n_in} stored answers kept ({len(keep_ids)} questions); dropped {dict(drop)}", flush=True)
    (out / "rollouts.part").rename(out / "rollouts.jsonl")
    P1_IDS.parent.mkdir(exist_ok=True)
    P1_IDS.write_text(json.dumps(sorted(keep_ids)))
    print(f"p1src: round 1 = {len(keep_ids)} questions", flush=True)
    drive_sync("save")


def dev_gram_set():
    """(11 Nov) The rare 8-grams of Dev-512 (a template shared by >= 3 dev questions is not evidence of a duplicate): a
    training question with >= 30% of its 8-grams in this set is a near-duplicate of a dev question."""
    from build_pool import ngrams, words
    dev_ids()
    seen = Counter()
    for l in (DEV_FILE.read_text().splitlines() if DEV_FILE.exists() else []):
        if l.strip():
            seen.update(ngrams(words(json.loads(l)["prompt"])))
    return {g for g, n in seen.items() if n < 3} | dev_svamp_gram_set()  # (12 Nov, consult17 bug 3) both Dev sets


def dev_svamp_gram_set():
    """(12 Nov, consult16) the rare 8-grams of Dev-SVAMP-200, as dev_gram_set() for Dev-512."""
    from build_pool import ngrams, words
    dev_svamp_ids()
    seen = Counter()
    for l in (Path(DEV_SVAMP_FILE).read_text().splitlines() if Path(DEV_SVAMP_FILE).exists() else []):
        if l.strip():
            seen.update(ngrams(words(json.loads(l)["prompt"])))
    return {g for g, n in seen.items() if n < 3}

def dev_ids():
    """(11 Nov) ids of the Dev-512 questions (never trained on once a run passes --rp-exclude-dev)."""
    drive_sync("restore", only=[str(DEV_FILE)])
    return {json.loads(l)["id"] for l in DEV_FILE.read_text().splitlines() if l.strip()} if DEV_FILE.exists() else set()


def devset(args):
    """Dev-512 (11 Nov, consult6 protocol): 512 own held-out questions for every choice (never the test, never the 720):
    math 140, code 100, IFEval 132 (66 with a counting rule, 66 without), tools 140 (60 one function, 35 several, 45 no fit).
    Drawn from the own pool OUTSIDE every question already drawn for training by rpE/rpG/rpH (seed 0), rpI (seed 1) and od1
    (seed 7, 4 rounds); later runs exclude it with --rp-exclude-dev. Then the ORIGINAL Qwen answers it x8 (the reference)."""
    from repair_pairs import counting
    from build_pool import part_of
    drive_sync("restore", only=[str(DEV_FILE)])
    if not DEV_FILE.exists():
        used = set()
        sizes0 = {"svamp": 2400, "mbpp": 900, "ifeval": 1500, "bfcl_simple_python": 300, "bfcl_multiple": 300, "bfcl_irrelevance": 300}
        od1 = {"svamp": 480 * 4, "mbpp": 180 * 4, "ifeval": 450 * 4, "bfcl_simple_python": 240, "bfcl_multiple": 240,
               "bfcl_irrelevance": 240}
        for seed, sizes in ((0, sizes0), (1, sizes0), (7, od1)):
            used |= {r["id"] for r in own_repair_questions(sizes, seed=seed)}
        every = own_repair_questions({p: 10 ** 6 for p in sizes0}, seed=99)
        want = {"svamp": 140, "mbpp": 100, "bfcl_simple_python": 60, "bfcl_multiple": 35, "bfcl_irrelevance": 45}
        picked, counts = [], defaultdict(int)
        for r in every:
            if r["id"] in used:
                continue
            p = part_of(r)
            if p == "ifeval":
                key = "ifeval-count" if counting(r) else "ifeval-plain"
                if counts[key] < 66:
                    counts[key] += 1
                    picked.append(r)
            elif counts[p] < want.get(p, 0):
                counts[p] += 1
                picked.append(r)
        print(f"devset: {len(picked)} questions {dict(counts)} (outside {len(used)} questions already drawn for training)", flush=True)
        DEV_FILE.parent.mkdir(exist_ok=True)
        DEV_FILE.write_text("".join(json.dumps(r) + "\n" for r in picked))
        drive_sync("save")
    dev_generate("base", "Qwen/Qwen3-0.6B")


def dev_generate(name, model, dev=""):
    dev_file, prefix, _ = DEV_SETS[dev]
    rows = [l for l in dev_file.read_text().splitlines() if l.strip()]
    out = BON / f"{prefix}-{name}"
    drive_sync("restore", only=[str(out)])
    if not (out / "rollouts.jsonl").exists() or sum(1 for _ in open(out / "rollouts.jsonl")) < 8 * len(rows):
        py("probe_curve.py", "--data", dev_file, "--sources", DEV_SOURCES, "--per-source", 1000000, "--samples", 8,
           "--model", model, "--out", out, "--dtype", "bfloat16", "--generate-only", "--chunk", 300,
           "--max-questions", len(rows), "--seed", 53)
        drive_sync("save")
    return out


def dev_score_model(name, spec, vs="", dev=""):
    """Score one model on a dev set x8 against the original: spec = a merged model / Hub id, or OUT=BASE:ADAPTER (merged here).
    vs (11 Nov): the name of an already scored run; dev_score.py then also prints the paired difference against it.
    dev (11 Nov): "" = Dev-512, "svamp200" = the SVAMP-like math slice (Dev-512's math is hard GSM8K and missed rpK's math loss)."""
    dev_file, prefix, label = DEV_SETS[dev]
    dev_ids()
    drive_sync("restore", only=[str(dev_file)])
    assert dev_file.exists(), f"{dev_file} is missing: run the {'devsvamp' if dev else 'devset'} stage first"
    base = dev_generate("base", "Qwen/Qwen3-0.6B", dev)
    model = spec
    if "=" in spec:
        model, rest = spec.split("=")
        merge_base, adapter = rest.split(":")
        if not (Path(model) / "config.json").exists():
            drive_sync("restore", only=[*([] if merge_base.startswith("Qwen/") else [merge_base]), adapter])
            py("merge.py", "--model", merge_base, "--out", model, "--adapters", f"{adapter}:1")
    else:
        drive_sync("restore", only=[spec] if not spec.startswith("Qwen/") else [])
    out = dev_generate(name, model, dev)
    if "=" in spec and model.startswith("/content/"):
        shutil.rmtree(model, ignore_errors=True)
    res = out / "dev_score.json"
    ref = BON / f"{prefix}-{vs}" / "rollouts.jsonl" if vs else None
    if ref is not None:
        drive_sync("restore", only=[str(ref.parent)])
        if not ref.exists():
            print(f"devscore {name}: no {label} answers of {vs} on Drive: scored without the paired difference", flush=True)
            ref = None
    run(sys.executable, "dev_score.py", "--base", base / "rollouts.jsonl", "--cand", out / "rollouts.jsonl", "--name", name,
        "--out", res, "--label", label, *(["--vs", ref] if ref is not None else []))
    drive_sync("save")
    d = json.loads(res.read_text())
    return (f"== {name} ({label} x8): saved {d['saved']:+.1f}% [{d['saved_ci'][0]:+.1f}, {d['saved_ci'][1]:+.1f}], "
            f"accuracy {d['acc']:+.2f} [{d['acc_ci'][0]:+.2f}, {d['acc_ci'][1]:+.2f}]")


def competitor_model(name, spec):
    """(11 Nov, user: compare the competitors' generalisation) A merged model from a competitor's PUBLIC weights, measured only:
    hf:REPO = a full model on the Hub; hflora:REPO = a LoRA adapter on the Hub, merged onto Qwen3-0.6B; zip:URL = a release
    archive holding a LoRA adapter (adapter_config.json), merged onto Qwen3-0.6B. Kept under /content/competitors (never Drive)."""
    import urllib.request
    import zipfile
    from huggingface_hub import snapshot_download
    root = Path("/content/competitors") / name
    merged = root / "merged"
    if (merged / "config.json").exists():
        return str(merged)
    root.mkdir(parents=True, exist_ok=True)
    kind, ref = spec.split(":", 1)
    if kind == "hf":
        return snapshot_download(ref, local_dir=str(root / "hub"),
                                 allow_patterns=["config.json", "generation_config.json", "model.safetensors", "tokenizer*", "vocab.json", "merges.txt"])
    if kind == "hflora":
        adapter = Path(snapshot_download(ref, local_dir=str(root / "hub"), allow_patterns=["adapter_config.json", "adapter_model.safetensors"]))
    else:
        archive = root / "release.zip"
        urllib.request.urlretrieve(ref, archive)
        with zipfile.ZipFile(archive) as z:
            z.extractall(root / "release")
        adapter = next(p.parent for p in (root / "release").rglob("adapter_config.json"))
    py("merge.py", "--model", "Qwen/Qwen3-0.6B", "--out", merged, "--adapters", f"{adapter}:1")
    return str(merged)


TRANSFER_FILE = Path("data/transfer.jsonl")
TRANSFER_SIZES = {"tr_math500": 1000, "tr_skywork": 200, "tr_arc_c": 200, "tr_csqa": 200, "tr_mmlu": 250, "tr_humaneval": 164}
# (12 Nov) the second, "generalization" set: the types the first lacks (tool calls: BFCL live; instructions: IFBench; more
# multiple choice: BBH option tasks, BoolQ). Used as --tr-sets v2 --tr-tag v2 (or spelled out as SOURCE=N,...).
# tr_bbh: 17 of the 27 BBH tasks have option answers, x 10 per task = 170 (270 asked; the other 10 answer yes/no, numbers or words).
TRANSFER_V2_SIZES = {"tr_ifbench": 300, "tr_bfcl_live_simple": 120, "tr_bfcl_live_multiple": 100, "tr_bfcl_live_irrelevance": 80,
                     "tr_bbh": 170, "tr_boolq": 200, "tr_math500": 311}


def hold(args):
    """(12 Nov) Keep the runtime alive and idle for --hold-minutes (a plan whose next line is `end`, or an empty queue for
    12 minutes, stops the runtime). Used to wait for the user's go between steps. The function was missing (KeyError)."""
    import time
    minutes = float(getattr(args, "hold_minutes", 30) or 30)
    print(f"hold: idle {minutes:.0f} min ({getattr(args, 'hold_tag', '')})", flush=True)
    end = time.time() + 60 * minutes
    while time.time() < end:
        time.sleep(min(300, max(1, end - time.time())))
        drive_sync("save")


def transfer(args):
    """(11 Nov, user: "how much does this transfer ... or is it benchmark overfitting?") Benchmarks our training never used nor
    imitated: MATH-500 (whole-number answers), Skywork-OR1 easy math, ARC-Challenge, CommonsenseQA, MMLU, HumanEval. Each model
    in --tr-models (NAME=SPEC, SPEC as devscore's) answers them x8 (user: as the test); transfer_score.py compares it with the original, per set."""
    global TRANSFER_FILE, TRANSFER_SIZES
    if args.tr_sets == "v2":  # (12 Nov) shorthand for the second transfer set
        args.tr_sets = ",".join(f"{k}={v}" for k, v in TRANSFER_V2_SIZES.items())
    if args.tr_sets:  # (12 Nov, user) e.g. AIME 2025 at a 32k cap, as arXiv 2602.20945 measures it
        TRANSFER_SIZES = {k: int(v) for k, v in (x.split("=") for x in args.tr_sets.split(","))}
        TRANSFER_FILE = Path(f"data/transfer-{args.tr_tag or 'x'}.jsonl")
    drive_sync("restore", only=[str(TRANSFER_FILE)])
    if not TRANSFER_FILE.exists():
        tmp = Path("data/transfer-all.jsonl")
        py("prepare_types.py", "--sources", ",".join(TRANSFER_SIZES), "--per-source", 1000000, "--out", tmp)
        rows = [json.loads(l) for l in tmp.read_text().splitlines() if l.strip()]
        rng = random.Random(5)
        by = defaultdict(list)
        for r in rows:
            by[r["source"]].append(r)
        out = []
        for src, n in TRANSFER_SIZES.items():
            rng.shuffle(by[src])
            out += by[src][:n]
            print(f"transfer: {src} {min(n, len(by[src]))} of {len(by[src])}", flush=True)
        TRANSFER_FILE.write_text("".join(json.dumps(r) + "\n" for r in out))
        drive_sync("save")
    rows = [l for l in TRANSFER_FILE.read_text().splitlines() if l.strip()]
    tr_caps = args.tr_source_caps or (",".join(f"{k}=32768" for k in ("tr_math500", "tr_skywork") if k in TRANSFER_SIZES) if args.tr_tag else "")
    print(f"transfer: per-source caps: {tr_caps or 'none (4,096 / code 8,192)'}", flush=True)
    outs = []
    items = list(args.tr_models) + ["base=Qwen/Qwen3-0.6B"] if args.tr_wait_base else ["base=Qwen/Qwen3-0.6B"] + list(args.tr_models)
    for item in items:
        name, spec = item.split("=", 1)
        out = BON / (f"transfer-{args.tr_tag}-{name}" if args.tr_sets else f"transfer-{name}")
        drive_sync("restore", only=[str(out)])
        complete = lambda: (out / "rollouts.jsonl").exists() and sum(1 for _ in open(out / "rollouts.jsonl")) >= 8 * len(rows)  # noqa: E731
        if args.tr_wait_base and name == "base":
            import time
            for _ in range(14 * 6):  # the other GPU writes the original's answers; poll Drive every 10 minutes
                if complete():
                    break
                print("transfer: waiting for the original's answers from the other GPU", flush=True)
                time.sleep(600)
                drive_sync("restore", only=[str(out)])
        if not complete():
            model = spec
            if spec.startswith(("hf:", "hflora:", "zip:")):  # (11 Nov, user) a competitor's PUBLIC weights, for comparison only
                model = competitor_model(name, spec)
            elif "=" in spec:
                model, rest = spec.split("=")
                merge_base, adapter = rest.split(":")
                if not (Path(model) / "config.json").exists():
                    drive_sync("restore", only=[*([] if merge_base.startswith("Qwen/") else [merge_base]), adapter])
                    py("merge.py", "--model", merge_base, "--out", model, "--adapters", f"{adapter}:1")
            else:
                drive_sync("restore", only=[spec] if not spec.startswith("Qwen/") else [])
            py("probe_curve.py", "--data", TRANSFER_FILE, "--sources", ",".join(TRANSFER_SIZES), "--per-source", 1000000,
               "--samples", 8, "--model", model, "--out", out, "--dtype", "bfloat16", "--generate-only", "--chunk", 300,
               "--max-tokens", args.tr_max_tokens, "--source-max-tokens", tr_caps,
               "--max-questions", len(rows), "--seed", 61)
            drive_sync("save")
        outs.append((name, out))
    if args.tr_wait_base:
        outs = outs[-1:] + outs[:-1]  # the original first, as the scorer expects
    for name, out in outs[1:]:
        run(sys.executable, "transfer_score.py", "--base", outs[0][1] / "rollouts.jsonl", "--cand", out / "rollouts.jsonl",
            "--name", name, "--out", out / "transfer_score.json", "--questions", TRANSFER_FILE)
    drive_sync("save")


def devsvamp(args):
    """(11 Nov, consult10) Dev-SVAMP-200: 200 SVAMP-like math questions (the public pool's easy math, as the test's SVAMP)
    never used in round 1 (data/p1-round1-ids.json), not in Dev-512 nor near one of its questions, not trained on in rounds 1-4
    (own_repair_questions drops those). Dev-512's own math is full-difficulty GSM8K: it did not see rpK step 20's math loss.
    The original answers them x8 (the reference). Rows keep their pool id in "id" (for exclusions) and answer under dsv-<id>."""
    drive_sync("restore", only=[str(DEV_SVAMP_FILE), str(DEV_FILE), str(P1_IDS)])
    if not DEV_SVAMP_FILE.exists():
        ex = dev_ids() | (set(json.loads(P1_IDS.read_text())) if P1_IDS.exists() else set())
        rows = own_repair_questions({"svamp": 200}, seed=97, exclude=ex, public=True, avoid_grams=dev_gram_set())
        assert len(rows) >= 150, f"only {len(rows)} SVAMP-like questions left for Dev-SVAMP-200"
        DEV_SVAMP_FILE.parent.mkdir(exist_ok=True)
        DEV_SVAMP_FILE.write_text("".join(json.dumps(r | {"pool_id": f"dsv-{r['id']}"}) + "\n" for r in rows))
        print(f"devsvamp: {len(rows)} questions; sources {dict(Counter(r['source'] for r in rows))}", flush=True)
        drive_sync("save")
    dev_generate("base", "Qwen/Qwen3-0.6B", "svamp200")


def dev_svamp_ids():
    drive_sync("restore", only=[str(DEV_SVAMP_FILE)])
    return ({json.loads(l)["id"] for l in DEV_SVAMP_FILE.read_text().splitlines() if l.strip()} if DEV_SVAMP_FILE.exists() else set())


def devscore(args):
    print(dev_score_model(args.ds_name, args.ds_model, vs=getattr(args, "ds_vs", ""), dev=getattr(args, "ds_dev", "")), flush=True)


def ladder(args):
    """Budget ladder, phase 0 (BUDGET_LADDER_PLAN.md, user 10 Nov): measurement only, no training. The ORIGINAL Qwen answers
    --ld-questions own training questions (never an evaluation question) --ld-samples times; budget_ladder.py then cuts every
    answer at each per-type rung, one rung per call (saved to Drive between rungs), and prints the needed-budget distribution
    and the oracle curve with the go / no-go rule. Results: runs/bon/ld-<name>/ladder/summary.txt and ladder.jsonl."""
    name = args.name
    want = {k: int(v) for k, v in (x.split("=") for x in args.ld_questions.split(","))}
    roll = BON / f"ld-{name}"
    drive_sync("restore", only=[str(roll)])
    data = Path(f"data/ladder-{name}.jsonl")
    if not (roll / "rollouts.jsonl").exists():
        rows = own_repair_questions(want, seed=args.ld_seed)
        data.write_text("".join(json.dumps(r) + "\n" for r in rows))
        py("probe_curve.py", "--data", data, "--sources", "toolace,hermes_tools,glaive_tools,kodcode,gsm8k_matched,metamath_sv,"
           "gsm8k_easy,gsm8k_easy_test,metamath_easy,ifeval_like,ifeval_built", "--per-source", 1000000, "--samples", args.ld_samples,
           "--model", "Qwen/Qwen3-0.6B", "--out", roll, "--dtype", args.dtype, "--generate-only", "--chunk", 300,
           "--max-questions", len(rows), "--seed", 53)
        drive_sync("save")
    out = roll / "ladder"
    for _ in range(30):  # one rung per call, saved between rungs: a lost runtime resumes at the next rung
        if (out / "LADDER_DONE").exists():
            break
        py("budget_ladder.py", "--rollouts", roll / "rollouts.jsonl", "--out", out, "--dtype", args.dtype,
           "--tolerance", args.ld_tolerance, "--max-new-rounds", 1)
        drive_sync("save")
    if not (out / "summary.txt").exists():  # done on an earlier run: print the summary again
        py("budget_ladder.py", "--rollouts", roll / "rollouts.jsonl", "--out", out, "--dtype", args.dtype,
           "--tolerance", args.ld_tolerance)
        drive_sync("save")
    print((out / "summary.txt").read_text(), flush=True)


def mixscreen(args):
    """Merge per-type adapters at chosen strengths and score the merge on the 720 x4 (5 Nov, finishing lpt6 as the
    consultant specified): each part's adapter is the average of --mx-seeds (weight split across seeds), parts at
    --mx-weights; no training. Result: runs/mvtf/s4-<name>."""
    name = args.name
    weights = {k: float(v) for k, v in (x.split("=") for x in args.mx_weights.split(",") if x)}
    members = []
    for part, w in weights.items():
        if w <= 0:
            continue
        for seed in args.mx_seeds:
            a = Path("adapters") / f"{args.mx_base}-{part}-s{seed}"
            drive_sync("restore", only=[str(a)])
            assert (a / "adapter_config.json").exists(), f"{a} is not on Drive"
            members.append(f"{a}:{w / len(args.mx_seeds):.6f}")
    out = ROOT / f"s4-{name}"
    drive_sync("restore", only=[str(out), SCREEN, str(ROOT / "s4-base")])
    if not (out / "results.json").exists():
        model = Path("/content/lupin-models") / name
        py("merge.py", "--model", "Qwen/Qwen3-0.6B", "--out", model, "--adapters", *members)
        evaluate(out, samples=4, questions=SCREEN, model=model)
        shutil.rmtree(model, ignore_errors=True)
        drive_sync("save")
    print(score_line(f"{name} {weights} (screen 720 x 4)", out, base=ROOT / "s4-base"), flush=True)


def thinkprune(args):
    """ThinkPrune-style RL (2 Nov, user): reward = 1 if right AND finished under the question's own token cap, else 0
    (rl_train.py --reward cap), group advantages, a small KL anchor, nothing else. Caps per test part from the ORIGINAL
    model's right answers in the pool (math generous: --tp-cap-math quantile; the rest --tp-cap-other), printed with the
    share of questions keeping a right answer under the cap. Questions the original gets right > half the time, mixed
    like the test (25% each; BFCL 400/200/240). Starts from --tp-start ("raw" or an adapter, merged first). 10-step
    blocks (resumable), the 720 every --tp-score-every steps. --tp-smoke: 5 steps, small batch, no scoring, timed."""
    import statistics
    import time
    name = args.name
    # the same questions as v1's DPO data (3 Nov, user: same data for the DPO and GRPO comparisons): pool dp13 + top-ups
    extra = [x for x in args.tp_extra.split(",") if x]
    # (6 Nov, user) --tp-answers10: the original model's 10 answers per question (better solve rates / length means)
    answers = BON / (f"lupin-all-dp13-nofit1-tools1-code1-nofit2-{'-'.join(extra)}" + ("" if args.tp_answers10 else "-a4") + ".jsonl")
    drive_sync("restore", only=[str(answers), "data/pool-dp13.jsonl", *[str(BON / f"tooltop-{x}") for x in extra], SCREEN,
                                str(ROOT / "s4-base")])
    rows = {}
    for f in (Path("data/pool-dp13.jsonl"), *[BON / f"tooltop-{x}" / "questions.jsonl" for x in extra]):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if not re.search(r"prostring|stringname|stringName", line):  # the "file" -> "string" corruption
                    rows[r["pool_id"]] = r
    by_q = defaultdict(list)
    right_think = defaultdict(list)
    for line in open(answers):
        r = json.loads(line)
        by_q[r["id"]].append((bool(r["correct"]) and r["closed_thinking"] and not r["truncated"], r["output_tokens"]))
        if bool(r["correct"]) and r["closed_thinking"] and not r["truncated"]:  # (10 Nov) for --copd-early: the original's thinking
            # the 10-answer file has no reasoning_tokens (cp1d printed 0/15407): fall back to the thinking ids, else the whole
            # answer's length (an over-estimate by the short final answer: the 0.5x threshold stays a safe proxy)
            think = r.get("reasoning_tokens")
            if think is None:
                think = len(r["thinking_ids"]) if r.get("thinking_ids") else r["output_tokens"]
            right_think[r["id"]].append(int(think))
    from build_pool import part_of
    caps, lines = {}, [f"thinkprune {name}: caps from the original's right answers (math q{args.tp_cap_math:g}, rest q{args.tp_cap_other:g})"]
    eligible = defaultdict(list)
    for qid, got in by_q.items():
        rate = sum(ok for ok, _ in got) / len(got)
        low_ok = rate >= args.tp_min_solve if args.tp_include_unsolved else rate > args.tp_min_solve
        if qid in rows and low_ok and rate <= args.tp_max_solve:
            eligible[rows[qid]["part"]].append(qid)
    for part, ids in sorted(eligible.items()):
        lens = sorted(n for q in ids for ok, n in by_q[q] if ok)
        cap = lens[min(len(lens) - 1, int(len(lens) * (args.tp_cap_math if part == "svamp" else args.tp_cap_other)))]
        caps[part] = cap
        keep = statistics.fmean(any(ok and n < cap for ok, n in by_q[q]) for q in ids)
        lines.append(f"  {part}: cap {cap} tokens; {len(ids)} questions right > half the time; {keep:.0%} keep a right answer under the cap")
    print("\n".join(lines), flush=True)
    weights = {"svamp": 1, "mbpp": 1, "ifeval": 1, "bfcl_simple_python": 400 / 840, "bfcl_multiple": 200 / 840,
               "bfcl_irrelevance": 240 / 840}
    rng = random.Random(args.data_seed)
    for ids in eligible.values():
        rng.shuffle(ids)
    solve = {qid: sum(ok for ok, _ in got) / len(got) for qid, got in by_q.items()}
    if args.tp_mixed_share is not None:  # (9 Nov, user) a set share of "sometimes wrong" questions (solved 1-9 of 10)
        mix_lines = [f"thinkprune {name}: questions per part, sometimes wrong (1-9/10) / always right (10/10), "
                     f"{args.tp_mixed_share:.0%} sometimes-wrong in the order used"]
        for part, ids in eligible.items():
            some = [q for q in ids if 0 < solve[q] < 1]
            always = [q for q in ids if solve[q] >= 1]
            order, credit_some = [], 0.0
            while some or always:  # the share holds until one kind runs out, then the other fills in
                credit_some += args.tp_mixed_share
                if some and (credit_some >= 1 or not always):
                    order.append(some.pop())
                    credit_some -= 1
                elif always:
                    order.append(always.pop())
            eligible[part] = order[::-1]  # popped from the end: the first chosen is order[0]
            n_some = sum(0 < solve[q] < 1 for q in ids)
            mix_lines.append(f"  {part}: {n_some} sometimes wrong, {len(ids) - n_some} always right")
        print("\n".join(mix_lines), flush=True)
    pool, credit = [], defaultdict(float)  # interleaved by the test's shares
    while any(eligible.values()):
        for part, w in weights.items():
            credit[part] += w
            while credit[part] >= 1 and eligible.get(part):
                credit[part] -= 1
                q = rows[eligible[part].pop()]
                pool.append(q | {"id": q["pool_id"], "cap": caps[part], "solve10": round(solve[q["pool_id"]], 2),
                                 "base_think": statistics.median(right_think[q["pool_id"]]) if right_think[q["pool_id"]] else None})
    if args.tp_check:  # (6 Nov, user) the question selection only: counts per part vs what the run needs, no training
        need = args.tp_steps * args.tp_questions
        per = defaultdict(int)
        for q in pool:
            per[q["part"]] += 1
        shares = {p: w / sum(weights.values()) for p, w in weights.items()}
        print(f"thinkprune CHECK {name}: {len(pool)} eligible questions for {args.tp_steps} steps x {args.tp_questions} = {need} needed "
              f"({len(pool) / max(1, need):.1f}x)", flush=True)
        for p in weights:
            want = round(need * shares[p])
            print(f"thinkprune CHECK {name}:   {p}: {per[p]} eligible, ~{want} used" + ("  SHORT" if per[p] < want else ""), flush=True)
        return
    if args.tp_mixed_share is not None and not args.tp_smoke:
        # (9 Nov) rl_train re-orders its pool by source (balanced_order), which undid the 50/50 ORDER: keep exactly the
        # questions the run uses (steps x questions, taken in the mixed order), so the share holds whatever the order
        pool = pool[:args.tp_steps * args.tp_questions]
        some = sum(0 < q["solve10"] < 1 for q in pool)
        print(f"thinkprune {name}: {len(pool)} questions kept, {some} sometimes wrong ({some / max(1, len(pool)):.0%})", flush=True)
    has_think = sum(q.get("base_think") is not None for q in pool)
    print(f"thinkprune {name}: {has_think}/{len(pool)} questions carry the original's right-answer thinking (base_think)", flush=True)
    data = Path(f"data/tp-{name}.jsonl")
    data.write_text("".join(json.dumps(r) + "\n" for r in pool))
    base = "Qwen/Qwen3-0.6B"
    if args.tp_start != "raw":  # the trained start merged into the weights; RL trains a new LoRA on top
        base = str(Path("models") / f"{args.tp_start}-merged")
        drive_sync("restore", only=[base, str(Path("adapters") / args.tp_start)])
        if not (Path(base) / "config.json").exists():
            py("merge.py", "--model", "Qwen/Qwen3-0.6B", "--out", base, "--adapters", f"adapters/{args.tp_start}:1")
            drive_sync("save")
    steps, questions, group = (5, 16, 8) if args.tp_smoke else (args.tp_steps, args.tp_questions, args.tp_group)
    norm_args = []
    if args.tp_norm_means:  # (6 Nov) the gated reward's B per part = the original's mean generated tokens on the pool's
        # questions, using the POOL's part labels (gr2e's --gr-norms-from re-derived parts and got svamp = 82 tokens: wrong)
        tot, cnt = defaultdict(int), defaultdict(int)
        for qid, got in by_q.items():
            if qid in rows:
                for _, n in got:
                    tot[rows[qid]["part"]] += n
                    cnt[rows[qid]["part"]] += 1
        means = {p: round(tot[p] / cnt[p]) for p in tot if cnt[p]}
        print(f"thinkprune {name}: gated B = the original's mean generated tokens per part: " +
              ", ".join(f"{p} {v}" for p, v in sorted(means.items())), flush=True)
        assert all(v >= 100 for p, v in means.items()), f"implausible mean lengths {means}"
        norm_args = ["--gr-norms", ",".join(f"{p}={v}" for p, v in sorted(means.items()))]
    rl = Path("adapters") / f"{name}-rl"
    drive_sync("restore", only=[str(rl)])
    previous = None
    block = steps if args.tp_smoke else args.tp_block
    # (9 Nov) --tp-block: blocks of any size; checkpoints already on disk (e.g. a run extended from 20 to 40 steps with
    # 5-step blocks after 10-step ones) are kept and the next block continues from the latest of them
    have = sorted(int(d.name[5:]) for d in rl.glob("step-*") if (d / "adapter_config.json").exists())
    dones = [d for d in have if d <= steps] + [d for d in range(block, steps + 1, block) if d > max(have, default=0)]
    for done in dones:
        ckpt = rl / f"step-{done}"
        if not (ckpt / "adapter_config.json").exists():
            began = time.time()
            size = done - (previous_done if previous else 0) if not args.tp_smoke else steps
            py("rl_train.py", "--model", base, "--data", data, "--sources", "all", "--out", ckpt,
               "--steps", size, "--offset", done - size,
               *(["--init-adapter", previous] if previous else []), "--questions", questions, "--group", group,
               "--max-tokens", max(caps.values()) if args.tp_reward == "cap" else 4096, "--reward", args.tp_reward,
               "--advantage", "group", "--kl-coef", args.tp_kl, "--grader", "typed", "--redundant", "none", "--lr", args.tp_lr,
               "--dtype", args.dtype, "--rank", args.tp_rank,
               *(["--lp-lam-max", args.tp_lam_max, "--lp-ramp", args.tp_ramp, "--lp-gate", args.tp_gate]
                 if args.tp_reward == "lenpen" else []), *(["--dr-grpo"] if args.tp_dr_grpo else []),
               *(["--total-steps", steps] if args.tp_persist else []),
               *(["--init-state", previous / "resume.pt"] if args.tp_persist and previous else []),
               *[x for x in args.tp_rl.split(",") if x], *norm_args)
            print(f"thinkprune {name}: steps to {done} took {(time.time() - began) / 60:.1f} min", flush=True)
            drive_sync("save")
        print((ckpt / "rl_log.jsonl").read_text()[-3000:], flush=True)
        previous, previous_done = ckpt, done
        if not args.tp_smoke and (done in args.tp_score_steps if args.tp_score_steps else done % args.tp_score_every == 0):
            out = ROOT / f"s4-{name}-{done}"
            drive_sync("restore", only=[str(out)])
            if not (out / "results.json").exists():
                model = Path("/content/lupin-models") / f"{name}-{done}"
                py("merge.py", "--model", base, "--out", model, "--adapters", f"{ckpt}:1")
                evaluate(out, samples=4, questions=SCREEN, model=model)
                shutil.rmtree(model, ignore_errors=True)
                drive_sync("save")
            print(score_line(f"{name}-{done} (screen 720 x 4)", out, base=ROOT / "s4-base"), flush=True)


def chirurgical(args):
    """Chirurgical edits (3 Nov, user; "Sparks of Weight-Space Programming", Exp 8): remove the looping habit from a few
    weights of the raw Qwen (later v1) with no training: chirurgical.py first lets the model answer 4,000 fresh practice
    questions (as in the paper; 75% of them questions the stored answers show it loops on), mines the looping answers, finds the loop direction and the copying heads by teacher-forced reads, checks every candidate edit
    (head zeroed / halved, direction removed at doses 1-6, a random-direction control) for selectivity, then generates
    paired dev answers, picks the most thinking saved at >= -1 point and confirms it on held-out questions. The pick
    is screened on the 720 x4 (s4-chir-NAME). --ch-smoke: tiny sizes, every phase, no screen."""
    name = args.name
    extra = [x for x in args.tp_extra.split(",") if x]
    answers = BON / f"lupin-all-dp13-nofit1-tools1-code1-nofit2-{'-'.join(extra)}-a4.jsonl"  # which questions loop
    out = Path("runs/chir") / name
    drive_sync("restore", only=[str(answers), "data/pool-dp13.jsonl", *[str(BON / f"tooltop-{x}") for x in extra], SCREEN,
                                str(ROOT / "s4-base"), str(out)])
    base = "Qwen/Qwen3-0.6B"
    if args.ch_start != "raw":
        base = str(Path("models") / f"{args.ch_start}-merged")
        drive_sync("restore", only=[base, str(Path("adapters") / args.ch_start)])
        if not (Path(base) / "config.json").exists():
            py("merge.py", "--model", "Qwen/Qwen3-0.6B", "--out", base, "--adapters", f"adapters/{args.ch_start}:1")
            drive_sync("save")
    small = ["--fresh", 200, "--pairs", 24, "--normal", 24, "--dev", 16, "--holdout", 16, "--attn-pairs", 8, "--check-pairs", 8,
             "--check-normal", 8, "--doses", 2, 6, "--heads", 1, "--max-tokens", 2048] if args.ch_smoke else []
    pools = ["data/pool-dp13.jsonl", *[str(BON / f"tooltop-{x}" / "questions.jsonl") for x in extra]]
    for phase in ("mine", "probe", "mech", "gen"):  # one process per phase: vLLM and the HF reads never share the GPU
        py("chirurgical.py", "--answers", answers, "--pool", *pools, "--model", base, "--out", out, "--phases", phase,
           *small, *args.ch_extra.replace(",", " ").split())
        drive_sync("save")
    pick = json.loads((out / "pick.json").read_text()) if (out / "pick.json").exists() else {}
    print(f"chirurgical {name}: pick {pick.get('best')}", flush=True)
    if args.ch_smoke or not pick.get("best"):
        return
    screen = ROOT / f"s4-chir-{name}"
    drive_sync("restore", only=[str(screen)])
    if not (screen / "results.json").exists():
        evaluate(screen, samples=4, questions=SCREEN, model=Path("/content/chir-models") / pick["best"])
        drive_sync("save")


def astra(args):
    """Astra weight-space programming (3 Nov, user; the paper's Exp 8 as it was run): gpt-6-astra gets Qwen3-0.6B's weights
    in a CPU kernel, practice questions and --as-passes forward passes, and does the diagnosis and the edit itself
    (astra_harness.py). HARD spending stop --as-cap-usd. The key comes only from the environment variable
    OPENROUTER_API_KEY (Colab Secrets, set in a notebook cell before the plan cell); without it this stage only prints
    that and returns. Afterwards OUR checks, never shown to Astra: a paired holdout (original vs Astra's edit, same
    seeds) and the 720 x4 screen (s4-astra-NAME)."""
    import random
    name = args.name
    key_var = {"openrouter": "OPENROUTER_API_KEY", "delos": "DELOS_API_KEY", "google": "GOOGLE_API_KEY"}[args.as_provider]
    if not os.environ.get(key_var) and not args.as_smoke:
        print(f"astra: {key_var} is not set in this runtime (Colab Secrets + the key cell); skipped, nothing sent", flush=True)
        return
    extra = [x for x in args.tp_extra.split(",") if x]
    out = Path("runs/astra") / name
    drive_sync("restore", only=["data/pool-dp13.jsonl", *[str(BON / f"tooltop-{x}") for x in extra], SCREEN,
                                str(ROOT / "s4-base"), str(out)])
    from build_pool import part_of
    from huggingface_hub import snapshot_download
    model_dir = snapshot_download("Qwen/Qwen3-0.6B")
    if args.as_start != "raw":  # on-top: Astra edits v1's merged weights
        model_dir = str(Path("models") / f"{args.as_start}-merged")
        drive_sync("restore", only=[model_dir, str(Path("adapters") / args.as_start)])
        if not (Path(model_dir) / "config.json").exists():
            py("merge.py", "--model", "Qwen/Qwen3-0.6B", "--out", model_dir, "--adapters", f"adapters/{args.as_start}:1")
            drive_sync("save")
    data_dir = ""
    if args.as_data:
        data_dir = str(Path("runs/astradata") / args.as_data)
        drive_sync("restore", only=[data_dir])
        assert (Path(data_dir) / "summary.json").exists(), f"{data_dir}: run the astradata stage first"
    rows = {}
    for f in (Path("data/pool-dp13.jsonl"), *[BON / f"tooltop-{x}" / "questions.jsonl" for x in extra]):
        for line in f.read_text().splitlines():
            if line.strip() and not re.search(r"prostring|stringname|stringName", line):
                r = json.loads(line)
                if part_of(r):
                    rows[r["pool_id"]] = r | {"id": r["pool_id"], "type": part_of(r), "benchmark": part_of(r)}
    ids = sorted(rows)
    random.Random(7).shuffle(ids)
    out.mkdir(parents=True, exist_ok=True)
    held = out / "holdout.json"  # ours, fixed before the session, never in Astra's practice file
    if args.as_data:  # the data folder's holdout (fixed when the data was made: its answers never include these)
        held = Path(data_dir) / "holdout.json"
    if not held.exists():
        held.write_text(json.dumps(ids[:args.as_holdout]))
    holdout = json.loads(held.read_text())
    practice = Path(f"/content/astra/{name}/practice.jsonl")
    practice.parent.mkdir(parents=True, exist_ok=True)
    keep = [rows[i] for i in ids if i not in set(holdout)]
    practice.write_text("".join(json.dumps(r) + "\n" for r in keep))
    work = Path(f"/content/astra/{name}/work")
    if args.as_smoke:
        py("astra_harness.py", "smoke", "--work", work, "--model-dir", model_dir, "--practice", practice,
           "--scratch", f"/content/astra/{name}/models")
        return
    if (out / "work_edited").exists() and not (work / "edited").exists():  # a lost runtime: the edits come back
        shutil.copytree(out / "work_edited", work / "edited")
    live = (DRIVE / out) if Path("/content/drive/MyDrive").exists() else out  # logs written straight to Drive, turn by turn
    live.mkdir(parents=True, exist_ok=True)
    py("astra_harness.py", "run", "--name", name, "--out", live, "--work", work, "--model-dir", model_dir,
       "--practice", practice, "--practice-n", len(keep), "--passes", args.as_passes, "--cap-usd", args.as_cap_usd,
       "--provider", args.as_provider, *(["--worker", args.as_worker] if args.as_worker else []),
       *(["--llm", args.as_llm] if args.as_llm else []),
       *(["--data", data_dir, "--goal", args.as_goal] if data_dir else []),
       *args.as_extra.replace(",", " ").split())
    for small in ("notes.md",):
        if (work / small).exists():
            shutil.copy(work / small, out / small)
    if (work / "edited").exists():
        shutil.rmtree(out / "work_edited", ignore_errors=True)
        shutil.copytree(work / "edited", out / "work_edited")
    for g in work.glob("gen_*.jsonl"):
        shutil.copy(g, out / g.name)
    drive_sync("save")
    if not list((work / "edited").glob("*.pt")):
        print(f"astra {name}: no edit submitted", flush=True)
        return
    # our checks: Astra's final edit vs the original on our holdout (same seeds), then the 720
    import types
    import chirurgical
    hargs = types.SimpleNamespace(model_dir=model_dir, work=str(work), scratch=f"/content/astra/{name}/models")
    import astra_harness
    edited_dir = astra_harness.edited_model_dir(hargs)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    gargs = types.SimpleNamespace(gen_dtype="bfloat16", gpu_memory=0.85, max_tokens=8192)
    check = out / "holdout_check.json"
    if not check.exists():
        orig = chirurgical.generate_graded(gargs, model_dir, holdout, rows, tok, 0)
        cand = chirurgical.generate_graded(gargs, edited_dir, holdout, rows, tok, 0)
        res = chirurgical.summarize(orig, cand, holdout, {q: rows[q]["type"] for q in holdout})
        check.write_text(json.dumps(res, indent=1))
        a = res["all"]
        print(f"astra {name} holdout (ours): accuracy {a['diff_pp']:+.1f} +- {a['se_pp']:.1f}, thinking saved "
              f"{a['saved_pct']:+.1f}%, cut {100 * a['cap_original']:.1f}% -> {100 * a['cap_candidate']:.1f}%", flush=True)
        drive_sync("save")
    screen = ROOT / f"s4-astra-{name}"
    drive_sync("restore", only=[str(screen)])
    if not (screen / "results.json").exists():
        evaluate(screen, samples=4, questions=SCREEN, model=edited_dir)
        drive_sync("save")


def astradata(args):
    """The data for the Astra data experiments (3 Nov, user): the answers the raw Qwen and v1 already gave to the training
    questions, one line per answer (id, type, sample, right, cut, thinking_tokens, tokens, full_text = chat template +
    question + answer, ready for a teacher-forced read), our holdout questions removed (fixed here, used to judge Astra's
    edit); per model and type a summary. NOT the 720 set or the full test (the judges: Astra must never see them).
    Raw answers: --ad-raw (default v1's training file, 4 answers a question); v1's answers: polans --ad-v1 (both halves)."""
    import random
    import statistics
    from transformers import AutoTokenizer
    from build_pool import part_of
    name = args.name
    out = Path("runs/astradata") / name
    drive_sync("restore", only=[str(out)])
    if (out / "summary.json").exists():
        print(f"astradata {name}: done", flush=True)
        return
    extra = [x for x in args.tp_extra.split(",") if x]
    raw_files = [BON / f for f in args.ad_raw.split(",") if f] or [BON / f"lupin-all-dp13-nofit1-tools1-code1-nofit2-{'-'.join(extra)}-a4.jsonl"]
    v1_files = [BON / f"onp-{x}-{h}" / "rollouts.jsonl" for x in args.ad_v1.split(",") if x for h in "ab"]
    drive_sync("restore", only=[*map(str, raw_files), *[str(f.parent) for f in v1_files], "data/pool-dp13.jsonl",
                                *[str(BON / f"tooltop-{x}") for x in extra], SCREEN, str(ROOT / "s4-base")])
    rows = {}
    for f in (Path("data/pool-dp13.jsonl"), *[BON / f"tooltop-{x}" / "questions.jsonl" for x in extra]):
        for line in f.read_text().splitlines():
            if line.strip() and not re.search(r"prostring|stringname|stringName", line):
                r = json.loads(line)
                if part_of(r):
                    rows[r["pool_id"]] = r
    ids = sorted(rows)
    random.Random(7).shuffle(ids)
    holdout = set(ids[:args.as_holdout])
    out.mkdir(parents=True, exist_ok=True)
    (out / "holdout.json").write_text(json.dumps(sorted(holdout)))
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    close_id = tok.convert_tokens_to_ids("</think>")
    summary = {}
    for label, files in (("raw", raw_files), ("v1", v1_files)):
        if not files:
            continue
        seen, lines = defaultdict(int), []
        for f in files:
            covered = set(seen)  # questions an earlier file already gave (the 10-answer file before the 4-answer top-ups)
            for line in open(f):
                r = json.loads(line)
                q = r["id"]
                if q not in rows or q in holdout or q in covered:
                    continue
                ids_ = r["output_ids"]
                lines.append({"id": q, "type": part_of(rows[q]), "sample": seen[q], "right": bool(r.get("correct")) and not r.get("truncated"),
                              "cut": bool(r.get("truncated")), "thinking_tokens": ids_.index(close_id) if close_id in ids_ else len(ids_),
                              "tokens": len(ids_), "full_text": tok.decode(r["prompt_ids"] + ids_, skip_special_tokens=False)})
                seen[q] += 1
        (out / f"{label}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
        by = defaultdict(list)
        for x in lines:
            by[x["type"]].append(x)
        summary[label] = {t: {"answers": len(v), "questions": len({x["id"] for x in v}),
                              "accuracy": round(sum(x["right"] for x in v) / len(v), 4),
                              "mean_thinking_tokens": round(statistics.fmean(x["thinking_tokens"] for x in v), 1),
                              "cut_share": round(sum(x["cut"] for x in v) / len(v), 4)} for t, v in sorted(by.items())}
        print(f"astradata {name}: {label} {len(lines)} answers on {len(seen)} questions", flush=True)
    if args.ad_v1_train:  # (user, 4 Nov) how v1 was made: its pairs, its settings, its checkpoints every 10 steps
        name_t, seed = args.ad_v1_train, args.ad_v1_seed
        tdir = out / "v1_training"
        (tdir / "checkpoints").mkdir(parents=True, exist_ok=True)
        pairs_dir, raw_dir = BON / f"pairs-{name_t}", BON / f"pairs-{name_t}-raw"
        adapter = Path("adapters") / f"{name_t}-s{seed}"
        drive_sync("restore", only=[str(pairs_dir), str(raw_dir), str(adapter)])
        src = pairs_dir / "pairs.jsonl" if (pairs_dir / "pairs.jsonl").exists() else raw_dir / "pairs.jsonl"
        decoded = []
        for line in open(src):
            r = json.loads(line)
            if r["id"] in holdout:
                continue
            decoded.append({"id": r["id"], "part": r.get("part"), "kind": r.get("kind"),
                            "prompt": tok.decode(r["prompt_ids"], skip_special_tokens=False),
                            "chosen": tok.decode(r["chosen_ids"], skip_special_tokens=False),
                            "rejected": tok.decode(r["rejected_ids"], skip_special_tokens=False),
                            "chosen_tokens": len(r["chosen_ids"]), "rejected_tokens": len(r["rejected_ids"])})
        (tdir / "pairs.jsonl").write_text("".join(json.dumps(x) + "\n" for x in decoded))
        steps = sorted(adapter.glob("step-*"), key=lambda d: int(d.name.split("-")[1]))
        for d in steps + [adapter]:
            dst = tdir / "checkpoints" / (d.name if d != adapter else "final")
            dst.mkdir(exist_ok=True)
            for f in ("adapter_model.safetensors", "adapter_config.json"):
                if (d / f).exists():
                    shutil.copy(d / f, dst / f)
        md = Path("DATASET.md").read_text()
        recipe = md[md.index("## 4. The pairs"):md.index("## 6.")] if "## 6." in md else md
        (tdir / "recipe.md").write_text(
            f"# How v1 was made ({name_t}, seed {seed})\n\nv1 = the raw Qwen3-0.6B + a LoRA adapter trained by DPO on the pairs in "
            "pairs.jsonl (one line per pair: the question, the chosen answer, the rejected answer, the pair kind: shorten = "
            "a right answer beats a much longer right answer; guard = the shortest right answer beats the longest wrong or "
            "capped one; at most as many shorten as guard pairs per part). michal-bida's exact settings: float32 base + "
            "float16 autocast with a loss scaler, warm-up 5% of the steps, cosine learning rate to 0, one pass in 339 steps, "
            "no pair held out. checkpoints/step-N (every 10 steps) and checkpoints/final are the LoRA adapters (PEFT: "
            "adapter_config.json + adapter_model.safetensors, lora_A / lora_B per projection); a forward pass can run any of "
            "them with \"model\": \"checkpoint:step-N\" or \"checkpoint:final\".\n\n" + recipe)
        print(f"astradata {name}: v1 training: {len(decoded)} pairs, {len(steps)} checkpoints", flush=True)
    card_stats(out, rows, holdout, raw_files, tok)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    drive_sync("save")


def card_stats(out, rows, holdout, raw_files, tok):
    """test_card.md for Astra (user, 4 Nov): the test described in detail WITHOUT its questions. Per test part, side by
    side for the test (the 720 screen: the raw Qwen's 4 answers a question, s4-base) and for our training data (the raw
    Qwen's answers to the practice questions): question length, the raw model's accuracy, the spread of per-question solve
    rates (difficulty), the thinking-length distribution and the share cut at the limit. Only aggregates leave the test."""
    import statistics
    from build_pool import part_of

    def q(xs, f):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(f * len(xs)))] if xs else 0

    def describe(qlen, groups):
        right = [a for g in groups.values() for a in g]
        rates = [sum(r for r, _, _ in g) / len(g) for g in groups.values() if g]
        think = [t for _, t, _ in right]
        return {"questions": len(groups), "q_len_p50": q(qlen, .5), "q_len_p90": q(qlen, .9),
                "accuracy": round(100 * sum(r for r, _, _ in right) / max(1, len(right)), 1),
                "always_right": round(100 * sum(x == 1 for x in rates) / max(1, len(rates))),
                "sometimes": round(100 * sum(0 < x < 1 for x in rates) / max(1, len(rates))),
                "never_right": round(100 * sum(x == 0 for x in rates) / max(1, len(rates))),
                "think_p25": q(think, .25), "think_p50": q(think, .5), "think_p75": q(think, .75), "think_p90": q(think, .9),
                "cut_pct": round(100 * sum(c for _, _, c in right) / max(1, len(right)), 1)}
    close_id = tok.convert_tokens_to_ids("</think>")
    stats = {"test": {}, "train": {}}
    base = ROOT / "s4-base" / "records.jsonl"
    qfile = Path(SCREEN) / "questions.jsonl"
    if base.exists() and qfile.exists():
        qrows = {}
        for line in open(qfile):
            r = json.loads(line)
            qrows[r["id"]] = r
        by = defaultdict(lambda: defaultdict(list))
        for line in open(base):
            r = json.loads(line)
            ds = r["dataset"] if r["dataset"] != "bfcl" else r.get("subset", "bfcl")
            think = int(r.get("reasoning_tokens") or 0)
            by[ds][r["id"]].append((str(r["correct"]) == "True", think, str(r["truncated"]) == "True"))
        for ds, groups in by.items():
            qlen = [len(tok.encode(json.dumps(qrows[i].get("messages") or qrows[i].get("prompt") or qrows[i].get("question", "")),
                                   add_special_tokens=False)) for i in groups if i in qrows]
            stats["test"][ds] = describe(qlen, groups)
    tby = defaultdict(lambda: defaultdict(list))
    seen = set()
    for f in raw_files:
        covered = set(seen)
        for line in open(f):
            r = json.loads(line)
            if r["id"] not in rows or r["id"] in holdout or r["id"] in covered:
                continue
            seen.add(r["id"])
            ids_ = r["output_ids"]
            tby[part_of(rows[r["id"]])][r["id"]].append((bool(r.get("correct")) and not r.get("truncated"),
                                                        ids_.index(close_id) if close_id in ids_ else len(ids_), bool(r.get("truncated"))))
    for part, groups in tby.items():
        qlen = [len(tok.encode(json.dumps(rows[i]["prompt"]), add_special_tokens=False)) for i in groups]
        stats["train"][part] = describe(qlen, groups)
    (out / "card_stats.json").write_text(json.dumps(stats, indent=1))
    cols = ["questions", "q_len_p50", "q_len_p90", "accuracy", "always_right", "sometimes", "never_right",
            "think_p25", "think_p50", "think_p75", "think_p90", "cut_pct"]
    lines = ["", "## The test vs our training data, in numbers (no question shown)",
             "The raw Qwen3-0.6B on each. Test: the 720-question screen of the test (4 answers a question). Train: the raw",
             "model's answers to the practice questions. q_len = question tokens; always / sometimes / never = % of questions",
             "the raw model gets right in every / some / none of its answers (difficulty); think = thinking tokens per answer",
             "(quantiles); cut = % of answers cut at the limit.", "",
             "| part | set | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 2)]
    for part in sorted(set(stats["test"]) | set(stats["train"])):
        for which in ("test", "train"):
            if part in stats[which]:
                d = stats[which][part]
                lines.append(f"| {part} | {which} | " + " | ".join(str(d[c]) for c in cols) + " |")
    card = Path("astra_test_card.md").read_text() + "\n".join(lines) + "\n"
    (out / "test_card.md").write_text(card)
    print(f"astradata: test card with stats for {len(stats['test'])} test parts and {len(stats['train'])} training parts", flush=True)


def consultpack(args):
    """The consultant pack (4 Nov, user: raw Qwen data + the test's profile, nothing of our work, for an independent strategy
    analysis by Argon; CPU only): test_profile.md (astra_test_card.md + the ORIGINAL model on the FULL test, full-base
    2,759 x 8: per part accuracy, thinking quantiles, cut share, solve-rate spread, question lengths; numbers only),
    raw_meta.jsonl (every raw answer to the training questions: numbers only), raw_text_sample.jsonl (3,000 full answers,
    stratified by type x outcome), train_profile.md. Saved to runs/consult/<name> (Drive)."""
    import random
    import statistics
    from transformers import AutoTokenizer
    from build_pool import part_of
    name = args.name
    out = Path("runs/consult") / name
    drive_sync("restore", only=[str(out)])
    if (out / "test_profile.md").exists():
        print(f"consultpack {name}: done", flush=True)
        return
    extra = [x for x in "nofit1,tools1,code1,nofit2,nofit4,math1,math2".split(",")]
    raw_files = [BON / "lupin-all-dp13-nofit1-tools1-code1-nofit2.jsonl",
                 BON / "lupin-all-dp13-nofit1-tools1-code1-nofit2-nofit4-math1-math2-a4.jsonl"]
    drive_sync("restore", only=[*map(str, raw_files), "data/pool-dp13.jsonl", *[str(BON / f"tooltop-{x}") for x in extra],
                                str(ROOT / "full-base"), "questions"])
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    close_id = tok.convert_tokens_to_ids("</think>")

    def q(xs, f):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(f * len(xs)))] if xs else 0

    def profile(groups, qlen):
        ans = [a for g in groups.values() for a in g]
        rates = [sum(r for r, _, _ in g) / len(g) for g in groups.values() if g]
        th = [t for _, t, _ in ans]
        return {"questions": len(groups), "answers": len(ans), "q_len_p50": q(qlen, .5), "q_len_p90": q(qlen, .9),
                "accuracy_pct": round(100 * sum(r for r, _, _ in ans) / max(1, len(ans)), 1),
                "always_right_pct": round(100 * sum(x == 1 for x in rates) / max(1, len(rates))),
                "sometimes_pct": round(100 * sum(0 < x < 1 for x in rates) / max(1, len(rates))),
                "never_right_pct": round(100 * sum(x == 0 for x in rates) / max(1, len(rates))),
                "think_mean": round(statistics.fmean(th), 0) if th else 0, "think_p10": q(th, .1), "think_p25": q(th, .25),
                "think_p50": q(th, .5), "think_p75": q(th, .75), "think_p90": q(th, .9), "think_p99": q(th, .99),
                "cut_pct": round(100 * sum(c for _, _, c in ans) / max(1, len(ans)), 1),
                "think_right_p50": q([t for r, t, _ in ans if r], .5), "think_wrong_p50": q([t for r, t, _ in ans if not r], .5)}
    cols = ["questions", "answers", "q_len_p50", "q_len_p90", "accuracy_pct", "always_right_pct", "sometimes_pct",
            "never_right_pct", "think_mean", "think_p10", "think_p25", "think_p50", "think_p75", "think_p90", "think_p99",
            "cut_pct", "think_right_p50", "think_wrong_p50"]

    def table(stats, title):
        lines = [title, "", "| part | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
        lines += [f"| {k} | " + " | ".join(str(v[c]) for c in cols) + " |" for k, v in sorted(stats.items())]
        return "\n".join(lines) + "\n"
    # the test: the original model on the FULL test (numbers only)
    qtext = {}
    qfile = Path("questions/questions.jsonl")
    if qfile.exists():
        for line in open(qfile):
            r = json.loads(line)
            qtext[r["id"]] = json.dumps(r.get("messages") or r.get("prompt") or r.get("question", ""))
    tg = defaultdict(lambda: defaultdict(list))
    for line in open(ROOT / "full-base" / "records.jsonl"):
        r = json.loads(line)
        tg[r["dataset"]][r["id"]].append((str(r["correct"]) == "True", int(r.get("reasoning_tokens") or 0), str(r["truncated"]) == "True"))
    test = {ds: profile(g, [len(tok.encode(qtext[i], add_special_tokens=False)) for i in g if i in qtext]) for ds, g in tg.items()}
    # the training questions: the original model's answers (the 10-answer file first, the 4-answer top-ups after)
    rows = {}
    for f in (Path("data/pool-dp13.jsonl"), *[BON / f"tooltop-{x}" / "questions.jsonl" for x in extra]):
        for line in f.read_text().splitlines():
            if line.strip() and not re.search(r"prostring|stringname|stringName", line):
                r = json.loads(line)
                if part_of(r):
                    rows[r["pool_id"]] = r
    out.mkdir(parents=True, exist_ok=True)
    seen, meta, keep = set(), [], []
    trg = defaultdict(lambda: defaultdict(list))
    qlen_t = defaultdict(list)
    for f in raw_files:
        covered = set(seen)
        for line in open(f):
            r = json.loads(line)
            qid = r["id"]
            if qid not in rows or qid in covered:
                continue
            if qid not in seen:
                qlen_t[part_of(rows[qid])].append(len(r["prompt_ids"]))
            seen.add(qid)
            ids_ = r["output_ids"]
            th = ids_.index(close_id) if close_id in ids_ else len(ids_)
            right = bool(r.get("correct")) and not r.get("truncated")
            part = part_of(rows[qid])
            trg[part][qid].append((right, th, bool(r.get("truncated"))))
            m = {"id": qid, "type": part, "source": rows[qid].get("source"), "sample": len(trg[part][qid]) - 1, "right": right,
                 "cut": bool(r.get("truncated")), "thinking_tokens": th, "tokens": len(ids_), "question_tokens": len(r["prompt_ids"])}
            meta.append(m)
            keep.append((part, "right" if right else "cut" if m["cut"] else "wrong", qid, r))
    (out / "raw_meta.jsonl").write_text("".join(json.dumps(x) + "\n" for x in meta))
    train = {p: profile(g, qlen_t[p]) for p, g in trg.items()}
    rng = random.Random(0)
    strata = defaultdict(list)
    for part, outcome, qid, r in keep:
        strata[(part, outcome)].append((qid, r))
    sample = []
    per = max(1, 3000 // max(1, len(strata)))
    for key, items in sorted(strata.items()):
        rng.shuffle(items)
        for qid, r in items[:per]:
            sample.append({"id": qid, "type": key[0], "outcome": key[1],
                           "text": tok.decode(r["prompt_ids"] + r["output_ids"], skip_special_tokens=False)})
    (out / "raw_text_sample.jsonl").write_text("".join(json.dumps(x) + "\n" for x in sample))
    (out / "test_profile.md").write_text(Path("astra_test_card.md").read_text().split("## What the practice")[0] +
                                         "\n" + table(test, "## The ORIGINAL model on the full test (2,759 x 8), numbers only"))
    (out / "train_profile.md").write_text(table(train, "## The ORIGINAL model on the training questions") + "\n" +
                                          table(test, "## ... and on the test, for comparison"))
    sizes = {f.name: round(f.stat().st_size / 1e6, 1) for f in out.iterdir()}
    print(f"consultpack {name}: {len(meta)} answers, {len(sample)} texts, files MB {sizes}", flush=True)
    drive_sync("save")


def wanda(args):
    """SPARSE arm C on a GPU (30 Oct: the 8 GB Mac swapped): wanda_prune.py with the dp11 check answers as calibration
    (128 of Qwen's own right answers) and damage check, levels --wd-sparsity, pruned models saved under models/ for the
    real eval; then the coverage check at 30% with 384 calibration answers (user: is 128 enough?)."""
    data = BON / "probe-poolcand-dp11-check" / "rollouts.jsonl"
    drive_sync("restore", only=[str(data.parent)])
    out = Path("runs/sparse") / args.name
    if not (out / "summary.txt").exists():
        py("wanda_prune.py", "--rollouts", data, "--calib", 128, "--per-part", 80, "--pairs-per-part", 30, "--sparsity",
           *args.wd_sparsity, "--device", "cuda", "--out", out)
        drive_sync("save")
    print((out / "summary.txt").read_text(), flush=True)
    cover = Path("runs/sparse") / f"{args.name}-c384"
    if not (cover / "summary.txt").exists():
        py("wanda_prune.py", "--rollouts", data, "--calib", 384, "--per-part", 80, "--pairs-per-part", 30, "--sparsity", 0.3,
           "--device", "cuda", "--out", cover, "--mask-compare", out / "mask-30.pt")
        drive_sync("save")
    print((cover / "summary.txt").read_text(), flush=True)


def wandaeval(args):
    """SPARSE arm C, the real test (user, 30 Oct): rebuild the Wanda models at --wd-sparsity (same calibration, so the
    same masks as wanda-c1: Wanda is deterministic), on local scratch (never on Drive), and score each on the choosing
    set x4 against the original Qwen (v4-base): accuracy AND thinking length per type (does pruning make it think
    longer?)."""
    data = BON / "probe-poolcand-dp11-check" / "rollouts.jsonl"
    drive_sync("restore", only=[str(data.parent), "runs/mvtf/v4-base"])
    scratch = Path("/content/wanda-models") / args.name
    val = f"questions_val{VAL_SEED}"
    todo = [s_ for s_ in args.wd_sparsity if not (ROOT / f"v4-{args.name}-{round(s_ * 100)}" / "results.json").exists()]
    if todo:
        py("wanda_prune.py", "--rollouts", data, "--calib", 128, "--per-part", 5, "--pairs-per-part", 2, "--sparsity", *todo,
           "--device", "cuda", "--save", "--out", scratch)
    for s_ in args.wd_sparsity:
        out = ROOT / f"v4-{args.name}-{round(s_ * 100)}"
        if not (out / "results.json").exists():
            evaluate(out, samples=4, questions=val, model=scratch / f"wanda-{round(s_ * 100)}")
            drive_sync("save")
        print(score_line(f"{args.name}-{round(s_ * 100)}% pruned (choosing 240 x 4)", out, base=ROOT / "v4-base"), flush=True)
    shutil.rmtree(scratch, ignore_errors=True)


def lupin_pool(args):
    """The checked pool LUPIN trains on: --lp-pool, or 'auto' = dp11 when dp10 was superseded (code still off), else dp10."""
    if args.lp_pool != "auto":
        return args.lp_pool
    drive_sync("restore", only=[str(BON / "dp10-superseded")])
    return "dp11" if (BON / "dp10-superseded").exists() else "dp10"


def topiccheck(args):
    """Topic check of the pool LUPIN trains on (user, 28 Oct; report only, never a gate: evaluation questions share their
    authors' style, so the evaluation is always closer to itself than to any training pool). topic_check.py."""
    pool_name = lupin_pool(args)
    out = BON / f"topic-{pool_name}"
    if not (out / "report.txt").exists():
        drive_sync("restore", only=[f"data/pool-{pool_name}.jsonl"])
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            run(sys.executable, "-m", "pip", "install", "-q", "sentence-transformers")
        py("topic_check.py", "--pool", f"data/pool-{pool_name}.jsonl", "--questions", "questions",
           "--exclude", f"questions_fresh{FRESH_SEED}", "--out", out)
        drive_sync("save")
    print((out / "report.txt").read_text(), flush=True)


def lupin(args):
    """LUPIN round 1 screen (spec 2.2 / 2.5, user 29 Oct: "run all these steps"; pair variations as decided 27 Oct:
    the model's own natural answers, difficulty protection, no prompted answers; code-only cut pairs come later).
    1. Waits for the 10 answers (lupin-answers-<pool>-a/b: written only if the pool's check opened the gate).
    2. Every answer to the pool's questions (4-8 selection answers + 6 new) -> one file; lupin_pairs.py --recipe
       --lp-recipe (--lp-min-solve: difficulty protection) -> raw pairs.
    3. lupin_select.py: pairs matched to the evaluation per part and per solve x thinking cell (spec 2.0), --lp-total.
    4. DPO from the ORIGINAL model (NLL anchor, dpo.py defaults: beta 0.1, lr 2e-5, LoRA r32, 16 pairs per step),
       --seeds, a checkpoint every --lp-save steps.
    5. Every checkpoint in --dpo-steps merged (local scratch, never on Drive) and scored on the CHOOSING set x4
       (decisions only on it); with --lp-fresh, the seeds' soups on the fresh set x8 instead (finalists)."""
    import statistics
    import time
    pool_name = lupin_pool(args)
    name, base = args.name, "Qwen/Qwen3-0.6B"
    if args.lp_gate:  # user, 29 Oct: train only on a pool that is representative of the whole evaluation
        report = BON / f"poolcheck-{pool_name}-both" / "report.txt"
        for _ in range(288):
            drive_sync("restore", only=[str(report.parent), f"data/pool-{pool_name}.jsonl"])
            if report.exists():
                break
            print(f"lupin {name}: waiting for {report}", flush=True)
            time.sleep(300)
        text = report.read_text() if report.exists() else ""
        ok = "verdict: ALL OK" in text or near_misses_only(text)[0]
        print(f"lupin {name}: pool {pool_name} check {'passes' if ok else 'FAILS'}: "
              + (text.strip().splitlines()[-1] if text else "no report"), flush=True)
        if not ok:
            return
    halves = [BON / f"lupin-answers-{args.lp_answers_pool or pool_name}-{h}" / "rollouts.jsonl" for h in "ab"]
    if args.lp_answers_pool:  # the pool's questions come from several answer sets: every top-up is a source too
        args.lp_extra_q = args.lp_extra_q or "nofit1,tools1,code1"
    for _ in range(288):  # the 10 answers take ~6 h on two GPUs: wait up to 24 h
        drive_sync("restore", only=[str(h.parent) for h in halves] + [f"data/pool-{pool_name}.jsonl", "runs/mvtf/v4-base",
                                                                     "runs/mvtf/eval-base"])
        if all(h.exists() for h in halves):
            break
        print(f"lupin {name}: waiting for {pool_name}'s 10 answers on Drive", flush=True)
        time.sleep(300)
    assert all(h.exists() for h in halves), f"{pool_name}'s answers are not on Drive"
    extras = [BON / f"tooltop-{x}" for x in args.lp_extra_q.split(",") if x]
    allfile = BON / (f"lupin-all-{pool_name}{''.join('-' + x for x in args.lp_extra_q.split(',') if x)}"
                     + (f"-a{args.lp_answers}" if args.lp_answers != 10 else "") + ".jsonl")  # 4 answers: its own file
    if not allfile.exists():
        ids = {json.loads(l)["pool_id"] for l in Path(f"data/pool-{pool_name}.jsonl").read_text().splitlines() if l.strip()}
        for extra in extras:  # top-up questions (harder no-fit / tools / code) join the pool for pairs; same matching
            for _ in range(288):  # another runtime may still be answering them: wait up to 24 h
                drive_sync("restore", only=[str(extra)])
                if (extra / "rollouts.jsonl").exists():
                    break
                print(f"lupin {name}: waiting for {extra} on Drive", flush=True)
                time.sleep(300)
            # unselected top-up questions join; a selected pool (dp12) already lists its own, EXCEPT the sets named in
            # --lp-add-q (1 Nov: nofit3 was silently left out because of --lp-answers-pool)
            if not args.lp_answers_pool or extra.name.removeprefix("tooltop-") in args.lp_add_q.split(","):
                ids |= {json.loads(l)["pool_id"] for l in (extra / "questions.jsonl").read_text().splitlines() if l.strip()}
        # at most --lp-answers answers per question, newest first (user, 29 Oct: "14 or 10?"): the 10-answer run, then
        # the extra-answer rounds (newest first), then the 4 selection answers. The pool was CHOSEN on the earliest
        # answers (lucky draws), so those are the biased ones; and an equal count per question keeps every type's
        # pairs equally strong (the shortest right of 14 is shorter than of 10).
        sources = [*halves, *[extra / "rollouts.jsonl" for extra in extras],
                   *[BON / f"probe-poolmore-{x}-{h}" / "rollouts.jsonl" for x in ("dp11", "dp10", "dp9", "dp8") for h in "ab"],
                   BON / "probe-poolcand-dp9n-all" / "rollouts.jsonl"]
        drive_sync("restore", only=[str(f.parent) for f in sources])
        count, taken = Counter(), Counter()
        with open(allfile, "w") as out:
            for f in sources:
                if not f.exists():
                    continue
                for line in open(f):
                    if not line.strip():
                        continue
                    i = json.loads(line)["id"]
                    if i in ids and taken[i] < args.lp_answers:
                        taken[i] += 1
                        out.write(line if line.endswith("\n") else line + "\n")
                        count[f.parent.name] += 1
        per_q = Counter(json.loads(l)["id"] for l in open(allfile))
        print(f"lupin: {len(per_q)} pool questions, answers per question min {min(per_q.values())} median "
              f"{statistics.median(per_q.values())}; from {dict(count)}", flush=True)
        drive_sync("save")
    if args.lp_own_answers:  # round 2: the trained model's own answers to the same questions replace the original's
        halves = [BON / f"onp-{args.lp_own_answers}-{h}" / "rollouts.jsonl" for h in "ab"]
        for _ in range(288):
            drive_sync("restore", only=[str(h.parent) for h in halves])
            if all(h.exists() for h in halves):
                break
            print(f"lupin {name}: waiting for onp-{args.lp_own_answers}-a/b on Drive", flush=True)
            time.sleep(300)
        allfile = BON / f"lupin-own-{args.lp_own_answers}.jsonl"
        with open(allfile, "w") as out:
            for h in halves:
                out.write("".join(l if l.endswith("\n") else l + "\n" for l in open(h) if l.strip()))
        print(f"lupin {name}: {sum(1 for _ in open(allfile))} own answers of {args.lp_own_answers}", flush=True)
    raw, pairs = BON / f"pairs-{name}-raw", BON / f"pairs-{name}"
    drive_sync("restore", only=[str(raw), str(pairs)])  # the SAME pair files as before a lost runtime (31 Oct)
    for x in filter(None, (args.lp_overcut + "," + args.lp_trained).split(",")):
        for _ in range(288):  # the other runtime may still be answering its half: wait up to 24 h
            drive_sync("restore", only=[str(BON / f"onp-{x}")])
            if (BON / f"onp-{x}" / "rollouts.jsonl").exists():
                break
            print(f"lupin {name}: waiting for onp-{x} on Drive", flush=True)
            time.sleep(300)
        assert (BON / f"onp-{x}" / "rollouts.jsonl").exists(), f"onp-{x} is not on Drive"
    if not (raw / "pairs.jsonl").exists():
        py("lupin_pairs.py", "--rollouts", allfile, "--recipe", args.lp_recipe, "--out", raw,
           *(["--min-solve", args.lp_min_solve] if args.lp_min_solve else []), *args.lp_extra.replace(",", " ").split(),
           *(["--easy-ratio", args.lp_easy_ratio] if args.lp_easy_ratio else []),
           *(["--shorten-cap", args.lp_shorten_cap] if args.lp_shorten_cap != 1.0 else []),
           *(["--min-solve-part", args.lp_min_solve_part] if args.lp_min_solve_part else []),
           *(["--right-parts", args.lp_right_parts] if args.lp_right_parts else []),
           *(["--median-shorten-part", args.lp_median_part] if args.lp_median_part else []),
           *(["--band", args.lp_band] if args.lp_band else []),
           *(["--overcut", *[BON / f"onp-{x}" / "rollouts.jsonl" for x in args.lp_overcut.split(",")]] if args.lp_overcut else []),
           *(["--trained", *[BON / f"onp-{x}" / "rollouts.jsonl" for x in args.lp_trained.split(",")]] if args.lp_trained else []))
    if args.lp_min_overcut:  # 1 Nov (user): the math data is checked BEFORE any training
        got = sum(int(l.rsplit(":", 1)[1]) for l in (raw / "summary.txt").read_text().splitlines() if "overcut (trained)" in l)
        print(f"lupin {name}: {got} trained overcut pairs (gate {args.lp_min_overcut})", flush=True)
        if got < args.lp_min_overcut:
            shutil.rmtree(raw, ignore_errors=True)  # rebuilt next time, with more onpolicy answers
            print(f"lupin {name}: GATE NOT MET, no training", flush=True)
            return
    if args.lp_no_match and not (pairs / "pairs.jsonl").exists():  # michal-bida control (30 Oct): his pairs as they come
        pairs.mkdir(parents=True, exist_ok=True)
        shutil.copy(raw / "pairs.jsonl", pairs / "pairs.jsonl")
        if args.lp_cap_like:  # (5 Nov, Jasmin data b) unmatched pairs, cut at random to another set's count per part x kind
            like = BON / f"pairs-{args.lp_cap_like}" / "pairs.jsonl"
            drive_sync("restore", only=[str(like.parent)])
            cap = Counter((json.loads(l)["part"], json.loads(l)["kind"]) for l in open(like) if l.strip())
            rows = [l for l in open(raw / "pairs.jsonl") if l.strip()]
            random.Random(0).shuffle(rows)
            got, keep = Counter(), []
            for l in rows:
                r = json.loads(l)
                if got[(r["part"], r["kind"])] < cap[(r["part"], r["kind"])]:
                    got[(r["part"], r["kind"])] += 1
                    keep.append(l)
            (pairs / "pairs.jsonl").write_text("".join(keep))
        kept = Counter((json.loads(l)["part"], json.loads(l)["kind"]) for l in open(pairs / "pairs.jsonl"))
        (pairs / "summary.txt").write_text("no matching (raw pairs): " + ", ".join(f"{p}/{k} {n}" for (p, k), n in sorted(kept.items())) + "\n")
        drive_sync("save")
    if not (pairs / "pairs.jsonl").exists():
        py("lupin_select.py", "--pairs", raw / "pairs.jsonl", "--rollouts", allfile, "--reference", *references(),
           "--total", args.lp_total, "--out", pairs,
           *(["--max-shorten-ratio", args.lp_balance] if args.lp_balance else []),
           *(["--shorten-ratio-part", args.lp_pressure] if args.lp_pressure else []),
           *(["--kind-share-part", args.lp_kind_share] if args.lp_kind_share else []),
           *(["--loose-parts", args.lp_loose] if args.lp_loose else []),
           *(["--part-share", args.lp_part_share] if args.lp_part_share else []),
           *(["--solve-tilt", args.lp_solve_tilt] if args.lp_solve_tilt else []),
           *(["--think-tilt-bfcl", args.lp_think_tilt_bfcl] if args.lp_think_tilt_bfcl != 1.0 else []))
        drive_sync("save")
    print((pairs / "summary.txt").read_text(), flush=True)
    rows = [json.loads(l) for l in (pairs / "pairs.jsonl").read_text().splitlines() if l.strip()]
    counts = [sum(r["kind"] == k for r in rows) for k in ("shorten", "guard", "hard", "right")]
    if args.lp_pairs_only:  # (5 Nov; moved into lupin 6 Nov) build and report the pairs, no training: per part x kind counts and median lengths
        per = Counter((r["part"], r["kind"]) for r in rows)
        print(f"lupin {name}: PAIRS ONLY, {len(rows)} pairs, {len({r['id'] for r in rows})} questions", flush=True)
        for (part, kind), n in sorted(per.items()):
            ch = sorted(len(r["chosen_ids"]) for r in rows if r["part"] == part and r["kind"] == kind)
            rj = sorted(len(r["rejected_ids"]) for r in rows if r["part"] == part and r["kind"] == kind)
            print(f"  {part} / {kind}: {n} pairs, median chosen {ch[len(ch) // 2]} vs rejected {rj[len(rj) // 2]} tokens",
                  flush=True)
        return
    start = Path("adapters") / f"fresh-{name}"
    if args.lp_init:  # two-stage (29 Oct): start from an earlier adapter (policy AND DPO reference), e.g. the "be right" round
        start = Path(args.lp_init)
        drive_sync("restore", only=[str(start)])
        assert (start / "adapter_config.json").exists(), f"{start} not found"
    else:  # the same fresh (zero) LoRA start every seed used before, if it is on Drive
        drive_sync("restore", only=[str(start)])
        if not (start / "adapter_config.json").exists():
            py("sft.py", "--init-only", "--model", base, "--out", start)
    gentle = [*(["--beta", args.lp_beta] if args.lp_beta else []), *(["--nll", args.lp_nll] if args.lp_nll is not None else []),
              *(["--lr", args.lp_lr] if args.lp_lr else []),
              *(["--lr-schedule", args.lp_lr_schedule] if args.lp_lr_schedule else [])]
    # a fresh runtime has an empty disk (31 Oct: both runtimes retrained finished seeds from scratch): bring back every
    # seed's adapter and checkpoints from Drive first, so a finished seed is skipped and an unfinished one resumes
    for seed in args.lp_reset_seeds:  # a damaged seed (31 Oct, lpc-runb-s1): start it over, here and on Drive, ONCE
        marker = BON / f"reset-{name}-s{seed}.done"
        drive_sync("restore", only=[str(marker)])
        if marker.exists():
            continue
        for d in (Path("adapters") / f"{name}-s{seed}", DRIVE / "adapters" / f"{name}-s{seed}"):
            shutil.rmtree(d, ignore_errors=True)
        # the soups had this seed in them: drop their scores so they are scored again (here and on Drive)
        for base_dir in (ROOT, DRIVE / ROOT):
            for d in [*base_dir.glob(f"f8-{name}-soup-*"), *base_dir.glob(f"s4-{name}-soup-*"), *base_dir.glob(f"r4-{name}-soup-*")]:
                shutil.rmtree(d, ignore_errors=True)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("reset\n")
        drive_sync("save")
        print(f"lupin: seed {seed} reset", flush=True)
    drive_sync("restore", only=[str(Path("adapters") / f"{name}-s{seed}") for seed in args.seeds])
    for seed in args.seeds:
        adapter = Path("adapters") / f"{name}-s{seed}"
        if not (adapter / "adapter_config.json").exists():
            py("dpo.py", "--model", base, "--pairs", pairs / "pairs.jsonl", "--init-adapter", start, "--out", adapter,
               "--mix", ",".join(f"{c / max(1, sum(counts)):.4f}" for c in counts), "--save-every", args.lp_save,
               "--epochs", args.lp_epochs, "--dtype", args.dtype, "--seed", seed, *args.lp_dpo.replace(",", " ").split(), *gentle)
            drive_sync("save")
    val, fresh_q = f"questions_val{VAL_SEED}", f"questions_fresh{FRESH_SEED}"
    scratch = Path("/content/lupin-models")
    if args.lp_fresh:
        evaluate(ROOT / "f8-base", samples=8, questions=fresh_q)
    else:
        evaluate(ROOT / "v4-base", samples=4, questions=val)
    for step in args.dpo_steps:
        members = [Path("adapters") / f"{name}-s{seed}" / f"step-{step}" for seed in args.seeds]
        members = [m for m in members if (m / "adapter_config.json").exists()]
        if not members:
            continue
        tag = f"{name}-{'soup-' if len(members) > 1 else ''}{step}"
        out = ROOT / (f"f8-{tag}" if args.lp_fresh else f"v4-{tag}")
        if not (out / "results.json").exists():
            model = scratch / tag
            if not (model / "config.json").exists():
                py("merge.py", "--model", base, "--out", model, "--adapters", *[f"{m}:{1 / len(members):.6f}" for m in members])
            if args.lp_fresh:
                evaluate(out, samples=8, questions=fresh_q, model=model)
            else:
                evaluate(out, samples=4, questions=val, model=model)
            shutil.rmtree(model, ignore_errors=True)  # rebuildable from the adapters; Drive keeps no merged copy
            drive_sync("save")
        ref = ROOT / ("f8-base" if args.lp_fresh else "v4-base")
        print(score_line(f"{tag} ({'fresh 240 x 8' if args.lp_fresh else 'choosing 240 x 4'})", out, base=ref), flush=True)


def codecut(args):
    """OPTIONAL module (user, 29 Oct): code-only cut pairs, strict labels, small dose, on top of a screened recipe, and
    dropped automatically if it costs accuracy or spills onto other types.
    1. Code questions of the pool solved >= half the time (the protected ones never get length pressure), up to
       --cc-questions: label_cuts.py on each one's longest right answer (thinking cut at up to 6 paragraph ends, the
       model answers 4 times from each cut).
    2. targeted_pairs.py, code only, --safe 1.0: a cut is safe only if ALL 4 answers from it pass the tests; pair =
       earliest safe cut + a passing answer > the full answer, kept when >= 25% shorter.
    3. The base recipe's matched pairs (--cc-base) with --cc-dose of its code pairs swapped for cut pairs: every
       other type's pairs are byte-identical to the base run's, and code keeps its share.
    4. Same DPO (seed 0) and checkpoints as the screen, scored on the choosing set x4.
    5. Verdict against the base run at the same checkpoints (mean over the steps both have): KEEP only if code thinks
       >= 3% less AND overall accuracy >= base - 1 AND code accuracy >= base - 2.5 AND non-code accuracy >= base - 1
       (no spill); else DROPPED, and the module is not pursued."""
    import statistics
    name, base, pool_name = args.name, args.cc_base, lupin_pool(args)
    allfile = BON / f"lupin-all-{pool_name}.jsonl"
    drive_sync("restore", only=[str(allfile), str(BON / f"pairs-{base}"), f"data/pool-{pool_name}.jsonl",
                                f"runs/mvtf/v4-{base}-*", "runs/mvtf/v4-base"])
    keep = Path(f"data/{name}-keep.jsonl")
    if not keep.exists():
        code = {json.loads(l)["pool_id"] for l in Path(f"data/pool-{pool_name}.jsonl").read_text().splitlines()
                if l.strip() and json.loads(l)["part"] == "mbpp"}
        rate = defaultdict(list)
        for line in open(allfile):
            r = json.loads(line)
            if r["id"] in code:
                rate[r["id"]].append(bool(r["correct"]) and r["closed_thinking"] and not r["truncated"])
        ok = sorted(i for i, v in rate.items() if sum(v) / len(v) >= 0.5)
        random.Random(args.data_seed).shuffle(ok)
        keep.write_text("".join(json.dumps({"id": i}) + "\n" for i in ok[:args.cc_questions]))
    cal, cut = BON / f"cuts-{name}", BON / f"pairs-{name}-cut"
    if not (cal / "verdicts.jsonl").exists():
        py("label_cuts.py", "--rollouts", allfile, "--keep", keep, "--out", cal, "--max-cuts", 6, "--samples", 4,
           "--dtype", args.dtype)
        drive_sync("save")
    if not (cut / "pairs.jsonl").exists():
        py("targeted_pairs.py", "--cal", cal, "--rollouts", allfile, "--keep", keep, "--out", cut,
           "--shorten-sources", "kodcode", "--long-sources", "none", "--guard-sources", "none", "--safe", 1.0,
           "--min-save", 0.25)
    pairs = BON / f"pairs-{name}"
    if not (pairs / "pairs.jsonl").exists():
        rng = random.Random(args.data_seed)
        rows = [json.loads(l) for l in (BON / f"pairs-{base}" / "pairs.jsonl").read_text().splitlines() if l.strip()]
        cuts = [json.loads(l) | {"part": "mbpp", "kind": "shorten"} for l in (cut / "pairs.jsonl").read_text().splitlines()
                if l.strip() and json.loads(l)["kind"] == "shorten"]
        code_short = [k for k, r in enumerate(rows) if r["part"] == "mbpp" and r["kind"] == "shorten"]
        n = min(len(cuts), round(args.cc_dose * sum(r["part"] == "mbpp" for r in rows)), len(code_short))
        rng.shuffle(code_short)
        rng.shuffle(cuts)
        drop = set(code_short[:n])
        mixed = [r for k, r in enumerate(rows) if k not in drop] + cuts[:n]
        rng.shuffle(mixed)
        pairs.mkdir(parents=True, exist_ok=True)
        (pairs / "pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in mixed))
        (pairs / "summary.txt").write_text(f"{name}: {base}'s {len(rows)} matched pairs with {n} code shorten pairs "
                                           f"swapped for strict cut pairs ({len(cuts)} available); other types unchanged\n")
        drive_sync("save")
    print((pairs / "summary.txt").read_text(), flush=True)
    rows = [json.loads(l) for l in (pairs / "pairs.jsonl").read_text().splitlines() if l.strip()]
    counts = [sum(r["kind"] == k for r in rows) for k in ("shorten", "guard", "hard", "right")]
    start = Path("adapters") / f"fresh-{name}"
    if not (start / "adapter_config.json").exists():
        py("sft.py", "--init-only", "--model", "Qwen/Qwen3-0.6B", "--out", start)
    adapter = Path("adapters") / f"{name}-s0"
    if not (adapter / "adapter_config.json").exists():
        py("dpo.py", "--model", "Qwen/Qwen3-0.6B", "--pairs", pairs / "pairs.jsonl", "--init-adapter", start, "--out",
           adapter, "--mix", ",".join(f"{c / max(1, sum(counts)):.4f}" for c in counts), "--save-every", 25,
           "--epochs", 1, "--dtype", args.dtype, "--seed", 0)
        drive_sync("save")
    val = f"questions_val{VAL_SEED}"
    diffs = []
    for step in args.dpo_steps:
        member = adapter / f"step-{step}"
        theirs = ROOT / f"v4-{base}-{step}"
        if not (member / "adapter_config.json").exists() or not (theirs / "records.jsonl").exists():
            continue
        out = ROOT / f"v4-{name}-{step}"
        if not (out / "results.json").exists():
            model = Path("/content/lupin-models") / f"{name}-{step}"
            py("merge.py", "--model", "Qwen/Qwen3-0.6B", "--out", model, "--adapters", f"{member}:1.0")
            evaluate(out, samples=4, questions=val, model=model)
            shutil.rmtree(model, ignore_errors=True)
            drive_sync("save")
        print(score_line(f"{name}-{step} (choosing 240 x 4)", out, base=ROOT / "v4-base"), flush=True)
        mine, ref = compare(out, ROOT / "v4-base")["results"], compare(theirs, ROOT / "v4-base")["results"]
        acc = lambda res, parts: statistics.fmean(res[p]["accuracy_delta"] for p in parts) * 100  # noqa: E731
        others = [p for p in mine if p != "mbpp"]
        diffs.append({"step": step,
                      "code_saved": 100 * ((mine["mbpp"]["token_reduction"] or 0) - (ref["mbpp"]["token_reduction"] or 0)),
                      "overall_acc": acc(mine, list(mine)) - acc(ref, list(ref)),
                      "code_acc": 100 * (mine["mbpp"]["accuracy_delta"] - ref["mbpp"]["accuracy_delta"]),
                      "other_acc": acc(mine, others) - acc(ref, others)})
    if diffs:
        m = {k: statistics.fmean(d[k] for d in diffs) for k in ("code_saved", "overall_acc", "code_acc", "other_acc")}
        keep_it = m["code_saved"] >= 3 and m["overall_acc"] >= -1 and m["code_acc"] >= -2.5 and m["other_acc"] >= -1
        text = (f"codecut {name} vs {base} (mean over steps {[d['step'] for d in diffs]}): code saving "
                f"{m['code_saved']:+.1f} pts, overall accuracy {m['overall_acc']:+.1f}, code accuracy {m['code_acc']:+.1f}, "
                f"other types {m['other_acc']:+.1f} -> {'KEEP (optional add to the recipe)' if keep_it else 'DROPPED (not pursued)'}")
        (BON / f"{name}-verdict.txt").write_text(text + "\n" + json.dumps(diffs) + "\n")
        drive_sync("save")
        print(text, flush=True)


def lupinanswers(args):
    """LUPIN step 1: 10 answers per question on the final pool (user, 28 Oct), by the ORIGINAL model. The pool's 4
    selection answers are reused; this answers 6 more (another sampling seed), half the pool per runtime (--dp-step).
    GATE (user, 28 Oct, "very important"): runs only when the pool's held-out check (poolcheck-<pool>-both) reads
    "verdict: ALL OK"; otherwise it says so and does nothing. Data only: no training here."""
    import time
    name = args.la_pool
    for marker in ((f"{name}-skipped", f"{name}-superseded") if not args.la_force else ()):  # dp10 / dp11: only the pool that is used gets answers
        drive_sync("restore", only=[str(BON / marker)])
        if (BON / marker).exists():
            print(f"lupinanswers: {name} {marker.split('-')[-1]} ({(BON / marker).read_text().strip()}); nothing to do", flush=True)
            return
    report = BON / f"poolcheck-{name}-both" / "report.txt"
    pool = Path(f"data/pool-{name}.jsonl")
    for _ in range(96):  # the other runtime may still be selecting and checking this pool: up to 8 hours
        drive_sync("restore", only=[str(BON / f"poolcheck-{name}-both"), str(pool)])
        if report.exists() and pool.exists():
            break
        print(f"waiting for {report} on Drive", flush=True)
        time.sleep(300)
    near, misses = near_misses_only(report.read_text()) if report.exists() else (False, [])
    if args.la_force:  # an explicit decision, logged (29 Oct: dp10 35/36 chosen over dp11 30/36)
        print(f"GATE: {name} forced open: {args.la_force}", flush=True)
    elif report.exists() and "verdict: ALL OK" not in report.read_text() and near:
        print(f"GATE: {name} not ALL OK but only near misses (user: close enough, go): " + " | ".join(misses), flush=True)
    elif not report.exists() or "verdict: ALL OK" not in report.read_text():
        print(f"GATE: {name}'s pool check is not ALL OK; no answers generated. {report.read_text()[-300:] if report.exists() else ''}",
              flush=True)
        return
    rows = [json.loads(l) for l in pool.read_text().splitlines() if l.strip()]
    parity = 0 if args.dp_step == "a" else 1
    data = Path(f"data/lupin-ans-{name}-{args.dp_step}.jsonl")
    data.write_text("".join(json.dumps(r) + "\n" for k, r in enumerate(rows) if k % 2 == parity))
    n = sum(1 for l in data.read_text().splitlines() if l.strip())
    out = BON / f"lupin-answers-{name}-{args.dp_step}"
    drive_sync("restore", only=[str(out)])
    if args.la_target and not (out / "rollouts.jsonl").exists():
        # top up to --la-target answers per question (29 Oct: most pool questions already had 8 selection answers, so
        # 6 more each was ~3x the work needed). Answers already made by the 6-answer run (the .partial file) are kept.
        sources = [BON / "probe-poolcand-dp9n-all" / "rollouts.jsonl",
                   *[BON / f"probe-poolmore-{x}-{h}" / "rollouts.jsonl" for x in ("dp8", "dp9", "dp10", "dp11") for h in "ab"]]
        drive_sync("restore", only=[str(f.parent) for f in sources])
        mine = [json.loads(l) for l in data.read_text().splitlines() if l.strip()]
        ids = {r["pool_id"] for r in mine}
        have = Counter()
        partial = out / "rollouts.jsonl.partial"
        for f in [*sources, partial]:
            if f.exists():
                for line in open(f):
                    if line.strip():
                        i = json.loads(line)["id"]
                        if i in ids:
                            have[i] += 1
        need = {r["pool_id"]: max(0, args.la_target - have[r["pool_id"]]) for r in mine}
        groups = Counter(need.values())
        print(f"lupinanswers {name} {args.dp_step}: answers to make per question {dict(sorted(groups.items()))} "
              f"(target {args.la_target}; {sum(1 for l in open(partial)) if partial.exists() else 0} already in the 6-answer run)", flush=True)
        made = []
        for k in sorted(v for v in groups if v > 0):
            sub = Path(f"data/lupin-ans-{name}-{args.dp_step}-k{k}.jsonl")
            sub.write_text("".join(json.dumps(r) + "\n" for r in mine if need[r["pool_id"]] == k))
            m = sum(1 for _ in open(sub))
            probe = BON / f"lupin-answers-{name}-{args.dp_step}-k{k}"
            drive_sync("restore", only=[str(probe)])
            if not (probe / "rollouts.jsonl").exists():
                py("probe_curve.py", "--data", sub, "--sources", "gsm8k_easy,gsm8k_easy_test,metamath_easy,gsm8k_matched,metamath_sv,"
                   "kodcode,toolace,hermes_tools,glaive_tools,ifeval_like,ifeval_built", "--per-source", 1000000, "--samples", k,
                   "--model", "Qwen/Qwen3-0.6B", "--out", probe, "--dtype", args.dtype, "--generate-only", "--chunk", 150,
                   "--max-questions", m, "--seed", 11 + k)
                drive_sync("save")
            made.append(probe / "rollouts.jsonl")
        with open(out / "rollouts.jsonl.tmp", "w") as handle:
            for f in [partial, *made]:
                if f.exists():
                    for line in open(f):
                        if line.strip():
                            handle.write(line if line.endswith("\n") else line + "\n")
        (out / "rollouts.jsonl.tmp").rename(out / "rollouts.jsonl")
        drive_sync("save")
        return
    if not (out / "rollouts.jsonl").exists() or sum(1 for _ in (out / "rollouts.jsonl").open()) < args.la_samples * n:
        py("probe_curve.py", "--data", data, "--sources", "gsm8k_easy,gsm8k_easy_test,metamath_easy,gsm8k_matched,metamath_sv,"
           "kodcode,toolace,hermes_tools,glaive_tools,ifeval_like,ifeval_built", "--per-source", 1000000, "--samples", args.la_samples,
           "--model", "Qwen/Qwen3-0.6B", "--out", out, "--dtype", args.dtype, "--generate-only", "--chunk", 150,
           "--max-questions", n, "--seed", 11)
    drive_sync("save")


def promptsearch(args):
    """Candidate prompts for prompt distillation (user, 27 Oct; Lightning McQwen3's ex-draft step): the ORIGINAL model
    answers --ps-per-source training questions per type of lot --ps-lot x --ps-samples under each prompt of
    prompt_candidates.py; prompt_search.py reports tokens saved, accuracy change and leak rate against no prompt.
    Measurement only, on training questions: the evaluation sets are never used to choose the prompt."""
    data = Path(f"data/lot-{args.ps_lot}.jsonl")
    if not data.exists():
        drive_sync("restore", only=[str(data)])
    cmd = ["--data", data, "--per-source", args.ps_per_source, "--samples", args.ps_samples,
           "--model", args.start or "Qwen/Qwen3-0.6B", "--dtype", args.dtype, "--out", BON / f"ps-{args.name}"]
    py("prompt_search.py", *cmd, *(["--only", args.ps_only] if args.ps_only else []))


def tokenskip(args):
    """TokenSkip (arXiv 2502.12067) raw, from the original model (user, 27 Oct: "try it raw, maybe we will be surprised"):
    one right answer per question of the label lots' original-Qwen rollouts (training sources only), its thinking cut to
    --tsk-ratios of its words by LLMLingua-2 (the reply kept verbatim), kept only if the original model, handed the cut
    thinking, still answers right (tokenskip.py --verify); a LoRA SFT per ratio, a checkpoint per epoch, each merged and
    scored on the fresh set x8. Ratio 1.0 is the control: the same SFT on the uncut answers, so the effect of the cut is
    the difference."""
    name, base = args.name, args.start or "Qwen/Qwen3-0.6B"
    ensure_lots(args.tsk_lots)
    drive_sync("restore", only=[f"runs/bon/tskip-{name}-*", f"adapters/{name}-*", f"runs/mvtf/f8-{name}-*"])
    try:
        import llmlingua  # noqa: F401
    except ImportError:
        run(sys.executable, "-m", "pip", "install", "-q", "llmlingua")
    rollouts = [p for l in args.tsk_lots for kind in ("probe", "kprobe")
                if (p := BON / f"{kind}-{l}" / "rollouts.jsonl").exists()]
    fresh_q = f"questions_fresh{FRESH_SEED}"
    evaluate(ROOT / "f8-base", samples=8, questions=fresh_q)
    for ratio in args.tsk_ratios:
        tag = f"{name}-r{round(ratio * 100)}"
        data = BON / f"tskip-{tag}"
        if not (data / "sft.jsonl").exists():
            py("tokenskip.py", "--rollouts", *rollouts, "--ratio", ratio, "--pick", args.tsk_pick, "--model", base,
               "--dtype", args.dtype, "--out", data, *(["--ratio-by", args.tsk_ratio_by] if args.tsk_ratio_by else []),
               *(["--verify"] if ratio < 1 else []))
            drive_sync("save")
        print((data / "summary.txt").read_text(), flush=True)
        adapter = Path("adapters") / f"{tag}-sft"
        n = sum(1 for l in (data / "sft.jsonl").read_text().splitlines() if l.strip())
        if not (adapter / "adapter_config.json").exists():
            py("sft.py", "--model", base, "--data", data / "sft.jsonl", "--out", adapter, "--epochs", args.tsk_epochs,
               "--lr", args.tsk_lr, "--save-every", math.ceil(n / 16), "--dtype", args.dtype)
            drive_sync("save")
        checkpoints = sorted(adapter.glob("step-*"), key=lambda p: int(p.name.split("-")[1])) + [adapter]
        for epoch, ckpt in enumerate(checkpoints, 1):
            model = Path("models") / f"{tag}-e{epoch}"
            if not (model / "config.json").exists():
                py("merge.py", "--model", base, "--out", model, "--adapters", f"{ckpt}:1")
            res = ROOT / f"f8-{tag}-e{epoch}"
            evaluate(res, samples=8, questions=fresh_q, model=model)
            print(score_line(f"{tag}-e{epoch} (fresh 240 x 8)", res, base=ROOT / "f8-base"), flush=True)


def mathfix(args):
    """Repair a model's math with its OWN answers (user, 26 Oct: rc1-rl-40 loses mostly on SVAMP, -4.1): it answers the
    SVAMP-like math questions of label lots (--probe-extra-from; matched GSM8K + MetaMathQA) x8; pairs = "be right" (its
    right reply beats its wrong one of similar length) + "keep thinking" (its longest right beats its shortest wrong when
    the right one is longer); nothing touches the other types. Contamination-checked again here. Then targeted()'s DPO
    (fresh LoRA on --start, NLL anchor, seeds, soups scored on fresh x8). No cut labelling needed."""
    name, base = args.name, args.start
    pairs = BON / f"pairs-{name}"
    if not (pairs / "pairs.jsonl").exists():
        math = ("gsm8k_matched", "metamath_sv")
        data = Path(f"data/mathfix-{name}.jsonl")
        if not data.exists():
            lots = [Path(f"data/lot-{lot}.jsonl") for lot in args.probe_extra_from]
            if not all(p.exists() for p in lots):
                drive_sync("restore", only=[str(p) for p in lots])
            rows = [l for p in lots for l in p.read_text().splitlines() if l.strip() and json.loads(l)["source"] in math]
            data.write_text("".join(l + "\n" for l in rows))
            print(f"{name}: {len(rows)} math questions from {', '.join(args.probe_extra_from)}", flush=True)
        probe = BON / f"probe-{name}"
        n = sum(1 for l in data.read_text().splitlines() if l.strip())
        if not (probe / "rollouts.jsonl").exists() or sum(1 for _ in (probe / "rollouts.jsonl").open()) < 8 * n:
            py("probe_curve.py", "--data", data, "--sources", ",".join(math), "--per-source", 100000, "--samples", 8,
               "--model", base, "--out", probe, "--dtype", args.dtype, "--generate-only", "--chunk", 150,
               "--max-questions", n)
        keep = BON / f"{name}-clean.jsonl"
        if not keep.exists():
            ids = {}
            for l in (probe / "rollouts.jsonl").read_text().splitlines():
                if l.strip():
                    r = json.loads(l)
                    ids.setdefault(r["id"], {"id": r["id"], "source": r["source"]})
            keep.write_text("".join(json.dumps(r) + "\n" for r in ids.values()))
            py("contamination.py", "--data", keep, "--rollouts", probe / "rollouts.jsonl", "--share", 0.3)
        cal = BON / f"pairs-{name}-nocal"  # no cut labels: only the rollout-based kinds are built
        cal.mkdir(parents=True, exist_ok=True)
        for f in ("replies.jsonl", "verdicts.jsonl", "cuts.jsonl"):
            (cal / f).touch()
        py("targeted_pairs.py", "--cal", cal, "--rollouts", probe / "rollouts.jsonl", "--keep", keep, "--out", pairs,
           "--shorten-sources", "none", "--long-sources", ",".join(math), "--right-sources", ",".join(math))
    targeted(args)  # pairs exist: straight to DPO + soups


def early_ratio(lots, source):
    """A type's settling speed from label lots' summaries: forced-answer success at 20-40% of the thinking over success
    with the full thinking (cut-weighted over the lots). SVAMP 0.34 / 0.73 = 0.47 (builds up slowly); practice GSM-IC
    0.88 and GSM-Plus 1.0 (settled early: stop pairs there teach stops SVAMP punishes)."""
    early = full = weight = 0.0
    for lot in lots:
        path = BON / f"pairs-{lot}" / "cal" / "summary.txt"
        for line in (path.read_text().splitlines() if path.exists() else []):
            parts = line.split()
            if parts and parts[0] == source and len(parts) > 4:
                vals = [x for x in parts[4:] if x != "/"][:6]  # 0-20 / 20-40 / 40-60 / 60-80 / 80-<100 / full
                if len(vals) == 6 and vals[1] != "-" and vals[5] != "-":
                    cuts = float(parts[3])
                    early, full, weight = early + cuts * float(vals[1]), full + cuts * float(vals[5]), weight + cuts
                break
    return early / full if weight and full else None


def targeted(args):
    """Targeted DPO from a label lot (targeted_pairs.py): shorten code and instructions where the lot's cuts show the
    answer already holds; make math think longer (full right beats an early cut that fails, long right beats short wrong);
    other types untouched. Fresh LoRA on --start, 3 seeds, soups at --dpo-steps, every soup scored on the fresh set x8."""
    name, base = args.name, args.start
    if args.tg_skip_if:  # a conditional stage: only run when the earlier run fell short
        other, want_saved, max_loss = args.tg_skip_if.split(":")
        for res in sorted(ROOT.glob(f"f8-{other}-soup-*")):
            if (res / "compare.json").exists():
                avg = json.loads((res / "compare.json").read_text())["average"]
                if (avg["token_reduction"] or 0) >= float(want_saved) and 100 * avg["accuracy_delta"] >= -float(max_loss):
                    print(f"{name}: skipped, {res.name} already saved {avg['token_reduction']:+.1%} at "
                          f"{100 * avg['accuracy_delta']:+.1f} points", flush=True)
                    return
        print(f"{name}: no {other} soup reached {float(want_saved):.0%} at >= -{max_loss} points: running", flush=True)
    pairs = BON / f"pairs-{name}"
    if not (pairs / "pairs.jsonl").exists():
        ensure_lots(args.probe_extra_from)
        shorten = args.tg_shorten
        for src in [s for s in args.tg_shorten_if_svamp.split(",") if s and s != "none"]:
            ratio = early_ratio(args.probe_extra_from, src)
            ok = ratio is not None and ratio <= 0.6
            print(f"{name}: {src} answer at 20-40% of the thinking / full = "
                  f"{'n/a' if ratio is None else f'{ratio:.2f}'} (SVAMP 0.47, GSM-IC 0.88; bar 0.6): "
                  f"{'stop pairs ON' if ok else 'stop pairs OFF'}", flush=True)
            if ok:
                shorten += f",{src}"
        rows = []
        for lot in args.probe_extra_from:  # one or more label lots of the same model
            part = pairs / lot
            if not (part / "pairs.jsonl").exists():
                py("targeted_pairs.py", "--cal", BON / f"pairs-{lot}" / "cal", "--rollouts",
                   BON / f"probe-{lot}" / "rollouts.jsonl", BON / f"kprobe-{lot}" / "rollouts.jsonl",
                   "--keep", BON / f"{lot}-clean.jsonl", "--out", part,
                   "--shorten-sources", shorten, "--long-sources", args.tg_long, "--rule", args.tg_rule,
                   "--min-samples", args.tg_min_samples, "--right-sources", args.tg_right,
                   "--guard-mode", args.tg_guard_mode, "--guard-sources", args.tg_guard or args.tg_long,
                   "--hard-ratio", args.tg_hard_ratio, "--hard-cap-sources", args.tg_hard_cap)
            print((part / "summary.txt").read_text(), flush=True)
            rows += [l for l in (part / "pairs.jsonl").read_text().splitlines() if l.strip()]
        note = ""
        if args.tg_drop_kinds or args.tg_brake_share < 1:  # rebalance: drop pair kinds; cap "keep thinking" per type
            drop = set(filter(None, args.tg_drop_kinds.split(",")))
            by_src = {}
            for l in rows:
                r = json.loads(l)
                if r["kind"] not in drop:
                    by_src.setdefault(r["source"], []).append((r["kind"], l))
            rng = random.Random(1)
            rows = []
            for src, group in sorted(by_src.items()):
                hard = [l for k, l in group if k == "hard"]
                other = [l for k, l in group if k != "hard"]
                rng.shuffle(hard)
                share = args.tg_brake_share
                rows += other + hard[:len(hard) if share >= 1 else int(round(share / (1 - share) * len(other)))]
        if args.tg_quota:  # cap each type (a random share of its pairs, so its kinds keep their proportions)
            quota = dict((k, int(v)) for k, v in (x.split("=") for x in args.tg_quota.split(",")))
            by_src = {}
            for l in rows:
                by_src.setdefault(json.loads(l)["source"], []).append(l)
            rng = random.Random(0)
            rows = []
            for src, group in sorted(by_src.items()):
                rng.shuffle(group)
                rows += group[:quota.get(src, len(group))]
            kinds = Counter((json.loads(l)["source"], json.loads(l)["kind"]) for l in rows)
            note = "".join(f"  {s} / {k}: {n}\n" for (s, k), n in sorted(kinds.items()))
            rng.shuffle(rows)
        (pairs / "pairs.jsonl").write_text("".join(l + "\n" for l in rows))
        (pairs / "summary.txt").write_text(f"{len(rows)} pairs from {', '.join(args.probe_extra_from)}\n{note}")
    print((pairs / "summary.txt").read_text(), flush=True)
    rows = [json.loads(l) for l in (pairs / "pairs.jsonl").read_text().splitlines() if l.strip()]
    counts = [sum(r["kind"] == k for r in rows) for k in ("shorten", "guard", "hard", "right")]
    start = Path("adapters") / f"fresh-{name}"
    if not (start / "adapter_config.json").exists():
        py("sft.py", "--init-only", "--model", base, "--out", start)
    for seed in args.seeds:
        adapter = Path("adapters") / f"{name}-s{seed}"
        if not (adapter / "adapter_config.json").exists():
            py("dpo.py", "--model", base, "--pairs", pairs / "pairs.jsonl", "--init-adapter", start, "--out", adapter,
               "--mix", ",".join(f"{c / max(1, sum(counts)):.4f}" for c in counts), "--save-every", args.tg_save_every,
               "--epochs", 1, "--dtype", args.dtype, "--seed", seed, "--max-steps", max(args.dpo_steps))
    fresh_q = f"questions_fresh{FRESH_SEED}"
    if not Path(fresh_q, "questions.jsonl").exists():
        py("subset_questions.py", "--per-dataset", 40, "--seed", FRESH_SEED, "--exclude", QUESTIONS, "--out", fresh_q)
    evaluate(ROOT / "f8-base", samples=8, questions=fresh_q)
    for step in args.dpo_steps:
        members = [Path("adapters") / f"{name}-s{seed}" / f"step-{step}" for seed in args.seeds]
        members = [m for m in members if (m / "adapter_config.json").exists()]
        if not members:
            continue
        model = Path("models") / f"{name}-soup-{step}"
        if not (model / "config.json").exists():
            py("merge.py", "--model", base, "--out", model, "--adapters", *[f"{m}:{1 / len(members):.6f}" for m in members])
        out = ROOT / f"f8-{name}-soup-{step}"
        evaluate(out, samples=8, questions=fresh_q, model=model)
        print(score_line(f"{name}-soup-{step} (fresh 240 x 8)", out, base=ROOT / "f8-base"), flush=True)


def zoo_margin(summary):
    """From a zoo summary's pooled trade-off table (TEST, <= 2 pt column): best classifier saving minus the best baseline
    saving (position, own confidence, answer stability), in points, and the two winners."""
    import re
    lines = summary.splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith("Trade-off, one threshold"))
    rows = {}
    for l in lines[i + 2:]:
        if not l.strip():
            break
        name = re.split(r"\s{2,}", l.strip())[0]
        vals = re.findall(r"(\d+)% / (-?[\d.]+)|(?<=\s)-(?=\s|$)", l)
        col = [(int(v[0]), float(v[1])) if v[0] else (0, 0.0) for v in vals]
        if len(col) >= 3:
            rows[name] = col[2]  # budgets 0.5, 1, 2, 3, 5 points: the third is <= 2 (saving %, real test loss)
    base = {k: v[0] for k, v in rows.items() if k in ("position", "own confidence", "answer stability")}
    # a classifier counts only if its real loss on the held-out questions also stays near the budget (<= 3 points)
    probes = {k: v[0] for k, v in rows.items() if k not in base and not k.startswith("confidence") and v[1] <= 3.0}
    bb, bp = max(base.items(), key=lambda x: x[1], default=("-", 0)), max(probes.items(), key=lambda x: x[1], default=("-", 0))
    return bp[1] - bb[1], bp, bb


def decide(args):
    """The user's rule (24 Oct): if a classifier clearly beats the simple signals in a zoo (>= --margin points more saving
    on the held-out questions at <= 2 points of loss, one threshold for all types), train the model itself to stop (lot B,
    original Qwen, every type at its real safe cuts); otherwise the targeted round on rc1-rl-40 (lot A)."""
    import time
    zoos = [BON / "zoo-zA" / "summary.txt", BON / "zoo-zB" / "summary.txt"]
    for _ in range(36):  # the other runtime's zoo: wait for it on Drive (up to 3 h)
        if all(z.exists() for z in zoos):
            break
        drive_sync("restore", only=[f"runs/bon/{z.parent.name}" for z in zoos] + ["runs/bon/pairs-plotA", "runs/bon/pairs-plotB"])
        if all(z.exists() for z in zoos):
            break
        print("decide: waiting for both zoo summaries", flush=True)
        time.sleep(300)
    margins = {}
    for z in zoos:
        if z.exists():
            # the first zoo reports counted loss against "always right"; redo them against the natural stop (24 Oct fix)
            lot = "plotA" if z.parent.name == "zoo-zA" else "plotB"
            v2 = z.parent / "summary_v2.txt"
            if not v2.exists() and (z.parent / "scores.npz").exists():
                py("probe_zoo.py", "--cal", BON / f"pairs-{lot}" / "cal", "--rollouts", "none", "--keep", "none",
                   "--out", z.parent, "--report-only")
            text = (v2 if v2.exists() else z).read_text()
            print(text[text.index("Trade-off"):text.index("Stopping simulation")], flush=True)
            m, bp, bb = zoo_margin(text)
            margins[z.parent.name] = m
            print(f"decide: {z.parent.name}: best classifier {bp[0]} {bp[1]}% vs best simple signal {bb[0]} {bb[1]}% "
                  f"at <= 2 pts -> margin {m:+d} pts", flush=True)
    helps = any(m >= args.decide_margin for m in margins.values())
    print(f"decide: classifier {'HELPS' if helps else 'does not help'} (threshold {args.decide_margin} pts)", flush=True)
    sub = (["targeted", "--name", "stopB", "--start", "Qwen/Qwen3-0.6B", "--probe-extra-from", "plotB",
            "--tg-shorten", "kodcode,ifeval_like,toolace,arc_easy,commonsense_qa,gsm_ic,gsm_plus", "--tg-long", "none"]
           if helps else
           ["targeted", "--name", "tg1", "--start", "models/rc1-rl-40", "--probe-extra-from", "plotA"])
    (ROOT.parent / "viz").mkdir(parents=True, exist_ok=True)
    (ROOT.parent / "viz" / "decision.txt").write_text(f"margins {margins}; helps={helps}; running {' '.join(sub)}\n")
    if not helps:
        drive_sync("restore", only=["models/rc1-rl-40"])
    py("track.py", *sub, "--dpo-steps", *args.dpo_steps, "--seeds", *args.seeds)


def zooreport(args):
    """Redo the zoo reports from their saved scores (no features, no fitting) with the current loss accounting."""
    for name, lot in (("zA", "plotA"), ("zB", "plotB")):
        out = BON / f"zoo-{name}"
        if not (out / "scores.npz").exists():
            drive_sync("restore", only=[f"runs/bon/zoo-{name}", f"runs/bon/pairs-{lot}"])
        if (out / "scores.npz").exists():
            py("probe_zoo.py", "--cal", BON / f"pairs-{lot}" / "cal", "--rollouts", "none", "--keep", "none", "--out", out,
               "--report-only")
            text = (out / "summary_v2.txt").read_text()
            print(f"== {name}\n" + text[text.index("Trade-off"):text.index("Cut point by cut point")], flush=True)


def caltable(args):
    """Confidence vs real outcome per data type and cut point, for every calibration run done so far (cal_table.py)."""
    dirs = [d for d in [BON / "pairs-rc1" / "cal", BON / "calcheck-rc1s30", BON / "calcheck-rc1rl40", BON / "pairs-rc2" / "cal"]
            if (d / "confidence.jsonl").exists()]
    out = ROOT.parent / "viz" / "cal_table.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = subprocess.run([sys.executable, "cal_table.py", *map(str, dirs)], capture_output=True, text=True).stdout
    out.write_text(text)
    print(text, flush=True)


def probeknow(args):
    """Does the model know when it already has the answer? A linear probe on its hidden state at each labelled cut of
    a calibration run (probe_knows.py), against the cut position, its confidence, and the 20/20 oracle."""
    frm = args.calcheck_from
    out = BON / f"probeknow-{args.name}"
    import time
    for run in (args.probe_extra_from or []):  # a lot made by the other runtime: wait for it on Drive (up to 4 h)
        for _ in range(48):
            if (BON / f"pairs-{run}" / "cal" / "summary.txt").exists():
                break
            print(f"probeknow: waiting for {run}'s labelled cuts", flush=True)
            time.sleep(300)
            drive_sync("restore")
    if not (out / "summary.txt").exists():
        # the model's own confidence (probability of ITS answer at each cut), for every set the probe reads
        sets = [BON / args.probe_cal] + [BON / f"pairs-{r}" / "cal" for r in (args.probe_extra_from or [])]
        sets += [BON / f"pairs-{args.probe_test_from}" / "cal"] if args.probe_test_from else []
        for cal in sets:
            if not (cal / "own_confidence.jsonl").exists():
                py("own_conf.py", "--model", args.probe_model, "--cal", cal, "--dtype", args.dtype)
        py("probe_knows.py", "--model", args.probe_model, "--cal", BON / args.probe_cal,
           "--rollouts", BON / f"probe-{frm}" / "rollouts.jsonl", BON / f"kprobe-{frm}" / "rollouts.jsonl",
           "--keep", BON / f"select-{frm}" / "sft.jsonl", BON / f"kselect-{frm}" / "sft.jsonl", "--out", out,
           "--dtype", args.dtype,
           *(["--extra-runs", *args.probe_extra_from] if args.probe_extra_from else []),
           *([] if not args.probe_test_from else
             ["--test-cal", BON / f"pairs-{args.probe_test_from}" / "cal",
              "--test-rollouts", BON / f"probe-{args.probe_test_from}" / "rollouts.jsonl",
              BON / f"kprobe-{args.probe_test_from}" / "rollouts.jsonl",
              "--test-keep", BON / f"select-{args.probe_test_from}" / "sft.jsonl",
              BON / f"kselect-{args.probe_test_from}" / "sft.jsonl"]))
    print((out / "summary.txt").read_text(), flush=True)


def fresh_rl_pool(name, seed, per_source, kod_file):
    """A new draw for RL: ToolACE, IFEval-like, ARC/CSQA, GSM-Plus, GSM-IC (--per-source each) plus KodCode tasks no
    run has answered, minus every question of an earlier draw, minus anything the contamination check flags (30% of
    8-grams or an exact 13-word run against the 2,759 evaluation questions). Ids as probe_curve gives them."""
    import random
    pool = Path(f"data/rl-{name}-fresh{seed}.jsonl")
    if pool.exists():
        return pool
    raw = Path(f"data/rlfresh-seed{seed}-raw.jsonl")
    py("prepare_types.py", "--sources", "toolace,ifeval_like,mc,gsm_plus,gsm_ic", "--per-source", per_source,
       "--seed", seed, "--out", raw)
    text = lambda r: r["prompt"][-1]["content"]
    used = [p for p in Path("data").glob("*.jsonl")
            if p.name.startswith(("bon", "bal", "chain", "trapmath", "rlcal", "rl-")) and "raw" not in p.name]
    seen = {text(json.loads(l)) for p in used for l in p.read_text().splitlines() if l.strip() and "prompt" in l}
    answered = {json.loads(l).get("question") for p in BON.glob("kprobe-*/rollouts.jsonl")
                for l in p.read_text().splitlines() if l.strip()}
    rows = [json.loads(l) for l in raw.read_text().splitlines() if l.strip()]
    kod = [json.loads(l) for l in kod_file.read_text().splitlines() if l.strip()]
    kod = [r for r in kod if text(r) not in answered]
    random.Random(seed).shuffle(kod)
    rows = [r for r in rows + kod[:per_source // 2] if text(r) not in seen]
    for index, row in enumerate(rows):
        row["id"] = f"{row['source']}-fresh{seed}-{index}"
    pool.write_text("".join(json.dumps(r) + "\n" for r in rows))
    questions = Path(f"data/rlfresh-seed{seed}-text.jsonl")  # contamination.py reads each id's question text
    questions.write_text("".join(json.dumps({"id": r["id"], "question": text(r)}) + "\n" for r in rows))
    py("contamination.py", "--data", pool, "--rollouts", questions, "--share", 0.3)
    print(f"{name}: fresh RL pool (seed {seed}): {sum(1 for l in pool.read_text().splitlines() if l.strip())} questions",
          flush=True)
    return pool


def rl_blocks(args, name, base, cal_tag, cal_model, data, kod_file, probes, keeps, choose, report):
    """RL from cal_model in blocks of --rl-block steps; after each block the LoRA is kept, merged and scored on the
    choosing set with 2 answers (a log line), so any block can be gone back to; a block more than 2 points below
    --floor stops the run. The best block (choosing set x4, within --floor) is reported x8 on usual and unseen."""
    small = args.smoke
    solved = defaultdict(list)
    for p in ([] if args.rl_fresh_seed else probes):
        for l in (p / "rollouts.jsonl").read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                solved[r["id"]].append(r["correct"])
    allowed = set() if args.rl_fresh_seed else \
        {json.loads(l)["id"] for k in keeps for l in (k / "sft.jsonl").read_text().splitlines() if l.strip()}
    pool = Path(f"data/rl-{name}.jsonl")
    if args.rl_fresh_seed:  # new questions, each seen about once over the run, instead of repeating the round's
        pool = fresh_rl_pool(name, args.rl_fresh_seed, args.rl_fresh_per_source, kod_file)
    if not pool.exists():
        rows = []
        for f in (data, kod_file):  # ids as probe_curve.py gives them: "<source>-<line index in the file>"
            for index, line in enumerate(f.read_text().splitlines()):
                if not line.strip():
                    continue
                row = json.loads(line)
                row["id"] = f"{row['source']}-{index}"
                if row["id"] in allowed and sum(solved[row["id"]]) >= 4:
                    rows.append(json.dumps(row))
        pool.write_text("\n".join(rows) + "\n")
        print(f"{name}: RL pool {len(rows)} questions (solved >= 4/8 by the start model)", flush=True)
    block = 1 if small else args.rl_block
    blocks = 2 if small else max(1, args.rl_steps // block)
    rl = Path("adapters") / f"{name}-rl"
    candidates, previous = [], None
    for b in range(1, blocks + 1):
        done = b * block
        ckpt = rl / f"step-{done}"
        if not (ckpt / "adapter_config.json").exists():
            py("rl_train.py", "--model", cal_model, "--data", pool, "--sources", "all", "--out", ckpt,
               "--steps", block, "--offset", done - block, *(["--init-adapter", previous] if previous else []),
               "--questions", 4 if small else 32, "--group", 8, "--max-tokens", args.rl_max_tokens, "--reward", "arora",
               "--adaptive", "--alpha", args.rl_alpha, "--advantage", "batch", "--kl-coef", args.rl_kl,
               "--grader", "typed", "--redundant", "none", "--lr", args.rl_lr, "--dtype", args.dtype)
        print((ckpt / "rl_log.jsonl").read_text(), flush=True)
        previous = ckpt
        model = Path("models") / f"{name}-rl-{done}"
        if not (model / "config.json").exists():
            py("merge.py", "--model", cal_model, "--out", model, "--adapters", f"{ckpt}:1")
        candidates.append((f"{name}-rl-{done}", model))
        if small:
            continue
        tag = f"{name}-rl-{done}"
        if args.judge == "fresh":  # every block straight on the fresh set x8
            fq = f"questions_fresh{FRESH_SEED}"
            out, basedir = ROOT / f"f8-{tag}", ROOT / "f8-base"
            evaluate(basedir, samples=8, questions=fq)
            evaluate(out, samples=8, questions=fq, model=model)
            line, delta = score_line(f"{tag} (fresh 240 x 8)", out, with_delta=True, base=basedir)
        else:
            out = ROOT / f"v2-{tag}"
            evaluate(ROOT / "v2-base", samples=2, questions=f"questions_val{VAL_SEED}")
            evaluate(out, samples=2, questions=f"questions_val{VAL_SEED}", model=model)
            line, delta = score_line(f"{tag} (choosing set x2, quick check)", out, with_delta=True, base=ROOT / "v2-base")
        print(line, flush=True)
        if delta < args.floor - 2:
            print(f"{name}: RL stopped after step {done}: accuracy {delta:+.1f} on the quick check", flush=True)
            break
    if small:
        print(f"{name}: smoke OK: every step ran ({len(candidates)} RL blocks merged)", flush=True)
        return
    tag, model = choose(f"{name} RL", [(cal_tag, cal_model)] + candidates)
    (ROOT / f"chain-{name}-rl.json").write_text(json.dumps({"model": model, "tag": tag}))
    report(tag, model)


def audit(args):
    """Every training question ever prepared (data/*.jsonl) against all 2,759 evaluation questions."""
    if Path("runs/audit/contamination_report.txt").exists():  # done before (restored from Drive)
        print(Path("runs/audit/contamination_report.txt").read_text(), flush=True)
        return
    files = sorted(str(p) for p in Path("data").glob("*.jsonl") if p.name not in ("val.jsonl",))
    py("audit_contamination.py", "--data", *files, "--out", "runs/audit")


def code_round(args):
    """Round 3's pairs plus code "stopped too early" pairs (code_pairs.py), DPO from --start, scored as bon_dpo."""
    name = args.name or "code-r3"
    start = args.start or "adapters/bon-dpo-r2g/step-10"
    sources = [(BON / "probe-bon-dpo-r3" / "rollouts.jsonl", BON / "select-bon-dpo-r3" / "sft.jsonl", "r3")]
    for data in args.code_data:  # more code: the start model answers every code question of these files 4 times
        stem = Path(data).stem
        probe, keep = BON / f"code-probe-{name}-{stem}", BON / f"code-select-{name}-{stem}"
        if not (probe / "rollouts.jsonl").exists():
            py("probe_curve.py", "--data", data, "--sources", "mbpp_train", "--per-source", 300, "--samples", 4,
               "--adapter", start, "--out", probe, "--dtype", args.dtype, "--generate-only", "--chunk", 150)
        if not (keep / "sft.jsonl").exists():
            py("select_shortest.py", "--rollouts", probe / "rollouts.jsonl", "--out", keep)
            py("contamination.py", "--data", keep / "sft.jsonl", "--rollouts", probe / "rollouts.jsonl")
        sources.append((probe / "rollouts.jsonl", keep / "sft.jsonl", stem))
    files = [BON / "pairs-bon-dpo-r3-drop-mbpp_train" / "pairs.jsonl"]
    for rollouts, keep, stem in sources:
        code = BON / f"code-{name}-{stem}"
        if not (code / "pairs.jsonl").exists():
            py("code_pairs.py", "--rollouts", rollouts, "--keep", keep, "--adapter", start, "--out", code,
               "--dtype", args.dtype)
        print((code / "summary.txt").read_text(), flush=True)
        files.append(code / "pairs.jsonl")
    pairs = BON / f"pairs-{name}"
    rows = [json.loads(l) for f in files for l in f.read_text().splitlines() if l.strip()]
    pairs.mkdir(parents=True, exist_ok=True)
    (pairs / "pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    counts = [sum(r["kind"] == k for r in rows) for k in ("shorten", "guard", "hard")]
    print(f"pairs: shorten {counts[0]}, guard {counts[1]}, code hard {counts[2]}", flush=True)
    adapter = Path("adapters") / name
    if not (adapter / "adapter_config.json").exists():  # every pair kept: shares = their counts
        py("dpo.py", "--pairs", pairs / "pairs.jsonl", "--init-adapter", start, "--out", adapter,
           "--mix", ",".join(f"{c / sum(counts):.4f}" for c in counts), "--save-every", args.save_every,
           "--epochs", args.dpo_epochs, "--dtype", args.dtype)
    score_checkpoints(name, adapter, args)


def confirm(args):
    """The run's 4-sample-checked checkpoint(s), scored again with 8 samples (finalists are compared at 8)."""
    evaluate(ROOT / "s8-base", samples=8)
    for run in sorted(ROOT.glob(f"s4-{args.name}-step-*")):
        step = run.name.rsplit("-step-", 1)[1]
        out = ROOT / f"s8-{args.name}-step-{step}"
        evaluate(out, Path("adapters") / args.name / f"step-{step}", samples=8)
        print(score_line(f"{args.name} step-{step} (8 samples)", out, base=ROOT / "s8-base"), flush=True)


FRESH_SEED = 20261022  # the fresh test set: 40 random questions per benchmark part, none from questions_40


def fresh(args):
    """Finalists on a fresh 240-question set never used for any choice, 8 samples, against the raw model."""
    questions = f"questions_fresh{FRESH_SEED}"
    if not Path(questions, "questions.jsonl").exists():
        py("subset_questions.py", "--per-dataset", 40, "--seed", FRESH_SEED, "--exclude", QUESTIONS, "--out", questions)
    base = ROOT / "f8-base"
    if os.environ.get("TRACK_SESSION") and not (base / "results.json").exists():
        import time  # a second runtime waits for session 1's raw-model score on this set instead of redoing it
        for _ in range(90):
            drive_sync("restore")
            if (base / "results.json").exists():
                break
            print("waiting for session 1's f8-base on Drive", flush=True)
            time.sleep(120)
    evaluate(base, samples=8, questions=questions)
    for adapter in map(resolve, args.inputs):
        if adapter.startswith("models/"):  # a merged model (soup, RL block), not a LoRA on the raw model
            name = adapter.removeprefix("models/").replace("/", "-")
            out = ROOT / f"f8-{name}"
            evaluate(out, samples=8, questions=questions, model=adapter)
        else:
            name = adapter.removeprefix("adapters/").replace("/", "-")
            out = ROOT / f"f8-{name}"
            evaluate(out, adapter, samples=8, questions=questions)
        print(score_line(f"{name} (fresh 240 questions x 8)", out, base=base), flush=True)


def peek(args):
    """Read-only: copy another folder's fresh-s8-* runs here and print their scores against its fresh-s8-base."""
    for folder in args.inputs:
        folder = Path(folder)
        runs = sorted(p for p in folder.glob("fresh-s8-*") if (p / "results.json").exists())
        print(f"{folder}: {len(runs)} finished fresh runs: {[p.name for p in runs]}", flush=True)
        local = ROOT / "peek"
        for run in runs:
            shutil.copytree(run, local / run.name, dirs_exist_ok=True)
        base = local / "fresh-s8-base"
        for run in runs:
            if run.name != "fresh-s8-base" and base.exists():
                print(score_line(f"{run.name} (their fresh questions 41-80 x 8)", local / run.name, base=base), flush=True)


def s1log(args):
    """Read-only: the score lines of the other runtime's log on Drive (its page may not show its output)."""
    log = DRIVE / "code" / "track_log.txt"
    if not log.exists():
        print(f"no log at {log}", flush=True)
        return
    lines = [l for l in log.read_text(errors="replace").splitlines() if l.startswith(("plan:", "== ", "   bfcl"))]
    print("\n".join(["session 1 log, latest lines:", *lines[-40:]]), flush=True)


def diag(args):
    """How a scored run fails on one benchmark against its base: lost rows by kind, with examples."""
    out, base = ROOT / args.name, ROOT / (args.name.split("-", 1)[0] + "-base")
    print(diagnose(out, dataset=args.dataset, examples=3, base=base), flush=True)


def new_questions(seed):
    """A fresh draw of training questions (prepare_types.py --seed), minus any in an earlier draw (data/bon*.jsonl)."""
    out = Path(f"data/bon-seed{seed}.jsonl")
    if not out.exists():
        raw = Path(f"data/bon-seed{seed}-raw.jsonl")
        py("prepare_types.py", "--sources", "gsm8k_all,mbpp_train,toolace,ifeval_like,mc", "--per-source", 300,
           "--seed", seed, "--out", raw)
        text = lambda r: r["prompt"][-1]["content"]
        used = [p for p in Path("data").glob("bon*.jsonl") if "raw" not in p.name and p != out]
        seen = {text(json.loads(l)) for p in used for l in p.read_text().splitlines() if l.strip()}
        rows = [json.loads(l) for l in raw.read_text().splitlines() if l.strip()]
        fresh = [r for r in rows if text(r) not in seen]
        out.write_text("".join(json.dumps(r) + "\n" for r in fresh))
        print(f"new questions (seed {seed}): {len(fresh)} of {len(rows)} not used before", flush=True)
    return out


def best_checkpoint(floor=-2.5, prefix=""):
    """The scored DPO checkpoint saving the most tokens within `floor` points (4 samples), of one run if prefix."""
    best = None
    for run in ROOT.glob(f"s4-{prefix}-step-*" if prefix else "s4-*-step-*"):
        if not (run / "results.json").exists():
            continue
        average = compare(run, ROOT / "s4-base")["average"]
        saved, delta = average["token_reduction"] or 0, 100 * average["accuracy_delta"]
        name, step = run.name.removeprefix("s4-").rsplit("-step-", 1)
        if delta >= floor and (best is None or saved > best[0]):
            best = (saved, Path("adapters") / name / f"step-{step}")
    if best is None:
        raise SystemExit(f"no scored checkpoint within {floor} points")
    print(f"best checkpoint: {best[1]} ({best[0]:+.1%})", flush=True)
    return best[1]


def steer_read(args):
    """Directions only (no pushing): every question and every finished reply of the best-of-8 rollouts."""
    out = Path("runs/steer-big")
    if not (out / "directions.json").exists():
        py("steer_dirs.py", "states", "--out", out, "--per-kind", 600, "--all-replies")
        py("steer_dirs.py", "directions", "--out", out)
    print((out / "directions.txt").read_text(), flush=True)


def steer(args):
    """Three "be brief" pushes (A mean difference, B within-question trend, C learned), kept small.

    Directions from the raw model's best-of-8 replies (100 questions per kind), a layer filter on
    held-out questions, then push tests on the 240-question sample with 2 samples: A at the 2 best
    layers x 2 strengths, B and C at the best layer x 2 strengths. The best within -3 points gets
    4 samples. Each kind's best also answers 25 fresh held-out questions per source x 4, compared
    with the raw model's replies to the same questions (steer_dirs.py compare).
    """
    out = Path("runs/steer")
    if not (out / "directions.json").exists():
        py("steer_dirs.py", "states", "--out", out)
        py("steer_dirs.py", "directions", "--out", out)
    print((out / "directions.txt").read_text(), flush=True)
    found = json.loads((out / "directions.json").read_text())
    best_layer = found["filter"]["A"][0]
    # strengths in units of |A| at the layer (about 1% of the residual norm there): 5 and 15 push by
    # about 6% and 18% of it. C's learned vector is about 0.6 |A|: 8 and 24 match those sizes.
    configs = [("A", l, s) for l in found["filter"]["A"] for s in (5, 15)]
    configs += [("B", found["filter"]["B"][0], s) for s in (5, 15)] + [("C", best_layer, s) for s in (8, 24)]
    if not (out / f"C-{best_layer}.pt").exists():
        py("steer_dirs.py", "learn", "--out", out, "--layer", best_layer)
    results, checked = {}, (out / "checked").exists()
    for kind, layer, strength in configs:
        name = f"{kind}-{layer}-{strength:g}"
        model, run = Path("steered") / name, ROOT / f"s2-steer-{name}"
        if not (run / "results.json").exists():
            if not (model / "config.json").exists():
                py("steer_dirs.py", "build", "--out", out, "--kind", kind, "--layer", layer, "--strength", strength,
                   "--target", model)
            if not checked:  # a strong push, well above the engines' rounding noise
                probe = Path("steered") / "check-A"
                py("steer_dirs.py", "build", "--out", out, "--kind", "A", "--layer", best_layer, "--strength", 40,
                   "--target", probe)
                py("steer_dirs.py", "check", "--model", probe)
                shutil.rmtree(probe)
                (out / "checked").write_text("ok\n")
                checked = True
            evaluate(run, samples=2, model=model)
        line, delta = score_line(f"steer {name} (2 samples)", run, with_delta=True)
        print(line, flush=True)
        results[name] = (compare(run)["average"]["token_reduction"] or 0, delta, kind)
    winners = {}
    for name, (saved, delta, kind) in results.items():
        best = winners.get(kind)
        key = (delta >= LIMIT, saved if delta >= LIMIT else delta)
        if best is None or key > best[0]:
            winners[kind] = (key, name)
    overall = max(winners.values())[1]
    evaluate(ROOT / "s4-base", samples=4)
    evaluate(ROOT / f"s4-steer-{overall}", samples=4, model=Path("steered") / overall)
    print(score_line(f"steer {overall} (4 samples)", ROOT / f"s4-steer-{overall}", base=ROOT / "s4-base"), flush=True)
    if not (out / "held.jsonl").exists():
        py("steer_dirs.py", "held", "--out", out, "--per-source", 25)
    rolls = []
    for _, name in sorted(winners.values(), key=lambda w: w[1]):
        roll = out / f"roll-{name}"
        if not (roll / "rollouts.jsonl").exists():
            py("probe_curve.py", "--data", out / "held.jsonl", "--model", Path("steered") / name, "--per-source", 25,
               "--samples", 4, "--out", roll, "--dtype", args.dtype, "--generate-only", "--chunk", 150)
        rolls.append(roll)
    py("steer_dirs.py", "compare", "--out", out, "--runs", *rolls)


def dpo_round(args):
    """One on-policy, verified MVT round: the current model's own replies -> verified cuts -> DPO -> test.

    Fresh questions (another --seed than earlier rounds), 4 replies each from --start, MVT cut plus an
    earlier and a later candidate verified by 4 re-answers (mvt_pairs.py), anchored DPO from --start
    (dpo.py), then every saved checkpoint scored on the 240-question test sample with 4 samples.
    """
    name = args.name or "dpo1"
    root = Path("runs") / name
    questions, probe, pairs = root / "questions.jsonl", root / "probe", root / "pairs"
    adapter = Path("adapters") / name
    if not questions.exists():
        root.mkdir(parents=True, exist_ok=True)
        py("prepare_types.py", "--sources", "gsm8k_all,mbpp_train,toolace,ifeval_like,mc", "--per-source",
           args.per_source, "--seed", args.seed, "--out", questions)
    if not (probe / "rollouts.jsonl").exists():
        py("probe_curve.py", "--data", questions, "--sources", BON_SOURCES, "--per-source", args.per_source,
           "--samples", 4, "--adapter", args.start, "--out", probe, "--dtype", args.dtype, "--generate-only",
           "--chunk", 150)
    if not (pairs / "pairs.jsonl").exists():
        py("mvt_pairs.py", "--rollouts", probe / "rollouts.jsonl", "--adapter", args.start, "--out", pairs,
           "--dtype", args.dtype)
        py("contamination.py", "--data", pairs / "pairs.jsonl", "--rollouts", probe / "rollouts.jsonl")
    print((pairs / "summary.txt").read_text(), flush=True)
    if not (adapter / "adapter_config.json").exists():
        py("dpo.py", "--pairs", pairs / "pairs.jsonl", "--init-adapter", args.start, "--out", adapter,
           "--max-steps", args.steps, "--dtype", args.dtype)
    base = ROOT / "s4-base"
    evaluate(base, samples=4)
    checkpoints = sorted(adapter.glob("step-*"), key=lambda p: int(p.name.split("-")[1]))
    for checkpoint in [c for c in checkpoints if int(c.name.split("-")[1]) % 50 == 0] or checkpoints[-1:]:
        out = ROOT / f"s4-{name}-{checkpoint.name}"
        evaluate(out, checkpoint, samples=4)
        print(score_line(f"{name} {checkpoint.name} (4 samples)", out, base=base), flush=True)
    print(diagnose(out, base=base))


def gridtable(args):
    """The method x kind table: each run's effect on every benchmark, its own benchmark first."""
    rows = [json.loads(l) for l in (GRID / "grid.jsonl").read_text().splitlines() if l.strip()]
    benchmarks = ["svamp", "mbpp", "bfcl", "ifeval"]

    def cell(row, bench):
        parts = [v for d, v in row["benchmarks"].items() if (d.startswith("bfcl") if bench == "bfcl" else d == bench)]
        if not parts:
            return "-"
        saved = sum(p["saved"] or 0 for p in parts) / len(parts)
        delta = 100 * sum(p["delta"] for p in parts) / len(parts)
        return f"{saved:+.0%} / {delta:+.1f}"

    print(f"{'run':<28}" + "".join(f"{b:>18}" for b in benchmarks) + f"{'average':>18}")
    for row in sorted(rows, key=lambda r: (r["type"], r["name"])):
        own = TYPES.get(row["type"], (None, None))[1]
        cells = [(">" if b == own else " ") + cell(row, b) for b in benchmarks]
        average = f"{row['average']['saved']:+.0%} / {100 * row['average']['delta']:+.1f}"
        print(f"{row['name']:<28}" + "".join(f"{c:>18}" for c in cells) + f"{average:>18}")
    print("\ncells: tokens saved / accuracy change in points vs the raw model (4 samples); > marks the "
          "benchmark of the kind trained on, the others show spill-over")


def recheck(args):
    """Score the base or one adapter with more samples per question: a tighter interval on the same subset."""
    base = ROOT / f"s{args.samples}-base"
    evaluate(base, samples=args.samples)
    if args.adapter:
        name = args.adapter.removeprefix("adapters/").replace("/", "-")
        out = ROOT / f"s{args.samples}-{name}"
        evaluate(out, args.adapter, samples=args.samples)
        print(score_line(f"{name} ({args.samples} samples)", out, base=base))
        print(diagnose(out, base=base))


def compare(out, base=BASE):
    json_path = Path(out) / "compare.json"
    if not json_path.exists():
        py("compare.py", base, out, "--out", json_path)
    return json.loads(json_path.read_text())


FULL_BFCL = {"bfcl_simple_python": 400, "bfcl_multiple": 200, "bfcl_irrelevance": 240}  # the full evaluation's sizes


def full_weighted(result):
    """The average as the FULL evaluation would weight it (user, 30 Oct): our subsets hold 40 questions per BFCL part,
    the full evaluation 400 / 200 / 240, so inside BFCL one-call counts twice as much as multiple. BFCL's accuracy change
    and token ratio are re-pooled with those weights (tokens weighted by each part's base mean), then averaged over the 4
    benchmarks exactly as compare.py does (mean of accuracy changes; 1 - geometric mean of token ratios)."""
    import math
    res = result["results"]
    deltas, ratios = [], []
    for d in ("svamp", "mbpp", "ifeval"):
        if d in res and res[d]["token_reduction"] is not None:
            deltas.append(res[d]["accuracy_delta"])
            ratios.append(1 - res[d]["token_reduction"])
    parts = [d for d in FULL_BFCL if d in res and res[d]["token_reduction"] is not None]
    if parts:
        w = {d: FULL_BFCL[d] for d in parts}
        deltas.append(sum(w[d] * res[d]["accuracy_delta"] for d in parts) / sum(w.values()))
        base_tok = {d: res[d]["base"].get("mean_output_tokens") or 1.0 for d in parts}
        before = sum(w[d] * base_tok[d] for d in parts)
        after = sum(w[d] * base_tok[d] * (1 - res[d]["token_reduction"]) for d in parts)
        ratios.append(after / before)
    if len(deltas) < 2:
        return None, None
    saved = 1 - math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    return saved, sum(deltas) / len(deltas)


def score_line(name, out, with_delta=False, base=BASE):
    result = compare(out, base)
    parts = []
    for dataset, r in sorted(result["results"].items()):
        saved = r["token_reduction"]
        parts.append(f"{dataset} {'n/a' if saved is None else f'{saved:+.0%}'} / {100 * r['accuracy_delta']:+.1f}")
    average = result["average"]
    saved, delta = average["token_reduction"], 100 * average["accuracy_delta"]
    line = (f"== {name}: tokens saved {'n/a' if saved is None else f'{saved:+.1%}'}, "
            f"accuracy {delta:+.1f} pts ({'within' if delta >= LIMIT else 'OUTSIDE'} the limit)\n   "
            + "; ".join(parts))
    fw_saved, fw_delta = full_weighted(result)
    if fw_saved is not None:
        line += f"\n   full-eval weighting (BFCL 400/200/240): tokens saved {fw_saved:+.1%}, accuracy {100 * fw_delta:+.1f} pts"
    try:  # one table of every score (user, 29 Oct: "log the results and be able to plot them")
        table = BON / "results.csv"
        table.parent.mkdir(parents=True, exist_ok=True)
        if not table.exists():
            table.write_text("name,run_dir,base_dir,saved,accuracy," + ",".join(f"{d}_saved,{d}_acc" for d in sorted(result["results"])) + "\n")
        if f",{out}," not in table.read_text():
            cells = [f"{'' if r['token_reduction'] is None else round(100 * r['token_reduction'], 2)},{round(100 * r['accuracy_delta'], 2)}"
                     for _, r in sorted(result["results"].items())]
            with table.open("a") as handle:
                handle.write(f"\"{name}\",{out},{base},{'' if saved is None else round(100 * saved, 2)},{round(delta, 2)},"
                             + ",".join(cells) + "\n")
    except Exception as error:  # the table must never break a stage
        print(f"results.csv: {error}", flush=True)
    return (line, delta) if with_delta else line


def diagnose(out, dataset="svamp", examples=2, base=BASE):
    """How the candidate fails where the base succeeds on one benchmark: lengths, error kinds, examples."""
    def load(directory):
        rows = [json.loads(l) for l in (Path(directory) / "records.jsonl").read_text().splitlines() if l.strip()]
        return {(r["id"], r["sample_index"]): r for r in rows if r["dataset"] == dataset}
    base, new = load(base), load(out)
    lost = [(base[k], new[k]) for k in base if k in new and base[k].get("correct") and not new[k].get("correct")]
    gained = sum(1 for k in base if k in new and not base[k].get("correct") and new[k].get("correct"))
    kinds = {}
    for _, r in lost:
        kind = "truncated" if r["truncated"] else (r.get("parse_error") or "wrong number")
        kinds[kind] = kinds.get(kind, 0) + 1
    mean = lambda xs: sum(xs) / max(1, len(xs))
    lines = [f"   {dataset}: lost {len(lost)}, gained {gained}; lost by {kinds}; reasoning tokens on lost rows "
             f"base {mean([b['reasoning_tokens'] for b, _ in lost]):.0f} -> {mean([r['reasoning_tokens'] for _, r in lost]):.0f}"]
    for _, r in lost[:examples]:
        lines.append("   --- lost example: " + r["text"][-500:].replace("\n", " / "))
    return "\n".join(lines)


def plan(args):
    """Run every line of plan.txt (track.py arguments) in order, rereading it after each stage.

    plan.txt comes from the branch, so pushing a line queues a new stage without touching the
    notebook. Stages skip finished work, so a line already done costs nothing.
    """
    import threading
    import time
    if not Path("/content/drive/MyDrive").exists():
        print("drive: NOT mounted; results live only on this runtime's disk", flush=True)
    import re
    plan_text = Path(args.plan_file).read_text() if Path(args.plan_file).exists() else ""
    wanted = {m for l in plan_text.splitlines() if l.strip() and not l.startswith("#")
              for m in re.findall(r"models/([\w.\-]+)", l)}
    stages = {l.split()[0] for l in plan_text.splitlines() if l.strip() and not l.startswith("#")} - {"end"}
    if stages and stages <= {"callot", "probezoo", "targeted", "decide", "zooreport", "benchlot", "drpo", "teach", "wronglen", "nothink", "mathmatch", "balround", "stopgold", "mathfix", "budgetprompt", "promptsearch", "buildpool", "buildpool2", "poolfix", "poolmore", "lupinanswers", "lupin", "topiccheck", "refsets", "codecut", "crosscheck", "fullsoup", "fullmodel", "screenset", "rescreen", "dial", "pertype", "tooltopup", "poolup", "newtop", "onpolicy", "typeonly", "polans", "ptrepair", "odpo", "ladder", "devset", "devscore", "devsvamp", "transfer", "p1src", "hold", "mixscreen", "thinkprune", "chirurgical", "astra", "astradata", "consultpack", "wanda", "wandaeval", "tokenskip", "dropmodels", "drivereport", "drivetidy", "breakdown", "struggle", "copddemo", "ifevalrules", "copdanalysis", "devset", "devscore", "devsvamp", "transfer", "p1src", "hold"}:  # label lots + zoo need only their own files, the draws and the model
        # (4 Nov) old-line adapters (tg/stop/mf/drpo/sb/teach/tsk) no longer restored at plan start: hours on a fresh runtime
        drive_sync("restore", only=["data/*.jsonl", "runs/bon/kprobe-*/rollouts.jsonl", "runs/bon/*plot*",
                                    "runs/bon/probe-plot*", "runs/bon/kprobe-plot*", "runs/bon/pairs-plot*", "runs/bon/zoo-*",
                                    "runs/bon/pairs-tg*", "runs/bon/pairs-stop*", "runs/mvtf/f8-base", "questions_fresh*",
                                    "runs/bon/*bench*", "runs/bon/probe-mathmatch", "runs/bon/ps-*", "runs/bon/poolcheck-*", "runs/bon/probe-poolcheck-*", "runs/bon/probe-poolcand-*", "runs/bon/lupin-answers-*", "runs/mvtf/v4-base", "runs/mvtf/eval-base", "questions_40", "questions_val*", "questions_ref*", "runs/mvtf/v4-base-*", "runs/mvtf/r4-*", "runs/bon/probe-mf*", "runs/bon/pairs-mf*", "runs/bon/tskip-*", *[f"models/{m}" for m in wanted]])
    else:
        drive_sync("restore", models=wanted)

    keep_saving(300)
    done, idle_since = set(), None
    while True:
        plan_file = Path(args.plan_file)
        lines = [l.split("#")[0].strip() for l in plan_file.read_text().splitlines()] if plan_file.exists() else []
        todo = [l for l in lines if l and l not in done]
        if todo and todo[0] == "end":  # the queue's last line: save everything, let Drive upload it, stop
            finish()
            return
        if not todo:
            idle_since = idle_since or time.time()
            if time.time() - idle_since > args.wait_hours * 3600:
                return
            time.sleep(300)
            update_files()
            continue
        idle_since = None
        print(f"plan: {todo[0]}", flush=True)
        if subprocess.run([sys.executable, "track.py", *todo[0].split()], env=os.environ | {"TRACK_UPDATED": ""}).returncode:
            print(f"plan: '{todo[0]}' failed; moving on", flush=True)
        done.add(todo[0])
        drive_sync("save")
        update_files()


def keep_saving(seconds):
    """Copy everything to Drive every `seconds` in a background thread (results, models, data, logs)."""
    import threading
    import time

    def loop():
        while True:
            time.sleep(seconds)
            drive_sync("save")
    threading.Thread(target=loop, daemon=True).start()


def end(args):
    """The queue's last line as a stage (a worker started before `end` existed runs it like any other line):
    save everything, flush Drive, then stop the worker that launched this stage, so the notebook moves on
    to releasing the runtime instead of polling an empty queue for hours."""
    import signal
    log = Path("/content/track_log.txt")
    if log.exists() and Path("/content/drive/MyDrive").exists():  # every score line, easy to read from Drive
        lines = [l for l in log.read_text(errors="replace").splitlines() if l.startswith(("plan:", "== ", "   bfcl"))]
        name = "SUMMARY_" + (os.environ.get("TRACK_SESSION") or "runtime1") + ".txt"
        DRIVE.mkdir(parents=True, exist_ok=True)
        (DRIVE / name).write_text("\n".join(lines) + "\n")
        print(f"drive: wrote {name}", flush=True)
    finish()
    parent = os.getppid()
    print(f"plan: stopping the worker (pid {parent})", flush=True)
    os.kill(parent, signal.SIGTERM)


def finish():
    """Final save: everything to Drive, flushed to the cloud, so the runtime can be released safely."""
    import time
    drive_sync("save")
    subprocess.run(["sync"])
    # drive.flush_and_unmount() hangs outside the notebook kernel: give Drive's uploader time here, and let the
    # notebook's last cell flush and release the runtime
    time.sleep(180)
    print("plan: end of queue; everything is on Drive", flush=True)


def update_files():
    os.environ.pop("TRACK_UPDATED", None)
    update(restart=False)


def scores(args):
    plan(args)  # the notebook's last cell: work through the queued plan first
    for out in sorted(ROOT.glob("eval-*")):
        if out != BASE and (out / "results.json").exists():
            print(score_line(out.name.removeprefix("eval-"), out))
            print(diagnose(out, examples=0))
    for base in sorted(ROOT.glob("s*-base")):
        for out in sorted(ROOT.glob(base.name.removesuffix("base") + "*")):
            if out != base and (out / "results.json").exists():
                print(score_line(out.name, out, base=base))


if __name__ == "__main__":
    main()
