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
from collections import defaultdict
import filecmp
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("runs/mvtf")
QUESTIONS = "questions_40"
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
                                          "grid_data", "grid", "gridtable", "combine", "merge", "fullbest", "stats", "bon", "bon_dpo", "code_round", "chain", "repair", "rlcal", "calcheck", "caltable", "callot", "cutexamples", "probezoo", "targeted", "decide", "zooreport", "wronglen", "benchlot", "drpo", "teach", "probeknow", "audit", "balanced", "full2", "clean", "confirm", "fresh", "peek", "s1log", "end", "diag", "steer_read", "steer", "dpo_round", "scores"))
    parser.add_argument("--travel", type=float, default=256)
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
    parser.add_argument("--rl-dynamic", type=float, default=0.0,
                        help="drpo: DLER dynamic sampling factor (draw this many times the questions, keep those with a right reply)")
    parser.add_argument("--rl-variant", choices=("drpo", "shorterbetter"), default="drpo",
                        help="drpo stage: DRPO objective, or GRPO with the ShorterBetter reward (alpha*correct - beta*|len-SOL|)")
    parser.add_argument("--drpo-steps", type=int, default=60, help="drpo: RL steps (--drpo-questions x 8 replies each)")
    parser.add_argument("--drpo-questions", type=int, default=48, help="drpo: questions per step, each used once")
    parser.add_argument("--drpo-score", type=int, nargs="*", default=[40, 60], help="drpo: checkpoints scored on fresh x8")
    parser.add_argument("--teach-questions", type=int, default=3000, help="teach: fresh questions answered once each")
    parser.add_argument("--teach-variants", nargs="*", default=["rel:0.05", "rel:0.1"],
                        help="teach: stop targets, one model each (fixed:T = yes/no judge above T; rel:M = soft judge's "
                             "predicted success within M of the full thinking's)")
    parser.add_argument("--teach-lots", nargs="*", default=["plotB", "plotB2", "plotB3"],
                        help="teach: label lots of --start the judge is fitted on")
    parser.add_argument("--lot-samples", type=int, default=8, help="callot: full replies per question (20 = a reliable question rate)")
    parser.add_argument("--tg-shorten", default="kodcode,ifeval_like", help="targeted: types whose safe early cuts are taught")
    parser.add_argument("--tg-long", default="gsm_ic,gsm_plus", help="targeted: types taught to think longer (math)")
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
    for source in [*(clone / "thinkingcap").glob("*.py"), *(clone / "thinkingcap").glob("plan*.txt")]:
        if not Path(source.name).exists() or not filecmp.cmp(source, source.name, shallow=False):
            shutil.copy(source, source.name)
            changed |= source.name == "track.py"
    print("scripts at " + subprocess.run(["git", "-C", str(clone), "log", "-1", "--format=%h %s"],
                                         capture_output=True, text=True).stdout.strip(), flush=True)
    if changed and restart:
        os.execve(sys.executable, [sys.executable, *sys.argv], os.environ | {"TRACK_UPDATED": "1"})


DRIVE = Path("/content/drive/MyDrive/thinkingcap-mvtf")


def drive_sync(direction, models=None, only=None):
    """Copy runs/, adapters/ and data/ to Google Drive ("save") or back ("restore"), when Drive is mounted.

    Colab wipes the runtime disk when it recycles a runtime; Drive keeps every result, adapter and rollout.
    """
    if not Path("/content/drive/MyDrive").exists():
        return
    DRIVE.mkdir(parents=True, exist_ok=True)
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
        seen = {text(json.loads(l)) for p in used for l in p.read_text().splitlines() if l.strip() and "prompt" in l}
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
               "--keep", clean, "--out", cal, "--include-wrong", "--dtype", args.dtype)
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


def mixed_pool(name, need, data_seed, kod_file):
    """A fresh question pool mixed like the evaluation (GSM-IC/Plus, IFEval-like, ToolACE, Hermes, MBPP-train,
    KodCode), shuffled so every batch mixes the types; earlier draws removed, contamination-checked."""
    import random
    pool = Path(f"data/rl-{name}-mixed.jsonl")
    if not pool.exists():
        raw = Path(f"data/rl-{name}-raw.jsonl")
        per = {"gsm_ic": 0.17, "gsm_plus": 0.17, "ifeval_like": 0.24, "toolace": 0.2, "hermes_tools": 0.1}  # no MBPP (user)
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
        answered = {json.loads(l).get("question") for p in BON.glob("kprobe-*/rollouts.jsonl")
                    for l in p.read_text().splitlines() if l.strip()}
        kod = [json.loads(l) for l in kod_file.read_text().splitlines() if l.strip()]
        random.Random(data_seed).shuffle(kod)
        rows += [r for r in kod if text(r) not in answered][:int(need * 0.16) + 50]  # code share incl. the former MBPP 6%
        used = [p for p in Path("data").glob("*.jsonl") if p.name.startswith(("bon", "bal", "chain", "trapmath", "rlcal",
                "rl-", "lot-")) and "raw" not in p.name and p != pool]
        seen = {text(r) for p in used for l in p.read_text().splitlines() if l.strip()
                for r in [json.loads(l)] if isinstance(r.get("prompt"), list) and r["prompt"]}
        rows = [r for r in rows if text(r) not in seen]
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
    draws removed). Checkpoints every 20 steps; the --drpo-score ones are merged and scored on the fresh set x8."""
    import random
    name, base = args.name, args.start
    if base.startswith("models/") and not (Path(base) / "config.json").exists():  # queued after this runtime's restore
        drive_sync("restore", only=[base])
    need = math.ceil(args.drpo_steps * args.drpo_questions * max(1.0, args.rl_dynamic))
    pool = mixed_pool(name, need, args.data_seed, args.kod_file)
    out = Path("adapters") / f"{name}-{args.rl_variant}"
    variant = (["--objective", "drpo", "--drpo-lambda", args.drpo_lambda] if args.rl_variant == "drpo"
               else ["--objective", "grpo", "--reward", "shorterbetter"])
    if not (out / "adapter_config.json").exists():
        py("rl_train.py", "--model", base, "--data", pool, "--sources", "all", "--out", out,
           "--steps", args.drpo_steps, "--questions", args.drpo_questions, "--group", 8, "--max-tokens", 4096,
           *variant, "--updates", 2, "--grader", "typed", "--dynamic-sampling", args.rl_dynamic,
           "--redundant", "none", "--lr", args.rl_lr, "--save-every", 20, "--dtype", args.dtype)
    print((out / "rl_log.jsonl").read_text(), flush=True)
    fresh_q = f"questions_fresh{FRESH_SEED}"
    evaluate(ROOT / "f8-base", samples=8, questions=fresh_q)
    for step in args.drpo_score:
        ckpt = out / f"step-{step}"
        if not (ckpt / "adapter_config.json").exists():
            continue
        model = Path("models") / f"{name}-{step}"
        if not (model / "config.json").exists():
            py("merge.py", "--model", base, "--out", model, "--adapters", f"{ckpt}:1")
        res = ROOT / f"f8-{name}-{step}"
        evaluate(res, samples=8, questions=fresh_q, model=model)
        print(score_line(f"{name}-{step} (fresh 240 x 8)", res, base=ROOT / "f8-base"), flush=True)


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
    pool = mixed_pool(name, args.teach_questions, args.data_seed, args.kod_file)
    probe = BON / f"probe-{name}"
    if not (probe / "rollouts.jsonl").exists():
        n = sum(1 for l in pool.read_text().splitlines() if l.strip())
        py("probe_curve.py", "--data", pool, "--per-source", 100000, "--samples", 1, "--model", base, "--out", probe,
           "--dtype", args.dtype, "--generate-only", "--chunk", 300, "--max-questions", min(n, args.teach_questions))
    out = Path("adapters") / f"{name}-teach"
    py("teach_stop.py", "--model", base, "--cal", *[BON / f"pairs-{l}" / "cal" for l in args.teach_lots],
       "--rollouts", probe / "rollouts.jsonl", "--out", out, "--variants", *args.teach_variants, "--dtype", args.dtype)
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


def wronglen(args):
    """Spill-over check (CPU, seconds): thinking length of right, wrong and never-solved answers on the fresh set x8,
    every scored model vs the original, read straight from the Drive copy (wrong_lengths.py)."""
    py("wrong_lengths.py", "--root", DRIVE / "runs" / "mvtf", "--out", ROOT / "wrong_lengths.txt")


def targeted(args):
    """Targeted DPO from a label lot (targeted_pairs.py): shorten code and instructions where the lot's cuts show the
    answer already holds; make math think longer (full right beats an early cut that fails, long right beats short wrong);
    other types untouched. Fresh LoRA on --start, 3 seeds, soups at --dpo-steps, every soup scored on the fresh set x8."""
    name, base = args.name, args.start
    pairs = BON / f"pairs-{name}"
    if not (pairs / "pairs.jsonl").exists():
        ensure_lots(args.probe_extra_from)
        rows = []
        for lot in args.probe_extra_from:  # one or more label lots of the same model
            part = pairs / lot
            if not (part / "pairs.jsonl").exists():
                py("targeted_pairs.py", "--cal", BON / f"pairs-{lot}" / "cal", "--rollouts",
                   BON / f"probe-{lot}" / "rollouts.jsonl", BON / f"kprobe-{lot}" / "rollouts.jsonl",
                   "--keep", BON / f"{lot}-clean.jsonl", "--out", part,
                   "--shorten-sources", args.tg_shorten, "--long-sources", args.tg_long, "--rule", args.tg_rule)
            print((part / "summary.txt").read_text(), flush=True)
            rows += [l for l in (part / "pairs.jsonl").read_text().splitlines() if l.strip()]
        (pairs / "pairs.jsonl").write_text("".join(l + "\n" for l in rows))
        (pairs / "summary.txt").write_text(f"{len(rows)} pairs from {', '.join(args.probe_extra_from)}\n")
    print((pairs / "summary.txt").read_text(), flush=True)
    rows = [json.loads(l) for l in (pairs / "pairs.jsonl").read_text().splitlines() if l.strip()]
    counts = [sum(r["kind"] == k for r in rows) for k in ("shorten", "guard", "hard")]
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
    if stages and stages <= {"callot", "probezoo", "targeted", "decide", "zooreport", "benchlot", "drpo", "teach", "wronglen"}:  # label lots + zoo need only their own files, the draws and the model
        drive_sync("restore", only=["data/*.jsonl", "runs/bon/kprobe-*/rollouts.jsonl", "runs/bon/*plot*",
                                    "runs/bon/probe-plot*", "runs/bon/kprobe-plot*", "runs/bon/pairs-plot*", "runs/bon/zoo-*",
                                    "runs/bon/pairs-tg*", "adapters/fresh-tg*", "adapters/tg*", "models/tg*",
                                    "runs/bon/pairs-stop*", "adapters/fresh-stop*", "adapters/stop*", "runs/mvtf/f8-stop*",
                                    "runs/mvtf/f8-base", "runs/mvtf/f8-tg*", "questions_fresh*",
                                    "runs/bon/*bench*", "adapters/drpo*", "runs/mvtf/f8-drpo*", "adapters/*-shorterbetter", "runs/mvtf/f8-sb*", "adapters/*-teach", "runs/mvtf/f8-*teach*",
                                    *[f"models/{m}" for m in wanted]])
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
