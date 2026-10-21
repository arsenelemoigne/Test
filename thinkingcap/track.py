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
import filecmp
import json
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("stage", choices=("setup", "base", "rollouts", "mvt", "report", "fade", "plan", "recheck", "mix", "full",
                                          "grid_data", "grid", "gridtable", "combine", "merge", "fullbest", "stats", "bon", "bon_dpo", "code_round", "full2", "clean", "confirm", "fresh", "peek", "diag", "steer_read", "steer", "dpo_round", "scores"))
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
    parser.add_argument("--drop-sources", default="", help="bon_dpo: sources with no pairs at all")
    parser.add_argument("--dpo-epochs", type=int, default=2, help="bon_dpo: passes over the pairs")
    parser.add_argument("--eval-steps", type=int, nargs="*", default=[], help="bon_dpo: checkpoints scored (default all)")
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
    parser.add_argument("--word-scorer", default="lingua", help="fading: lingua (experiment 2) or marginal (experiment 3)")
    args = parser.parse_args()
    update()
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


def drive_sync(direction):
    """Copy runs/, adapters/ and data/ to Google Drive ("save") or back ("restore"), when Drive is mounted.

    Colab wipes the runtime disk when it recycles a runtime; Drive keeps every result, adapter and rollout.
    """
    if not Path("/content/drive/MyDrive").exists():
        return
    DRIVE.mkdir(parents=True, exist_ok=True)
    names = ["runs", "adapters", "data", "models", *[p.name for p in Path(".").glob("questions_*") if p.is_dir()]]
    if direction == "restore":
        names = ["runs", "adapters", "data", "models", *[p.name for p in DRIVE.glob("questions_*") if p.is_dir()]]
    for name in names:
        source, target = (Path(name), DRIVE / name) if direction == "save" else (DRIVE / name, Path(name))
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
    if not Path("data/train.jsonl").exists():
        py("utils/train/prepare_data.py")
    if not Path("questions/questions.jsonl").exists():
        py("utils/eval/benchmarks.py")
    if not Path(QUESTIONS, "questions.jsonl").exists():
        py("subset_questions.py", "--per-dataset", 40, "--out", QUESTIONS)


def evaluate(out, adapter=None, samples=2, questions=QUESTIONS, model=None):
    if not (Path(out) / "results.json").exists():
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
    drive_sync("restore")

    def keep_saving():  # mid-stage saves too: a recycle loses at most 20 minutes
        while True:
            time.sleep(1200)
            drive_sync("save")
    threading.Thread(target=keep_saving, daemon=True).start()
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


def finish():
    """Final save: everything to Drive, flushed to the cloud, so the runtime can be released safely."""
    import time
    drive_sync("save")
    subprocess.run(["sync"])
    try:
        from google.colab import drive
        drive.flush_and_unmount()
        print("drive: flushed and unmounted", flush=True)
    except Exception as error:  # outside the notebook kernel: give Drive's uploader time instead
        print(f"drive: flush not available here ({error}); waiting 5 minutes for the upload", flush=True)
        time.sleep(300)
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
