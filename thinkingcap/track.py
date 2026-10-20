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
                                          "grid_data", "grid", "gridtable", "combine", "merge", "fullbest", "stats", "bon", "scores"))
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
    parser.add_argument("--type", choices=("math", "code", "tool", "toolh", "text", "choice"), help="grid: the kind of data trained on")
    parser.add_argument("--method", choices=("whole", "mvt", "marginal", "fade"), help="grid: how its traces are shortened")
    parser.add_argument("--unit", default="paragraph", help="grid: paragraph, sentence or tokensN")
    parser.add_argument("--keep-whole", default="", help="grid: categories kept uncut, e.g. irrelevance")
    parser.add_argument("--per-source", type=int, default=250, help="grid_data: questions per kind")
    parser.add_argument("--inputs", nargs="*", default=[],
                        help="combine: sft.jsonl[:source[:source...]] files whose rows (of those sources) are pooled")
    parser.add_argument("--margin", type=float, default=-2.5,
                        help="fullbest: lowest 4-sample accuracy change (points) a candidate may have")
    parser.add_argument("--plan-file", default="plan.txt", help="plan: the queue this runtime works through")
    parser.add_argument("--wait-hours", type=float, default=0,
                        help="plan: when the queue is empty, keep polling the branch this long; a running cell "
                             "also keeps Colab from disconnecting an idle runtime and wiping its disk")
    parser.add_argument("--samples", type=int, default=4, help="recheck: samples per question")
    parser.add_argument("--max-per-source", type=int, default=70, help="mvt: SFT rows per dataset")
    parser.add_argument("--min-steps", type=int, default=1, help="mvt: never cut below this many paragraphs")
    parser.add_argument("--repeat", default="", help="mvt: dataset:times for SFT, e.g. gsm8k:2")
    parser.add_argument("--full-sources", default="", help="mvt: datasets whose traces stay whole, e.g. gsm8k")
    parser.add_argument("--word-scorer", default="lingua", help="fading: lingua (experiment 2) or marginal (experiment 3)")
    args = parser.parse_args()
    update()
    globals()[args.stage](args)


def update(restart=True):
    """On Colab, pull this branch's scripts before every stage, so a pushed fix needs no restart."""
    if os.environ.get("TRACK_UPDATED") or not Path("/content").exists():
        return
    clone = Path("/content/T")
    shutil.rmtree(clone, ignore_errors=True)
    branch = os.environ.get("TRACK_BRANCH", "mvt-fading")
    if subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", branch,
                       "https://github.com/arsenelemoigne/Test", str(clone)]).returncode:
        print("could not update the scripts; running the local copies", flush=True)
        return
    changed = False
    for source in [*(clone / "thinkingcap").glob("*.py"), *(clone / "thinkingcap").glob("plan*.txt")]:
        if not Path(source.name).exists() or not filecmp.cmp(source, source.name, shallow=False):
            shutil.copy(source, source.name)
            changed |= source.name == "track.py"
    print("scripts at " + subprocess.run(["git", "-C", str(clone), "log", "-1", "--format=%h %s"],
                                         capture_output=True, text=True).stdout.strip(), flush=True)
    if changed and restart:
        os.execve(sys.executable, [sys.executable, *sys.argv], os.environ | {"TRACK_UPDATED": "1"})


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
           "--dtype", args.dtype, "--generate-only")


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
             "text": ("ifeval_like", "ifeval"), "choice": ("arc_easy,commonsense_qa", None),
             "toolh": ("hermes_tools", "bfcl")}  # last night's tool data under tonight's settings (a control)
    TYPE_DATA = Path("data/matched.jsonl")
    EXTRA_DATA = {"hermes_tools": Path("data/hermes.jsonl")}  # own file: matched.jsonl stays as sampled
    PREPARE = ["--sources", "gsm8k_all,mbpp_train,toolace,ifeval_like,mc", "--per-source", 300]
    EASY = {"math": 700}  # keep GSM8K questions with a correct reply of at most 700 tokens
else:
    GRID = Path("runs/grid")
    TYPES = {"math": ("gsm8k", "svamp"), "code": ("mbpp_train", "mbpp"), "tool": ("hermes_tools", "bfcl"),
             "text": ("ifeval_like", "ifeval")}
    TYPE_DATA = Path("data/types.jsonl")
    EXTRA_DATA = {}
    PREPARE = ["--per-source", 400]
    EASY = {}


def grid_data(args):
    """Training questions of every kind, and the model's own replies to them (2 per question)."""
    wanted = [s for sources, _ in TYPES.values() for s in sources.split(",")]
    questions = TYPE_DATA
    have = {json.loads(l)["source"] for l in questions.read_text().splitlines() if l.strip()} \
        if questions.exists() else set()
    if set(wanted) - have - set(EXTRA_DATA):
        py("prepare_types.py", *PREPARE, "--out", questions)
    for source, path in EXTRA_DATA.items():
        if not path.exists():
            py("prepare_types.py", "--sources", source, "--per-source", args.per_source, "--out", path)
    probe = GRID / "probe"
    rollouts = probe / "rollouts.jsonl"
    done = {json.loads(l)["source"] for l in rollouts.read_text().splitlines() if l.strip()} \
        if rollouts.exists() else set()
    missing = [source for source in wanted if source not in done]
    files = {}
    for source in missing:
        files.setdefault(EXTRA_DATA.get(source, questions), []).append(source)
    for data, sources in files.items():  # sample only what is missing and append it, so earlier replies stay
        extra = GRID / "probe-extra"
        shutil.rmtree(extra, ignore_errors=True)
        py("probe_curve.py", "--data", data, "--sources", ",".join(sources), "--per-source", args.per_source,
           "--samples", 2, "--out", extra, "--dtype", args.dtype, "--generate-only")
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
        + (f"-{unit}" if unit != "paragraph" else "") + (f"-whole-{args.keep_whole}" if args.keep_whole else "")
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
        rule += ["--full-sources", args.keep_whole] if args.keep_whole else []
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
    base, out = ROOT / "s4-base", GRID / f"eval-merge-{args.name}"
    evaluate(base, samples=4)
    evaluate(out, samples=4, model=model)
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
           "--samples", 8, "--out", probe, "--dtype", args.dtype, "--generate-only")
    if not (select / "sft.jsonl").exists():
        py("select_shortest.py", "--rollouts", probe / "rollouts.jsonl", "--out", select,
           "--easy-tokens", "gsm8k_all=700")
        py("contamination.py", "--data", select / "sft.jsonl", "--rollouts", probe / "rollouts.jsonl")
    print((select / "summary.txt").read_text(), flush=True)
    adapter = Path("adapters") / f"bon-{name}"
    if not (adapter / "adapter_config.json").exists():
        py("sft.py", "--data", select / "sft.jsonl", "--out", adapter, "--epochs", 1, "--lr", 5e-5,
           "--max-per-source", args.max_per_source, "--dtype", args.dtype)
    base, out = ROOT / "s4-base", ROOT / f"s4-bon-{name}"
    evaluate(base, samples=4)
    evaluate(out, adapter, samples=4)
    print(score_line(f"bon-{name} (4 samples, 240 test questions)", out, base=base), flush=True)
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
    import time
    done, idle_since = set(), None
    while True:
        plan_file = Path(args.plan_file)
        lines = [l.split("#")[0].strip() for l in plan_file.read_text().splitlines()] if plan_file.exists() else []
        todo = [l for l in lines if l and l not in done]
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
        update_files()


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
