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
    parser.add_argument("stage", choices=("setup", "base", "rollouts", "mvt", "report", "fade", "scores"))
    parser.add_argument("--travel", type=float, default=256)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--order", default="gain")
    parser.add_argument("--floor-travel", type=float)
    parser.add_argument("--word-step", type=float, default=0.1, help="fading: words compacted per round; 0 = paragraphs only")
    parser.add_argument("--tag", default="", help="suffix for the fading run's directories")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--word-scorer", default="lingua", help="fading: lingua (experiment 2) or marginal (experiment 3)")
    args = parser.parse_args()
    update()
    globals()[args.stage](args)


def update():
    """On Colab, pull this branch's scripts before every stage, so a pushed fix needs no restart."""
    if os.environ.get("TRACK_UPDATED") or not Path("/content").exists():
        return
    clone = Path("/content/T")
    shutil.rmtree(clone, ignore_errors=True)
    if subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", "mvt-fading",
                       "https://github.com/arsenelemoigne/Test", str(clone)]).returncode:
        print("could not update the scripts; running the local copies", flush=True)
        return
    changed = False
    for source in (clone / "thinkingcap").glob("*.py"):
        if not Path(source.name).exists() or not filecmp.cmp(source, source.name, shallow=False):
            shutil.copy(source, source.name)
            changed |= source.name == "track.py"
    print("scripts at " + subprocess.run(["git", "-C", str(clone), "log", "-1", "--format=%h %s"],
                                         capture_output=True, text=True).stdout.strip(), flush=True)
    if changed:
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


def evaluate(out, adapter=None):
    if not (Path(out) / "results.json").exists():
        py("evaluate.py", "--questions", QUESTIONS, "--samples", 2, "--out", out,
           *(["--adapter", adapter] if adapter else []))


def base(args):
    evaluate(BASE)


def rollouts(args):
    if not ROLLOUTS.exists():
        # The same training slice as SFT rounds 1-2: 150 questions per source, 2 samples, 2,048 tokens.
        py("probe_curve.py", "--data", "data/train.jsonl", "--sources", "gsm8k,arc_easy,commonsense_qa",
           "--per-source", 150, "--samples", 2, "--max-tokens", 2048, "--out", ROLLOUTS.parent,
           "--dtype", args.dtype, "--generate-only")


def mvt(args):
    name = f"mvt{args.travel:g}"
    prune, adapter, out = ROOT / f"prune-{name}", Path("adapters") / name, ROOT / f"eval-{name}"
    if not (prune / "sft.jsonl").exists():
        py("prune_traces.py", "--rollouts", ROLLOUTS, "--out", prune, "--rule", "mvt", "--travel", args.travel,
           "--check-samples", 4, "--dtype", args.dtype)
    print((prune / "summary.txt").read_text())
    if not (adapter / "adapter_config.json").exists():
        py("sft.py", "--data", prune / "sft.jsonl", "--out", adapter, "--max-per-source", 70, "--dtype", args.dtype)
    evaluate(out, adapter)
    print(score_line(name, out))


def report(args):
    logs = sorted(ROOT.glob("prune-mvt*/log.jsonl"))
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
        if delta < LIMIT:
            print(f"round {k} is past the {LIMIT:+.0f}-point limit: stop; the answer is round {k - 1}")
            return
        previous = (data, adapter)


def compare(out):
    json_path = Path(out) / "compare.json"
    if not json_path.exists():
        py("compare.py", BASE, out, "--out", json_path)
    return json.loads(json_path.read_text())


def score_line(name, out, with_delta=False):
    result = compare(out)
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


def scores(args):
    for out in sorted(ROOT.glob("eval-*")):
        if out != BASE and (out / "results.json").exists():
            print(score_line(out.name.removeprefix("eval-"), out))


if __name__ == "__main__":
    main()
