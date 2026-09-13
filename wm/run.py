"""
The experiment runner.

    python -m wm.run inputs                 # print each condition's input, no API calls
    python -m wm.run prose                  # build + cache the prose twin (1 frontier call)
    python -m wm.run trial A4 <model> [n]   # one arm, n seeds
    python -m wm.run all [n]                # every arm x every model, n seeds
    python -m wm.run report                 # table of results so far

Everything lands in wm/runs/<condition>__<model>__seed<k>/.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from . import conditions, judge, llm, render
from .abstraction import Decision, Disposition, check

RUNS = Path(__file__).resolve().parent / "runs"
PROSE_CACHE = RUNS / "_prose_twin.txt"


def parse_decisions(text: str) -> list[Decision]:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        raise ValueError("no JSON array in model output")
    out = []
    for row in json.loads(m.group(0)):
        try:
            out.append(Decision(
                issue_id=row["issue_id"],
                disposition=Disposition(row["disposition"].upper()),
                counter=row.get("counter", ""),
                rationale=row.get("rationale", ""),
            ))
        except (KeyError, ValueError):
            continue          # malformed rows are dropped, and that shows up as a miss
    return out


def one_trial(condition: str, model_name: str, call, seed: int, prose: str | None) -> dict:
    d = RUNS / f"{condition}__{model_name}__seed{seed}"
    d.mkdir(parents=True, exist_ok=True)

    prompt = conditions.build(condition, prose=prose)
    (d / "prompt.txt").write_text(prompt)

    raw = call(prompt)
    (d / "raw_response.txt").write_text(raw)

    try:
        decisions = parse_decisions(raw)
    except ValueError as e:
        (d / "error.txt").write_text(str(e))
        return {"condition": condition, "model": model_name, "seed": seed, "failed": True}

    (d / "decisions.json").write_text(json.dumps(
        [{"issue_id": x.issue_id, "disposition": x.disposition.value,
          "counter": x.counter, "rationale": x.rationale} for x in decisions], indent=2))

    violations = check(decisions)
    (d / "violations.json").write_text(json.dumps(
        [{"issue_id": v.issue_id, "message": v.message} for v in violations], indent=2))

    deliverables = {
        "counter-turn-redline-dsa.docx": render.redline(decisions),
        "cover-note-to-calyx.docx": render.cover_note(decisions),
    }
    for k, v in deliverables.items():
        (d / k.replace(".docx", ".txt")).write_text(v)

    meta = {"condition": condition, "model": model_name, "seed": seed,
            "n_decisions": len(decisions), "n_violations": len(violations),
            "prompt_chars": len(prompt), "failed": False}
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def grade(run_dir: Path, judge_model: str) -> dict:
    deliverables = {
        "counter-turn-redline-dsa.docx": (run_dir / "counter-turn-redline-dsa.txt").read_text(),
        "cover-note-to-calyx.docx": (run_dir / "cover-note-to-calyx.txt").read_text(),
    }
    j = llm.claude(judge_model) if judge_model.startswith("claude") else llm.open_weights(judge_model)
    s = judge.score(deliverables, j)
    (run_dir / f"scores__{judge_model}.json").write_text(json.dumps(s, indent=2))
    return s


# --------------------------------------------------------------------------

def cmd_inputs() -> None:
    prose = PROSE_CACHE.read_text() if PROSE_CACHE.exists() else None
    print(f"{'condition':<12}{'chars':>10}{'~tokens':>10}   note")
    for c in conditions.CONDITIONS:
        if c == "A2" and prose is None:
            print(f"{c:<12}{'-':>10}{'-':>10}   run `prose` first")
            continue
        p = conditions.build(c, prose=prose)
        print(f"{c:<12}{len(p):>10,}{len(p)//4:>10,}")


def cmd_prose() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    text = conditions.prose_twin(PROSE_CACHE, llm.claude(llm.FRONTIER))
    wm = conditions.worldmodel()
    print(f"prose twin cached: {len(text):,} chars   (world model: {len(wm):,} chars)")
    ratio = len(text) / len(wm)
    print(f"length ratio {ratio:.2f} -- regenerate if this is outside roughly 0.8-1.25")


def cmd_trial(condition: str, model_name: str, seeds: int = 1) -> None:
    prose = PROSE_CACHE.read_text() if PROSE_CACHE.exists() else None
    call = llm.claude(model_name) if model_name.startswith("claude") else llm.open_weights(model_name)
    for s in range(seeds):
        m = one_trial(condition, model_name, call, s, prose)
        print(m)


def cmd_all(seeds: int = 3) -> None:
    for model_name in (llm.FRONTIER, llm.SMALL_CLAUDE):
        for c in conditions.CONDITIONS:
            cmd_trial(c, model_name, seeds)


def cmd_report() -> None:
    rows = []
    for d in sorted(RUNS.glob("*__*__seed*")):
        meta_f = d / "meta.json"
        if not meta_f.exists():
            continue
        meta = json.loads(meta_f.read_text())
        scores = [json.loads(f.read_text()) for f in d.glob("scores__*.json")]
        rate = sum(s["criterion_pass_rate"] for s in scores) / len(scores) if scores else None
        rows.append((meta["condition"], meta["model"], meta["seed"],
                     meta["n_violations"], meta["prompt_chars"] // 4, rate))
    if not rows:
        print("no runs yet")
        return
    print(f"{'cond':<6}{'model':<22}{'seed':>5}{'authority viol.':>17}{'~in tok':>10}{'pass rate':>11}")
    for c, m, s, v, t, r in rows:
        print(f"{c:<6}{m:<22}{s:>5}{v:>17}{t:>10,}{(f'{r:.3f}' if r is not None else '-'):>11}")
    print()
    print("Primary comparison is A2 vs A4 on the SAME model. A0 vs A4 is confounded.")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        print(__doc__)
    elif a[0] == "inputs":
        cmd_inputs()
    elif a[0] == "prose":
        cmd_prose()
    elif a[0] == "trial":
        cmd_trial(a[1], a[2], int(a[3]) if len(a) > 3 else 1)
    elif a[0] == "all":
        cmd_all(int(a[1]) if len(a) > 1 else 3)
    elif a[0] == "report":
        cmd_report()
    else:
        print(__doc__)
