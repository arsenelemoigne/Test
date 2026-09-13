"""
The experiment runner.

    python -m wm.run selftest               # exercise the whole pipeline offline
    python -m wm.run gravity                # per-issue gravity from the authority memo
    python -m wm.run blind                  # lexical change detection, no API calls
    python -m wm.run encode                 # tied encoder on both docs -> latent diff
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
    j = llm.model(judge_model)
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
    text = conditions.prose_twin(PROSE_CACHE, llm.model(llm.FRONTIER))
    wm = conditions.worldmodel()
    print(f"prose twin cached: {len(text):,} chars   (world model: {len(wm):,} chars)")
    ratio = len(text) / len(wm)
    print(f"length ratio {ratio:.2f} -- regenerate if this is outside roughly 0.8-1.25")


def cmd_trial(condition: str, model_name: str, seeds: int = 1) -> None:
    prose = PROSE_CACHE.read_text() if PROSE_CACHE.exists() else None
    call = llm.model(model_name)
    for s in range(seeds):
        m = one_trial(condition, model_name, call, s, prose)
        print(m)


def cmd_all(seeds: int = 3) -> None:
    for model_name in llm.ARMS:
        for c in conditions.CONDITIONS:
            cmd_trial(c, model_name, seeds)


def stub(prompt: str, max_tokens: int = 16000) -> str:
    """
    Offline model stand-in. Answers the JSON contract correctly so the plumbing
    (build -> parse -> authority check -> render) can be exercised without an API
    key. It is NOT a model: its decisions are fixed, so it proves the pipeline
    runs, not that anything is any good.
    """
    from .abstraction import ISSUES
    fixed = {
        "I01": ("MODIFY", "Fixed aggregate cap of $3,500,000. No fee-multiple formulation."),
        "I02": ("MODIFY", "Retention of 18 months from receipt of each delivery."),
        "I03": ("REJECT", "Mandatory deletion or return at expiry; anonymisation-in-place not accepted."),
        "I04": ("REJECT", "Officer-signed certification of destruction within 10 business days retained."),
        "I05": ("REJECT", "Absolute prohibition retained, no exceptions, technique-agnostic."),
        "I06": ("MODIFY", "Notification within 48 hours of Discovery; Discovery means first reasonable suspicion."),
        "I07": ("REJECT", "Vendor Security Assessment required for every subprocessor before access."),
        "I08": ("REJECT", "Unrestricted right to share audit findings with regulators retained."),
        "I09": ("REJECT", "On-site audit capability preserved; frequency once per calendar year."),
        "I10": ("REJECT", "Residuals clause deleted in its entirety."),
        "I11": ("REJECT", "Three-part prerequisite retained: prior written approval, completed TIA, transfer mechanism."),
        "I12": ("MODIFY", "Use limited to aggregated Benchmarking Reports; no ML training on the Shared Data Set."),
        "I13": ("REJECT", "Delaware governing law maintained."),
        "I14": ("REJECT", "Delaware Court of Chancery venue maintained."),
        "I15": ("REJECT", "2-year initial term, 1-year renewals and 90 days' non-renewal notice retained."),
    }
    rows = []
    for i in ISSUES:
        disp, counter = fixed.get(i.id, ("REJECT", "Reverted to the Carden initial draft."))
        rows.append({"issue_id": i.id, "disposition": disp, "counter": counter,
                     "rationale": f"Carden's position on {i.name.lower()}, consistent with the initial draft."})
    return json.dumps(rows, indent=2)


def cmd_selftest() -> None:
    """Exercise every stage offline. No network, no key, no spend."""
    from .abstraction import ISSUES
    prose = PROSE_CACHE.read_text() if PROSE_CACHE.exists() else "(prose twin not built)"
    ok = True
    for c in conditions.CONDITIONS:
        prompt = conditions.build(c, prose=prose)
        raw = stub(prompt)
        decisions = parse_decisions(raw)
        viols = check(decisions)
        r = render.redline(decisions)
        n = render.cover_note(decisions)
        good = len(decisions) == len(ISSUES) and not viols and len(r) > 500 and len(n) > 500
        ok &= good
        print(f"  {c:<4} prompt {len(prompt):>7,} chars | {len(decisions):>2} decisions | "
              f"{len(viols)} violations | redline {len(r):,} | note {len(n):,}  "
              f"{'OK' if good else 'FAIL'}")
        if viols:
            for v in viols:
                print(f"        ! {v.issue_id} {v.message}")
    print()
    print("pipeline:", "OK -- build/parse/check/render all work end to end" if ok else "FAILED")
    print("not exercised offline: the model call and the judge call.")


def cmd_encode() -> None:
    """Run the tied encoder on both documents and cache the latent diff."""
    from . import encoder
    T = Path(__file__).resolve().parent / "task"
    draft = (T / "carden-initial-draft-dsa.txt").read_text()
    markup = (T / "luminos-first-markup-dsa.txt").read_text()
    enc = llm.model(llm.FRONTIER)
    RUNS.mkdir(parents=True, exist_ok=True)
    s_x = encoder.encode(draft, enc)
    s_y = encoder.encode(markup, enc)
    (RUNS / "_latent_draft.json").write_text(json.dumps(s_x, indent=2))
    (RUNS / "_latent_markup.json").write_text(json.dumps(s_y, indent=2))
    deltas = encoder.latent_diff(s_x, s_y)
    blind = encoder.blind_spots(draft, markup)
    text = encoder.render_latent(deltas, blind)
    (RUNS / "_latent_diff.txt").write_text(text)
    print(text)
    print()
    print(llm.spend_report())


def cmd_gravity() -> None:
    """Per-issue gravity, read from the client's authority memo. No API calls."""
    from . import gravity
    from .abstraction import ISSUES
    print(gravity.report({i.id: i.name for i in ISSUES}))


def cmd_blind() -> None:
    """Lexical channel only -- no API calls."""
    from . import encoder
    T = Path(__file__).resolve().parent / "task"
    draft = (T / "carden-initial-draft-dsa.txt").read_text()
    markup = (T / "luminos-first-markup-dsa.txt").read_text()
    moved = encoder.section_diff(draft, markup)
    blind = encoder.blind_spots(draft, markup)
    print(f"{len(moved)} sections moved; {len(blind)} fall outside the slot schema\n")
    for b in blind:
        state = ("deleted" if not b.present_in_markup else
                 "inserted" if not b.present_in_draft else
                 f"rewritten (sim {b.similarity:.3f})")
        print(f"  Section {b.section:<8} {state}")


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
    elif a[0] == "selftest":
        cmd_selftest()
    elif a[0] == "encode":
        cmd_encode()
    elif a[0] == "gravity":
        cmd_gravity()
    elif a[0] == "blind":
        cmd_blind()
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
