"""
The experiment runner.

    python -m wm.run preflight              # check key + model slugs (3 tiny calls)
    python -m wm.run selftest               # exercise the whole pipeline offline
    python -m wm.run bridge                 # value vs NO contract, Shapley per clause
    python -m wm.run cardinal               # money view + Cox diagnostic + packages
    python -m wm.run validate [reps]        # groundedness, stability, model agreement
    python -m wm.run autoschema             # FULL contract, no playbook: schema+weights+value
    python -m wm.run gravity                # per-issue gravity from the authority memo
    python -m wm.run blind                  # lexical change detection, no API calls
    python -m wm.run encode                 # tied encoder on both docs -> latent diff
    python -m wm.run inputs                 # print each condition's input, no API calls
    python -m wm.run prose                  # build + cache the prose twin (1 frontier call)
    python -m wm.run trial A4 <model> [n]   # one arm, n seeds
    python -m wm.run all [n]                # every arm x every model, n seeds
    python -m wm.run gradeall               # grade every ungraded run (parallel)
    python -m wm.run report                 # table of results so far

Everything lands in wm/runs/<condition>__<model>__seed<k>/.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

from . import conditions, judge, llm, render
from . import taskctx
from .abstraction import Decision, Disposition
from .taskctx import check

# Resolved at import: WM_TASK_DIR is set before the process starts.
RUNS = taskctx.runs_dir()
PROSE_CACHE = RUNS / "_prose_twin.txt"
PROSE_STAMP = RUNS / "_prose_twin.source.sha"


def slug(model_name: str) -> str:
    """Model ids contain '/', which would nest the run directory and break the
    report glob. Flatten it."""
    return model_name.replace("/", "--")


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


def load_prose() -> str | None:
    """The A2 control, but only if it was built from THIS task's world model.

    The cache used to be one global file. Running a second task reused the first
    task's prose twin, so the control arm was answering about a different
    contract entirely - and scored accordingly, which looks exactly like "prose
    is worse than structure".
    """
    if not PROSE_CACHE.exists():
        return None
    want = _wm_hash()
    have = PROSE_STAMP.read_text().strip() if PROSE_STAMP.exists() else ""
    if have != want:
        print(f"REFUSING the cached prose twin: it was built from a different "
              f"world model\n  ({PROSE_CACHE})\n"
              f"  expected source {want[:12]}, cache carries "
              f"{have[:12] or 'no stamp'}.\n"
              f"  Delete it and run `python -m wm.run prose` again.", flush=True)
        return None
    return PROSE_CACHE.read_text()


def _wm_hash() -> str:
    import hashlib
    return hashlib.sha256(conditions.worldmodel().encode()).hexdigest()


LOOP_CONDITIONS = {"A5", "A5N", "A0L", "A0LN", "B0L", "B0LN",
                   "B1L", "B1LN"}


def one_trial(condition: str, model_name: str, call, seed: int, prose: str | None) -> dict:
    d = RUNS / f"{condition}__{slug(model_name)}__seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    # Scores belong to the generation they graded. Leaving them behind makes
    # gradeall skip the directory as already-graded, and the report then shows
    # an old pass rate beside a new violation count for the same run.
    for stale in d.glob("scores__*.json"):
        stale.unlink()

    prompt = conditions.build(condition, prose=prose)
    (d / "prompt.txt").write_text(prompt)

    print(f"  {condition}/{model_name}/seed{seed}: sending {len(prompt):,} chars ...",
          end="", flush=True)
    t0 = time.time()

    n_calls, loop_trace = 1, None
    # Any arm whose name ends in N is the blind control for the arm without it.
    if condition in LOOP_CONDITIONS:
        from . import loop
        rounds = int(os.environ.get("WM_ROUNDS", "3"))
        res = loop.run(prompt, call, parse_decisions, taskctx.issues(), check,
                       rounds=rounds, blind=condition.endswith("N"))
        decisions, raw = res["decisions"], res["raw"]
        n_calls, loop_trace = res["calls"], res["trace"]
        print(f" {time.time()-t0:.0f}s, {n_calls} calls, "
              f"{len(raw):,} chars back", flush=True)
        (d / "loop_trace.json").write_text(json.dumps(loop_trace, indent=2))
        (d / "raw_response.txt").write_text(raw)
        if not decisions:
            (d / "error.txt").write_text("loop produced no parseable decisions")
            return {"condition": condition, "model": model_name, "seed": seed,
                    "failed": True}
    else:
        raw = call(prompt)
        print(f" {time.time()-t0:.0f}s, {len(raw):,} chars back", flush=True)
        (d / "raw_response.txt").write_text(raw)
        try:
            decisions = parse_decisions(raw)
        except ValueError as e:
            (d / "error.txt").write_text(str(e))
            return {"condition": condition, "model": model_name, "seed": seed,
                    "failed": True}

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

    meta = {"generated_at": time.time(), "n_calls": n_calls,
            "condition": condition, "model": model_name, "seed": seed,
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
    # slug(): the judge id contains "/", which would write into a subdirectory
    # that does not exist - and would do so AFTER all 29 judge calls were paid for.
    (run_dir / f"scores__{slug(judge_model)}.json").write_text(json.dumps(s, indent=2))
    return s


# --------------------------------------------------------------------------

def cmd_inputs() -> None:
    prose = load_prose()
    print(f"{'condition':<12}{'chars':>10}{'~tokens':>10}   note")
    for c in conditions.CONDITIONS:
        if c == "A2" and prose is None:
            print(f"{c:<12}{'-':>10}{'-':>10}   run `prose` first")
            continue
        p = conditions.build(c, prose=prose)
        print(f"{c:<12}{len(p):>10,}{len(p)//4:>10,}")


def cmd_prose() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    if PROSE_CACHE.exists() and load_prose() is not None:
        print(f"cached already: {PROSE_CACHE}")
    else:
        print(f"generating the prose twin with {llm.FRONTIER} - "
              f"this takes 30-90s, leave it running ...", flush=True)
    text = conditions.prose_twin(PROSE_CACHE, llm.model(llm.FRONTIER))
    PROSE_STAMP.write_text(_wm_hash())
    wm = conditions.worldmodel()
    print(f"prose twin cached: {len(text):,} chars   (world model: {len(wm):,} chars)")
    ratio = len(text) / len(wm)
    print(f"length ratio {ratio:.2f} -- regenerate if this is outside roughly 0.8-1.25")


def cmd_trial(condition: str, model_name: str, seeds: int = 1) -> None:
    if not model_name or not model_name.strip():
        model_name = llm.FRONTIER
        print(f"(no model given - defaulting to {model_name})")
    prose = load_prose()
    call = llm.model(model_name)
    for s in range(seeds):
        m = one_trial(condition, model_name, call, s, prose)
        print(m)


# Approximate OpenRouter list prices, USD per MILLION tokens (in, out).
# These move. Check https://openrouter.ai/models before trusting a total.
PRICES = {
    "anthropic/claude-opus-4.1":      (15.00, 75.00),
    "anthropic/claude-opus-4":        (15.00, 75.00),
    "anthropic/claude-sonnet-4.5":     (3.00, 15.00),
    "anthropic/claude-3.5-sonnet":     (3.00, 15.00),
    "qwen/qwen-2.5-72b-instruct":      (0.12,  0.39),
    "qwen/qwen3-32b":                  (0.10,  0.30),
    "google/gemini-2.5-flash":         (0.30,  2.50),
    "google/gemini-2.5-pro":           (1.25, 10.00),
    "z-ai/glm-4.6":                    (0.40,  1.75),
    "z-ai/glm-4.5-air":                (0.15,  0.85),
    "deepseek/deepseek-chat":          (0.25,  1.00),
    "moonshotai/kimi-k2":              (0.50,  2.00),
}
TOK = 4.0          # chars per token, near enough for English prose


def _price(name):
    return PRICES.get(name, (5.00, 20.00))      # unknown model: assume mid-range


def cmd_cost(seeds: int = 3) -> None:
    """What will `all` and `gradeall` cost, before you spend it?

    Builds every prompt for real and counts it. Output tokens are estimated
    from the deliverable sizes the self-test measures.
    """
    prose = load_prose()
    if prose is None:
        print("no prose twin cached - A2 estimated at 11,000 chars\n")
        prose = "x" * 11_000

    OUT_TOK = 2_700          # redline + cover note, measured
    print(f"conditions: {', '.join(conditions.CONDITIONS)}   seeds: {seeds}")
    print(f"arms      : {', '.join(llm.ARMS)}")
    print(f"judge     : {llm.JUDGE}\n")
    print(f"{'cond':<6}{'model':<32}{'runs':>5}{'in tok':>11}{'out tok':>10}{'USD':>9}")
    print("-" * 73)

    total = 0.0
    gen_runs = 0
    for m in llm.ARMS:
        pin, pout = _price(m)
        for c in conditions.CONDITIONS:
            try:
                prompt = conditions.build(c, prose=prose)
            except Exception as e:                      # noqa: BLE001
                print(f"{c:<6}{m:<32}{'-':>5}  cannot build: {str(e)[:24]}")
                continue
            # A5 stops when clean (2 calls is typical); A5N has no stopping
            # signal and always runs the full round count. Follow-up calls carry
            # the prompt again plus the previous answer and the evaluation.
            rounds = int(os.environ.get("WM_ROUNDS", "3"))
            calls = ((rounds + 1) if c.endswith("N")
                     else 2 if c in LOOP_CONDITIONS else 1)
            per_in = len(prompt) / TOK
            tin = int(per_in + (calls - 1) * (per_in + OUT_TOK * 1.3)) * seeds
            tout = OUT_TOK * calls * seeds
            usd = tin / 1e6 * pin + tout / 1e6 * pout
            total += usd
            gen_runs += seeds
            tag = f"  x{calls} calls" if calls > 1 else ""
            print(f"{c:<6}{m[:31]:<32}{seeds:>5}{tin:>11,}{tout:>10,}{usd:>9.2f}{tag}")

    # judging: one call per (deliverable, criterion), each carrying the deliverable
    n_crit = len(judge.criteria())
    jin, jout = _price(llm.JUDGE)
    jt_in = gen_runs * n_crit * int(OUT_TOK * 1.2)
    jt_out = gen_runs * n_crit * 120
    judge_usd = jt_in / 1e6 * jin + jt_out / 1e6 * jout
    total += judge_usd
    print("-" * 73)
    print(f"{'judge':<6}{llm.JUDGE[:31]:<32}{gen_runs * n_crit:>5}"
          f"{jt_in:>11,}{jt_out:>10,}{judge_usd:>9.2f}")
    print(f"\n{'ESTIMATED TOTAL':<43}{'':>21}{total:>9.2f} USD")
    print("\nRough. Prices move and token counts are char/4. Treat it as the order")
    print("of magnitude, not the bill. Narrow the run with WM_CONDITIONS=A2,A4,A4G")
    print("and swap WM_FRONTIER for a cheaper model to cut this sharply.")


def cmd_all(seeds: int = 3, workers: int = 5) -> None:
    """All arms. Runs are independent, so they go concurrently."""
    from concurrent.futures import ThreadPoolExecutor
    prose = load_prose()
    if prose is None and "A2" in conditions.CONDITIONS:
        print("ABORT: no prose twin cached, so A2 - the PRIMARY comparison - cannot run.")
        print("       Run `python -m wm.run prose` first and let it finish (60-150s).")
        return
    jobs = [(c, m, s) for m in llm.ARMS for c in conditions.CONDITIONS for s in range(seeds)]
    print(f"{len(jobs)} runs, {workers} at a time")

    def go(job):
        c, m, s = job
        try:
            return one_trial(c, m, llm.model(m), s, prose)
        except Exception as e:                       # noqa: BLE001
            print(f"  {c}/{m}/seed{s} FAILED: {str(e)[:140]}", flush=True)
            return {"condition": c, "model": m, "seed": s, "failed": True}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        done = list(pool.map(go, jobs))
    bad = [d for d in done if d.get("failed")]
    print(f"\n{len(done) - len(bad)}/{len(done)} runs succeeded")


def cmd_gradeall(workers: int = 5) -> None:
    """Grade every run that has not been graded yet."""
    from concurrent.futures import ThreadPoolExecutor
    dirs = [d for d in sorted(RUNS.glob("*__*__seed*"))
            if (d / "meta.json").exists() and not list(d.glob("scores__*.json"))]
    if not dirs:
        print("nothing to grade")
        return

    # Probe every precondition BEFORE spending on 29 judge calls per run.
    probe = dirs[0]
    for f in ("counter-turn-redline-dsa.txt", "cover-note-to-calyx.txt"):
        if not (probe / f).exists():
            print(f"ABORT: {probe.name} has no {f} - the runs are incomplete.")
            return
    try:
        t = probe / f"scores__{slug(llm.JUDGE)}.json"
        t.write_text("{}")
        t.unlink()
    except OSError as e:
        print(f"ABORT: cannot write the scores file ({e}). Fix this before grading.")
        return
    print(f"grading {len(dirs)} runs with {llm.JUDGE}, {workers} at a time "
          f"({len(dirs) * 29} judge calls)")

    def go(d):
        try:
            s = grade(d, llm.JUDGE)
            print(f"  {d.name:<52} {s['n_passed']}/{s['n_criteria']}"
                  + (f"  ({s['n_unparseable']} unparseable)" if s.get("n_unparseable") else ""),
                  flush=True)
        except Exception as e:                       # noqa: BLE001
            print(f"  {d.name:<52} FAILED: {str(e)[:110]}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(go, dirs))


def stub(prompt: str, max_tokens: int = 16000) -> str:
    """
    Offline model stand-in. Answers the JSON contract correctly so the plumbing
    (build -> parse -> authority check -> render) can be exercised without an API
    key. It is NOT a model: its decisions are fixed, so it proves the pipeline
    runs, not that anything is any good.
    """
    ISSUES = taskctx.issues()
    gen = taskctx.gen_issues()
    if gen is not None:
        # A ported task: answer each issue with a position built from its own
        # limits, so the self-test checks that the limits are satisfiable
        # rather than that a DSA answer happens to fit another contract.
        from .issuegen import compliant_counter
        rows = [{"issue_id": g.id, "disposition": "MODIFY",
                 "counter": compliant_counter(g),
                 "rationale": f"Position on {g.name.lower()} per the client's mandate."}
                for g in gen]
        return json.dumps(rows, indent=2)

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
    # The controls below need no task at all. Run them even when the selected
    # task has no issue list yet - that is precisely when you want to know the
    # prompts format, since `issues` is the call about to use them.
    try:
        ISSUES = taskctx.issues()
    except RuntimeError as e:
        ISSUES = None
        print(f"  no issue list for this task yet, so the arm-by-arm build is")
        print(f"  skipped. Run `python -m wm.run issues` first for that part.")
        print(f"  ({str(e).splitlines()[0]})")
        print()

    prose = load_prose() or "(prose twin not built)"
    ok = True
    for c in (conditions.CONDITIONS if ISSUES else []):
        try:
            prompt = conditions.build(c, prose=prose)
        except RuntimeError as e:
            # an arm whose input has not been built yet is not a failure
            print(f"  {c:<4} skipped -- {str(e).splitlines()[0]}")
            continue
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
    # NEGATIVE CONTROL. Every run currently reports zero violations. That is
    # either full compliance or a checker that cannot detect anything, and the
    # two look identical from the outside. So feed it decisions that plainly
    # breach the memo and require that it objects.
    from .abstraction import ISSUES as _DSA_ISSUES, check as _dsa_check
    breaches = {
        "I01": "Aggregate cap set at three times the fees paid in the prior twelve months.",
        "I02": "Retention of thirty-six (36) months from receipt.",
        "I06": "Notification within seventy-two (72) hours of confirmation of a breach.",
        "I13": "Governing law changed to Maryland.",
    }
    bad = [Decision(issue_id=i.id, disposition=Disposition.REJECT,
                    counter=breaches.get(i.id, "Reverted to the Carden initial draft."),
                    rationale="") for i in _DSA_ISSUES]
    caught = {v.issue_id for v in _dsa_check(bad)}
    missed = set(breaches) - caught
    print()
    print(f"  negative control: {len(breaches) - len(missed)}/{len(breaches)} "
          f"planted breaches caught"
          + (f"  MISSED {', '.join(sorted(missed))}" if missed else ""))
    ok &= not missed

    # Every prompt template must survive .format(). A literal brace in a
    # template - "{+ins+}" describing a tracked change, a JSON example written
    # with single braces - raises KeyError only when the prompt is first sent,
    # which is after the model call has been set up and paid for.
    bad_templates = []
    import importlib, string
    for mod in ("autoschema", "bridge", "cardinal", "conditions", "encoder",
                "issuegen", "judge", "valuation"):
        m = importlib.import_module(f".{mod}", package="wm")
        for name in dir(m):
            if not name.endswith("PROMPT"):
                continue
            tpl = getattr(m, name)
            if not isinstance(tpl, str):
                continue
            try:
                fields = {f for _, f, _, _ in string.Formatter().parse(tpl) if f}
                tpl.format(**{f: "x" for f in fields})
            except Exception as e:                     # noqa: BLE001
                bad_templates.append(f"{mod}.{name}: {type(e).__name__} {e}")
    print(f"  prompt templates: "
          + (f"{len(bad_templates)} BROKEN" if bad_templates else "all format cleanly"))
    for b in bad_templates:
        print(f"      {b}")
    ok &= not bad_templates

    g = taskctx.gen_issues()
    if g is not None:
        from .issuegen import unsatisfiable
        bad = unsatisfiable(g)
        print(f"  limit consistency: "
              + (f"{len(bad)} issues have contradictory limits" if bad
                 else f"all {sum(len(i.limits) for i in g)} limits are satisfiable"))
        for iid, name, why in bad:
            print(f"      {iid} {name}: {why}")
        ok &= not bad

    # The closed loop must converge when the feedback carries information and
    # must NOT when it does not. If the blind arm improves too, the evaluator is
    # not what is doing the work and the A5/A5N comparison means nothing.
    from . import loop as _loop
    from .issuegen import GenIssue as _GI, check_generic as _cg
    _iss = [_GI(id="L1", name="cap", question="?", section="11.1", limits=[
                {"kind": "max_quantity", "unit": "month", "value": 18, "message": "cap"},
                {"kind": "require_quantity", "unit": "month",
                 "message": "no month figure stated"}])]
    _bad = '[{"issue_id":"L1","disposition":"MODIFY","counter":"Two times total fees.","rationale":"r"}]'
    _good = '[{"issue_id":"L1","disposition":"MODIFY","counter":"Fees paid in the eighteen (18) month trailing period.","rationale":"r"}]'

    def _mk():
        st = {"t": _bad}
        def c(prompt, max_tokens=16000):
            if "\n  L1 " in prompt:        # only repairs what the evaluator names
                st["t"] = _good
            return st["t"]
        return c

    def _p(t):
        return [Decision(issue_id=r["issue_id"], disposition=Disposition(r["disposition"]),
                         counter=r["counter"], rationale=r["rationale"])
                for r in json.loads(re.search(r"\[.*\]", t, re.S).group(0))]

    _ev = _loop.run("P", _mk(), _p, _iss, lambda d: _cg(d, _iss), rounds=3, blind=False)
    _bl = _loop.run("P", _mk(), _p, _iss, lambda d: _cg(d, _iss), rounds=3, blind=True)
    _ev_ok = _ev["trace"][-1]["clean"] and _ev["calls"] < _bl["calls"]
    _bl_ok = not _bl["trace"][-1]["clean"]
    print(f"  closed loop     : evaluated {_ev['calls']} calls -> "
          f"{'clean' if _ev['trace'][-1]['clean'] else 'STILL DIRTY'}; "
          f"blind {_bl['calls']} calls -> "
          f"{'clean (CONTROL BROKEN)' if _bl['trace'][-1]['clean'] else 'still dirty, as it must be'}")
    ok &= _ev_ok and _bl_ok

    # the generic checker too, since a ported task uses that path instead
    from .issuegen import GenIssue, check_generic
    gi = [GenIssue(id="G1", name="cap", question="?", limits=[
              {"kind": "max_money", "value": 1_000_000, "message": "cap too high"}]),
          GenIssue(id="G2", name="term", question="?", limits=[
              {"kind": "max_quantity", "unit": "month", "value": 12, "message": "too long"}])]
    gbad = [Decision(issue_id="G1", disposition=Disposition.REJECT,
                     counter="Cap of FIVE MILLION DOLLARS.", rationale="they wanted 1m"),
            Decision(issue_id="G2", disposition=Disposition.REJECT,
                     counter="Term of thirty-six (36) months.", rationale="")]
    ggood = [Decision(issue_id="G1", disposition=Disposition.REJECT,
                      counter="Cap of $750,000.", rationale="they wanted five million"),
             Decision(issue_id="G2", disposition=Disposition.REJECT,
                      counter="Term of twelve (12) months.", rationale="thirty-six refused")]
    nb, ng = len(check_generic(gbad, gi)), len(check_generic(ggood, gi))
    print(f"  generic checker : {nb}/2 breaches caught, {ng} false positives on "
          f"compliant text")
    ok &= (nb == 2 and ng == 0)

    print()
    if ISSUES is None:
        print("controls:", "OK" if ok else "FAILED",
              "-- the prompt and checker controls pass; the pipeline itself")
        print("          was not exercised because this task has no issue list yet.")
    else:
        print("pipeline:", "OK -- build/parse/check/render all work end to end"
              if ok else "FAILED")
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


def cmd_autoschema() -> None:
    """Full contract, no playbook: schema -> decomposed assessment -> couplings
    -> value under several profiles. Encodes draft and markup so the per-clause
    delta shows how much each change moved the contract."""
    from . import autoschema, valuation
    T = Path(__file__).resolve().parent / "task"
    RUNS.mkdir(parents=True, exist_ok=True)
    enc = llm.model(llm.FRONTIER)
    PARTY = "Carden Analytics (the data discloser)"

    state = {}
    for label, fname in (("draft", "carden-initial-draft-dsa.txt"),
                         ("markup", "luminos-first-markup-dsa.txt")):
        doc = (T / fname).read_text()
        t0 = time.time()
        print(f"[{label}] enumerating decisions ...", flush=True)
        schema = autoschema.build_schema(doc, enc)
        print(f"[{label}] {len(schema)} decisions ({time.time()-t0:.0f}s); assessing ...",
              flush=True)
        t0 = time.time()
        asmts = valuation.assess(schema, enc, PARTY)
        print(f"[{label}] {len(asmts)} assessed ({time.time()-t0:.0f}s); finding couplings ...",
              flush=True)
        t0 = time.time()
        cpls = valuation.find_couplings(schema, enc)
        print(f"[{label}] {len(cpls)} couplings ({time.time()-t0:.0f}s)", flush=True)
        (RUNS / f"_auto_{label}.json").write_text(valuation.dump(asmts, cpls))
        state[label] = (asmts, cpls)

    prof = valuation.DATA_DISCLOSER
    for label in ("draft", "markup"):
        a, c = state[label]
        total, contribs = valuation.score(a, c, prof)
        print()
        print("=" * 78)
        print(f"### {label.upper()}")
        print(valuation.render(total, contribs, prof))

    da, dc = state["draft"]
    ma, mc = state["markup"]
    d_total, d_contrib = valuation.score(da, dc, prof)
    m_total, m_contrib = valuation.score(ma, mc, prof)
    print()
    print("=" * 78)
    print(f"DRAFT {d_total:+.1f}  ->  MARKUP {m_total:+.1f}   "
          f"(the markup moved the contract {m_total - d_total:+.1f})")
    print()
    print(valuation.across_profiles(ma, mc))
    print()
    print("=" * 78)
    low = [a for a in ma if not a.confident]
    print(f"{len(low)}/{len(ma)} decisions scored with LOW CONFIDENCE - these are")
    print("where a firm playbook would actually change the answer:")
    for a in low[:20]:
        print(f"  {a.id}  {a.name[:56]:<58} {a.basis[:70]}")
    print()
    print(llm.spend_report())


def cmd_bridge() -> None:
    """What is the contract worth against having no contract at all?

    Elicits, per clause, the value with it and the value under the default rule
    that would apply without it; finds the pairs whose value is joint; then
    attributes the difference across clauses by Shapley value.
    """
    from . import autoschema, bridge
    T = Path(__file__).resolve().parent / "task"
    RUNS.mkdir(parents=True, exist_ok=True)
    doc = (T / "luminos-first-markup-dsa.txt").read_text()
    PARTY = "Carden Analytics (the data discloser)"
    CONTEXT = ("a B2B data sharing agreement; Carden licenses a pseudonymised "
               "consumer dataset to Luminos for benchmarking analytics. Assume a "
               "mid-size contract value and a US regulatory footprint.")
    enc = llm.model(llm.FRONTIER)

    print("enumerating decisions ...", flush=True)
    schema = autoschema.build_schema(doc, enc)
    print(f"{len(schema)} decisions; eliciting default rules and values ...", flush=True)
    effects = bridge.elicit(schema, enc, PARTY, CONTEXT)
    print(f"{len(effects)} valued; finding joint-value pairs ...", flush=True)
    inter = bridge.elicit_interactions(effects, enc)
    print(f"{len(inter)} interactions\n", flush=True)
    (RUNS / "_bridge.json").write_text(bridge.dump(effects, inter))

    rel = bridge.Relationship(effects, inter)
    attrs = bridge.shapley(rel, samples=6000)
    print(bridge.waterfall(rel, attrs))
    print()

    # the biggest adverse contributors, reverted to their default rule
    worst = [a for a in attrs if a.shapley < 0][:4]
    changes = [(a.id, rel.clauses[a.id].without_clause, f"revert {a.name[:34]}")
               for a in worst]
    print(bridge.ladder(rel, changes))
    print()
    low = [c for c in effects if not c.confident]
    print(f"{len(low)}/{len(effects)} figures marked low confidence.")
    print()
    print("DEFAULT RULES the model identified (the baseline is a legal question,")
    print("and these are its answers - check them):")
    for c in sorted(effects, key=lambda x: x.naive_delta)[:10]:
        print(f"  {c.name[:38]:<40}without it: {c.default_rule[:60]}")
    print()
    print(llm.spend_report())


def cmd_bridge_demo() -> None:
    """The same value bridge, on the hand-filled worked example. Costs nothing.

    Run this first. It shows exactly what `bridge` produces and lets you argue
    with the model before paying anything to populate it.
    """
    from . import bridge, example
    rel = example.relationship()
    attrs = bridge.shapley(rel, samples=8000)

    print("WORKED EXAMPLE - the numbers below are HAND-SET, not elicited.")
    print("The clauses and the default rules are real; the values are")
    print("illustrative. Read the shape, not the digits.")
    print()
    print(bridge.waterfall(rel, attrs))
    print()
    loo, surplus, gap = bridge.loo_gap(rel)
    print(f"NON-ADDITIVITY: leave-one-out values sum to {bridge._m(loo)}, the "
          f"surplus is {bridge._m(surplus)}.")
    print(f"The {bridge._m(abs(gap))} gap is what take-one-out double counts. "
          f"Where that gap is")
    print("small, Shapley is overkill and instinct would have been fine.")
    print()
    engine = "sampled (precedence constraints in play)" if rel.has_precedence \
        else "exact closed form (2-additive, no sampling)"
    print(f"engine: {engine}")
    print()

    # Unwind the Luminos markup, one change at a time, in the order a
    # negotiator would actually ask for them.
    changes = [
        ("C10R", 0,        "drop remote-only audit limit (10.3)"),
        ("C12", -20_000,   "convenience termination -> 90 days notice (12.4)"),
        ("C14C", -60_000,  "restore Carden's re-identification carve-out (14.3)"),
        ("C15", -10_000,   "governing law back to Delaware (15.1)"),
    ]
    print(bridge.ladder(rel, changes))
    print()
    print("How to read it: the ladder is a negotiation sequence and the waterfall")
    print("is an attribution. The ladder answers 'what do I get if I win these")
    print("asks, in this order'. The waterfall answers 'which clauses account for")
    print("the value I already have', and it is the one that sums correctly.")


def cmd_cardinal() -> None:
    """Money-denominated view: expected annual cost per clause, the Cox
    diagnostic on the ordinal model, and the non-modular packages."""
    from . import autoschema, cardinal, valuation
    T = Path(__file__).resolve().parent / "task"
    RUNS.mkdir(parents=True, exist_ok=True)
    doc = (T / "luminos-first-markup-dsa.txt").read_text()
    PARTY = "Carden Analytics (the data discloser)"
    CONTEXT = ("a B2B data sharing agreement; Carden licenses a pseudonymised "
               "consumer dataset to Luminos for benchmarking analytics. Assume a "
               "mid-size contract value and a US regulatory footprint.")
    enc = llm.model(llm.FRONTIER)

    print("enumerating decisions ...", flush=True)
    schema = autoschema.build_schema(doc, enc)
    print(f"{len(schema)} decisions\n", flush=True)

    print("ordinal assessment (for the Cox diagnostic) ...", flush=True)
    ordinal = valuation.assess(schema, enc, PARTY)
    cpls = valuation.find_couplings(schema, enc)
    print()
    print("=" * 78)
    print(cardinal.cox_diagnostic(ordinal))
    print()
    print("=" * 78)
    print(cardinal.modularity_report(ordinal, cpls))

    print("=" * 78)
    print("cardinal assessment (money) ...", flush=True)
    exps = cardinal.assess_cardinal(schema, enc, PARTY, CONTEXT)
    (RUNS / "_cardinal.json").write_text(cardinal.dump(exps))
    print()
    print(cardinal.top_exposures(exps, 5))
    print()
    low = [e for e in exps if not e.confident]
    print(f"{len(low)}/{len(exps)} figures marked low confidence.")
    print()
    print(llm.spend_report())


def cmd_validate(reps: int = 3) -> None:
    """Measure whether the encoding can be trusted: groundedness, stability,
    cross-model agreement. Roughly reps+2 frontier calls plus 1 small-model call."""
    from . import autoschema, valuation, validate
    T = Path(__file__).resolve().parent / "task"
    RUNS.mkdir(parents=True, exist_ok=True)
    doc = (T / "luminos-first-markup-dsa.txt").read_text()
    PARTY = "Carden Analytics (the data discloser)"

    enc = llm.model(llm.FRONTIER)
    print("enumerating decisions ...", flush=True)
    schema = autoschema.build_schema(doc, enc)
    print(f"{len(schema)} decisions\n", flush=True)

    print("=" * 78)
    print(validate.grounding_report(validate.check_quotes(schema, doc)))

    print()
    print("=" * 78)
    runs = []
    for i in range(reps):
        print(f"assessment run {i+1}/{reps} ...", flush=True)
        runs.append(valuation.assess(schema, enc, PARTY))
    print()
    print(validate.stability_report(validate.stability(runs), runs))

    print()
    print("=" * 78)
    print(f"second opinion from {llm.SMALL} ...", flush=True)
    other = valuation.assess(schema, llm.model(llm.SMALL), PARTY)
    print()
    print(validate.agreement(runs[0], other))
    print()
    print(llm.spend_report())


def cmd_preflight() -> None:
    llm.preflight()


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


def cmd_recheck() -> None:
    """Re-run the authority check over every saved run. No API calls.

    decisions.json is kept for each run, so the deterministic check can be
    replayed whenever the rules change - and the counter text behind any
    surviving violation is printed, because a violation you cannot read is a
    violation you cannot trust.
    """
    dirs = [d for d in sorted(RUNS.glob("*__*__seed*")) if (d / "decisions.json").exists()]
    if not dirs:
        print("no runs with decisions.json")
        return
    print(f"{'run':<48}{'was':>5}{'now':>5}")
    changed = []
    for d in dirs:
        rows = json.loads((d / "decisions.json").read_text())
        decisions = [Decision(issue_id=r["issue_id"],
                              disposition=Disposition(r["disposition"]),
                              counter=r["counter"], rationale=r["rationale"])
                     for r in rows]
        now = check(decisions)
        was = json.loads((d / "violations.json").read_text()) \
            if (d / "violations.json").exists() else []
        mark = "" if len(now) == len(was) else "   <-- changed"
        print(f"{d.name[:47]:<48}{len(was):>5}{len(now):>5}{mark}")
        if len(now) != len(was):
            changed.append(d)
        (d / "violations.json").write_text(json.dumps(
            [{"issue_id": v.issue_id, "message": v.message} for v in now], indent=2))
        meta_f = d / "meta.json"
        if meta_f.exists():
            meta = json.loads(meta_f.read_text())
            meta["n_violations"] = len(now)
            meta.setdefault("generated_at", meta_f.stat().st_mtime)
            meta_f.write_text(json.dumps(meta, indent=2))

    print()
    if changed:
        print(f"{len(changed)} runs changed. meta.json updated - re-run `report`.")
    else:
        print("no run changed: the rule edit did not move any result.")

    print()
    print("SURVIVING VIOLATIONS, with the text that triggered them:")
    print("=" * 74)
    for d in dirs:
        viols = json.loads((d / "violations.json").read_text())
        if not viols:
            continue
        rows = {r["issue_id"]: r for r in json.loads((d / "decisions.json").read_text())}
        print(f"\n{d.name}")
        for v in viols:
            r = rows.get(v["issue_id"], {})
            print(f"  {v['issue_id']}  {v['message']}")
            print(f"      counter: {r.get('counter', '(none)')[:150]}")


def cmd_pack(task_path: str) -> None:
    """Extract a Harvey LAB task folder to text this pipeline can read.

        export WM_LABS=~/harvey-labs
        python -m wm.run pack ip-licensing/license-agreement-first-turn-redline/scenario-04
    """
    import os
    from . import taskpack
    src = Path(task_path).expanduser()
    if not src.is_absolute() or not src.exists():
        labs = Path(os.environ.get("WM_LABS", "~/harvey-labs")).expanduser()
        src = labs / "tasks" / "contracts" / task_path
    if not (src / "task.json").exists():
        print(f"ABORT: no task.json in {src}")
        print("       Set WM_LABS to your harvey-labs checkout, or pass an absolute path.")
        return

    out = Path(__file__).resolve().parent / "tasks" / src.parent.name if src.name.startswith("scenario") else None
    out = (Path(__file__).resolve().parent / "tasks" /
           ("-".join(src.parts[-2:]) if src.name.startswith("scenario") else src.name))
    sizes = taskpack.pack(src, out)
    print(f"packed {len([v for v in sizes.values() if v > 0])} documents -> {out}\n")
    for k, v in sorted(sizes.items(), key=lambda kv: -kv[1]):
        print(f"  {v:>9,}  {k}" if v > 0 else f"  {'SKIPPED':>9}  {k}")

    r = taskpack.roles(list(sizes))
    (out / "_roles.json").write_text(json.dumps(r, indent=2))
    print()
    for role in ("template", "markup", "memo", "policy", "pricing", "email", "other"):
        if r.get(role):
            print(f"  {role:<10} {', '.join(r[role])}")
    missing = [x for x in ("markup", "memo") if not r.get(x)]
    if missing:
        print(f"\n  WARNING no file matched {', '.join(missing)}. This pipeline needs")
        print("  a counterparty markup and a client mandate; edit _roles.json by hand.")
    n = len(judge_criteria_count(out))
    print(f"\n  rubric: {n} criteria "
          f"(the original DSA task had 29 - more criteria means more headroom)")
    print(f"\nNext:  export WM_TASK_DIR={out}")
    print( "       python -m wm.run issues")


def judge_criteria_count(task_dir: Path) -> list:
    try:
        d = json.loads((task_dir / "task.json").read_text())
        return d.get("rubric") or d.get("criteria") or []
    except Exception:                                   # noqa: BLE001
        return []


def cmd_tighten() -> None:
    """Add the missing presence requirement beside each bounded limit. Free."""
    from . import issuegen, taskctx
    f = taskctx.issues_file()
    if not f.exists():
        print(f"ABORT: {f} does not exist. Run `issues` first.")
        return
    issues = issuegen.load(f.read_text())
    before = sum(len(i.limits) for i in issues)
    n = issuegen.tighten(issues)
    f.write_text(issuegen.dump(issues))
    print(f"added {n} presence requirements ({before} -> {before + n} limits)\n")
    print(issuegen.summary(issues))


def cmd_audit() -> None:
    """Check every generated limit against the memo it was read from. Free."""
    from . import issuegen, taskctx
    T, f = taskctx.task_dir(), taskctx.issues_file()
    if not f.exists():
        print(f"ABORT: {f} does not exist. Run `issues` first.")
        return
    rf = T / "_roles.json"
    r = json.loads(rf.read_text()) if rf.exists() else {}
    memo = "\n\n".join((T / (Path(n).stem + ".txt")).read_text()
                        for n in (r.get("memo") or [])
                        if (T / (Path(n).stem + ".txt")).exists())
    if not memo:
        print("ABORT: no memo text found for this task.")
        return
    print(issuegen.audit(issuegen.load(f.read_text()), memo))


def cmd_issues() -> None:
    """Read the client's mandate and the counterparty markup; produce the issue
    list and its machine-checkable limits. One model call, then cached."""
    from . import issuegen, taskctx, taskpack
    T = taskctx.task_dir()
    rf = T / "_roles.json"
    if not rf.exists():
        print(f"ABORT: no _roles.json in {T}. Run `pack` first.")
        return
    r = json.loads(rf.read_text())

    def read(role):
        names = r.get(role) or []
        return "\n\n".join((T / (Path(n).stem + ".txt")).read_text()
                            for n in names
                            if (T / (Path(n).stem + ".txt")).exists())

    memo, markup = read("memo"), read("markup")
    if not memo or not markup:
        print(f"ABORT: memo {len(memo):,} chars, markup {len(markup):,} chars - "
              f"both are required.")
        return
    print(f"memo {len(memo):,} chars, markup {len(markup):,} chars -> "
          f"{llm.FRONTIER}", flush=True)
    issues = issuegen.build_issues(memo, markup, llm.model(llm.FRONTIER))
    taskctx.issues_file().write_text(issuegen.dump(issues))
    print()
    print(issuegen.summary(issues))
    print()
    print(f"written to {taskctx.issues_file()}")
    print("READ IT. Everything downstream trusts this file, and it was written")
    print("by a model from the memo - check the limits against the memo yourself.")
    print()
    print(llm.spend_report())


def cmd_drift_demo() -> None:
    """The drift valuation on the worked example. No mandate, no API calls."""
    from . import drift, drift_example
    d = drift_example.drift()
    print("WORKED EXAMPLE - changes are real, NUMBERS ARE HAND-SET.\n")
    print(drift.report(d))
    print()
    print("No mandate was used. The baseline is our own template, which is the")
    print("counterfactual ASC 805 and IFRS 3 already require for acquired")
    print("contracts: PV(actual terms) - PV(reference terms). Auditors sign that")
    print("number; nobody audits 'what if there were no contract'.")


def cmd_model_elicit() -> None:
    """Construit le contrat parametrique a partir du modele et du markup."""
    from . import parametric, taskctx
    T = taskctx.task_dir()
    r = json.loads((T / "_roles.json").read_text()) if (T / "_roles.json").exists() else {}

    def read(role):
        return "\n\n".join((T / (Path(n).stem + ".txt")).read_text()
                            for n in (r.get(role) or [])
                            if (T / (Path(n).stem + ".txt")).exists())

    template, markup = read("template"), read("markup")
    if not template or not markup:
        print(f"ABORT: modele {len(template):,} car., markup {len(markup):,} car. "
              f"- les deux sont requis.")
        return
    PARTY = os.environ.get("WM_PARTY", "notre client, qui a envoye le modele")
    CONTEXT = os.environ.get("WM_CONTEXT", "un contrat commercial B2B")
    print(f"modele {len(template):,} car., markup {len(markup):,} car. -> "
          f"{llm.FRONTIER}", flush=True)
    raw_path = T / "_parametric_raw.txt"
    call = llm.model(llm.FRONTIER)
    print("passe 1/2 : les variables, leurs domaines, NOTRE vecteur ...", flush=True)
    c = parametric.build_model(template, markup, call, PARTY, CONTEXT,
                               raw_out=raw_path)
    if c.params:
        print(f"passe 2/2 : leur vecteur, depuis leur siege, sans voir le notre "
              f"({len(c.params)} variables) ...", flush=True)
        n = parametric.elicit_theirs(c, call, PARTY, CONTEXT,
                                     raw_out=T / "_parametric_theirs_raw.txt")
        print(f"  {n} redactions valorisees cote adverse", flush=True)
    if not c.params:
        print(f"\nAUCUNE VARIABLE RETENUE. La reponse brute est dans\n  {raw_path}\n"
              f"Regarde sa premiere variable : le schema attendu met des "
              f"IDENTIFIANTS d'option\ndans ours_option/theirs_option au niveau "
              f"variable, et des VECTEURS dans\nours/theirs au niveau option.")
        return
    (T / "parametric.json").write_text(parametric.dump(c))
    print(f"\n{len(c.params)} variables, {c.size():,} contrats possibles, "
          f"{len(c.couplings)} couplages\n")
    print(parametric.report(c, min_concessions=2))
    print()
    print(parametric.sanity_report(c))
    print()
    print(f"ecrit dans {T / 'parametric.json'}")
    if parametric.model_sanity(c):
        print("RELIS LE RAPPORT CI-DESSUS AVANT DE LANCER B0. Un modele qui "
              "declenche ces\ncontroles ne vaut pas mieux qu'un score unique, "
              "et B0 ne mesurera rien.")
    print(llm.spend_report())


def cmd_parametric_demo() -> None:
    """Le contrat comme objet parametrique. Aucun appel API."""
    from . import parametric, parametric_example
    print(parametric.report(parametric_example.contract(), min_concessions=2))


def cmd_report() -> None:
    import datetime as _dt
    rows = []
    for d in sorted(RUNS.glob("*__*__seed*")):
        meta_f = d / "meta.json"
        if not meta_f.exists():
            continue
        meta = json.loads(meta_f.read_text())
        scores = [json.loads(f.read_text()) for f in d.glob("scores__*.json")]
        rate = sum(s["criterion_pass_rate"] for s in scores) / len(scores) if scores else None
        # generated_at, not the file mtime: recheck rewrites meta.json and would
        # otherwise make every run look freshly generated.
        rows.append((meta["condition"], meta["model"], meta["seed"],
                     meta["n_violations"], meta["prompt_chars"] // 4, rate,
                     meta.get("generated_at") or meta_f.stat().st_mtime,
                     meta.get("n_calls", 1)))
    if not rows:
        print("no runs yet")
        return

    # Runs accumulate across sessions. A row generated days ago by different code
    # sitting next to a fresh one, with nothing to tell them apart, is how you
    # end up comparing two different experiments and calling it a result.
    newest = max(r[6] for r in rows)
    stale = [r for r in rows if newest - r[6] > 3600]

    print(f"{'cond':<6}{'model':<26}{'seed':>5}{'viol.':>7}{'calls':>6}{'~in tok':>9}"
          f"{'pass rate':>11}{'age':>9}")
    for c, m, sd, v, t, r, ts, nc in rows:
        age = newest - ts
        aged = "now" if age < 3600 else (f"{age/3600:.0f}h" if age < 86400
                                         else f"{age/86400:.0f}d")
        mark = " <-- older run" if age > 3600 else ""
        print(f"{c:<6}{m[:25]:<26}{sd:>5}{v:>7}{nc:>6}{t:>9,}"
              f"{(f'{r:.3f}' if r is not None else '-'):>11}{aged:>9}{mark}")
    print()
    if stale:
        conds = sorted({f"{r[0]}/{r[1].split('/')[0]}" for r in stale})
        print(f"WARNING {len(stale)} of {len(rows)} rows are from an EARLIER session: "
              f"{', '.join(conds)}.")
        print("They were generated by a different revision of this code. Compare")
        print("them with the fresh rows only if you have checked that the prompt")
        print("builder has not changed since. `git log --oneline -- wm/conditions.py`")
        print()

    print(f"temperature is 0.0 and the seed is NOT sent to the model - it only")
    print("names the directory. Seeds are therefore REPLICATES of one computation,")
    print("not independent samples. Identical scores across seeds show provider")
    print("determinism, not robustness. Set WM_TEMP=0.7 to get real variation.")
    print()
    print("A4 vs A5   : does an EXECUTABLE model in the loop help? (the hypothesis)")
    print("A5 vs A5N  : ... or was it just the extra passes? A5N spends the same")
    print("             calls with the feedback replaced by 'improve your answer'.")
    print("A2 vs A4   : does FORM help, holding information constant?")
    print("A4 vs A4G  : does telling the model which issues are walk-aways help?")
    print("A0 vs A4   : confounded by preprocessing and context length - not a result.")
    print()
    print("Read the calls column alongside the pass rate. A5 winning on fewer")
    print("calls than A5N is the strong result; winning on more is a weaker one")
    print("and must be reported as such.")
    print()
    print("Read the violation column as carefully as the pass rate. The rubric asks")
    print("whether the issues were addressed; the authority check asks whether the")
    print("client's stated limits were respected. A run can score 29/29 and still")
    print("have exceeded its mandate, and for a law firm that is the worse failure.")


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
    elif a[0] == "bridge":
        if len(a) > 1 and a[1] in ("--demo", "demo"):
            cmd_bridge_demo()
        else:
            cmd_bridge()
    elif a[0] == "cardinal":
        cmd_cardinal()
    elif a[0] == "validate":
        cmd_validate(int(a[1]) if len(a) > 1 else 3)
    elif a[0] == "autoschema":
        cmd_autoschema()
    elif a[0] == "preflight":
        cmd_preflight()
    elif a[0] == "gravity":
        cmd_gravity()
    elif a[0] == "blind":
        cmd_blind()
    elif a[0] == "prose":
        cmd_prose()
    elif a[0] == "trial":
        cmd_trial(a[1],
                  a[2] if len(a) > 2 else "",
                  int(a[3]) if len(a) > 3 else 1)
    elif a[0] == "cost":
        cmd_cost(int(a[1]) if len(a) > 1 else 3)
    elif a[0] == "all":
        cmd_all(int(a[1]) if len(a) > 1 else 3)
    elif a[0] == "gradeall":
        cmd_gradeall(int(a[1]) if len(a) > 1 else 5)
    elif a[0] == "pack":
        cmd_pack(a[1] if len(a) > 1 else "")
    elif a[0] == "issues":
        if len(a) > 1 and a[1] in ("--tighten", "tighten"):
            cmd_tighten()
        elif len(a) > 1 and a[1] in ("--audit", "audit"):
            cmd_audit()
        else:
            cmd_issues()
    elif a[0] == "model":
        if len(a) > 1 and a[1] in ("--elicit", "elicit"):
            cmd_model_elicit()
        else:
            cmd_parametric_demo()
    elif a[0] == "drift":
        if len(a) > 1 and a[1] in ("--demo", "demo"):
            cmd_drift_demo()
        else:
            print("only `drift --demo` is wired so far - the elicited version "
                  "needs the template/markup pair.")
    elif a[0] == "recheck":
        cmd_recheck()
    elif a[0] == "report":
        cmd_report()
    else:
        print(__doc__)
