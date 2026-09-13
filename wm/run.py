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
import re
import sys
import time
from pathlib import Path

from . import conditions, judge, llm, render
from .abstraction import Decision, Disposition, check

RUNS = Path(__file__).resolve().parent / "runs"
PROSE_CACHE = RUNS / "_prose_twin.txt"


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


def one_trial(condition: str, model_name: str, call, seed: int, prose: str | None) -> dict:
    d = RUNS / f"{condition}__{slug(model_name)}__seed{seed}"
    d.mkdir(parents=True, exist_ok=True)

    prompt = conditions.build(condition, prose=prose)
    (d / "prompt.txt").write_text(prompt)

    print(f"  {condition}/{model_name}/seed{seed}: sending {len(prompt):,} chars ...",
          end="", flush=True)
    t0 = time.time()
    raw = call(prompt)
    print(f" {time.time()-t0:.0f}s, {len(raw):,} chars back", flush=True)
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
    # slug(): the judge id contains "/", which would write into a subdirectory
    # that does not exist - and would do so AFTER all 29 judge calls were paid for.
    (run_dir / f"scores__{slug(judge_model)}.json").write_text(json.dumps(s, indent=2))
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
    if PROSE_CACHE.exists():
        print(f"cached already: {PROSE_CACHE}")
    else:
        print(f"generating the prose twin with {llm.FRONTIER} - "
              f"this takes 30-90s, leave it running ...", flush=True)
    text = conditions.prose_twin(PROSE_CACHE, llm.model(llm.FRONTIER))
    wm = conditions.worldmodel()
    print(f"prose twin cached: {len(text):,} chars   (world model: {len(wm):,} chars)")
    ratio = len(text) / len(wm)
    print(f"length ratio {ratio:.2f} -- regenerate if this is outside roughly 0.8-1.25")


def cmd_trial(condition: str, model_name: str, seeds: int = 1) -> None:
    if not model_name or not model_name.strip():
        model_name = llm.FRONTIER
        print(f"(no model given - defaulting to {model_name})")
    prose = PROSE_CACHE.read_text() if PROSE_CACHE.exists() else None
    call = llm.model(model_name)
    for s in range(seeds):
        m = one_trial(condition, model_name, call, s, prose)
        print(m)


def cmd_all(seeds: int = 3, workers: int = 5) -> None:
    """All arms. Runs are independent, so they go concurrently."""
    from concurrent.futures import ThreadPoolExecutor
    prose = PROSE_CACHE.read_text() if PROSE_CACHE.exists() else None
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
    print("A2 vs A4   : does FORM help, holding information constant? (primary)")
    print("A4 vs A4G  : does telling the model which issues are walk-aways help?")
    print("A0 vs A4   : confounded by preprocessing and context length - not a result.")


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
    elif a[0] == "all":
        cmd_all(int(a[1]) if len(a) > 1 else 3)
    elif a[0] == "gradeall":
        cmd_gradeall(int(a[1]) if len(a) > 1 else 5)
    elif a[0] == "report":
        cmd_report()
    else:
        print(__doc__)
