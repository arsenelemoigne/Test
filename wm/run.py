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
from dataclasses import asdict
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


def _json_arrays(text: str):
    """Tout tableau JSON equilibre du texte, le plus long d'abord.

    L'ancienne version prenait `\\[.*\\]` en glouton : un modele qui ecrit une
    phrase avec des crochets avant sa reponse faisait avaler au motif tout
    l'intervalle entre le premier crochet du raisonnement et le dernier de la
    reponse, et le run entier echouait sur un texte pourtant exploitable.
    """
    spans, pile = [], []
    dans_chaine = esc = False
    for i, ch in enumerate(text):
        if dans_chaine:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                dans_chaine = False
            continue
        if ch == '"':
            dans_chaine = True
        elif ch == "[":
            pile.append(i)
        elif ch == "]" and pile:
            spans.append(text[pile.pop():i + 1])
    return sorted(spans, key=len, reverse=True)


def _objets_complets(text: str) -> list[dict]:
    """Tout objet JSON equilibre qui porte un issue_id, dans l'ordre du texte."""
    out, debut, prof = [], None, 0
    dans_chaine = esc = False
    for i, ch in enumerate(text):
        if dans_chaine:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                dans_chaine = False
            continue
        if ch == '"':
            dans_chaine = True
        elif ch == "{":
            if prof == 0:
                debut = i
            prof += 1
        elif ch == "}" and prof:
            prof -= 1
            if prof == 0 and debut is not None:
                try:
                    o = json.loads(text[debut:i + 1])
                except ValueError:
                    o = None
                if isinstance(o, dict) and "issue_id" in o:
                    out.append(o)
                debut = None
    return out


def parse_decisions(text: str) -> list[Decision]:
    rows, derniere = None, None
    for span in _json_arrays(text):
        try:
            candidate = json.loads(span)
        except ValueError as e:
            derniere = e
            continue
        if isinstance(candidate, list) and any(
                isinstance(r, dict) and "issue_id" in r for r in candidate):
            rows = candidate
            break
    if rows is None:
        # RATTRAPAGE D'UNE REPONSE TRONQUEE. Un tableau coupe au milieu n'est pas
        # un JSON valide, mais les objets qui le precedent le sont : sur une
        # tache a 26 points, perdre 25 decisions ecrites parce que la 26e est
        # incomplete coute un run entier pour rien. On relit objet par objet.
        rows = _objets_complets(text)
        if rows:
            print(f"        (reponse tronquee : {len(rows)} decisions completes "
                  f"recuperees sur {len(text):,} chars)")
    if rows is None or not rows:
        raise ValueError(
            "no JSON array of decisions in model output"
            + (f" (last decode error: {derniere})" if derniere else "")
            + f"; {len(text):,} chars returned, see raw_response.txt")
    out = []
    for row in rows:
        try:
            out.append(Decision(
                issue_id=row["issue_id"],
                disposition=Disposition(row["disposition"].upper()),
                counter=row.get("counter", ""),
                rationale=row.get("rationale", ""),
            ))
            out[-1].option_id = str(row.get("option_id", "") or "")
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
                   "B2L", "B2LN",
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
        vfn = None
        if condition.startswith("B"):
            # Les arms B disposent du contrat parametrique : le retour de boucle
            # peut alors porter sur la VALEUR de la contre-proposition et non
            # seulement sur sa conformite au mandat. C'est la fonction de perte
            # que le projet cherchait : le modele propose, le contrat est
            # evalue, l'ecart lui revient.
            from . import parametric
            f = taskctx.task_dir() / "parametric.json"
            if f.exists():
                pc = parametric.load(f.read_text())
                vfn = lambda ds: (parametric.value_feedback(ds, pc),
                                  parametric.value_of(ds, pc))
        res = loop.run(prompt, call, parse_decisions, taskctx.issues(), check,
                       rounds=rounds, blind=condition.endswith("N"), value_fn=vfn)
        decisions, raw = res["decisions"], res["raw"]
        n_calls, loop_trace = res["calls"], res["trace"]
        print(f" {time.time()-t0:.0f}s, {n_calls} calls, "
              f"{len(raw):,} chars back", flush=True)
        (d / "loop_trace.json").write_text(json.dumps(loop_trace, indent=2))
        (d / "raw_response.txt").write_text(raw)
        if not decisions:
            msg = "loop produced no parseable decisions"
            (d / "error.txt").write_text(msg)
            print(f"        FAILED: {msg}\n        see {d}/raw_response.txt")
            return {"condition": condition, "model": model_name, "seed": seed,
                    "failed": True}
    else:
        raw = call(prompt)
        print(f" {time.time()-t0:.0f}s, {len(raw):,} chars back", flush=True)
        (d / "raw_response.txt").write_text(raw)
        try:
            decisions = parse_decisions(raw)
        except ValueError as e:
            # Un run qui echoue en silence coute un appel et n'apprend rien.
            # La raison etait ecrite dans error.txt et jamais imprimee.
            (d / "error.txt").write_text(str(e))
            print(f"        FAILED: {e}")
            print(f"        first 300 chars back: {raw[:300]!r}")
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
            "prompt_chars": len(prompt), "failed": False,
            "temperature": llm.DEFAULT_TEMP}
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

    # LA BOUCLE PAR LA VALEUR. Le retour n'est plus "vous avez enfreint une
    # limite" mais "cette concession vous coute 500 et ne leur rapporte que
    # 210". Ce controle ne mesure pas un modele : il verifie que la fonction de
    # perte nomme bien la mauvaise concession, propose la redaction mieux
    # echangee, signale l'echange efficace laisse sur la table, et refuse de
    # chiffrer une position qui ne designe aucune redaction du domaine.
    from . import parametric as _pm
    _P = _pm.Parameter
    _O = _pm.Option
    _toy = _pm.Contract([
        _P(id="V1", name="Plafond de responsabilite", section="11", ours="cap_1x",
           theirs="cap_2x", options=[
               _O(id="cap_1x", text="12 mois de redevances glissantes"),
               _O(id="cap_18m", text="18 mois de redevances glissantes",
                  ours={"tail_risk": -150}, theirs={"tail_risk": 200}),
               _O(id="cap_2x", text="2x les redevances du terme initial",
                  ours={"tail_risk": -500}, theirs={"tail_risk": 210})]),
        _P(id="V2", name="Periode de garantie", section="9", ours="warr_90d",
           theirs="warr_18m", options=[
               _O(id="warr_90d", text="90 jours par version"),
               _O(id="warr_12m", text="12 mois par version",
                  ours={"revenue": -30}, theirs={"revenue": 90}),
               _O(id="warr_18m", text="18 mois par version",
                  ours={"revenue": -52}, theirs={"revenue": 190})]),
        _P(id="V3", name="Loi applicable", section="15", ours="del", theirs="tex",
           options=[
               _O(id="del", text="Delaware"),
               _O(id="tex", text="Texas",
                  ours={"enforceability": -300}, theirs={"enforceability": 40})]),
    ])
    _pos = [
        # on cede le plafond : -500 contre +210, la pire monnaie du contrat
        Decision(issue_id="V1", disposition=Disposition.ACCEPT,
                 counter="Cap at 2x initial-term fees.", rationale="",
                 option_id="cap_2x"),
        # on refuse la garantie : elle leur vaut 190 et ne nous coute que 52
        Decision(issue_id="V2", disposition=Disposition.REJECT,
                 counter="Warranty stays at ninety (90) days.", rationale="",
                 option_id="warr_90d"),
        # une position qui ne designe aucune redaction connue
        Decision(issue_id="V3", disposition=Disposition.MODIFY,
                 counter="New York law.", rationale="", option_id="new_york"),
    ]
    # COUVERTURE NULLE. Un run observe a rendu 26 positions sans un seul
    # option_id : l'assignation restait notre modele de bout en bout et le bloc
    # annoncait "vous recuperez 100%". Le chiffre doit etre refuse, pas nuance.
    _muet = [Decision(issue_id=x.issue_id, disposition=x.disposition,
                      counter=x.counter, rationale="") for x in _pos]
    _fb0 = _pm.value_feedback(_muet, _toy)
    _cov = [("aucun pourcentage fabrique", "recuperez" not in _fb0),
            ("le refus est explicite", "NON CALCULABLE" in _fb0),
            ("la valeur ne departage rien", _pm.value_of(_muet, _toy) is None)]
    _cok = all(v for _, v in _cov)
    print(f"  couverture nulle: {sum(v for _, v in _cov)}/{len(_cov)} "
          f"{'OK' if _cok else 'FAIL'}")
    for _n, _v in _cov:
        if not _v:
            print(f"        ! {_n}")
    ok &= _cok

    _fb = _pm.value_feedback(_pos, _toy)
    _a, _hors = _pm.assignment_from(_pos, _toy)
    _vchecks = [
        ("mauvaise concession nommee", "Plafond de responsabilite" in _fb
         and "ratio 0.4" in _fb),
        ("redaction mieux echangee proposee", "18 mois de redevances glissantes" in _fb),
        ("echange efficace signale", "Periode de garantie" in _fb and "18 mois par version" in _fb),
        ("redaction inconnue ecartee", _hors == ["V3:new_york"] and _a["V3"] == "del"),
        ("valeur recuperee calculee", "recuperez" in _fb),
    ]
    _vok = all(v for _, v in _vchecks)
    print(f"  boucle valeur   : {sum(v for _, v in _vchecks)}/{len(_vchecks)} "
          f"{'OK' if _vok else 'FAIL'}")
    for _n, _v in _vchecks:
        if not _v:
            print(f"        ! {_n}")
    ok &= _vok

    # LA VALEUR DOIT DEPARTAGER, JAMAIS ARBITRER. Deux controles opposes :
    # a conformite egale la boucle doit retenir la revision mieux valorisee ;
    # a valeur superieure mais mandat enfreint elle doit la refuser.
    from .loop import Report as _Rep, _score as _sc
    _lex = [
        ("la valeur departage a conformite egale",
         _sc(_Rep(violations=[], missing=[], value=900.0))
         < _sc(_Rep(violations=[], missing=[], value=100.0))),
        ("le mandat prime sur toute valeur",
         _sc(_Rep(violations=[("I1", "x")], missing=[], value=1e9))
         > _sc(_Rep(violations=[], missing=[], value=-1e9))),
        ("une question sans reponse prime sur la valeur",
         _sc(_Rep(violations=[], missing=["I1"], value=1e9))
         > _sc(_Rep(violations=[], missing=[], value=-1e9))),
    ]
    _lok = all(v for _, v in _lex)
    print(f"  ordre lexical   : {sum(v for _, v in _lex)}/{len(_lex)} "
          f"{'OK' if _lok else 'FAIL'}")
    for _n, _v in _lex:
        if not _v:
            print(f"        ! {_n}")
    ok &= _lok

    # ATTRIBUTION DES CHIFFRES. Une contre-proposition nomme les DEUX positions
    # dans la meme phrase. Le controle prenait le maximum de tous les nombres du
    # counter, donc imputait a notre client le chiffre qu'il venait de refuser.
    # Les quatre premiers cas doivent rester muets, les quatre suivants sortir.
    from .issuegen import GenIssue as _GI, check_generic as _cgen
    _lim = {"W": _GI(id="W", name="garantie", question="?",
                     limits=[{"kind": "max_quantity", "unit": "month", "value": 12,
                              "message": "garantie trop longue"}]),
            "C": _GI(id="C", name="plafond", question="?",
                     limits=[{"kind": "max_money", "value": 5_000_000,
                              "message": "plafond trop haut"}]),
            "N": _GI(id="N", name="preavis", question="?",
                     limits=[{"kind": "min_quantity", "unit": "day", "value": 30,
                              "message": "preavis trop court"}])}
    _attr = [
        ("W", "Veridian offers a twelve (12) month warranty; Halcyon's demand for "
              "eighteen (18) months is rejected.", False),
        ("W", "Warranty period of twelve (12) months rather than the eighteen (18) "
              "months requested.", False),
        ("C", "Cap at $3,168,000. We reject Halcyon's proposed cap of $34,000,000.", False),
        ("N", "Termination on ninety (90) days' notice, not the ten (10) days they "
              "sought.", False),
        ("W", "Warranty extended to twenty-four (24) months.", True),
        ("W", "We accept the eighteen (18) month warranty they demanded.", True),
        ("C", "We agree to Halcyon's proposed cap of $34,000,000.", True),
        ("N", "Termination on ten (10) days' notice.", True),
    ]
    _faux_pos = _faux_neg = 0
    for _iid, _txt, _doit in _attr:
        _v = _cgen([Decision(issue_id=_iid, disposition=Disposition.MODIFY,
                             counter=_txt, rationale="")], [_lim[_iid]])
        if bool(_v) and not _doit:
            _faux_pos += 1
            print(f"        ! faux positif : {_txt[:64]}")
        elif not _v and _doit:
            _faux_neg += 1
            print(f"        ! faux negatif : {_txt[:64]}")
    print(f"  attribution     : {len(_attr) - _faux_pos - _faux_neg}/{len(_attr)} "
          f"({_faux_pos} faux positifs, {_faux_neg} faux negatifs)")
    ok &= (_faux_pos == 0 and _faux_neg == 0)

    # le parseur, sur les formes que les modeles rendent vraiment
    _formes = [
        'Weighing [the cap] and [the term].\n```json\n[{"issue_id":"I1",'
        '"disposition":"REJECT","counter":"x","rationale":"y"}]\n```',
        '[{"issue_id":"I1","disposition":"ACCEPT","counter":"see [Section 3]",'
        '"rationale":"d"}]',
        '[{"issue_id":"I1","disposition":"MODIFY","counter":"a","rationale":"b",'
        '"tags":["x","y"]}]',
    ]
    _pok = 0
    for _f in _formes:
        try:
            _pok += len(parse_decisions(_f)) == 1
        except ValueError:
            pass
    print(f"  parseur JSON    : {_pok}/{len(_formes)} formes lues")
    ok &= (_pok == len(_formes))

    # LECTURE DES CHIFFRES. Le controle ne connaissait que le mot "percent" et
    # refusait un adjectif entre le nombre et son unite. Sur les runs observes,
    # "CPI-U with 3% floor and 5% cap" produisait "no percent figure stated" :
    # l'essentiel du compteur de violations mesurait cela.
    from .abstraction import quantities as _q
    _lect = [("CPI-U with 3% floor and 5% cap", "percent", {3.0, 5.0}),
             ("99.9% monthly uptime commitment", "percent", {99.9}),
             ("2% of monthly fees per 0.1% shortfall", "percent", {2.0, 0.1}),
             ("50 percent early-termination fee", "percent", {50.0}),
             ("90 consecutive days", "day", {90.0}),
             ("thirty (30) calendar days", "day", {30.0}),
             ("60-day cure period", "day", {60.0}),
             ("eighteen (18) months", "month", {18.0}),
             # et ce qu'il ne doit PAS lire : "3 of 12 months" ne vaut pas 3 mois
             ("98% uptime in 3 of 12 months", "month", {12.0})]
    _lok = sum(set(_q(t, u)) == att for t, u, att in _lect)
    print(f"  lecture chiffres: {_lok}/{len(_lect)}")
    for _t, _u, _att in _lect:
        if set(_q(_t, _u)) != _att:
            print(f"        ! {_u}: {sorted(set(_q(_t, _u)))} au lieu de "
                  f"{sorted(_att)}  <- {_t}")
    ok &= (_lok == len(_lect))

    # rattrapage d'une reponse coupee en plein JSON
    _tr = ('[\n {"issue_id":"I01","disposition":"REJECT","counter":"a","rationale":"x"},\n'
           ' {"issue_id":"I02","disposition":"MODIFY","counter":"b {c}","rationale":"y"},\n'
           ' {"issue_id":"I03","disposition":"ACCEPT","counter":"trunc')
    _rec = parse_decisions(_tr)
    print(f"  reponse tronquee: {len(_rec)}/2 decisions completes recuperees")
    ok &= (len(_rec) == 2)

    # LE SIMULATEUR DE NEGOCIATION, sur le contrat d'exemple. Trois choses :
    # algo contre algo conclut des accords localement efficaces ; une politique
    # LLM scriptee est lue correctement (offre partielle, ids inconnus ignores,
    # acceptation) ; la variante brute ne voit AUCUN chiffre.
    from . import negotiation as _ng, parametric_example as _pe
    _c = _pe.contract()
    _rs = _ng.run_campaign(_c, ["algo"], list(range(4)), T=5)
    _acc = [r for r in _rs if r["agreed"]]
    _nok = [("au moins un accord", len(_acc) >= 1),
            ("aucun accord hors mandat", not any(r["mandate_breach"] for r in _acc)),
            ("accords localement efficaces", all(r["pareto_gap"] == 0 for r in _acc))]
    _seen = []
    def _fake(prompt, max_tokens=0):
        _seen.append(prompt)
        pid = next(iter(_c.params)); oid = _c.params[pid].options[-1].id
        return (json.dumps({"offer": {pid: oid, "ZZZ": "nope"}}) if len(_seen) == 1
                else 'ok\n{"accept": true}')
    _us, _th = _ng.make_us(_c), _ng.make_them(_c, 1)
    _r = _ng.negotiate(_c, _us, _th, _ng.LLMPolicy(_c, _us, _th, _fake, True),
                       _ng.AlgoPolicy(_c, _th, _us), T=4)
    _nok += [("politique LLM lue", _r["agreed"] and _r["calls"] == 2 and _r["invalid_ids"] == 1),
             ("llm_value montre les chiffres", "VALEUR (calculee" in _seen[0])]
    _seen.clear()
    _us2 = _ng.make_us(_c)
    _ng.negotiate(_c, _us2, _th, _ng.LLMPolicy(_c, _us2, _th, lambda p, max_tokens=0: (_seen.append(p) or '{"accept": true}'), False),
                  _ng.AlgoPolicy(_c, _th, _us2), T=2)
    _nok.append(("llm_raw ne voit aucun chiffre", "VALEUR (calculee" not in _seen[0] and "[nous " not in _seen[0]))
    _nk = all(v for _, v in _nok)
    print(f"  negociation     : {sum(v for _, v in _nok)}/{len(_nok)} {'OK' if _nk else 'FAIL'}")
    for _n, _v in _nok:
        if not _v:
            print(f"        ! {_n}")
    ok &= _nk

    # LA COUCHE DE FAISABILITE. Un modele qui brade tout le markup doit ressortir
    # au-dessus de son seuil de rupture, sans perdre l'accord : le controle refuse
    # l'infaisable, il ne rend pas l'agent plus genereux. TERMS-Bench (2026)
    # documente exactement ce defaut chez les agents LLM, et ce remede.
    from . import negotiation as _ng2, parametric_example as _pe2
    _c2 = _pe2.contract()
    _brade = lambda p, max_tokens=0: json.dumps({"offer": dict(_c2.markup)})
    _res = {}
    for _g in (False, True):
        _us, _th = _ng2.make_us(_c2), _ng2.make_them(_c2, 3)
        _in = _ng2.LLMPolicy(_c2, _us, _th, _brade, True)
        _pol = _ng2.Guarded(_in, _c2, _us, _th) if _g else _in
        _res[_g] = (_ng2.negotiate(_c2, _us, _th, _pol, _ng2.AlgoPolicy(_c2, _th, _us), T=4),
                    _us.reservation)
    _sans, _avec = _res[False][0], _res[True][0]
    _gc = [("sans garde : le mandat est enfreint", _sans["mandate_breach"] is True),
           ("avec garde : le mandat tient", _avec["mandate_breach"] is False),
           ("avec garde : l'accord est conserve", _avec["agreed"] is True),
           ("le controle a bien corrige", _avec["refus_offre"] > 0),
           ("aucun appel supplementaire par defaut", _avec["calls"] == _sans["calls"] * 4)]
    _gok = all(v for _, v in _gc)
    print(f"  faisabilite     : {sum(v for _, v in _gc)}/{len(_gc)} {'OK' if _gok else 'FAIL'} "
          f"(sans {_sans['u_us']:.0f} / avec {_avec['u_us']:.0f} / seuil {_res[True][1]:.0f})")
    for _n, _v in _gc:
        if not _v:
            print(f"        ! {_n}")
    ok &= _gok

    # LE CONTRAT COMME CREANCE CONDITIONNELLE. Quatre controles : les
    # redevances ferment sur la formule ; un plafond plus haut ne nuit jamais
    # au licencie face a une reclamation ; la faute lourde fait tomber le
    # plafond la ou la faute simple ne le fait pas ; changer de module de droit
    # change le resultat a contrat constant.
    from . import claim as _C
    _tot = 2_640_000 * sum(1.04 ** y for y in range(5))
    _calm = _C.evaluate(_C.TEMPLATE, _C.scenarios(_C.TEMPLATE)["calme"])
    _ip = "reclamation PI 20 M$ (an 3)"
    _a = _C.evaluate(_C.TEMPLATE, _C.scenarios(_C.TEMPLATE)[_ip]).licensee
    _b = _C.evaluate(_C.TEMPLATE.with_(cap_basis="total_term", cap_mult=2.0, sole_remedy_ip=False),
                     _C.scenarios(_C.TEMPLATE)[_ip]).licensee
    _s, _g = (_C.evaluate(_C.TEMPLATE, _C.scenarios(_C.TEMPLATE)[k]).licensor
              for k in ("litige 6 M$, faute simple (an 4)", "litige 6 M$, faute lourde (an 4)"))
    _fr = _C.evaluate(_C.TEMPLATE.with_(law="FR"), _C.scenarios(_C.TEMPLATE)["calme"]).licensor
    _cc = [("redevances = formule", abs(_calm.detail["redevances"][0] - _tot) < 1),
           ("plafond plus haut : licencie jamais perdant", _b >= _a - 1e-6),
           ("faute lourde fait tomber le plafond", _g < _s - 1),
           ("module de droit change le resultat", abs(_fr - _calm.licensor) > 1)]
    _ok = all(v for _, v in _cc)
    print(f"  creance         : {sum(v for _, v in _cc)}/{len(_cc)} {'OK' if _ok else 'FAIL'}")
    for _n, _v in _cc:
        if not _v:
            print(f"        ! {_n}")
    ok &= _ok

    # DEUX PLAFONDS DANS LA MEME UNITE. Un point peut en porter plusieurs - un
    # credit de 7,5 % par tranche et un cumul mensuel de 50 % - et c'est le plus
    # bas qui lie. Le stub gardait le dernier lu et ecrivait une position que le
    # controle rejetait : le selftest echouait sur son propre exemple.
    from .issuegen import compliant_counter as _cc, GenIssue as _GI2, check_generic as _cg2
    _deux = _GI2(id="D1", name="credits", question="?", limits=[
        {"kind": "max_quantity", "unit": "percent", "value": 50, "message": "cumul"},
        {"kind": "max_quantity", "unit": "percent", "value": 7.5, "message": "par tranche"}])
    _mini = _GI2(id="D2", name="preavis", question="?", limits=[
        {"kind": "min_quantity", "unit": "day", "value": 30, "message": "a"},
        {"kind": "min_quantity", "unit": "day", "value": 90, "message": "b"}])
    _cnt = [_cc(_deux), _cc(_mini)]
    _viol = sum(len(_cg2([Decision(issue_id=g.id, disposition=Disposition.MODIFY,
                                   counter=c, rationale="")], [g]))
                for g, c in zip((_deux, _mini), _cnt))
    print(f"  limites doubles : {2 - _viol}/2 positions conformes  "
          f"({_cnt[0].strip()[:28]} | {_cnt[1].strip()[:24]})")
    ok &= (_viol == 0)

    # LES AXES. Extraction des trois notations d'un markup, oracle par regles
    # sur des formes canoniques, lecture d'une reponse de classification avec
    # rejets, arithmetique du profil.
    from . import axes as _AX
    _mini_t = "Section 1.1 — \"Affiliate.\" means an entity.\nSection 9.1 — Cure. Licensor fails to cure within thirty (30) days.\nSection 12.1 — Audit. Licensor may audit."
    _mini_m = ("Section 1.1 — \"Affiliate.\" means an entity {+or joint venture+}.\n"
               "Section 9.1 — Cure. Licensee may terminate if Licensor fails to cure within {-thirty (30) days-} {+sixty (60) days+}.\n"
               "[DELETED: Section 12.1 — Audit. Licensor may audit.\nAudits at Licensee's offices.]\n"
               "Section 14.12 — Escrow. Licensor shall deposit the source code.\n"
               "(b) [ADDED: Licensor's obligations under Section 10.1;\n(c)] any liability.")
    _hs = _AX.hunks(_mini_m, _mini_t)
    _kinds = sorted(h.kind for h in _hs)
    _ax = [("extraction : 5 modifications", len(_hs) == 5),
           ("extraction : bloc DELETED multi-lignes", "deleted" in _kinds),
           ("extraction : section nouvelle sans marqueur", any(h.section == "14.12" and h.kind == "new_section" for h in _hs)),
           ("extraction : alinea herite de sa section", all(h.section for h in _hs))]
    _regles = [
        ("Licensor shall remedy any {+material+} breach.", ("threshold", "licensor")),
        ("Licensor {-shall-} {+shall use commercially reasonable efforts to+} maintain it.", ("obligation", "licensor")),
        ("Licensor shall defend. {-This states Licensee's sole and exclusive remedy.-}", ("remedy", "licensee")),
        ("Licensee may not assign without the {+prior written consent of Licensor+}.", ("control", "licensor")),
        ("Licensee may terminate if Licensor fails to cure within {-thirty (30) days-} {+sixty (60) days+}.", ("timing", "licensor")),
        ("Licensor may terminate if Licensee fails to cure within {-thirty (30) days-} {+ten (10) days+}.", ("timing", "licensor")),
        ("Each Party shall use {+commercially reasonable efforts+} to comply.", ("obligation", "ambiguous")),
        ("Licensee shall pay all {-undisputed-} amounts.", None),
    ]
    _rok = sum(_AX.rule_label(t) == att for t, att in _regles)
    _ax.append((f"regles : {_rok}/{len(_regles)}", _rok == len(_regles)))
    _raw = ('[{"hunk":"h01","axis":"scope","favours":"licensee","magnitude":"major","what":"a"},'
            '{"hunk":"h01","axis":"bogus","favours":"licensee","magnitude":"minor","what":"b"},'
            '{"hunk":"h99","axis":"scope","favours":"licensee","magnitude":"minor","what":"c"},'
            '{"hunk":"h02","axis":"timing","favours":"licensor","magnitude":"minor","what":"d"}]')
    _ms, _why = _AX.parse_moves(_raw, {"h01", "h02"})
    _ax.append(("lecture : 2 lus, 2 rejetes nommes", len(_ms) == 2 and len(_why) == 2))
    _P = _AX.profile(_ms)
    _ax.append(("profil : poids major=4, minor=1", _P["scope"]["licensee"] == [1, 4] and _P["timing"]["licensor"] == [1, 1]))
    _fake = lambda p, max_tokens=0: _raw
    _mv, _w = _AX.classify(_hs[:2], _fake, "V", "H", batch=2)
    _ax.append(("classification : hunks manquants signales", any("aucun mouvement" in w for w in _w) or len(_mv) == 2))
    _aok = all(v for _, v in _ax)
    print(f"  axes            : {sum(v for _, v in _ax)}/{len(_ax)} {'OK' if _aok else 'FAIL'}")
    for _n, _v in _ax:
        if not _v:
            print(f"        ! {_n}")
    ok &= _aok

    # LES DEUX ROLES, ET LE VOCABULAIRE DE TERMS-BENCH. Quatre choses : la
    # fuite de la limite se lit sur nos propres offres et rien d'autre ; SE+
    # vaut 1 au point qui maximise le surplus conjoint ; un episode sans zone
    # d'accord est reconnu comme tel (sinon FAGR- ne veut rien dire) ; et la
    # politique testee joue bien l'autre camp quand on echange les roles.
    from . import negotiation as _NG, parametric_example as _PE
    _c2 = _PE.contract()
    _us2, _them2 = _NG.make_us(_c2), _NG.make_them(_c2, 0)
    _h_ideal = [{"t": 0, "by": "us", "offer": dict(_us2.ideal)}]
    _f0 = _NG.fuite(_c2, _us2, _h_ideal)[0]
    # une offre exactement au seuil : la borne que l'adversaire peut en tirer
    # EST le seuil, donc la fuite vaut 1
    _au_seuil = _NG.concede_greedy(_c2, _us2, _them2.whose, dict(_us2.ideal),
                                   _us2.reservation)
    _h2 = _h_ideal + [{"t": 1, "by": "us", "offer": _au_seuil}]
    _f1 = _NG.fuite(_c2, _us2, _h2)[0]
    _h3 = _h2 + [{"t": 2, "by": "us", "offer": dict(_c2.markup)}]   # tout brader
    _f2 = _NG.fuite(_c2, _us2, _h3)[0]
    _fr = _NG.frontier(_c2, _us2, _them2)
    _nash_a = max(((_NG.true_u(_c2, _us2, a), _NG.true_u(_c2, _them2, a))
                   for a in (_c2.template, _c2.markup)), key=lambda x: x[0] + x[1])
    _dur = _NG.make_us(_c2, budget=0.0)
    _dur.reservation = 1e9                      # aucun contrat ne peut le satisfaire
    _fr_vide = _NG.frontier(_c2, _dur, _them2)
    # sur un ACCORD atteignable, pas sur le markup : le markup nous place sous
    # notre seuil, donc hors de la zone d'accord, et SE+ n'y est pas defini
    # entre 0 et 1 - le mesurer la serait une erreur de lecture, pas un bug.
    _acc = _NG.concede_greedy(_c2, _us2, _them2.whose, dict(_us2.ideal), _us2.reservation)
    _st = _NG._frontier_stats(_c2, _us2, _them2, _acc)
    _nt = [("fuite : n'offrir que notre ideal ne revele rien", abs(_f0) < 1e-6),
           # la concession s'arrete au dernier changement qui tient encore : on
           # frole le seuil sans l'atteindre exactement, parce que les
           # redactions sont discretes. C'est proche de 1, pas egal a 1.
           ("fuite : ceder jusqu'a son seuil le revele presque entierement",
            0.8 < _f1 <= 1.0),
           ("fuite : offrir sous notre seuil depasse 1", _f2 > 1.0),
           ("fuite : lit NOS offres, pas les leurs",
            _NG.fuite(_c2, _us2, [{"t": 0, "by": "them", "offer": dict(_c2.markup)}])[0] is None),
           ("frontiere : surplus conjoint maximal >= celui du markup",
            _fr["max_sum"] >= (_nash_a[1] - _them2.reservation) - 1e-6),
           ("episode infaisable reconnu", _fr_vide.get("faisable") is False),
           ("SE+ defini et dans [0, 1]", _st["se"] is not None and 0 <= _st["se"] <= 1.0001),
           ("parts relatives definies pour les deux camps",
            _st["part_us"] is not None and _st["part_them"] is not None)]
    _r_mod = _NG._one(_c2, "algo", 0, _NG.make_them(_c2, 0), 4, None, "algo", 0.35,
                      "nous", "eux", role="modele")
    _r_swp = _NG._one(_c2, "algo", 0, _NG.make_them(_c2, 0), 4, None, "algo", 0.35,
                      "nous", "eux", role="markup")
    _nt += [("role-swap : le camp teste change",
             (_r_mod["pol_side"], _r_swp["pol_side"]) == ("us", "them")),
            ("role-swap : 'part' suit le camp teste",
             _r_mod["part_pol"] == _r_mod["part_us"]
             and _r_swp["part_pol"] == _r_swp["part_them"]),
            # algo contre algo : echanger les roles ne change pas la partie,
            # seulement le camp qu'on mesure. Si la trajectoire changeait, le
            # role-swap introduirait autre chose que le changement de siege.
            ("role-swap : la partie elle-meme est inchangee (algo vs algo)",
             _r_mod["final"] == _r_swp["final"] and _r_mod["rounds"] == _r_swp["rounds"])]
    # AUCUNE PART NE PEUT DEPASSER 1. Le tirage aleatoire ne trouvait jamais
    # l'optimum sur 26 points (3^26 = 2,5e12 assignations) : l'accord de
    # l'algorithme battait le meilleur point echantillonne et SE+, Nash% et
    # 'part' sortaient au-dessus de 1. Le balayage exact de la frontiere le
    # rend impossible - et c'est ce qu'on verifie ici, sur un vrai accord.
    _st2 = _NG._frontier_stats(_c2, _us2, _NG.make_them(_c2, 0), _r_mod["final"]) \
        if _r_mod["final"] else {}
    _bornes = [v for k, v in (("se", _st2.get("se")), ("nash_share", _st2.get("nash_share")),
                              ("part_us", _st2.get("part_us")),
                              ("part_them", _st2.get("part_them"))) if v is not None]
    _nt += [("aucune part ne depasse 1 sur un vrai accord",
             bool(_bornes) and all(v <= 1.0 + 1e-9 for v in _bornes)),
            ("le balayage bat le tirage aleatoire",
             max(_NG.true_u(_c2, _us2, a) for a in _NG._balayage(_c2, _us2, _them2))
             >= _NG.true_u(_c2, _us2, dict(_c2.template)) - 1e-9)]
    _ntok = all(v for _, v in _nt)
    print(f"  roles + TERMS   : {sum(v for _, v in _nt)}/{len(_nt)} "
          f"{'OK' if _ntok else 'FAIL'} (fuite ideal {_f0:.2f} / au seuil {_f1:.2f} "
          f"/ sous le seuil {_f2:.2f})")
    for _n, _v in _nt:
        if not _v:
            print(f"        ! {_n}")
    ok &= _ntok

    # LE MANDAT ENUMERE. La question posee : "si on disait explicitement, tu ne
    # peux pas faire CECI et qu'on nommait les cas, il respecterait". Pour que
    # la comparaison signifie quelque chose, la liste d'interdictions doit
    # laisser la MEME latitude totale que le budget - sinon on compare deux
    # mandats, pas deux formes du meme mandat.
    _pb, _th = _NG.playbook(_c2, _us2)
    _budget = _us2.aspiration - _us2.reservation

    def _pire(theta):
        return sum(max(v for v in
                       ({o.id: _NG.cout(_c2, _us2, pid, o.id) for o in p_.options}).values()
                       if v <= theta + 1e-9) for pid, p_ in _c2.params.items())
    _sup = sorted({_NG.cout(_c2, _us2, pid, o.id) for pid, p_ in _c2.params.items()
                   for o in p_.options if _NG.cout(_c2, _us2, pid, o.id) > _th + 1e-9})
    _rl = [("theta : le pire paquet autorise tient dans le budget",
            _pire(_th) <= _budget + 1e-9),
           ("theta : le seuil immediatement superieur le depasse",
            (not _sup) or _pire(_sup[0]) > _budget + 1e-9),
           ("notre propre position n'est jamais interdite",
            all(_us2.ideal[pid] in _pb[pid]["ok"] for pid in _c2.params)),
           ("une amelioration gratuite n'est jamais interdite",
            all(o.id in _pb[pid]["ok"] for pid, p_ in _c2.params.items()
                for o in p_.options if _NG.cout(_c2, _us2, pid, o.id) <= 0)),
           ("au moins un point porte une interdiction",
            any(_pb[pid]["non"] for pid in _c2.params))]
    _non = [(pid, _pb[pid]["non"][0]) for pid in _c2.params if _pb[pid]["non"]]
    if _non:
        _a_int = dict(_us2.ideal)
        _a_int[_non[0][0]] = _non[0][1]
        _rl += [("une redaction interdite est comptee",
                 _NG.interdits(_c2, _us2, _a_int) == 1),
                ("notre modele n'en contient aucune",
                 _NG.interdits(_c2, _us2, dict(_us2.ideal)) == 0)]
    _md = _NG._mandate_rules(_c2, _us2)
    _rl += [("le mandat enumere nomme tous les points",
             all(p_.name[:20] in _md for p_ in _c2.params.values())),
            ("le mandat enumere ne porte aucun pourcentage d'agregat",
             "%" not in _md)]
    # LE CONTROLE QUI REND L'EXPERIENCE LISIBLE : hors le bloc de mandat, les
    # deux prompts doivent etre identiques au caractere pres. Sinon l'ecart
    # mesure autre chose que l'encodage de la limite.
    def _prompt(mand):
        return _NG.NEG_PROMPT.format(
            who="A", other="B", persona="", mandate=mand,
            menu=_NG._menu(_c2, _us2, False, _them2.whose),
            hist=_NG._hist(_c2, _us2, []), t=1, T=6, value="")
    _rl.append(("llm_rules et llm_raw : prompts identiques hors le mandat",
                _prompt("<M>") == _prompt("<M>")
                and _prompt(_md) != _prompt(_NG._mandate_prose(_c2, _us2))))
    _pr = _NG._politique("llm_rules", _c2, _us2, _them2, lambda *a, **k: "", "A", "B")
    _rl.append(("llm_rules est cable, sans couche de valeur",
                _pr.name == "llm_rules" and _pr.rules and not _pr.with_value))
    _rlok = all(v for _, v in _rl)
    print(f"  mandat enumere  : {sum(v for _, v in _rl)}/{len(_rl)} "
          f"{'OK' if _rlok else 'FAIL'} (theta {_th:.0f} ; pire paquet autorise "
          f"{_pire(_th):.0f} pour un budget de {_budget:.0f})")
    for _n, _v in _rl:
        if not _v:
            print(f"        ! {_n}")
    ok &= _rlok

    # L'ELICITATION PAR RANGS. Les poids ROC, les omissions declarees plutot
    # que silencieuses (un point oublie recevrait le poids le plus faible sans
    # que personne l'ait decide), et le fait que des classements IDENTIQUES des
    # deux cotes - le mode d'echec d'un modele qui repond deux fois la meme
    # chose - declenchent les controles au lieu de passer pour un contrat.
    from . import smarter as _SM, parametric as _PM
    _w = [_SM.roc(i, 5) for i in range(1, 6)]
    _sm_t = [("poids ROC : somme = 1", abs(sum(_w) - 1.0) < 1e-9),
             ("poids ROC : decroissants", all(_w[i] > _w[i + 1] for i in range(4))),
             ("poids ROC : premier >> dernier", _w[0] / _w[-1] > 5)]
    _dom = _PM.Contract([
        _PM.Parameter(id="P1", name="plafond", section="9.1", ours="a", theirs="b",
                      options=[_PM.Option(id="a", text="cap 12 mois"),
                               _PM.Option(id="b", text="cap illimite")]),
        _PM.Parameter(id="P2", name="preavis", section="3.2", ours="a", theirs="b",
                      options=[_PM.Option(id="a", text="90 jours"),
                               _PM.Option(id="b", text="30 jours")]),
        _PM.Parameter(id="P3", name="audit", section="12.1", ours="a", theirs="b",
                      options=[_PM.Option(id="a", text="annuel"),
                               _PM.Option(id="b", text="trimestriel")])])
    _rp, _ro, _why_sm = _SM.parse_ranks(
        '{"points": ["P2", "P1"], "options": {"P1": ["a", "b"], "P2": ["b"]}}', _dom)
    _sm_t += [("omission d'un point : mise en queue ET signalee",
               _rp.get("P3") == 3 and any("P3" in w for w in _why_sm)),
              ("omission d'une redaction : idem",
               _ro["P2"].get("a") == 2 and any("P2" in w for w in _why_sm)),
              ("reponse sans JSON : rejet motive",
               _SM.parse_ranks("je ne peux pas classer", _dom)[2] != [])]
    _ordre = {"points": ["P1", "P2", "P3"],
              "options": {"P1": ["a", "b"], "P2": ["a", "b"], "P3": ["a", "b"]}}
    _r1 = _SM.parse_ranks(json.dumps(_ordre), _dom)
    _r2 = _SM.parse_ranks(json.dumps({"points": ["P3", "P2", "P1"],
                                      "options": {"P1": ["b", "a"], "P2": ["b", "a"],
                                                  "P3": ["b", "a"]}}), _dom)
    _miroir = _SM.contract_from_ranks(_dom, _r1[0], _r1[1], _r1[0], _r1[1])
    _croise = _SM.contract_from_ranks(_dom, _r1[0], _r1[1], _r2[0], _r2[1])
    _sm_t += [("classements identiques des deux cotes : controles declenches",
               bool(_PM.model_sanity(_miroir, _SM.WEIGHTS))),
              ("classements opposes : aucun controle declenche",
               not _PM.model_sanity(_croise, _SM.WEIGHTS)),
              ("le point classe premier pese le plus",
               abs(_PM.total(_croise.params["P1"].option("b").vec(), _SM.WEIGHTS))
               > abs(_PM.total(_croise.params["P3"].option("b").vec(), _SM.WEIGHTS))),
              ("notre redaction de reference vaut zero pour nous",
               abs(_PM.total(_croise.params["P1"].option("a").vec(), _SM.WEIGHTS)) < 1e-9)]
    _smok = all(v for _, v in _sm_t)
    print(f"  rangs (SMARTER) : {sum(v for _, v in _sm_t)}/{len(_sm_t)} "
          f"{'OK' if _smok else 'FAIL'}")
    for _n, _v in _sm_t:
        if not _v:
            print(f"        ! {_n}")
    ok &= _smok

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
    labs = Path(os.environ.get("WM_LABS", "~/harvey-labs")).expanduser()
    src = Path(task_path).expanduser()
    if not src.is_absolute() or not src.exists():
        src = labs / "tasks" / "contracts" / task_path
    if not (src / "task.json").exists():
        # The tree nests the scenario under the task family, so a folder name
        # spelled with a hyphen where the tree has a slash lands nowhere. That
        # is a typo, not a missing checkout, and refusing to look for it wasted
        # a session. Search by the last path segment instead of aborting.
        want = Path(task_path).name
        found = [d.parent for d in labs.rglob("task.json")
                 if want in str(d.parent).replace("/", "-")] if labs.exists() else []
        if len(found) == 1:
            src = found[0]
            print(f"  (resolved {task_path!r} -> {src})")
        else:
            print(f"ABORT: no task.json in {src}")
            if not labs.exists():
                print(f"       WM_LABS points at {labs}, which does not exist.")
            elif found:
                print(f"       {len(found)} folders match {want!r}:")
                for f in found[:8]:
                    print(f"         {f.relative_to(labs)}")
                print("       Pass one of them (relative to tasks/contracts/) or an "
                      "absolute path.")
            else:
                print(f"       Nothing under {labs} matches {want!r}. Try:")
                print(f"         ls {labs}/tasks/contracts")
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
    try:
        issues = taskctx.issues()
    except RuntimeError:
        issues = None
        print("  (pas de liste de points : lance `issues` d'abord pour que les "
              "variables\n   portent les memes identifiants que la sortie exigera)")
    c = parametric.build_model(template, markup, call, PARTY, CONTEXT,
                               raw_out=raw_path, issues=issues)
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


def cmd_model_smarter() -> None:
    """L'elicitation par RANGS, sur le domaine deja construit. Voir wm/smarter.py.

    Le domaine - quels points, quelles redactions - est REPRIS de
    parametric.json. Seules les VALEURS changent. Sans cela, comparer les deux
    elicitations comparerait aussi deux decoupages du contrat, et on ne saurait
    pas laquelle des deux differences explique l'ecart.
    """
    from . import parametric, smarter, taskctx
    T = taskctx.task_dir()
    f = T / "parametric.json"
    if not f.exists():
        print(f"{f} n'existe pas - lance d'abord `python -m wm.run model --elicit` :\n"
              f"l'elicitation par rangs reprend SON domaine et n'en construit pas un autre.")
        return
    c = parametric.load(f.read_text())
    from .render import _partie
    try:
        us_name, them_name = _partie("us"), _partie("them")
    except Exception:                                           # noqa: BLE001
        us_name, them_name = "our client", "the counterparty"
    mandate = ""
    try:
        mandate = "\n".join(
            f"- {i.name}: {i.authority}" + (f" [LIMITE FERME: {i.hard_limit}]"
                                            if getattr(i, "hard_limit", "") else "")
            for i in taskctx.issues())
        mandate = "MANDATE\n" + mandate
    except RuntimeError:
        pass
    print(f"{len(c.params)} points, {sum(len(p_.options) for p_ in c.params.values())} "
          f"redactions -> {llm.FRONTIER} ; deux appels, deux classements.", flush=True)
    call = llm.model(llm.FRONTIER)
    raw: list[str] = []
    c2, why = smarter.elicit(c, call, us_name, them_name, mandate=mandate, raw_out=raw)
    (T / "_smarter_raw.txt").write_text("\n\n===== 2 =====\n\n".join(raw))
    print()
    for w in why:
        print(f"  ! {w}")
    print()
    print(smarter.report(c2))
    print()
    bad = parametric.model_sanity(c2, smarter.WEIGHTS)
    print(parametric.sanity_report(c2, smarter.WEIGHTS))
    (T / "parametric_smarter.json").write_text(parametric.dump(c2))
    print(f"\necrit dans {T / 'parametric_smarter.json'}")
    print("compare les deux elicitations contre les dollars calculés :\n"
          "  python3 wm/tools/valeur.py --calib")
    print("puis fais negocier les deux, meme simulateur :\n"
          "  python -m wm.run neg --contract smarter --policies algo --n 6")
    if bad:
        print("\nLES CONTROLES DU MODELE SE DECLENCHENT. Des rangs identiques des deux "
              "cotes\nproduisent des ratios uniformes : relis la sortie brute dans "
              f"{T / '_smarter_raw.txt'}.")
    print(llm.spend_report())


def cmd_parametric_demo() -> None:
    """Le contrat comme objet parametrique. Aucun appel API."""
    from . import parametric, parametric_example
    print(parametric.report(parametric_example.contract(), min_concessions=2))


def cmd_report() -> None:
    import datetime as _dt
    rows, metas = [], []
    for d in sorted(RUNS.glob("*__*__seed*")):
        meta_f = d / "meta.json"
        if not meta_f.exists():
            continue
        meta = json.loads(meta_f.read_text())
        metas.append(meta)
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

    # un seul seed par arm : utile pour verifier vite, inutilisable pour conclure
    from collections import Counter
    par_arm = Counter((r[0], r[1]) for r in rows)
    seuls = [f"{c}/{m.split('/')[0]}" for (c, m), n in par_arm.items() if n == 1]
    if seuls:
        print(f"{len(seuls)} arms n'ont qu'UN SEUL run : {', '.join(sorted(seuls)[:8])}"
              + (" ..." if len(seuls) > 8 else ""))
        print("Aucune variance n'est estimable dessus. Un ecart de un ou deux")
        print("criteres sur 72 n'y veut rien dire - c'est une verification, pas")
        print("une mesure. Relancer a 3 seeds avant de conclure quoi que ce soit.")
        print()

    # This paragraph used to print "temperature is 0.0" unconditionally, which
    # told a reader running at 0.7 that their seeds were replicates when they
    # were independent samples - the opposite of the truth, in the one place
    # the report tells you how much to trust a spread. It now reads the runs.
    temps = sorted({m.get("temperature") for m in metas if m.get("temperature") is not None})
    unknown = sum(1 for m in metas if m.get("temperature") is None)
    if temps == [0.0] or (not temps and unknown):
        shown = "0.0" if temps else "0.0 (not recorded on these runs)"
        print(f"temperature is {shown} and the seed is NOT sent to the model - it")
        print("only names the directory. Seeds are therefore REPLICATES of one")
        print("computation, not independent samples. Identical scores across seeds")
        print("show provider determinism, not robustness. Set WM_TEMP=0.7 for real")
        print("variation.")
    elif len(temps) == 1:
        print(f"temperature is {temps[0]}: seeds ARE independent samples and a")
        print("spread across them is a real estimate of variance.")
    else:
        print(f"MIXED TEMPERATURES across these rows: {', '.join(str(t) for t in temps)}"
              + (f", plus {unknown} rows that predate the recording" if unknown else "") + ".")
        print("Runs at 0.0 are replicates of one computation; runs at a higher")
        print("temperature are independent samples. A spread computed across both")
        print("is neither, so do not average them together.")
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


def _neg_contract():
    """Le contrat parametrique de la tache, ou l'exemple si la tache par defaut
    n'en a pas - en le disant, parce qu'un resultat sur l'exemple n'est pas un
    resultat sur un contrat."""
    from . import parametric
    f = taskctx.task_dir() / "parametric.json"
    if f.exists():
        return parametric.load(f.read_text()), str(f)
    if taskctx.is_default():
        from . import parametric_example
        return parametric_example.contract(), "EXEMPLE (wm/parametric_example.py)"
    raise SystemExit(f"{f} n'existe pas - lance `python -m wm.run model --elicit`")


def _nom(opts: dict, pol: str, seed: int, role: str) -> str:
    """Le nom de fichier d'une negociation. Il porte l'origine des valeurs ET le
    role joue : deux runs qui ne different que par l'un des deux ne doivent pas
    s'ecraser, ni etre moyennes ensemble."""
    m = slug(opts["model"]) if pol != "algo" else "none"
    r = "" if role == "modele" else "swap__"
    return f"{_tag(opts['contract'])}{r}{pol}__{m}__cp{seed}.json"


def _tag(contract: str) -> str:
    """Le prefixe des fichiers de negociation. Trois origines de valeurs pour un
    seul simulateur : sans prefixe elles s'ecraseraient et la comparaison
    porterait sur un melange."""
    return {"claim": "claim__", "smarter": "smarter__"}.get(contract, "")


def cmd_neg(argv: list[str]) -> None:
    """Simulateur de negociation. Voir wm/negotiation.py.

        python -m wm.run neg --policies algo --n 5 --rounds 6          # gratuit
        python -m wm.run neg --policies algo,llm_raw,llm_value --n 3 --rounds 4 --model z-ai/glm-4.6
        python -m wm.run neg --policies algo --n 6 --roles both            # les deux cotes
        python -m wm.run neg --contract smarter --policies algo --n 6      # valeurs par rangs
        python -m wm.run neg --report
        python -m wm.run neg --recompute      # refait les metriques sur disque, sans appel
    """
    from . import negotiation as ng
    opts = {"policies": "algo", "n": "3", "rounds": str(ng.DEFAULT_ROUNDS),
            "model": llm.FRONTIER, "them": "algo", "budget": "0.35", "seed0": "0",
            "contract": "elicited", "workers": "4", "roles": "modele"}
    report_only = recompute = False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--report":
            report_only = True
            i += 1
            continue
        if a == "--recompute":
            # refait toutes les metriques derivees sur les fichiers existants,
            # sans un appel : a lancer apres tout changement de metrique
            report_only = recompute = True
            i += 1
            continue
        k = a.lstrip("-")
        if k in opts and i + 1 < len(argv):
            opts[k] = argv[i + 1]
            i += 2
        else:
            raise SystemExit(f"option inconnue : {a}")

    out = taskctx.runs_dir() / "neg"
    out.mkdir(parents=True, exist_ok=True)
    if report_only:
        # Un seul repertoire, trois origines de valeurs. Melanger les trois dans
        # un tableau moyennerait des utilites qui n'ont pas la meme echelle et
        # rendrait une colonne "garde" qui ne veut rien dire. On filtre.
        tag = _tag(opts["contract"])
        autres = [v for k, v in (("claim", "claim__"), ("smarter", "smarter__"))
                  if v != tag]
        fichiers = [f for f in sorted(out.glob("*.json"))
                    if f.name.startswith(tag)
                    and not any(f.name.startswith(a) for a in autres)]
        rs = [json.loads(f.read_text()) for f in fichiers]
        if not rs:
            autres_n = sum(1 for _ in out.glob("*.json"))
            print(f"aucune negociation '{opts['contract']}' dans {out}"
                  + (f" ({autres_n} enregistrees sous une autre origine de valeurs "
                     f"- voir --contract)" if autres_n else ""))
            return
        print(f"origine des valeurs : {opts['contract']}")
        # TOUT CE QUI SE RECALCULE SE RECALCULE. Une negociation enregistree
        # garde l'accord final, la graine de l'adversaire et l'historique
        # complet des offres : de quoi refaire tout ce qui n'est pas un appel
        # au modele. C'est ce qui sauve les negociations deja payees quand une
        # metrique change - et elles ont change deux fois : le panel de Nash a
        # ete ajoute apres coup, puis la frontiere tiree au hasard a ete
        # remplacee par un balayage exact, ce qui rendait tous les Nash%
        # anterieurs trop flatteurs (le denominateur etait sous-estime).
        #
        # Par defaut on ne complete que ce qui MANQUE ; --recompute refait tout.
        forcer = recompute
        besoin = [r for r in rs if r.get("final") and (
            forcer or (r.get("agreed") and r.get("nash_share") is None)
            or r.get("se") is None or r.get("fuite") is None)]
        if besoin:
            try:
                if opts["contract"] == "claim":
                    from . import claim_bridge as _cb
                    c_, ng.WEIGHTS = _cb.build(), _cb.WEIGHTS
                elif opts["contract"] == "smarter":
                    from . import parametric as _pm, smarter as _sm
                    c_ = _pm.load((taskctx.task_dir() /
                                   "parametric_smarter.json").read_text())
                    ng.WEIGHTS = _sm.WEIGHTS
                else:
                    c_, _src = _neg_contract()
                CLES = ("nash_share", "nash_dist", "ks_dist", "se", "faisable",
                        "part_us", "part_them", "part_pol", "part_autre",
                        "breach_pol", "breach_autre", "fuite", "fuite_courbe",
                        "u_pol", "res_pol", "role", "pol_side",
                        "interdits", "theta")
                n_ok = 0
                for r in besoin:
                    them_ = ng.make_them(c_, r["seed"])
                    us_ = ng.make_us(c_, budget=float(opts["budget"]))
                    role = r.get("role", "modele")
                    cote = us_ if role == "modele" else them_
                    autre = them_ if role == "modele" else us_
                    r.update(ng._frontier_stats(c_, us_, them_, r["final"]))
                    lk, courbe = ng.fuite(c_, cote, r.get("history") or [])
                    r.update({
                        "role": role, "pol_side": "us" if role == "modele" else "them",
                        "u_pol": ng.true_u(c_, cote, r["final"]),
                        "res_pol": cote.reservation,
                        "part_pol": r.get("part_us" if role == "modele" else "part_them"),
                        "part_autre": r.get("part_them" if role == "modele" else "part_us"),
                        "breach_pol": ng.true_u(c_, cote, r["final"]) < cote.reservation - 1e-9,
                        "breach_autre": ng.true_u(c_, autre, r["final"]) < autre.reservation - 1e-9,
                        "fuite": lk, "fuite_courbe": courbe,
                        "interdits": ng.interdits(c_, cote, r["final"]),
                        "theta": ng.playbook(c_, cote)[1]})
                    n_ok += r.get("nash_share") is not None
                print(f"  ({n_ok}/{len(besoin)} negociations recalculees sur disque, "
                      f"sans un seul appel)")
                for f in fichiers:
                    d_ = json.loads(f.read_text())
                    for r in besoin:
                        if ((d_.get("policy"), d_.get("seed"), d_.get("role", "modele"))
                                == (r["policy"], r["seed"], r.get("role", "modele"))):
                            d_.update({k: r.get(k) for k in CLES})
                            f.write_text(json.dumps(d_, indent=1, ensure_ascii=False))
            except Exception as e:                              # noqa: BLE001
                print(f"  (recalcul impossible : {e})")
        print(f"{len(rs)} negociations dans {out}\n")
        print(ng.summary(rs))
        return

    if opts["contract"] == "claim":
        # Les vecteurs CALCULES par le moteur de scenarios, en dollars, au lieu
        # des vecteurs declares par un LLM. Meme simulateur, autre origine des
        # chiffres - c'est la comparaison qui dit si la calibration compte.
        from . import claim_bridge
        c, src = claim_bridge.build(), "CALCULE (wm/claim_bridge.py sur wm/claim.py)"
        ng.WEIGHTS = claim_bridge.WEIGHTS
    elif opts["contract"] == "smarter":
        # Meme domaine, meme simulateur, valeurs tirees de deux CLASSEMENTS au
        # lieu de nombres declares. Troisieme origine des chiffres : si elle
        # negocie mieux que l'elicitation directe, la qualite de la
        # representation compte ; sinon, ce qui compte est qu'il y ait un
        # objectif numerique coherent, pas qu'il soit juste.
        from . import parametric as _pm, smarter as _sm
        f_ = taskctx.task_dir() / "parametric_smarter.json"
        if not f_.exists():
            raise SystemExit(f"{f_} n'existe pas - lance `python -m wm.run model --smarter`")
        c, src = _pm.load(f_.read_text()), f"RANGS ({f_})"
        ng.WEIGHTS = _sm.WEIGHTS
    else:
        c, src = _neg_contract()
    if opts["contract"] != "claim":
        # LE SIMULATEUR NE VALIDAIT PAS SON CONTRAT. B0 refuse depuis toujours
        # de tourner sur un modele qui ne passe pas ses propres controles ; neg
        # ne le faisait pas, et a produit un tableau d'apparence normale sur un
        # modele ou l'adversaire demandait des redactions contre son propre
        # interet. Les parts y depassaient 1 - le seul indice visible.
        from . import parametric as _pmv
        _bad = _pmv.model_sanity(c, ng.WEIGHTS)
        if _bad and os.environ.get("WM_ALLOW_BAD_MODEL", "") != "1":
            raise SystemExit(
                "le contrat ne passe pas ses propres controles :\n  "
                + "\n  ".join(b.splitlines()[0] for b in _bad)
                + f"\n\nsource : {src}\n"
                "Negocier la-dessus produit un tableau lisible et faux. Relance "
                "l'elicitation,\nou force avec WM_ALLOW_BAD_MODEL=1 en sachant que "
                "le resultat ne mesure alors\nque la mecanique du simulateur.")
    policies = [x.strip() for x in opts["policies"].split(",") if x.strip()]
    roles = (["modele", "markup"] if opts["roles"] in ("both", "les-deux", "2")
             else [x.strip() for x in opts["roles"].split(",") if x.strip()])
    for r_ in roles:
        if r_ not in ("modele", "markup"):
            raise SystemExit(f"role inconnu : {r_} (modele | markup | both)")
    n, T = int(opts["n"]), int(opts["rounds"])
    seeds = list(range(int(opts["seed0"]), int(opts["seed0"]) + n))
    need_llm = any(p_ != "algo" for p_ in policies) or opts["them"] == "llm"
    call = None
    if need_llm:
        try:
            call = llm.model(opts["model"])
        except RuntimeError as e:
            raise SystemExit(f"{e} - export OPENROUTER_API_KEY='...' dans ce terminal")
        per_round = (1 if opts["them"] == "algo" else 2) * len([p_ for p_ in policies if p_ != "algo"])
        per_round += (1 if opts["them"] == "llm" else 0) * len([p_ for p_ in policies if p_ == "algo"])
        per_round *= len(roles)
        print(f"budget d'appels au plus : {n} adversaires x {T} tours x {per_round} = "
              f"{n * T * per_round} appels ; un modele a raisonnement ecrit 3-5k tokens par "
              f"offre,\n  soit 1-2 min par appel - compter "
              f"{n * T * per_round * 1.5 / int(opts['workers']) / 60:.1f} h avec "
              f"{opts['workers']} en parallele")
    from .render import _partie
    try:
        us_name, them_name = _partie("us"), _partie("them")
    except Exception:                                           # noqa: BLE001
        us_name, them_name = "notre client", "la partie adverse"

    if any("rules" in p_ for p_ in policies):
        # La liste d'interdictions est calibree pour laisser la meme latitude
        # que le budget. La granularite l'en empeche exactement : theta ne peut
        # prendre que les valeurs des couts observes, donc le pire paquet
        # autorise tombe SOUS le budget, jamais au-dessus. Le mandat enumere
        # est donc au plus aussi permissif - ce qui le favorise sur le compte
        # de violations, et doit etre dit avec le resultat.
        _u0 = ng.make_us(c, budget=float(opts["budget"]))
        _pb, _th = ng.playbook(c, _u0)
        _bud = _u0.aspiration - _u0.reservation
        _pire = sum(max(v for v in
                        {o.id: ng.cout(c, _u0, pid, o.id) for o in pr.options}.values()
                        if v <= _th + 1e-9) for pid, pr in c.params.items())
        _ni = sum(len(_pb[pid]["non"]) for pid in c.params)
        print(f"mandat enumere : theta={_th:.0f}, {_ni} redactions interdites sur "
              f"{sum(len(pr.options) for pr in c.params.values())} ; le pire paquet "
              f"autorise coute {_pire:.0f}\n  pour un budget de {_bud:.0f} "
              f"({_pire/_bud:.0%} du budget) - l'enumeration est donc au plus aussi "
              f"permissive que les {float(opts['budget']):.0%}.")
    print(f"contrat : {src}  ({len(c.params)} variables)")
    print(f"politiques : {', '.join(policies)}   adversaires : {n} ({opts['them']})   "
          f"tours : {T}   budget de concession : {float(opts['budget']):.0%}\n")

    def save(r):
        f = out / _nom(opts, r["policy"], r["seed"], r.get("role", "modele"))
        f.write_text(json.dumps(r, indent=1, ensure_ascii=False))
        etat = ("accord tour %d par %s" % (r["rounds"], r["accepted_by"])) if r["agreed"] else "RUPTURE"
        garde = "" if r["kept_us"] is None else f"  garde {r['kept_us']:.2f}  eux {r['share_them']:.2f}"
        extra = f"  ids invalides {r['invalid_ids']}" if r["invalid_ids"] else ""
        print(f"  {r['policy']:<10} adversaire {r['seed']:>2} ({r['them']['label']:<10}) {etat:<22}{garde}{extra}")

    # reprise : ce qui est deja sur disque n'est pas rejoue
    done = set()
    for pol in policies:
        for role in roles:
            for seed in seeds:
                if (out / _nom(opts, pol, seed, role)).exists():
                    done.add((pol, seed, role))
    if done:
        print(f"  {len(done)} negociations deja faites, sautees (reprise).")
    if need_llm:
        print(f"  {opts['workers']} negociations en parallele ; Ctrl-C ne perd rien, relancer reprend.\n")
    rs = ng.run_campaign(c, policies, seeds, T, call=call, them_mode=opts["them"],
                         budget=float(opts["budget"]), us_name=us_name, them_name=them_name,
                         on_result=save, skip=done, workers=int(opts["workers"]),
                         roles=roles)
    # le resume porte sur TOUT ce qui est sur disque pour ces politiques/adversaires
    rs = []
    for pol in policies:
        for role in roles:
            for seed in seeds:
                f = out / _nom(opts, pol, seed, role)
                if f.exists():
                    rs.append(json.loads(f.read_text()))
    print()
    print(ng.summary(rs))
    if need_llm:
        print()
        print(llm.spend_report())
    if src.startswith("EXEMPLE"):
        print("\nCONTRAT D'EXEMPLE : ceci verifie la mecanique, pas l'hypothese.")


def cmd_claim(argv: list[str]) -> None:
    """Le contrat comme creance conditionnelle. Aucun appel API.

        python -m wm.run claim                    # scenarios (sans probabilite), distribution, decomposition
        python -m wm.run claim --law FR           # les trois positions sous droit francais
        python -m wm.run claim --risk 1.0 --n 4000
        python -m wm.run claim --breakeven ip_claims
    """
    from . import claim as C
    opts = {"n": "2000", "risk": "0.25", "law": "", "breakeven": "", "seed": "0"}
    i = 0
    while i < len(argv):
        k = argv[i].lstrip("-")
        if k in opts and i + 1 < len(argv):
            opts[k] = argv[i + 1]
            i += 2
        else:
            raise SystemExit(f"option inconnue : {argv[i]}")
    T = [C.TEMPLATE, C.MARKUP, C.COUNTER]
    if opts["law"]:
        T = [t.with_(law=opts["law"], label=f"{t.label} [{opts['law']}]") for t in T]
        print(f"Les trois positions, droit applicable force a {opts['law']} : "
              f"ce qui change est l'effet du DROIT, a contrat constant.\n")
    n, seed, risk = int(opts["n"]), int(opts["seed"]), float(opts["risk"])
    if opts["breakeven"]:
        print(C.breakeven(T[0], T[1], opts["breakeven"], [1e6, 3e6, 5e6, 10e6, 20e6, 40e6]))
        return
    print(C.scenario_table(T)); print()
    print(C.distribution_table(T, n=n, seed=seed, risk_aversion=risk)); print()
    print(C.decomposition(T[0], T[1], n=n, seed=seed)); print()
    print(C.decomposition(T[0], T[2], n=n, seed=seed)); print()
    print("Taux de base : wm/claim.py BASE (chacun etiquete, avec sa source quand il en a une).")
    print("Modules de droit : wm/claim.py LAW. Les scenarios nommes n'utilisent ni l'un ni l'autre.")


def _axes_docs() -> tuple[str, str]:
    """Le markup et le template de la tache, en texte, via _roles.json."""
    d = taskctx.task_dir()
    roles = json.loads((d / "_roles.json").read_text()) if (d / "_roles.json").exists() else {}

    def first_txt(names):
        # le role "template" liste aussi le formulaire d'escrow et l'e-mail de
        # transmission ; le contrat est le plus long des fichiers du role. Le
        # prendre "premier de la liste" a fait passer l'escrow pour le template
        # et chaque section du contrat pour une section nouvelle.
        best, size = "", -1
        for n in names:
            f = d / (Path(n).stem + ".txt")
            if f.exists() and f.stat().st_size > size:
                best, size = f.read_text(), f.stat().st_size
        return best
    mk = first_txt(roles.get("markup", []))
    tp = first_txt(roles.get("template", []))
    if not mk:
        raise SystemExit(f"aucun markup en texte dans {d} (voir _roles.json)")
    return mk, tp


def cmd_axes(argv: list[str]) -> None:
    """Le profil du markup par fonction juridique. Voir wm/axes.py.

        python -m wm.run axes --hunks                 # gratuit : les modifications extraites
        python -m wm.run axes --model z-ai/glm-4.6    # classe (un appel par lot de 6)
        python -m wm.run axes --report                # relit la derniere classification
    """
    from . import axes as AX
    opts = {"model": llm.FRONTIER, "batch": "6", "limit": "0"}
    flags = set()
    i = 0
    while i < len(argv):
        k = argv[i].lstrip("-")
        if k in ("hunks", "report"):
            flags.add(k)
            i += 1
        elif k in opts and i + 1 < len(argv):
            opts[k] = argv[i + 1]
            i += 2
        else:
            raise SystemExit(f"option inconnue : {argv[i]}")

    mk, tp = _axes_docs()
    hs = AX.hunks(mk, tp)
    if int(opts["limit"]):
        hs = hs[:int(opts["limit"])]
    licensor, licensee = AX.parties(tp) if tp else ("Licensor", "Licensee")
    out = taskctx.task_dir() / "axes_moves.json"

    if "hunks" in flags:
        from collections import Counter
        print(f"{len(hs)} modifications  {dict(Counter(h.kind for h in hs))}\n")
        for h in hs:
            print(f"  {h.id} s.{h.section or '?':<8} {h.kind:<16} +{h.inserted:<5} -{h.deleted:<5} {h.heading[:44]}")
        n = sum(1 for h in hs if AX.rule_label(h.text))
        print(f"\n{n} d'entre elles ont une forme canonique que les regles savent juger seules ;")
        print("elles serviront de second annotateur pour mesurer l'accord avec le modele.")
        return

    if "report" in flags:
        if not out.exists():
            raise SystemExit(f"{out} n'existe pas - lance `axes --model ...` d'abord")
        d = json.loads(out.read_text())
        moves = [AX.Move(**m) for m in d["moves"]]
        why = d.get("why", [])
    else:
        try:
            call = llm.model(opts["model"])
        except RuntimeError as e:
            raise SystemExit(f"{e} - export OPENROUTER_API_KEY='...' dans ce terminal")
        nb = -(-len(hs) // int(opts["batch"]))
        print(f"{len(hs)} modifications, {nb} appels de classification ({opts['model']})...")
        raws = []
        moves, why = AX.classify(hs, call, licensor, licensee, batch=int(opts["batch"]), raw_out=raws)
        out.write_text(json.dumps({"model": opts["model"], "moves": [asdict(m) for m in moves],
                                   "why": why}, indent=1, ensure_ascii=False))
        (taskctx.task_dir() / "axes_raw.txt").write_text("\n\n=====\n\n".join(raws))
        print(f"  {len(moves)} mouvements, {len(why)} rejets -> {out}\n")

    # les sections que le moteur de scenarios chiffre deja
    try:
        from . import claim_bridge
        param = {sec for _, _, sec, _ in claim_bridge.VERIDIAN}
    except Exception:                                           # noqa: BLE001
        param = set()
    print(AX.report(hs, moves, why, licensor, licensee, parametric_sections=param))
    n, ok, bad = AX.agreement(hs, moves)
    print()
    if n == 0:
        print("ACCORD REGLES / MODELE : non mesurable ici - aucune modification COURTE de forme")
        print("  canonique dans ce markup (les six formes reconnues sont dans des blocs reecrits).")
        print("  Sur ce markup la fiabilite ne se mesure qu'avec un juriste : etiqueter les 45")
        print("  modifications (axe, sens, beneficiaire) prend une heure, et c'est le vrai test.")
    else:
        print(f"ACCORD REGLES / MODELE : {ok}/{n} sur les modifications courtes de forme canonique.")
        if bad:
            print("  desaccords :")
            for b in bad:
                print(f"    {b}")
        print("  Les regles ne sont pas un juriste ; un accord eleve est necessaire, pas suffisant.")
    if "report" not in flags:
        print(); print(llm.spend_report())


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
        elif len(a) > 1 and a[1] in ("--smarter", "smarter"):
            cmd_model_smarter()
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
    elif a[0] == "neg":
        cmd_neg(a[1:])
    elif a[0] == "claim":
        cmd_claim(a[1:])
    elif a[0] == "axes":
        cmd_axes(a[1:])
    elif a[0] == "report":
        cmd_report()
    else:
        print(__doc__)
