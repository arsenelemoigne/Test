"""
The closed loop: the model proposes, the CODE evaluates, the model revises.

This is the arm the project was actually about. A0/A2/A4 differ only in how the
brief is formatted - prose, bullets, raw documents - and the model answers once,
with nothing checking it. Here the contract model is executable and sits inside
the loop:

    propose -> check() against the mandate
            -> which other issues this position moves
            -> what the package is worth
            -> feed the result back -> revise

The evaluator NEVER tells the model what to write. It reports what is wrong with
what the model wrote, using rules already stated in the mandate the model was
given. That distinction is what keeps the comparison honest: no information
enters the loop that was not in the one-shot prompt, only the consequences of
the model's own answer.

TWO ARMS, because "more passes" is not the same claim as "the evaluator helps":

    A5   evaluated loop  - real feedback, stops when clean
    A5N  blind revision  - same prompt, same number of extra passes, feedback
                           replaced by "review and improve your answer"

A5 beating A4 could just be extra thinking. A5 beating A5N cannot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class Report:
    violations: list[tuple[str, str]] = field(default_factory=list)  # (issue_id, message)
    missing: list[str] = field(default_factory=list)
    coupled: dict[str, list[str]] = field(default_factory=dict)
    value_note: str = ""

    @property
    def clean(self) -> bool:
        return not self.violations and not self.missing

    def render(self, positions: dict[str, str]) -> str:
        L = ["EVALUATION OF YOUR DRAFT", "=" * 60, ""]
        if self.violations:
            L.append(f"{len(self.violations)} of your positions fall outside the "
                     f"client's mandate:")
            L.append("")
            for iid, msg in self.violations:
                L.append(f"  {iid}  {msg}")
                pos = positions.get(iid, "")
                if pos:
                    L.append(f"        you wrote: \"{pos[:150]}\"")
            L.append("")
        if self.missing:
            L.append(f"{len(self.missing)} issues have no decision recorded: "
                     f"{', '.join(self.missing)}")
            L.append("")
        if self.coupled:
            L.append("Issues whose value depends on positions you changed:")
            for k, v in list(self.coupled.items())[:10]:
                L.append(f"  {k} -> {', '.join(v)}")
            L.append("  Check that your positions on these are consistent with "
                     "each other.")
            L.append("")
        if self.value_note:
            L += [self.value_note, ""]
        L.append("Revise your answer. Return the COMPLETE JSON array again, with "
                 "every issue, not just the ones named above.")
        return "\n".join(L)


BLIND_FEEDBACK = (
    "Review the answer you just gave. Revise anything you think can be improved.\n"
    "Return the COMPLETE JSON array again, with every issue."
)


def couplings_from_sections(issues) -> dict[str, list[str]]:
    """Issues sharing a contract article, as a coupling heuristic.

    A HEURISTIC, and labelled as one: 11.1 and 11.3 are the cap and its
    exceptions, which genuinely move together, but co-location is not
    entanglement. Where a task has elicited couplings this is replaced by them.
    """
    by_article: dict[str, list[str]] = {}
    for i in issues:
        art = str(getattr(i, "section", "") or "").split(".")[0].strip()
        if art:
            by_article.setdefault(art, []).append(i.id)
    out = {}
    for ids in by_article.values():
        if len(ids) < 2:
            continue
        for a in ids:
            out[a] = [b for b in ids if b != a]
    return out


def evaluate(decisions, issues, check_fn, couplings=None, value_fn=None) -> Report:
    """Run the deterministic model over a proposed set of positions."""
    r = Report()
    answered = {d.issue_id for d in decisions}
    r.missing = [i.id for i in issues if i.id not in answered]
    for v in check_fn(decisions):
        if "no decision recorded" not in v.message:
            r.violations.append((v.issue_id, v.message))

    cmap = couplings if couplings is not None else couplings_from_sections(issues)
    touched = {iid for iid, _ in r.violations} or answered
    r.coupled = {k: v for k, v in cmap.items() if k in touched}

    if value_fn is not None:
        try:
            r.value_note = value_fn(decisions)
        except Exception:                               # noqa: BLE001
            r.value_note = ""
    return r


def _score(rep: Report) -> tuple[int, int]:
    """Lower is better. Violations first, then unanswered issues."""
    return (len(rep.violations), len(rep.missing))


def run(prompt: str, call, parse, issues, check_fn, rounds: int = 3,
        blind: bool = False, couplings=None, value_fn=None) -> dict:
    """Propose, evaluate, revise - and KEEP THE BEST ANSWER, not the last one.

    Keeping the last one is how a working loop produces a worse result than no
    loop at all: an observed run went 6 violations -> 4 -> 3 and then shipped a
    fourth revision carrying 7. The evaluator is the objective function, so the
    answer to return is the one that scores best against it. Anything else is
    not optimisation, it is a random walk that happens to be evaluated.
    """
    trace = []
    raw = call(prompt)
    try:
        decisions = parse(raw)
    except ValueError:
        return {"decisions": [], "raw": raw, "rounds": 0, "calls": 1,
                "best_round": None, "trace": [{"round": 0, "error": "unparseable"}]}

    calls = 1
    rep = evaluate(decisions, issues, check_fn, couplings, value_fn)
    best, best_rep, best_round, best_raw = decisions, rep, 0, raw
    trace.append({"round": 0, "violations": len(rep.violations),
                  "missing": len(rep.missing), "clean": rep.clean, "kept": True})

    for n in range(1, rounds + 1):
        if rep.clean and not blind:
            break
        positions = {d.issue_id: d.counter for d in decisions}
        feedback = BLIND_FEEDBACK if blind else rep.render(positions)
        follow = (f"{prompt}\n\n=== YOUR PREVIOUS ANSWER ===\n"
                  f"{json.dumps([{'issue_id': d.issue_id, 'disposition': d.disposition.value, 'counter': d.counter, 'rationale': d.rationale} for d in decisions], indent=2)}"
                  f"\n\n{feedback}")
        raw = call(follow)
        calls += 1
        try:
            decisions = parse(raw)
        except ValueError:
            trace.append({"round": n, "error": "unparseable, keeping best so far"})
            break

        rep = evaluate(decisions, issues, check_fn, couplings, value_fn)
        improved = _score(rep) < _score(best_rep)
        trace.append({"round": n, "violations": len(rep.violations),
                      "missing": len(rep.missing), "clean": rep.clean,
                      "kept": improved})
        if improved:
            best, best_rep, best_round, best_raw = decisions, rep, n, raw

    trace.append({"round": "returned", "from_round": best_round,
                  "violations": len(best_rep.violations),
                  "missing": len(best_rep.missing), "clean": best_rep.clean})
    return {"decisions": best, "raw": best_raw, "rounds": len(trace) - 2,
            "calls": calls, "best_round": best_round, "trace": trace}
