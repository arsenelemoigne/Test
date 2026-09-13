"""
CLI for the Project Lantern NDA world model.

    python -m nda_world_model pre          balance report on Corrigan's markup
    python -m nda_world_model post         balance report on our counter-turn
    python -m nda_world_model diff         what the counter-turn moves
    python -m nda_world_model brief        the context block for the drafting LLM
    python -m nda_world_model brief-blind  same, instruction-blind (for the A/B)
    python -m nda_world_model checklist    completion checklist
    python -m nda_world_model coverage     audit vs ContractNLI 17 NDA hypotheses
    python -m nda_world_model sizes        token/character comparison vs raw docs
    python -m nda_world_model whatif       counterfactual rollouts
"""

from __future__ import annotations

import sys
from pathlib import Path

from . import coverage
from .brief import negotiation_brief, render_checklist
from .core import Action, Disposition, Position, Tier, diff, score
from .velantis_nda import (
    GATING_RULES,
    base_state,
    corrigan_state,
    counter_turn_state,
)

DOCS = Path(__file__).resolve().parents[1] / "task_extract"


def cmd_pre() -> None:
    print(score(corrigan_state(), GATING_RULES).render(verbose=True))


def cmd_post() -> None:
    print(score(counter_turn_state(), GATING_RULES).render(verbose=True))


def cmd_diff() -> None:
    print(diff(corrigan_state(), counter_turn_state()))


def cmd_brief() -> None:
    print(negotiation_brief(corrigan_state()))


def cmd_brief_blind() -> None:
    print(negotiation_brief(corrigan_state(), include_instructions=False))


def cmd_checklist() -> None:
    print(render_checklist(corrigan_state()))


def cmd_coverage() -> None:
    print(coverage.report(corrigan_state()))


def cmd_sizes() -> None:
    """Compression achieved, and what it costs."""
    raw = 0
    rows = []
    if DOCS.is_dir():
        for f in sorted(DOCS.glob("*.txt")):
            n = f.stat().st_size
            raw += n
            rows.append((f.name, n))
    brief = len(negotiation_brief(corrigan_state()))
    blind = len(negotiation_brief(corrigan_state(), include_instructions=False))
    chk = len(render_checklist(corrigan_state()))

    print("INPUT SIZE COMPARISON  (characters; ~4 chars/token)")
    print("=" * 68)
    for name, n in rows:
        print(f"  {name:<46s} {n:>8,d}")
    print(f"  {'TOTAL RAW TASK DOCUMENTS':<46s} {raw:>8,d}   ~{raw // 4:,d} tokens")
    print()
    print(f"  {'world-model brief (instructed)':<46s} {brief:>8,d}   ~{brief // 4:,d} tokens")
    print(f"  {'world-model brief (instruction-blind)':<46s} {blind:>8,d}   ~{blind // 4:,d} tokens")
    print(f"  {'completion checklist':<46s} {chk:>8,d}   ~{chk // 4:,d} tokens")
    print(f"  {'brief + checklist':<46s} {brief + chk:>8,d}   ~{(brief + chk) // 4:,d} tokens")
    if raw:
        print()
        print(f"  compression: {raw / (brief + chk):.1f}x")
        print()
        print("  NOTE: the brief is an INDEX, not a replacement. It carries locators and")
        print("  verbatim quotes so the drafting step still reads the clause it edits.")
        print("  Deploy it alongside the documents (condition A6), not instead of them.")


def cmd_whatif() -> None:
    """
    Counterfactual rollouts -- the thing raw text cannot do.
    Each is one line of code because the state is a data structure.
    """
    pre = corrigan_state()
    post = counter_turn_state()

    print("COUNTERFACTUAL ROLLOUTS")
    print("=" * 68)
    print()

    scenarios = [
        ("Accept their markup as-is", pre),
        ("Our counter-turn", post),
        ("Counter-turn, but concede the 3-year term",
         post.apply(Action("DP-18", Disposition.ACCEPT,
                           Position("three (3) years", -15, Tier.HARD_LINE),
                           "conceded"))),
        ("Counter-turn, but offer the standstill fallback now",
         post.apply(Action("DP-16", Disposition.PARTIAL,
                           Position("6-month standstill, fall-away on third-party bid, no DADW",
                                    -25, Tier.ESCALATION),
                           "fallback offered in first turn"))),
        ("Counter-turn, but drop residuals to buy goodwill",
         post.apply(Action("DP-07", Disposition.ACCEPT,
                           Position("Section 6 deleted", 0, Tier.HARD_LINE),
                           "conceded"))),
        ("Counter-turn, but accept JAMS arbitration",
         post.apply(Action("DP-20", Disposition.ACCEPT,
                           Position("binding JAMS arbitration, Richmond VA", -30, Tier.HARD_LINE),
                           "conceded"))),
    ]

    base = score(post, GATING_RULES).overall
    print(f"  {'scenario':<52s} {'balance':>8s} {'vs plan':>9s}  admissible")
    print("  " + "-" * 82)
    for label, st in scenarios:
        r = score(st, GATING_RULES)
        d = r.overall - base
        adm = "yes" if r.admissible else f"NO ({len(r.hard_line_violations)} hard-line)"
        print(f"  {label:<52s} {r.overall:>+8.1f} {d:>+9.1f}  {adm}")
    print()
    print("  Sensitivity ranking -- what each deal point is worth to us, by mutation:")
    print()
    rows = []
    for pid, dp in post.deal_points.items():
        if dp.proposed is None:
            continue
        mutated = post.apply(Action(pid, Disposition.ACCEPT, dp.proposed, "mutation test"))
        delta = score(mutated, GATING_RULES).overall - base
        rows.append((abs(delta), pid, dp.name, delta))
    for _, pid, name, delta in sorted(rows, reverse=True):
        print(f"    {pid}  {name:<48s} {delta:+7.2f}")
    print()
    print("  This ranking is |dL/dx| -- it says what to fight for, in the model's units.")


COMMANDS = {
    "pre": cmd_pre,
    "post": cmd_post,
    "diff": cmd_diff,
    "brief": cmd_brief,
    "brief-blind": cmd_brief_blind,
    "checklist": cmd_checklist,
    "coverage": cmd_coverage,
    "sizes": cmd_sizes,
    "whatif": cmd_whatif,
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        return 1
    COMMANDS[sys.argv[1]]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
