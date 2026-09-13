"""
Core engine for a contract world model.

A world model needs three things:
  STATE       -- an encoding of the contract you can inspect and compare
  TRANSITION  -- apply(state, action) -> state'   (a redline is an action)
  VALUE       -- score(state) -> BalanceReport    (how good is this for us)

Nothing here is NDA-specific. The NDA instantiation lives in `velantis_nda.py`.

Design decisions (see project notes):
  * The latent space is DESIGNED, not learned. Each DealPoint is one interpretable
    dimension. Favourability is signed and anchored at a reference point.
  * The anchor (favourability 0) is the MARKET-STANDARD position for this kind of
    instrument, not our own template. Anchoring on our template only measures
    "how much did they change our paper", which is not balance.
  * Authority tiers come from the negotiation playbook, which is a task input.
    They are what makes some moves forbidden regardless of score.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable


# --------------------------------------------------------------------------
# Authority tiers -- lifted directly from the Velantis playbook, Section 1.1
# --------------------------------------------------------------------------

class Tier(Enum):
    """Playbook authority tier for a given position."""

    PREFERRED = "preferred"          # our form language; opening position
    ACCEPTABLE = "acceptable"        # may be accepted without escalation
    ESCALATION = "escalation"        # needs VP/GC sign-off before accepting
    HARD_LINE = "hard_line"          # will not concede under any circumstances

    @property
    def rank(self) -> int:
        return {"preferred": 3, "acceptable": 2, "escalation": 1, "hard_line": 0}[self.value]

    @property
    def conceivable(self) -> bool:
        """Can an agent accept this position on its own authority?"""
        return self in (Tier.PREFERRED, Tier.ACCEPTABLE)


class Disposition(Enum):
    """What we do with the counterparty's proposed change to a deal point."""

    ACCEPT = "accept"            # take their language as-is
    REJECT = "reject"            # revert to our base form
    PARTIAL = "partial"          # accept with a qualifier we draft
    COUNTER = "counter"          # reject and propose different language
    RETAIN = "retain"            # they touched nothing; keep base form
    UNRESOLVED = "unresolved"    # not yet decided -- an incomplete state


class Holder(Enum):
    """Which side a position favours."""

    US = "us"
    THEM = "them"
    NEUTRAL = "neutral"


# --------------------------------------------------------------------------
# Provenance -- every value must point at text. No span, no value.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Provenance:
    document: str
    locator: str          # section number, or line ref in the extracted text
    quote: str            # verbatim supporting text

    def __str__(self) -> str:
        q = self.quote if len(self.quote) <= 90 else self.quote[:87] + "..."
        return f"{self.document} @ {self.locator}: “{q}”"


@dataclass(frozen=True)
class Position:
    """One party's position on one deal point."""

    label: str                       # human-readable: "3 years", "highest degree of care"
    favourability: int               # -100..+100, signed toward US, 0 = market standard
    tier: Tier                       # playbook classification of this position
    provenance: Provenance | None = None

    def __str__(self) -> str:
        sign = "+" if self.favourability > 0 else ""
        return f"{self.label} [{sign}{self.favourability}, {self.tier.value}]"


# --------------------------------------------------------------------------
# Deal points -- the dimensions of the latent space
# --------------------------------------------------------------------------

@dataclass
class DealPoint:
    """One negotiable dimension of the contract."""

    id: str
    name: str
    dimension: str                   # which balance dimension it rolls up into
    playbook_ref: str                # e.g. "Section 5 -- Confidentiality Term"
    weight: float                    # relative economic/strategic weight

    base: Position                   # our form (v6.2) position
    proposed: Position | None = None # what the counterparty asked for; None = untouched
    current: Position | None = None  # where the deal point stands in this state

    disposition: Disposition = Disposition.UNRESOLVED
    rationale: str = ""
    instruction: str = ""            # client instruction, if any, from the email

    # Interaction hooks: other deal points whose value changes this one's meaning.
    couples_with: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.current is None:
            self.current = self.base

    @property
    def touched(self) -> bool:
        return self.proposed is not None

    @property
    def delta(self) -> int:
        """Movement from our form to where we currently stand. Negative = we conceded."""
        return self.current.favourability - self.base.favourability

    @property
    def exposure(self) -> int:
        """What we'd lose if we simply accepted their markup."""
        if self.proposed is None:
            return 0
        return self.proposed.favourability - self.base.favourability

    def violates_hard_line(self) -> bool:
        return self.current.tier is Tier.HARD_LINE

    def needs_escalation(self) -> bool:
        return self.current.tier is Tier.ESCALATION


# --------------------------------------------------------------------------
# Actions -- the transition function
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Action:
    """A single redline move on a single deal point."""

    deal_point_id: str
    disposition: Disposition
    result: Position
    rationale: str = ""

    def __str__(self) -> str:
        return f"{self.deal_point_id}: {self.disposition.value.upper()} -> {self.result}"


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------

@dataclass
class ContractState:
    """The world model's state: the contract as a vector of deal points."""

    name: str
    instrument: str                            # which document this state describes
    deal_points: dict[str, DealPoint]
    parties: dict[str, str] = field(default_factory=dict)
    facts: dict[str, str] = field(default_factory=dict)   # dates, code names, entity details
    confidential: tuple[str, ...] = ()         # facts that must NOT leave the building

    # ---- transition ------------------------------------------------------

    def apply(self, action: Action) -> "ContractState":
        """Pure transition. Returns a new state; does not mutate."""
        if action.deal_point_id not in self.deal_points:
            raise KeyError(f"unknown deal point: {action.deal_point_id}")
        new_points = dict(self.deal_points)
        dp = new_points[action.deal_point_id]
        new_points[action.deal_point_id] = replace(
            dp,
            current=action.result,
            disposition=action.disposition,
            rationale=action.rationale or dp.rationale,
        )
        return replace(self, deal_points=new_points)

    def apply_all(self, actions: list[Action]) -> "ContractState":
        state = self
        for a in actions:
            state = state.apply(a)
        return state

    # ---- helpers ---------------------------------------------------------

    def by_dimension(self) -> dict[str, list[DealPoint]]:
        out: dict[str, list[DealPoint]] = {}
        for dp in self.deal_points.values():
            out.setdefault(dp.dimension, []).append(dp)
        return out

    def touched(self) -> list[DealPoint]:
        return [dp for dp in self.deal_points.values() if dp.touched]

    def unresolved(self) -> list[DealPoint]:
        return [
            dp for dp in self.deal_points.values()
            if dp.touched and dp.disposition is Disposition.UNRESOLVED
        ]


# --------------------------------------------------------------------------
# Value function
# --------------------------------------------------------------------------

@dataclass
class DimensionScore:
    dimension: str
    score: float
    weight: float
    points: list[DealPoint]


@dataclass
class BalanceReport:
    state_name: str
    overall: float
    dimensions: list[DimensionScore]
    hard_line_violations: list[DealPoint]
    escalations: list[DealPoint]
    unresolved: list[DealPoint]
    gating_notes: list[str]

    @property
    def admissible(self) -> bool:
        """Can this state be sent to the counterparty on the agent's own authority?"""
        return not self.hard_line_violations and not self.unresolved

    def render(self, verbose: bool = False) -> str:
        lines: list[str] = []
        lines.append(f"BALANCE REPORT -- {self.state_name}")
        lines.append("=" * 72)
        sign = "+" if self.overall > 0 else ""
        lines.append(f"OVERALL BALANCE: {sign}{self.overall:.1f}   (0 = market standard, + = favours Velantis)")
        lines.append("")
        lines.append("BY DIMENSION")
        for d in sorted(self.dimensions, key=lambda x: x.score):
            bar = _bar(d.score)
            s = "+" if d.score > 0 else ""
            lines.append(f"  {d.dimension:<28s} {s}{d.score:6.1f}  {bar}")
            if verbose:
                for dp in sorted(d.points, key=lambda p: p.current.favourability):
                    mark = "*" if dp.touched else " "
                    lines.append(f"      {mark} {dp.id:<8s} {dp.name:<44s} {dp.current}")
        lines.append("")
        if self.hard_line_violations:
            lines.append("HARD-LINE VIOLATIONS  (cannot be sent -- playbook forbids)")
            for dp in self.hard_line_violations:
                lines.append(f"  !! {dp.id} {dp.name}: {dp.current.label}  [{dp.playbook_ref}]")
            lines.append("")
        if self.escalations:
            lines.append("REQUIRES ESCALATION  (VP/GC sign-off before accepting)")
            for dp in self.escalations:
                lines.append(f"  ^  {dp.id} {dp.name}: {dp.current.label}  [{dp.playbook_ref}]")
            lines.append("")
        if self.unresolved:
            lines.append("UNRESOLVED  (counterparty moved; we have not responded)")
            for dp in self.unresolved:
                lines.append(f"  ?  {dp.id} {dp.name}")
            lines.append("")
        if self.gating_notes:
            lines.append("INTERACTION / GATING NOTES")
            for n in self.gating_notes:
                lines.append(f"  -> {n}")
            lines.append("")
        lines.append(f"ADMISSIBLE: {'YES' if self.admissible else 'NO'}")
        return "\n".join(lines)


def _bar(score: float, width: int = 24) -> str:
    """A simple signed bar around a zero centre."""
    half = width // 2
    n = int(round(abs(score) / 100 * half))
    n = min(n, half)
    if score >= 0:
        return " " * half + "|" + "#" * n
    return " " * (half - n) + "#" * n + "|"


GatingRule = Callable[[ContractState], str | None]


def score(state: ContractState, gating_rules: list[GatingRule] | None = None) -> BalanceReport:
    """
    Value function.

    Deliberately NOT a plain weighted sum of favourabilities: gating rules let a
    combination of deal points cap or override the naive aggregate, which is how
    the real interaction effects get in without a Monte Carlo simulator.
    """
    dims: list[DimensionScore] = []
    for dimension, points in state.by_dimension().items():
        total_w = sum(p.weight for p in points) or 1.0
        s = sum(p.current.favourability * p.weight for p in points) / total_w
        dims.append(DimensionScore(dimension, s, total_w, points))

    total_w = sum(d.weight for d in dims) or 1.0
    overall = sum(d.score * d.weight for d in dims) / total_w

    notes: list[str] = []
    for rule in (gating_rules or []):
        note = rule(state)
        if note:
            notes.append(note)

    return BalanceReport(
        state_name=state.name,
        overall=overall,
        dimensions=dims,
        hard_line_violations=[p for p in state.deal_points.values() if p.violates_hard_line()],
        escalations=[p for p in state.deal_points.values() if p.needs_escalation()],
        unresolved=state.unresolved(),
        gating_notes=notes,
    )


# --------------------------------------------------------------------------
# Diff -- the thing an LLM actually needs in context while rewriting a clause
# --------------------------------------------------------------------------

def diff(before: ContractState, after: ContractState) -> str:
    lines = [f"DIFF: {before.name}  ->  {after.name}", "=" * 72]
    moved = 0
    for pid, b in before.deal_points.items():
        a = after.deal_points[pid]
        if a.current.label == b.current.label:
            continue
        moved += 1
        d = a.current.favourability - b.current.favourability
        sign = "+" if d > 0 else ""
        lines.append(f"  {pid}  {a.name}")
        lines.append(f"        from: {b.current.label}")
        lines.append(f"          to: {a.current.label}")
        lines.append(f"        move: {sign}{d}   [{a.disposition.value}]")
        if a.rationale:
            lines.append(f"          why: {a.rationale}")
        lines.append("")
    before_score = score(before).overall
    after_score = score(after).overall
    lines.append(f"{moved} deal points moved.")
    lines.append(f"BALANCE: {before_score:+.1f}  ->  {after_score:+.1f}   ({after_score - before_score:+.1f})")
    return "\n".join(lines)
