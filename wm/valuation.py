"""
A richer value model: decomposed severity, typed inter-clause couplings, and
swappable value profiles.

WHAT THIS MODEL IS CALLED

It is not novel, and that is good news. Value = per-clause terms + pairwise
interaction terms is a **GAI decomposition of order 2**, equivalently a
**2-additive Choquet integral**, equivalently a pairwise factor graph over
clauses. Thirty years of decision theory applies to it, and every parameter has
a name.

  Robu, Somefun & La Poutre, "Modeling complex multi-issue negotiations using
  utility graphs" (AAMAS 2005) - utility over contract issues as node terms plus
  edge terms on an interdependency graph. This exact model, for contracts, in 2005.
  Bacchus & Grove (UAI-95) - conditional additive independence has an exact
  representation as separation in an undirected graph. The licence for drawing a
  clause interaction graph at all.
  Grabisch, 2-additive Choquet - gives the pairwise term a named, signed meaning
  (the Shapley interaction index: >0 synergy, <0 redundancy).
  Keeney & Raiffa - why plain addition is not available here: it requires mutual
  preferential independence, which contracts violate structurally (a liability
  cap has no value where there is no exposure). Their multiplicative fallback
  adds exactly ONE global interaction parameter, too coarse to say "cap and
  indemnity are complements while arbitration and forum are redundant".

TWO KINDS OF INTERACTION, AND ONLY ONE IS BILINEAR

This distinction matters more than the formula:

  CONDITIONAL / STRUCTURAL   B is inoperative unless A holds. A cap with the
                             relevant claim carved out is not "worth less" - it
                             does not apply. Modelled here as GATES and REQUIRES,
                             which are multiplicative gates, NOT cross terms.
                             A bilinear w_ij term fits this badly.
  VALUE INTERACTION          both adverse is worse than the sum, or two terms
                             cover the same loss. Modelled as AMPLIFIES and
                             SUBSTITUTES. This is the genuinely bilinear part.

Most apparent clause interactions in commercial contracts are the first kind.

NEVER FIT A DENSE INTERACTION MATRIX. n clauses give n(n-1)/2 cross terms - 27
clauses is 351 parameters, unidentifiable against any corpus you will have. The
couplings here are a sparse, expert-specified graph (about a dozen edges), which
is the route the utility-graph literature settled on.

WHY A SINGLE 0-10 IMPACT IS NOT ENOUGH

One number per clause cannot express any of these, and all four are ordinary:

  * a liability cap is worth nothing if the claim you actually face is carved
    out of it                                                      -> GATES
  * broad data-use scope x long retention x weak deletion is worse than the sum
    of the three                                                   -> AMPLIFIES
  * liquidated damages OR injunctive relief; you do not need both  -> SUBSTITUTES
  * an audit right is inert without an on-site modality            -> REQUIRES

So severity is decomposed into factors a model can actually assess separately,
and the couplings are first-class typed edges rather than something baked into a
scalar.

WHY PROFILES

The same contract is worth different things to different parties at different
times: buy-side vs sell-side, discloser vs recipient, a firm that has just been
through a data-protection audit vs one that has not. A profile is a set of
multipliers over the encoded contract. Re-scoring under a new profile is a
matrix multiply, not a re-read.

This is the one capability that is not available from a prose analysis at any
quality. Text cannot be re-weighted. If your risk appetite changes, a written
memo must be rewritten; an encoded contract is the same object with a different
profile applied. Whether structure produces *better drafting* is the open
question the A2-vs-A4 arms measure. Whether it produces *re-weightable* drafting
is not in question - it does, and text does not.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from enum import Enum


# ---------------------------------------------------------------------------
# Per-clause assessment: three factors, not one score
# ---------------------------------------------------------------------------

@dataclass
class Assessment:
    id: str
    name: str
    section: str
    category: str

    exposure: int        # 0-10  how much is at stake when this clause bites
    likelihood: int      # 0-10  how often it actually bites in a deal like this
    reversibility: int   # 0-10  how easily fixed later (10 = trivially renegotiated)
    mutuality: str       # "mutual" | "one_way_us" | "one_way_them"
    favours: str         # "us" | "them" | "neutral"
    basis: str
    confident: bool

    @property
    def severity(self) -> float:
        """
        Unsigned magnitude, 0-10.

        exposure x likelihood is the expected-loss shape (how bad x how often).
        Reversibility discounts it: a term you can renegotiate next year costs
        less than one that binds for the term. One-way terms count for more than
        mutual ones, because a mutual term is a cost you can also impose.
        """
        base = (self.exposure * self.likelihood) / 10.0
        base *= 1.0 - (self.reversibility / 25.0)          # up to a 40% discount
        if self.mutuality == "one_way_them":
            base *= 1.25
        elif self.mutuality == "mutual":
            base *= 0.85
        return round(min(base, 10.0), 2)

    @property
    def signed(self) -> float:
        return {"us": 1.0, "them": -1.0, "neutral": 0.0}[self.favours] * self.severity


# ---------------------------------------------------------------------------
# Couplings: the part a flat list of scores cannot represent
# ---------------------------------------------------------------------------

class Kind(str, Enum):
    GATES = "GATES"            # source adverse -> target's protection is neutralised
    AMPLIFIES = "AMPLIFIES"    # both adverse -> worse than the sum
    SUBSTITUTES = "SUBSTITUTES"  # either suffices -> do not double count
    REQUIRES = "REQUIRES"      # source is inert unless target holds


@dataclass
class Coupling:
    source: str
    target: str
    kind: Kind
    strength: float = 1.0      # 0-1, how completely the effect applies
    note: str = ""


# ---------------------------------------------------------------------------
# Profiles: many value distributions over one encoded contract
# ---------------------------------------------------------------------------

@dataclass
class Profile:
    name: str
    category_weights: dict[str, float] = field(default_factory=dict)
    overrides: dict[str, float] = field(default_factory=dict)   # per-clause multiplier
    risk_aversion: float = 1.0   # >1 penalises adverse terms superlinearly

    def multiplier(self, a: Assessment) -> float:
        return (self.category_weights.get(a.category, 1.0)
                * self.overrides.get(a.id, 1.0))


NEUTRAL = Profile("neutral")

DATA_DISCLOSER = Profile(
    "data discloser, post-audit",
    category_weights={"data": 1.6, "risk_allocation": 1.3, "money": 1.0,
                      "rights": 1.0, "time": 0.9, "process": 0.7, "boilerplate": 0.3},
    risk_aversion=1.4,
)

COST_FOCUSED = Profile(
    "cost-focused, low regulatory exposure",
    category_weights={"money": 1.8, "time": 1.2, "risk_allocation": 0.9,
                      "data": 0.6, "rights": 0.8, "process": 0.6, "boilerplate": 0.3},
    risk_aversion=1.0,
)

SPEED_FOCUSED = Profile(
    "speed to signature",
    category_weights={"process": 1.4, "time": 1.3, "money": 0.8, "data": 0.8,
                      "risk_allocation": 0.7, "rights": 0.7, "boilerplate": 0.2},
    risk_aversion=0.8,
)

PROFILES = {p.name: p for p in (NEUTRAL, DATA_DISCLOSER, COST_FOCUSED, SPEED_FOCUSED)}


# ---------------------------------------------------------------------------
# Scoring: apply couplings, then the profile
# ---------------------------------------------------------------------------

@dataclass
class Contribution:
    id: str
    name: str
    category: str
    raw: float          # signed severity before couplings
    coupled: float      # after couplings
    final: float        # after the profile
    notes: list[str] = field(default_factory=list)


def score(assessments: list[Assessment], couplings: list[Coupling],
          profile: Profile = NEUTRAL) -> tuple[float, list[Contribution]]:
    by_id = {a.id: a for a in assessments}
    val = {a.id: a.signed for a in assessments}
    notes: dict[str, list[str]] = {a.id: [] for a in assessments}

    # GATES: an adverse source neutralises the target's protective value.
    for c in couplings:
        if c.kind is not Kind.GATES:
            continue
        src, tgt = by_id.get(c.source), by_id.get(c.target)
        if src is None or tgt is None or val[tgt.id] <= 0 or src.signed >= 0:
            continue
        before = val[tgt.id]
        val[tgt.id] *= 1.0 - c.strength
        notes[tgt.id].append(
            f"gated by {c.source} ({before:+.1f} -> {val[tgt.id]:+.1f}): {c.note}")

    # REQUIRES: a positive term is inert if what it depends on does not hold.
    for c in couplings:
        if c.kind is not Kind.REQUIRES:
            continue
        src, tgt = by_id.get(c.source), by_id.get(c.target)
        if src is None or tgt is None or val[src.id] <= 0 or tgt.signed >= 0:
            continue
        before = val[src.id]
        val[src.id] *= 1.0 - c.strength
        notes[src.id].append(
            f"inert without {c.target} ({before:+.1f} -> {val[src.id]:+.1f}): {c.note}")

    # SUBSTITUTES: keep the stronger of the pair, discount the other.
    for c in couplings:
        if c.kind is not Kind.SUBSTITUTES:
            continue
        a_, b_ = val.get(c.source), val.get(c.target)
        if a_ is None or b_ is None:
            continue
        weaker = c.source if abs(a_) <= abs(b_) else c.target
        before = val[weaker]
        val[weaker] *= 1.0 - c.strength
        notes[weaker].append(
            f"substituted by the stronger of the pair ({before:+.1f} -> {val[weaker]:+.1f})")

    # AMPLIFIES: both adverse compounds. Charged to both, half each.
    for c in couplings:
        if c.kind is not Kind.AMPLIFIES:
            continue
        a_, b_ = val.get(c.source), val.get(c.target)
        if a_ is None or b_ is None or a_ >= 0 or b_ >= 0:
            continue
        extra = c.strength * (abs(a_) * abs(b_)) ** 0.5
        val[c.source] -= extra / 2
        val[c.target] -= extra / 2
        notes[c.source].append(f"compounds with {c.target} (-{extra/2:.1f}): {c.note}")
        notes[c.target].append(f"compounds with {c.source} (-{extra/2:.1f}): {c.note}")

    out: list[Contribution] = []
    total = 0.0
    for a in assessments:
        coupled = val[a.id]
        f = coupled * profile.multiplier(a)
        if f < 0 and profile.risk_aversion != 1.0:
            f = -(abs(f) ** profile.risk_aversion)
        total += f
        out.append(Contribution(a.id, a.name, a.category,
                                round(a.signed, 2), round(coupled, 2), round(f, 2),
                                notes[a.id]))
    return round(total, 1), sorted(out, key=lambda c: c.final)


def across_profiles(assessments: list[Assessment],
                    couplings: list[Coupling]) -> str:
    """The same encoded contract under several value distributions.

    This is the capability text does not have: one encoding, many valuations,
    no re-reading.
    """
    L = ["THE SAME CONTRACT UNDER DIFFERENT VALUE DISTRIBUTIONS",
         "=" * 78, "",
         f"{'profile':<38}{'value':>10}{'worst clause':>28}", "-" * 78]
    for p in PROFILES.values():
        total, contribs = score(assessments, couplings, p)
        worst = contribs[0].name[:26] if contribs else "-"
        L.append(f"{p.name[:36]:<38}{total:>+10.1f}{worst:>28}")
    L += ["", "Rank order changes between profiles: that is the point. A prose",
          "analysis fixes one weighting at writing time and cannot be re-weighted."]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# The same model, written as linear algebra
# ---------------------------------------------------------------------------

def feature_matrix(assessments: list[Assessment]) -> tuple[list[str], list[list[float]]]:
    """
    Each clause IS a vector. This returns the N x 5 matrix of them.

        v_i = [exposure, likelihood, reversibility, mutuality, direction]

    Which makes the whole value model a quadratic form:

        V = sum_i  w_i . f(v_i)          <- clause-level terms
          + sum_ij C_ij . g(v_i, v_j)     <- coupling terms

    The first sum is what a per-clause score gives you. The second is the part a
    flat list cannot express, and C is the coupling matrix below. So "a vector
    per clause" and "a coupling matrix" are not two ideas - they are the linear
    and quadratic halves of one model.
    """
    MUT = {"one_way_them": -1.0, "mutual": 0.0, "one_way_us": 1.0}
    DIR = {"them": -1.0, "neutral": 0.0, "us": 1.0}
    rows = [[float(a.exposure), float(a.likelihood), float(a.reversibility),
             MUT.get(a.mutuality, 0.0), DIR.get(a.favours, 0.0)] for a in assessments]
    return [a.id for a in assessments], rows


FEATURE_NAMES = ["exposure", "likelihood", "reversibility", "mutuality", "direction"]


def coupling_matrix(assessments: list[Assessment],
                    couplings: list[Coupling]) -> tuple[list[str], list[list[float]]]:
    """
    The N x N interaction matrix C. Signed by kind so the heatmap is readable:
    GATES and REQUIRES are destructive (negative), AMPLIFIES compounds
    (negative), SUBSTITUTES is redundancy (positive - it means you are double
    counting, not that it is good).
    """
    ids = [a.id for a in assessments]
    idx = {i: n for n, i in enumerate(ids)}
    M = [[0.0] * len(ids) for _ in ids]
    SIGN = {Kind.GATES: -1.0, Kind.REQUIRES: -1.0,
            Kind.AMPLIFIES: -1.0, Kind.SUBSTITUTES: 1.0}
    for c in couplings:
        if c.source in idx and c.target in idx:
            M[idx[c.source]][idx[c.target]] = SIGN[c.kind] * c.strength
    return ids, M


def render(total: float, contribs: list[Contribution], profile: Profile,
           top: int = 20) -> str:
    L = [f"CONTRACT VALUE: {total:+.1f}   [profile: {profile.name}]",
         "(ordinal, signed toward us; comparable only between versions of THIS contract)",
         "", f"{'':<8}{'clause':<42}{'raw':>7}{'coupled':>9}{'final':>8}", "-" * 76]
    for c in contribs[:top]:
        L.append(f"{c.id:<8}{c.name[:40]:<42}{c.raw:>7.1f}{c.coupled:>9.1f}{c.final:>8.1f}")
    moved = [c for c in contribs if abs(c.coupled - c.raw) > 0.05]
    if moved:
        L += ["", f"{len(moved)} clauses changed value because of their couplings:", ""]
        for c in moved[:12]:
            for n in c.notes:
                L.append(f"  {c.id}  {n}")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Model-facing prompts
# ---------------------------------------------------------------------------

ASSESS_PROMPT = """Assess each contract decision below for {party}.

You have NO access to {party}'s internal policies, risk appetite, or negotiation
authority. Judge only from general commercial and legal experience.

Score three things separately - do not collapse them.

EVERY SCALE POINT IS ANCHORED TO A NAMED EXAMPLE. Use the anchors. Unanchored
magnitude scales are extremely noisy even between raters who do not actually
disagree ("Noisy law: scaling without a modulus", J. Risk & Uncertainty 2024),
so a number without a referent is not a measurement.

  "exposure"      how much is at stake WHEN this clause bites
                    0  = a notices provision naming the wrong floor of a building
                    3  = a 30-day payment term instead of 45
                    5  = an audit capped at once a year instead of twice
                    8  = a liability cap set at 3x fees instead of a fixed sum
                   10  = unlimited liability for a data breach, or a term that
                         would make the deal unsignable

  "likelihood"    how often it actually bites in a deal of this kind
                    0  = has never been invoked in this kind of agreement
                    3  = invoked in an unusual deal, perhaps 1 in 20
                    5  = invoked in a meaningful minority of deals
                    8  = invoked in most deals that run their full term
                   10  = operative from signature, every deal, unavoidably

                  A catastrophic clause that almost never triggers scores HIGH
                  exposure and LOW likelihood. Say so rather than averaging.

  "reversibility" how easily this could be fixed later
                    0  = perpetual, survives termination, no amendment right
                    3  = fixed for the initial term
                    5  = revisited at renewal
                    8  = amendable on notice
                   10  = either party may change it at will

Then:
  "mutuality"  "mutual" | "one_way_us" | "one_way_them"   who it binds
  "favours"    "us" | "them" | "neutral"                  who the current value favours
  "basis"      one sentence, in market or exposure terms
  "confident"  true | false - FALSE where your answer depends on facts you do not
               have (deal size, sector norms, the party's risk appetite). Be honest;
               marking everything true is worse than useless.

Return ONLY a JSON array of objects with exactly those keys plus "id".

=== DECISIONS ===
{decisions}"""


COUPLE_PROMPT = """Identify where these contract clauses interact, so that their
combined effect is not the sum of their separate effects.

Four kinds:
  GATES        source being adverse NEUTRALISES the target's protective value.
               e.g. a carve-out list gates a liability cap.
  REQUIRES     source is inert unless the target holds.
               e.g. an audit right requires an on-site modality.
  SUBSTITUTES  either one suffices; counting both double-counts.
               e.g. liquidated damages vs injunctive relief.
  AMPLIFIES    both being adverse is worse than the sum.
               e.g. broad data-use scope plus long retention plus weak deletion.

Only report interactions that are real and material. Ten strong couplings are
worth more than fifty weak ones. Do not connect clauses merely because they are
in the same section.

Return ONLY a JSON array:
[{{"source": "D0xx", "target": "D0yy", "kind": "GATES",
   "strength": 0.8, "note": "one short sentence"}}]

strength is 0-1: how completely the effect applies.

=== CLAUSES ===
{decisions}"""


def _json_array(raw: str) -> list[dict]:
    m = re.search(r"\[.*\]", raw or "", re.S)
    if not m:
        raise ValueError("no JSON array in model output")
    return json.loads(m.group(0))


def assess(decisions, llm, party: str) -> list[Assessment]:
    payload = "\n".join(
        f"{d.id} [{d.category}, s.{d.section}] {d.name}: {d.value[:200]}" for d in decisions)
    rows = _json_array(llm(ASSESS_PROMPT.format(party=party, decisions=payload),
                           max_tokens=32000))
    by_id = {d.id: d for d in decisions}
    out = []
    for r in rows:
        d = by_id.get(r.get("id"))
        if d is None:
            continue
        try:
            out.append(Assessment(
                id=d.id, name=d.name, section=d.section, category=d.category,
                exposure=int(r["exposure"]), likelihood=int(r["likelihood"]),
                reversibility=int(r.get("reversibility", 5)),
                mutuality=r.get("mutuality", "mutual"),
                favours=r.get("favours", "neutral"),
                basis=r.get("basis", ""), confident=bool(r.get("confident", False))))
        except (KeyError, ValueError, TypeError):
            continue
    return out


def find_couplings(decisions, llm) -> list[Coupling]:
    payload = "\n".join(
        f"{d.id} [{d.category}] {d.name}: {d.value[:160]}" for d in decisions)
    try:
        rows = _json_array(llm(COUPLE_PROMPT.format(decisions=payload), max_tokens=16000))
    except ValueError:
        return []
    out = []
    ids = {d.id for d in decisions}
    for r in rows:
        try:
            if r["source"] not in ids or r["target"] not in ids:
                continue
            out.append(Coupling(r["source"], r["target"], Kind(r["kind"].upper()),
                                float(r.get("strength", 1.0)), r.get("note", "")))
        except (KeyError, ValueError):
            continue
    return out


def dump(assessments: list[Assessment], couplings: list[Coupling]) -> str:
    return json.dumps({"assessments": [asdict(a) for a in assessments],
                       "couplings": [{**asdict(c), "kind": c.kind.value} for c in couplings]},
                      indent=2)
