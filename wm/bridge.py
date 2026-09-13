"""
What is the contract WORTH, measured against not having one?

This replaces the absolute score, which Cox's theorem shows cannot be trusted,
with a marginal quantity that can be:

    v(no contract)  ->  v(contract)  ->  v(contract + one change)

A DCF does not tell you what a company "is". It tells you what the cash flows
are worth against the alternative of not owning it. Same move here: a contract
has no absolute value, only a value relative to the relationship you would have
had without it.

THE BASELINE IS NOT ZERO, AND THIS IS THE WHOLE POINT

The absent-clause baseline is the hardest problem in Shapley-style attribution -
in machine learning it is unsolved and contested (what does a "missing feature"
even mean?). In contracts it has an exact answer:

    a clause being absent does not mean nothing happens.
    it means THE DEFAULT RULE APPLIES.

No limitation of liability clause means unlimited liability at common law. No
governing-law clause means conflict-of-laws analysis picks one. No termination
clause means the contract runs to term or is terminable on reasonable notice.
Gap-fillers are real law - UCC Art. 2 for goods, implied terms at common law,
droit suppletif in civil systems - and they are what the parties get if they
write nothing.

So the counterfactual is not hypothetical. It is a legal question with a real
answer, and it is exactly the anchor the EU/German/French tests already use:
"significant imbalance" under Dir. 93/13 art.3(1) is measured against the
dispositive law that would apply absent the clause (CJEU Aziz C-415/11), and
BGB s.307(2) presumes unreasonableness where a term departs from the
wesentliche Grundgedanken der gesetzlichen Regelung.

WHY SHAPLEY AND NOT JUST "VALUE WITH MINUS VALUE WITHOUT"

Because clauses interact. The naive marginal contribution of a liability cap
depends on whether you remove it before or after the carve-outs. Take-one-out
attribution double counts or under counts whenever value is non-additive.

The Shapley value is the unique attribution that (a) averages a clause's
marginal contribution over every possible order of adding clauses, and (b)
satisfies efficiency: the parts sum EXACTLY to v(contract) - v(no contract).
That second property is what makes the waterfall close, and it is why this is
the right formalism rather than a convenient one.
"""

from __future__ import annotations

import itertools
import json
import random
import re
from dataclasses import dataclass, asdict, field


# ---------------------------------------------------------------------------
# A clause, and what happens without it
# ---------------------------------------------------------------------------

@dataclass
class ClauseEffect:
    """One clause, valued against the default rule it displaces."""
    id: str
    name: str
    category: str

    with_clause: float      # value to us per year, in currency, with the clause in force
    without_clause: float   # value to us per year if this clause were simply absent
    default_rule: str       # WHAT governs in its absence - the legal answer, cited
    confident: bool = True

    @property
    def naive_delta(self) -> float:
        """Take-one-out contribution. Correct only if nothing interacts."""
        return self.with_clause - self.without_clause


@dataclass
class Interaction:
    """Value that exists only when two clauses are BOTH present (or both absent).

    Positive = the pair is worth more together than apart (complements).
    Negative = the pair overlaps; together they are worth less than the sum
    (redundancy), or together they compound a loss.
    """
    a: str
    b: str
    joint: float
    note: str = ""


# ---------------------------------------------------------------------------
# The characteristic function
# ---------------------------------------------------------------------------

class Relationship:
    """
    v(S) = what the relationship is worth with clause subset S in force and
    everything else on its default rule.

    v(empty set) is the no-contract baseline: every clause absent, every default
    rule applying. v(all) is the contract as drafted.
    """

    def __init__(self, clauses: list[ClauseEffect], interactions: list[Interaction] | None = None):
        self.clauses = {c.id: c for c in clauses}
        self.interactions = interactions or []

    def v(self, S) -> float:
        S = set(S)
        total = sum(c.with_clause if c.id in S else c.without_clause
                    for c in self.clauses.values())
        # interaction value accrues only when both members are in force
        for it in self.interactions:
            if it.a in S and it.b in S:
                total += it.joint
        return total

    @property
    def baseline(self) -> float:
        """v(no contract at all) - everything on default rules."""
        return self.v(set())

    @property
    def drafted(self) -> float:
        """v(the contract as it stands)."""
        return self.v(set(self.clauses))

    @property
    def surplus(self) -> float:
        """What the contract is worth. THE number, and it is a difference."""
        return self.drafted - self.baseline


# ---------------------------------------------------------------------------
# Shapley attribution
# ---------------------------------------------------------------------------

@dataclass
class Attribution:
    id: str
    name: str
    category: str
    shapley: float          # marginal contribution, averaged over all orderings
    naive: float            # take-one-out, for comparison
    stderr: float = 0.0

    @property
    def interaction_gap(self) -> float:
        """How much the naive number misleads. Large gap = this clause's value is
        entangled with others and cannot be quoted on its own."""
        return self.naive - self.shapley


def shapley(rel: Relationship, samples: int = 4000, seed: int = 0) -> list[Attribution]:
    """
    Monte Carlo permutation sampling (Castro et al.).

    Exact Shapley is 2^n - 27 clauses is 134 million subsets. Sampling random
    orderings and averaging each clause's marginal contribution converges at
    1/sqrt(samples), and the standard error is reported per clause so you can
    see whether a difference between two clauses is real.
    """
    rng = random.Random(seed)
    ids = list(rel.clauses)
    n = len(ids)
    sums = {i: 0.0 for i in ids}
    sumsq = {i: 0.0 for i in ids}

    for _ in range(samples):
        order = ids[:]
        rng.shuffle(order)
        S: set[str] = set()
        prev = rel.v(S)
        for cid in order:
            S.add(cid)
            cur = rel.v(S)
            marg = cur - prev
            sums[cid] += marg
            sumsq[cid] += marg * marg
            prev = cur

    out = []
    for cid in ids:
        mean = sums[cid] / samples
        var = max(sumsq[cid] / samples - mean * mean, 0.0)
        c = rel.clauses[cid]
        out.append(Attribution(cid, c.name, c.category, mean, c.naive_delta,
                               (var / samples) ** 0.5))
    return sorted(out, key=lambda a: a.shapley)


def check_efficiency(rel: Relationship, attrs: list[Attribution]) -> tuple[float, float, float]:
    """
    The Shapley axiom that makes the waterfall close: contributions sum exactly
    to v(all) - v(none). With sampling it should be near-exact; a large residual
    means too few samples.
    """
    total = sum(a.shapley for a in attrs)
    return total, rel.surplus, total - rel.surplus


# ---------------------------------------------------------------------------
# The three questions the user actually asks
# ---------------------------------------------------------------------------

def what_if(rel: Relationship, clause_id: str, new_with: float) -> float:
    """'contract + one change' - the value after moving one clause."""
    import copy
    r2 = Relationship(list(copy.deepcopy(rel.clauses).values()), rel.interactions)
    r2.clauses[clause_id].with_clause = new_with
    return r2.surplus


def ladder(rel: Relationship, changes: list[tuple[str, float, str]]) -> str:
    """
    no contract  ->  contract  ->  contract + change 1  ->  + change 2 ...

    changes: (clause_id, new with_clause value, label)
    """
    L = ["VALUE LADDER", "=" * 74, "",
         f"{'':<44}{'value/yr':>13}{'change':>13}", "-" * 74]
    L.append(f"{'no contract (default rules only)':<44}{_m(rel.baseline):>13}{'':>13}")
    prev = rel.baseline
    L.append(f"{'the contract as drafted':<44}{_m(rel.drafted):>13}"
             f"{_m(rel.drafted - prev):>13}")
    prev = rel.drafted

    import copy
    working = Relationship([copy.deepcopy(c) for c in rel.clauses.values()], rel.interactions)
    for cid, new_val, label in changes:
        if cid not in working.clauses:
            continue
        working.clauses[cid].with_clause = new_val
        cur = working.drafted
        L.append(f"{'  + ' + label[:40]:<44}{_m(cur):>13}{_m(cur - prev):>13}")
        prev = cur
    L += ["", f"The contract is worth {_m(rel.surplus)}/yr more than no contract.",
          "That difference is the only defensible number here. The absolute levels",
          "depend on the baseline assumptions and should not be quoted alone."]
    return "\n".join(L)


def waterfall(rel: Relationship, attrs: list[Attribution], top: int = 18) -> str:
    """From the no-contract baseline, clause by clause, to the drafted contract."""
    L = ["VALUE BRIDGE: no contract -> contract as drafted", "=" * 78, "",
         f"{'':<38}{'contribution':>14}{'+/-':>7}  {'naive':>11}  gap", "-" * 78,
         f"{'BASELINE (default rules only)':<38}{_m(rel.baseline):>14}", ""]
    shown = attrs[:top // 2] + attrs[-(top - top // 2):] if len(attrs) > top else attrs
    for a in shown:
        gap = a.interaction_gap
        flag = " <-- entangled" if abs(gap) > max(abs(a.shapley) * 0.3, 1.0) else ""
        L.append(f"  {a.name[:34]:<36}{_m(a.shapley):>14}{'±' + _m(a.stderr):>7}"
                 f"  {_m(a.naive):>11}  {_m(gap):>8}{flag}")
    if len(attrs) > top:
        L.append(f"  ... {len(attrs) - len(shown)} clauses omitted")
    total, surplus, resid = check_efficiency(rel, attrs)
    L += ["", f"{'CONTRACT AS DRAFTED':<38}{_m(rel.drafted):>14}", "",
          f"  sum of contributions {_m(total)}  vs  v(contract) - v(no contract) "
          f"{_m(surplus)}   residual {_m(resid)}"]
    if abs(resid) > max(abs(surplus) * 0.01, 1.0):
        L.append("  residual is large - increase `samples`")
    else:
        L.append("  the bridge closes: Shapley's efficiency axiom holds to sampling error")
    entangled = [a for a in attrs if abs(a.interaction_gap) > max(abs(a.shapley) * 0.3, 1.0)]
    if entangled:
        L += ["", f"{len(entangled)} clauses are ENTANGLED - their take-one-out value differs",
              "materially from their true contribution, because it depends on which",
              "other clauses are present. Never quote a take-one-out figure for these:"]
        for a in entangled[:8]:
            L.append(f"    {a.name[:44]:<46}naive {_m(a.naive):>10}  true {_m(a.shapley):>10}")
    return "\n".join(L)


def _m(x: float) -> str:
    a = abs(x)
    s = "-" if x < 0 else ""
    if a >= 1_000_000:
        return f"{s}{a/1_000_000:.2f}M"
    if a >= 1_000:
        return f"{s}{a/1_000:.0f}k"
    return f"{s}{a:.0f}"


# ---------------------------------------------------------------------------
# Eliciting the two numbers per clause
# ---------------------------------------------------------------------------

BASELINE_PROMPT = """For each contract clause below, answer two questions for {party}.

(1) WHAT IS THE DEFAULT RULE? If this clause were simply deleted and the parties
    wrote nothing in its place, what would govern? Name the actual fallback: a
    statutory gap-filler, an implied term, a conflict-of-laws outcome, "unlimited
    liability at common law", "terminable on reasonable notice", and so on. If
    deleting the clause genuinely leaves nothing - no default applies - say so.

(2) WHAT IS EACH STATE WORTH per year, in currency, to {party}?
      "with_clause"     the expected annual value with the clause as drafted
      "without_clause"  the expected annual value under the default rule instead
    Both are signed: negative means the state costs {party} money. These are
    expected values - probability-weighted, not worst cases.

The DIFFERENCE between them is what the clause buys. A clause that merely restates
the default rule has with_clause ~= without_clause and is worth nothing, however
important it sounds. Say that plainly when it is true.

Deal context: {context}

Return ONLY a JSON array:
[{{"id": "...", "default_rule": "...", "with_clause": 0, "without_clause": 0,
   "confident": true}}]

"confident": false wherever the figure depends on facts you do not have.

=== CLAUSES ===
{clauses}"""


INTERACTION_PROMPT = """Which pairs of these clauses have value that exists ONLY when
both are in force?

Report a pair only where the joint effect is real and material - where the two
together are worth meaningfully more, or meaningfully less, than the sum of their
separate effects.

  positive "joint"  complements: together they are worth MORE than separately
                    (an audit right plus a records-retention covenant)
  negative "joint"  redundancy or compounding: together they are worth LESS
                    (two remedies covering the same loss; or a broad data-use
                    right plus a long retention period, which compound a loss)

"joint" is in the same currency per year as the clause values.

Ten real pairs are worth more than fifty speculative ones. Do not pair clauses
merely because they sit in the same section.

Return ONLY a JSON array:
[{{"a": "...", "b": "...", "joint": 0, "note": "one sentence"}}]

=== CLAUSES ===
{clauses}"""


def _json_array(raw: str) -> list[dict]:
    m = re.search(r"\[.*\]", raw or "", re.S)
    if not m:
        raise ValueError("no JSON array in model output")
    return json.loads(m.group(0))


def elicit(decisions, llm, party: str, context: str) -> list[ClauseEffect]:
    payload = "\n".join(
        f"{d.id} [{d.category}, s.{d.section}] {d.name}: {d.value[:200]}" for d in decisions)
    rows = _json_array(llm(BASELINE_PROMPT.format(party=party, context=context,
                                                  clauses=payload), max_tokens=32000))
    by_id = {d.id: d for d in decisions}
    out = []
    for r in rows:
        d = by_id.get(r.get("id"))
        if d is None:
            continue
        try:
            out.append(ClauseEffect(
                id=d.id, name=d.name, category=d.category,
                with_clause=float(r["with_clause"]),
                without_clause=float(r["without_clause"]),
                default_rule=r.get("default_rule", ""),
                confident=bool(r.get("confident", False))))
        except (KeyError, ValueError, TypeError):
            continue
    return out


def elicit_interactions(effects: list[ClauseEffect], llm) -> list[Interaction]:
    payload = "\n".join(f"{c.id} [{c.category}] {c.name}" for c in effects)
    ids = {c.id for c in effects}
    try:
        rows = _json_array(llm(INTERACTION_PROMPT.format(clauses=payload), max_tokens=16000))
    except ValueError:
        return []
    out = []
    for r in rows:
        try:
            if r["a"] in ids and r["b"] in ids and r["a"] != r["b"]:
                out.append(Interaction(r["a"], r["b"], float(r["joint"]), r.get("note", "")))
        except (KeyError, ValueError, TypeError):
            continue
    return out


def dump(effects: list[ClauseEffect], interactions: list[Interaction]) -> str:
    return json.dumps({"clauses": [asdict(c) for c in effects],
                       "interactions": [asdict(i) for i in interactions]}, indent=2)
