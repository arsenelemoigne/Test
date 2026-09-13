"""
A worked example of the value bridge, with the numbers filled in by hand.

`python -m wm.run bridge --demo` runs the whole Shapley machinery on this and
spends nothing. It exists for two reasons:

  1. so the mechanism can be inspected and argued with before anyone pays a
     model to produce numbers, and
  2. so there is a fixture the arithmetic can be regression-tested against.

THE NUMBERS BELOW ARE HAND-SET AND ILLUSTRATIVE. They are not elicited, not
benchmarked, and not evidence about this deal. What is real is the structure:
the clauses are the actual sections of the Luminos markup, the default rules
named in each entry are the actual rules that govern in that clause's absence,
and the interaction list is where a lawyer would say "those two only work
together". Read the shape of the answer, not the digits.

Perspective: Carden Analytics, the data discloser. Values are USD per year,
positive = value to Carden. The aggregate cap in 14.1 is mutual; Carden's own
tail exposure as the source of the consumer data is larger than Luminos's, so a
mutual cap nets out positive for Carden. That is a judgement, and it is the
kind of judgement the whole exercise is meant to make visible and contestable.
"""

from __future__ import annotations

from .bridge import ClauseEffect, Interaction, Relationship


CLAUSES: list[ClauseEffect] = [
    ClauseEffect(
        id="C05", name="5.1 Data-Access Fee", category="commercial",
        with_clause=900_000, without_clause=120_000,
        default_rule="No express price term. Recovery in restitution / quantum "
                     "meruit for the reasonable value of data actually supplied "
                     "(cf. UCC 2-305 open price term), discounted heavily for "
                     "proof and collection risk."),
    ClauseEffect(
        id="C14", name="14.1 Aggregate Cap (mutual, $3.5M)", category="liability",
        with_clause=-290_000, without_clause=-1_150_000,
        default_rule="Uncapped compensatory damages, bounded only by "
                     "foreseeability (Hadley v Baxendale) and mitigation."),
    ClauseEffect(
        id="C14C", name="14.3 Carve-Outs from the cap", category="liability",
        with_clause=-180_000, without_clause=-40_000,
        default_rule="No carve-outs: the cap, if any, applies to everything.",
        # A carve-out from a deleted cap is not a clause. Without 14.1 this is a
        # null player, and no coalition containing it should be priced as if it
        # were operative.
        requires="C14"),
    ClauseEffect(
        id="C10", name="10.1 Audit Right", category="enforcement",
        with_clause=35_000, without_clause=0,
        default_rule="No right of inspection. Courts do not imply one into a "
                     "commercial supply agreement; discovery is available only "
                     "once litigation has started."),
    ClauseEffect(
        id="C10R", name="10.3 Notice / remote-only audit (Luminos markup)",
        category="enforcement",
        with_clause=-15_000, without_clause=0,
        default_rule="Absent this limitation the audit right in 10.1 is "
                     "unqualified as to method."),
    ClauseEffect(
        id="C07", name="7.1 Absolute Prohibition on Re-Identification",
        category="data protection",
        with_clause=260_000, without_clause=40_000,
        default_rule="FTC Act s.5 and state UDAP statutes reach deceptive "
                     "re-identification, but Carden has no privity remedy and "
                     "no control over whether a regulator acts."),
    ClauseEffect(
        id="C06", name="6.1 Retention Period", category="data protection",
        with_clause=95_000, without_clause=0,
        default_rule="No implied duty to delete. A recipient lawfully in "
                     "possession may retain indefinitely absent a term."),
    ClauseEffect(
        id="C08", name="8.2 Breach Notification Trigger", category="data protection",
        with_clause=70_000, without_clause=25_000,
        default_rule="State breach-notification statutes bind the holder of the "
                     "data and run to regulators and consumers, not to the "
                     "upstream source, and on their own statutory clocks."),
    ClauseEffect(
        id="C09", name="9.1 Subprocessor Prior Written Consent", category="data protection",
        with_clause=60_000, without_clause=0,
        default_rule="A party is free to subcontract performance absent a "
                     "restriction; only the duty, not the risk, stays home."),
    ClauseEffect(
        id="C12", name="12.4 Termination for Convenience (Luminos, 30 days)",
        category="commercial",
        with_clause=-120_000, without_clause=-20_000,
        default_rule="An agreement of indefinite duration is terminable on "
                     "reasonable notice; what is reasonable here would likely "
                     "exceed 30 days given the quarterly delivery cycle."),
    ClauseEffect(
        id="C15", name="15.1 Governing Law (Maryland, per markup)", category="procedural",
        with_clause=-25_000, without_clause=-10_000,
        default_rule="Forum and law determined by a conflicts analysis, with "
                     "the attendant cost and uncertainty of litigating it."),
    ClauseEffect(
        id="C13", name="13.1 Indemnification by Luminos", category="liability",
        with_clause=210_000, without_clause=0,
        default_rule="No indemnity. Carden bears its own defence costs and is "
                     "left to a damages claim, plus whatever contribution the "
                     "forum's apportionment rules allow."),
]


INTERACTIONS: list[Interaction] = [
    Interaction("C14", "C14C", 40_000,
                "Residual complementarity beyond the precedence gate: the pair "
                "is the tail protection, and reads as pure cost priced apart."),
    Interaction("C10", "C10R", -30_000,
                "Remote-only audits on notice cannot detect the thing 7.1 "
                "prohibits. The limitation guts the right it qualifies."),
    Interaction("C07", "C10", 45_000,
                "A prohibition you cannot inspect for is a prohibition you "
                "cannot prove was broken."),
    Interaction("C07", "C13", 60_000,
                "The prohibition is worth more when a breach of it triggers a "
                "defence-and-hold-harmless rather than a damages claim."),
    Interaction("C06", "C08", 25_000,
                "A deletion deadline shrinks the window the notification duty "
                "has to cover; each makes the other cheaper to live with."),
    Interaction("C12", "C05", -90_000,
                "A 30-day exit at will turns the annual fee into a 30-day fee. "
                "The price term is worth what the term commitment makes it worth."),
    Interaction("C13", "C14", -300_000,
                "The indemnity is subject to the aggregate cap. Most of what it "
                "would pay is already inside the cap, so holding both is worth "
                "far less than the sum of the two."),
]


def relationship() -> Relationship:
    return Relationship(CLAUSES, INTERACTIONS)
