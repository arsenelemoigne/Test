"""
How grave is each change?

This is the piece that was missing. We had two encoded documents and a diff, but
no answer to "how much does this markup matter" -- only a binary authority check.

The weights are NOT invented. The client's negotiation authority memo classifies
every issue itself, in its own headings:

    2.5  Walk-Away            11.3 Walk-Away
    4.2  Authorized Range     10.2 Escalation Requirement
    4.4  Walk-Away            14.2 Authorized Fallback
    5.1  Zero-Tolerance       14.3 Walk-Away
    6.1  Policy Maximum        7.2 Authorized Concession
    ...

So gravity is read out of the source document rather than assigned by an
engineer. That matters: a hand-tuned weight is a number a lawyer can argue with
and nobody can verify; a tier lifted from the memo is auditable against a line
of text the client wrote.

Gravity of a proposed change = tier of the issue x whether the counterparty
crossed the stated limit. Moving inside an authorised range is not grave even on
an important issue; breaching a walk-away is grave even on a small one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Tier(IntEnum):
    """The memo's own authority tiers, ordered."""
    NEGOTIABLE = 1        # "discretion ... within reasonable commercial bounds"
    AUTHORIZED = 2        # "Authorized Range" / "Authorized Concession" / "Fallback"
    ESCALATION = 3        # "Escalation Requirement" - VP, and for some the CPO too
    WALK_AWAY = 4         # "Walk-Away" - no authority to concede at the table


TIER_LABEL = {
    Tier.NEGOTIABLE: "negotiable",
    Tier.AUTHORIZED: "authorised range",
    Tier.ESCALATION: "escalation",
    Tier.WALK_AWAY: "WALK-AWAY",
}


@dataclass(frozen=True)
class Weight:
    slot_id: str
    tier: Tier
    memo_ref: str          # where in the memo this tier comes from
    crossed: bool          # did the counterparty's position breach the stated limit?
    note: str = ""

    @property
    def gravity(self) -> int:
        """0-8. Tier alone is potential exposure; crossing doubles it into actual."""
        return int(self.tier) * (2 if self.crossed else 1)


# Read out of carden-negotiation-authority-memo.txt. `crossed` is read out of the
# Luminos markup: did they land outside what the memo permits?
WEIGHTS: list[Weight] = [
    Weight("I01", Tier.WALK_AWAY, "memo 2.3, 2.5", True,
           "fee-multiple formulation is prohibited in any form; also uncapped above $3.5M"),
    Weight("I02", Tier.WALK_AWAY, "memo 4.1, 4.4", True,
           "36 months against an 18-month policy maximum that counsel cannot waive"),
    Weight("I03", Tier.WALK_AWAY, "memo 4.3, 4.4", True,
           "anonymisation-in-place is named as a walk-away"),
    Weight("I04", Tier.AUTHORIZED, "memo 4 (certification)", True,
           "certification weakened; no walk-away language but outside the draft position"),
    Weight("I05", Tier.WALK_AWAY, "memo 5.1, 5.3", True,
           "zero-tolerance; ANY exception is a walk-away absent Board action"),
    Weight("I06", Tier.WALK_AWAY, "memo 6.1, 6.3", True,
           "both dimensions breached, and the memo forbids trading one against the other"),
    Weight("I07", Tier.WALK_AWAY, "memo 7.1, 7.4", True,
           "pre-approved list grants access before VSA completion"),
    Weight("I08", Tier.ESCALATION, "memo 8.1", True,
           "regulatory cooperation must be preserved"),
    Weight("I09", Tier.ESCALATION, "memo 8.2, 8.3", True,
           "on-site capability non-negotiable; frequency itself is negotiable"),
    Weight("I10", Tier.ESCALATION, "memo 10.2, 10.3", True,
           "residuals escalate to VP *and* Chief Privacy Officer - Board-level policy"),
    Weight("I11", Tier.WALK_AWAY, "memo 9.1, 9.3", True,
           "SCCs alone, without prior approval and a TIA, is a walk-away"),
    Weight("I12", Tier.ESCALATION, "memo 13.1-13.3", True,
           "ML training and derivative-work ownership broadened"),
    Weight("I13", Tier.WALK_AWAY, "memo 14.1, 14.3", True,
           "Delaware law non-negotiable; may not be traded against venue"),
    Weight("I14", Tier.WALK_AWAY, "memo 14.3", True,
           "Maryland venue may not be accepted even if Delaware law is preserved"),
    # Added after the lexical channel flagged sections 12.1/12.2 as moved with no
    # slot to hold them -- and the memo turns out to govern them at s.12.
    Weight("I15", Tier.AUTHORIZED, "memo 12.1-12.3", True,
           "term 2->3 years, renewal 1->2 years, non-renewal notice 90->180 days; "
           "found by the recall channel, not by the playbook-derived schema, and "
           "absent from all 29 rubric criteria"),
]

WEIGHTS_BY_ID = {w.slot_id: w for w in WEIGHTS}


def report(names: dict[str, str]) -> str:
    """names: slot_id -> human name."""
    rows = sorted(WEIGHTS, key=lambda w: (-w.gravity, w.slot_id))
    L = ["GRAVITY OF EACH PROPOSED CHANGE",
         "(tier from the client's own authority memo; x2 where the counterparty "
         "crossed the stated limit)",
         "=" * 86, "",
         f"{'':<6}{'issue':<42}{'tier':<18}{'crossed':>8}{'gravity':>9}",
         "-" * 86]
    for w in rows:
        L.append(f"{w.slot_id:<6}{names.get(w.slot_id, '?')[:40]:<42}"
                 f"{TIER_LABEL[w.tier]:<18}{('yes' if w.crossed else 'no'):>8}{w.gravity:>9}")
    L += ["", f"total exposure: {sum(w.gravity for w in WEIGHTS)} "
              f"across {len(WEIGHTS)} issues; "
              f"{sum(1 for w in WEIGHTS if w.tier is Tier.WALK_AWAY)} are walk-aways", ""]
    L.append("Walk-aways, with the memo line each is read from:")
    for w in rows:
        if w.tier is Tier.WALK_AWAY:
            L.append(f"  {w.slot_id}  [{w.memo_ref}]  {w.note}")
    return "\n".join(L)
