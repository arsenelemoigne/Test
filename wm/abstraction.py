"""
STEP 2 -- the abstraction.

The contract is reduced to a list of Issues. One Issue = one thing the parties
disagree about. That's it. This is the "world model": a small typed object you
can read, compare and validate, instead of 250KB of prose.

Built ONLY from documents the agent is given (initial draft, counterparty
markup, negotiation authority memo). The 29 rubric criteria were not consulted.
"""

from __future__ import annotations

import re

from dataclasses import dataclass, field
from enum import Enum


class Disposition(str, Enum):
    ACCEPT = "ACCEPT"   # take their language
    REJECT = "REJECT"   # revert to our draft
    MODIFY = "MODIFY"   # neither; we propose something else


@dataclass(frozen=True)
class Issue:
    id: str
    name: str
    section: str               # where it lives in the agreement
    ours: str                  # our initial draft position
    theirs: str                # what Luminos proposed
    theirs_quote: str          # verbatim, so the model can see the real words
    authority: str             # what the negotiation memo permits
    hard_limit: str | None = None   # machine-checkable ceiling, if any


@dataclass
class Decision:
    issue_id: str
    disposition: Disposition
    counter: str               # the position we are taking
    rationale: str             # why -- goes in the cover note


# ---------------------------------------------------------------------------
# The abstracted contract: Carden <-> Luminos Data Sharing Agreement
# ---------------------------------------------------------------------------

ISSUES: list[Issue] = [
    Issue(
        id="I01", name="Liability cap", section="14.1",
        ours="$2,000,000 fixed aggregate cap",
        theirs="greater of $2,000,000 or 3x fees paid in the preceding 12 months",
        theirs_quote="THE LESSER OF: (A) ACTUAL DIRECT DAMAGES ...; OR (B) {-TWO MILLION "
                     "DOLLARS-} {+THE GREATER OF TWO MILLION DOLLARS OR THREE TIMES THE FEES "
                     "PAID+}",
        authority="Fee-multiple formulations are prohibited in any form. A fixed dollar "
                  "aggregate cap is required. Ceiling $3,500,000. Insisting on a fee multiple "
                  "or a cap above $3.5M is a walk-away requiring escalation.",
        hard_limit="fixed_dollar_amount <= 3500000; no fee multiple",
    ),
    Issue(
        id="I02", name="Data retention period", section="6.1",
        ours="12 months from receipt of each quarterly delivery",
        theirs="36 months",
        theirs_quote="for a period not to exceed {-twelve (12) months-} {+thirty-six (36) "
                     "months+} from the date of receipt",
        authority="Third-Party Data Sharing Policy s.3.1 sets an absolute maximum of 18 months. "
                  "s.3.4 makes it non-negotiable: no deviation may be authorised by counsel. "
                  "36 months is a walk-away.",
        hard_limit="retention_months <= 18",
    ),
    Issue(
        id="I03", name="Deletion obligation", section="6.2",
        ours="mandatory secure deletion of all copies across all environments at expiry",
        theirs="anonymisation-in-place as an alternative to deletion; anonymised residual "
               "retained indefinitely",
        theirs_quote="(revised 6.2/6.4) anonymization-in-place ... retain the anonymized "
                     "residual data indefinitely",
        authority="Anonymisation-in-place is a walk-away. Anonymisation is not infallible and "
                  "re-identification risk grows over time; it would grant an indefinite "
                  "retention right. Mandatory deletion or return must be maintained.",
        hard_limit="mandatory_deletion == True",
    ),
    Issue(
        id="I04", name="Deletion certification", section="6.3",
        ours="written certification of destruction signed by an officer within 10 business days",
        theirs="weakened/removed certification requirement",
        theirs_quote="(revised 6.3) certification ... {-permanently and irreversibly deleted or "
                     "destroyed in accordance with the requirements-}",
        authority="Certification of deletion must be preserved. It is the only evidence Carden "
                  "has that deletion occurred.",
        hard_limit="certification_required == True",
    ),
    Issue(
        id="I05", name="Re-identification prohibition", section="7.2",
        ours="absolute prohibition, no exceptions; any attempt is a material breach",
        theirs="good-faith research exception; material breach only for activity outside that "
               "exception",
        theirs_quote="7.3 ... {+other than activity permitted under Section 7.2,+} shall "
                     "constitute a material breach",
        authority="Zero-tolerance Board-level policy. ANY exception, qualification, safe harbour "
                  "or carve-out is a walk-away absent Board action -- including intent-based "
                  "('shall not intentionally'), materiality-based, or technique-limited "
                  "reformulations. Must remain absolute, technique-agnostic, intent-independent.",
        hard_limit="no_exceptions == True",
    ),
    Issue(
        id="I06", name="Breach notification window and trigger", section="8.1, 8.2, 1.8",
        ours="24 hours after Discovery; Discovery = first reasonable suspicion, irrespective of "
             "confirmation",
        theirs="72 hours; Discovery redefined as confirmation through reasonable investigation",
        theirs_quote="no later than {-twenty-four (24)-} {+seventy-two (72)+} hours after "
                     "Discovery ... 1.8 'Discovery' means {+the point at which a Security "
                     "Incident has been confirmed through reasonable investigation+}",
        authority="Policy s.5.2 sets a 48-hour maximum after Discovery. Both the 72-hour window "
                  "and the confirmation trigger are outside authority, and the memo is explicit "
                  "that they may not be traded against each other: 48h with a confirmation "
                  "trigger is NOT an acceptable compromise, nor is 72h with a Discovery trigger.",
        hard_limit="notification_hours <= 48 AND trigger == 'discovery'",
    ),
    Issue(
        id="I07", name="Subprocessor pre-approved list", section="9.1, 9.2, 1.18",
        ours="prior written consent required for every subprocessor; VSA before consent",
        theirs="Schedule 1 'Pre-Approved Subprocessors' deemed approved at signing with no VSA",
        theirs_quote="1.18 'Pre-Approved Subprocessors' means those Subprocessors identified in "
                     "Schedule 1 ... deemed approved by Carden as of the Effective Date without "
                     "requirement for a Vendor-Security Assessment",
        authority="Policy s.6 requires every subprocessor to pass the Vendor Security Assessment "
                  "before access, and s.6.2 addresses pre-approved lists specifically. Any "
                  "mechanism granting access before VSA completion -- deemed approval, interim "
                  "access, conditional approval -- is a walk-away. VSA must precede access with "
                  "no exception.",
        hard_limit="vsa_required_for_all == True",
    ),
    Issue(
        id="I08", name="Audit: sharing findings with regulators", section="10.6",
        ours="unrestricted right to share audit findings with regulators; no Luminos consent",
        theirs="restriction on Carden's disclosure of audit findings to regulators",
        theirs_quote="10.6 Regulatory Disclosure. {-Nothing in this Agreement shall restrict ... "
                     "Carden's disclosure of audit findings ... to any applicable regulatory "
                     "authority-}",
        authority="Regulatory cooperation must be preserved. Carden is subject to banking and "
                  "data-protection supervision and cannot contract away its ability to share "
                  "findings with its regulators.",
        hard_limit="regulator_sharing_unrestricted == True",
    ),
    Issue(
        id="I09", name="Audit: on-site modality", section="10.1, 10.4",
        ours="on-site inspections, remote assessments and documentation reviews all available",
        theirs="audits limited to remote means; access to systems via remote only",
        theirs_quote="Audits may {-include ... on-site inspections at Luminos's facilities-} "
                     "{+be limited to remote+} ... access to its {+systems (via remote means)+}",
        authority="On-site audit capability must be preserved. Frequency is negotiable within "
                  "commercial bounds (twice per year may be reduced to once) but a minimum of "
                  "one audit per calendar year is non-negotiable.",
        hard_limit="onsite_audit_available == True",
    ),
    Issue(
        id="I10", name="Residuals clause", section="4.5, 1.19",
        ours="no residuals clause",
        theirs="new residuals clause: unaided-memory knowledge usable for any purpose, and "
               "residuals are not Confidential Information",
        theirs_quote="4.5 Residuals. Nothing in this Agreement shall restrict Luminos or its "
                     "personnel from using Residuals for any purpose ... The parties acknowledge "
                     "that Residuals do not constitute Confidential Information",
        authority="Policy position is against residuals clauses in data sharing agreements. "
                  "Residuals matters require escalation to both the VP and the Chief Privacy "
                  "Officer -- Board-level policy commitments. Recommended position: reject.",
        hard_limit="residuals_clause == False",
    ),
    Issue(
        id="I11", name="Cross-border transfer prerequisites", section="11.1, 11.2",
        ours="prior written Carden approval + Transfer Impact Assessment + SCCs (three-part "
             "prerequisite)",
        theirs="SCC execution alone sufficient; UK transfer deemed pre-authorised",
        theirs_quote="(c) Standard Contractual Clauses ... executed ... or such other transfer "
                     "mechanism as Carden deems adequate; 11.2 Obligation to Inform (30 days' "
                     "notice)",
        authority="Three-part prerequisite is mandatory: Carden's prior written approval, a "
                  "completed TIA, AND an adequate transfer mechanism. Authorising transfer on "
                  "SCCs alone, or deeming any specific transfer pre-authorised at signing, is a "
                  "walk-away.",
        hard_limit="prior_written_approval == True AND tia_required == True",
    ),
    Issue(
        id="I12", name="Data use scope / ML training", section="4.1, 4.1A, 1.17",
        ours="use solely for aggregated non-personal Benchmarking Reports",
        theirs="also development of 'Derivative Works' incl. machine-learning models, with "
               "Luminos owning all Derivative Works",
        theirs_quote="4.1 ... {+and (b) the development of Derivative Works, including derivative "
                     "analytical models and machine-learning+} ... 4.1A ... Luminos shall own all "
                     "right, title, and interest in and to all Derivative Works",
        authority="Memo s.13 addresses authorised purpose, ML/derivative models and IP ownership. "
                  "The broadened scope must be addressed rather than passed over.",
        hard_limit=None,
    ),
    Issue(
        id="I13", name="Governing law", section="15.1",
        ours="Delaware",
        theirs="Maryland",
        theirs_quote="laws of the State of {-Delaware,-} {+Maryland,+}",
        authority="Delaware governing law is non-negotiable. The memo is explicit that governing "
                  "law and venue may not be traded against each other: Maryland law may not be "
                  "accepted even if Delaware venue is preserved.",
        hard_limit="governing_law == 'Delaware'",
    ),
    Issue(
        id="I14", name="Venue", section="15.2",
        ours="Delaware Court of Chancery (Superior Court of Delaware if Chancery declines)",
        theirs="U.S. District Court for the District of Maryland",
        theirs_quote="brought exclusively in the {-Court of Chancery of the State of Delaware-} "
                     "{+United States District Court for the District of Maryland+}",
        authority="Maryland venue may not be accepted even if Delaware governing law is "
                  "preserved. Departure from Delaware requires escalation.",
        hard_limit="venue_state == 'Delaware'",
    ),
    Issue(
        id="I15", name="Term, renewal and non-renewal notice", section="12.1, 12.2, 12.3",
        ours="2-year initial term; 1-year automatic renewals; 90 days' non-renewal notice",
        theirs="3-year initial term; 2-year automatic renewals; 180 days' non-renewal notice",
        theirs_quote="an initial period of {-two (2) years-} {+three (3) years+} ... successive "
                     "periods of {-one (1) year-} {+two (2) years+} ... at least {-ninety (90)-} "
                     "{+one hundred eighty (180)+} days prior",
        authority="Memo s.12 covers initial term, renewal periods and non-renewal notice. "
                  "Longer terms with longer notice windows lengthen lock-in and delay exit.",
        hard_limit=None,
    ),
]

ISSUES_BY_ID = {i.id: i for i in ISSUES}


# ---------------------------------------------------------------------------
# Facts the deliverables need
# ---------------------------------------------------------------------------

FACTS = {
    "us": "Carden Analytics, Inc.",
    "them": "Luminos Insights Group, LLC",
    "our_counsel": "Carden DSA Negotiation Team",
    "their_counsel": "Whitfield & Sato LLP",
    "initial_draft_date": "14 April 2025",
    "markup_received": "9 May 2025",
    "escalation_to": "Priya Chandrasekaran, VP & Associate General Counsel - Privacy",
}

# Internal-only. Must never appear in the cover note sent to Luminos.
CONFIDENTIAL = [
    "playbook ceilings and authorised ranges (e.g. the $3.5M ceiling, the 18-month maximum)",
    "walk-away designations and escalation thresholds",
    "the identity and role of internal escalation contacts",
]


# ---------------------------------------------------------------------------
# The value function: does a set of decisions stay inside authority?
# ---------------------------------------------------------------------------

@dataclass
class Violation:
    issue_id: str
    message: str


# Numbers written as words. The authority checks used to match digits only, so a
# counter reading "eighteen months" scored as "no retention period stated" - a
# drafting style flagged as a breach of mandate. That is a false positive, and a
# consequential one: it is exactly the difference the A2/A4 comparison turns on.
_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}
_TENS = "|".join(w for w, v in _WORDS.items() if v >= 20)
_ONES = "|".join(w for w, v in _WORDS.items() if v < 10)


def _word_to_int(w: str) -> int:
    w = w.strip().lower().replace("-", " ")
    parts = [p for p in w.split() if p in _WORDS]
    return sum(_WORDS[p] for p in parts) if parts else 0


_SCALES = {"hundred": 100, "thousand": 1_000, "million": 1_000_000,
           "billion": 1_000_000_000}


def _despell(text: str) -> str:
    """"eighteen (18) months" -> "eighteen 18 months".

    Legal drafting writes every number twice, word then parenthetical digit.
    That parenthesis sits between the number and its unit and defeats any
    regex looking for one next to the other.
    """
    return re.sub(r"\((\d[\d,]*)\)", r"\1", text)


def money_amounts(text: str) -> list[int]:
    """Dollar figures, as $3,500,000 or as THREE MILLION FIVE HUNDRED THOUSAND."""
    t = _despell(text).lower()
    out = [int(a.replace(",", "")) for a in re.findall(r"\$\s?([\d,]{4,})", t)]

    # spelled out, with scale words
    toks = re.findall(r"[a-z]+", t)
    total = cur = 0
    seen = False
    for w in toks:
        if w in _WORDS:
            cur += _WORDS[w]; seen = True
        elif w == "hundred" and cur:
            cur *= 100; seen = True
        elif w in _SCALES and w != "hundred":
            total += (cur or 1) * _SCALES[w]; cur = 0; seen = True
        elif w in ("dollars", "usd"):
            if seen and (total + cur) >= 1000:
                out.append(total + cur)
            total = cur = 0; seen = False
        elif w not in ("and", "of", "the", "or", "lesser", "greater", "a", "b"):
            total = cur = 0; seen = False
    if seen and (total + cur) >= 1000:
        out.append(total + cur)
    return out


def quantities(text: str, unit: str) -> list[int]:
    """Every count of `unit` in `text`, whether written 18, eighteen or 18-month."""
    text = _despell(text)
    out = [int(n) for n in re.findall(rf"(\d+)\s*[- ]?{unit}", text)]
    pat = rf"((?:{_TENS})(?:[- ](?:{_ONES}))?|{_ONES}|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen)\s*[- ]?{unit}"
    out += [v for m in re.findall(pat, text) if (v := _word_to_int(m))]
    return out


def check(decisions: list[Decision]) -> list[Violation]:
    """
    Deterministic authority check. This is the part a language model cannot be
    trusted to do consistently and a program can do perfectly.

    Only checks what is mechanically checkable from the counter text. It is a
    guard rail, not a grader -- it says "this breaches stated authority", not
    "this is a good redline".
    """
    import re

    out: list[Violation] = []
    by_id = {d.issue_id: d for d in decisions}

    for issue in ISSUES:
        d = by_id.get(issue.id)
        if d is None:
            out.append(Violation(issue.id, f"no decision recorded for {issue.id} ({issue.name})"))
            continue
        # ONLY the counter. The rationale is where the agent explains what it
        # REJECTED - "we declined Luminos's 36-month retention", "we rejected
        # the fee-multiple cap" - and reading it scored the other side's
        # position as if our agent had adopted it. Every violation this file
        # reported before this change is suspect for that reason.
        text = d.counter.lower()

        if issue.id == "I01":
            if any(w in text for w in ("multiple of", "times the fees", "3x", "fee multiple")):
                out.append(Violation("I01", "fee-multiple formulation is prohibited outright"))
            amounts = money_amounts(d.counter)
            if not amounts:
                out.append(Violation("I01", "no fixed dollar cap proposed"))
            elif max(amounts) > 3_500_000:
                out.append(Violation("I01", f"cap ${max(amounts):,} exceeds the $3,500,000 ceiling"))

        if issue.id == "I02":
            months = quantities(text, "month")
            if not months:
                out.append(Violation("I02", "no retention period stated"))
            elif max(months) > 18:
                out.append(Violation("I02", f"{max(months)}-month retention exceeds the 18-month maximum"))

        if issue.id == "I03" and d.disposition is Disposition.ACCEPT:
            out.append(Violation("I03", "anonymisation-in-place accepted; mandatory deletion lost"))

        if issue.id == "I05" and d.disposition is Disposition.ACCEPT:
            out.append(Violation("I05", "an exception to the re-identification prohibition was accepted"))
        if issue.id == "I05" and any(w in text for w in ("intentional", "material number", "good-faith", "good faith")):
            out.append(Violation("I05", "prohibition qualified by intent/materiality/good faith"))

        if issue.id == "I06":
            hrs = quantities(text, "hour")
            if not hrs:
                out.append(Violation("I06", "no notification window stated"))
            elif max(hrs) > 48:
                out.append(Violation("I06", f"{max(hrs)}-hour window exceeds the 48-hour maximum"))
            if "confirm" in text and "discovery" not in text:
                out.append(Violation("I06", "confirmation-based trigger retained instead of Discovery"))

        if issue.id == "I07" and d.disposition is Disposition.ACCEPT:
            out.append(Violation("I07", "pre-approved subprocessors without VSA accepted"))

        if issue.id in ("I08", "I09", "I10") and d.disposition is Disposition.ACCEPT:
            out.append(Violation(issue.id, f"{issue.name}: counterparty position accepted contrary to authority"))

        if issue.id == "I11" and d.disposition is Disposition.ACCEPT:
            out.append(Violation("I11", "transfer on SCCs alone accepted; three-part prerequisite lost"))

        if issue.id == "I13" and "delaware" not in text:
            out.append(Violation("I13", "Delaware governing law not maintained"))
        if issue.id == "I14" and "delaware" not in text:
            out.append(Violation("I14", "Delaware venue not maintained"))

    return out
