"""
World model instantiation for Harvey LAB task:
    tasks/contracts/commercial-vendor-customer/non-disclosure-agreement-first-turn-redline

Three states are built:
    BASE          -- Velantis Mutual NDA v.6.2 (our form, the opening position)
    CORRIGAN      -- v.6.2 as marked up by Corrigan on 19 May 2025  (the incoming state)
    COUNTER_TURN  -- our counter-turn                                (the target state)

PROVENANCE OF THIS MODEL -- important for the experiment:
Every deal point, tier and instruction below is derived ONLY from documents the
agent is legitimately given:
    * velantis-standard-nda-v6-2.docx        (our form)
    * corrigan-first-markup-nda.docx         (their markup, tracked changes)
    * velantis-nda-negotiation-playbook.docx (tiers: preferred/acceptable/escalation/hard line)
    * email-sung-to-orenbach-review-request.eml (client instructions)
    * corrigan-transmittal-email.eml, project-lantern-timeline.xlsx
The 50 rubric criteria in task.json were NOT consulted when choosing the deal
points, the dimensions, the weights or the checklist. Keeping that separation is
what makes the A/B comparison meaningful.

FAVOURABILITY NUMBERS are illustrative. In a production build these are set by
the negotiating lawyers (conjoint / pairwise elicitation), not by an engineer.
They encode *Velantis's playbook preferences*, anchored at 0 = market standard
for an M&A NDA -- not a universal notion of fairness.
"""

from __future__ import annotations

from .core import (
    Action,
    ContractState,
    DealPoint,
    Disposition,
    Position,
    Provenance,
    Tier,
)

M = "corrigan-first-markup-nda.docx"
B = "velantis-standard-nda-v6-2.docx"
P = "velantis-nda-negotiation-playbook.docx"


def _p(doc: str, loc: str, quote: str) -> Provenance:
    return Provenance(document=doc, locator=loc, quote=quote)


# ==========================================================================
# Deal points: our form position, and what Corrigan proposed
# ==========================================================================

DEAL_POINTS: list[DealPoint] = [
    # ---- Scope of protection --------------------------------------------
    DealPoint(
        id="DP-01",
        name="CI exception: industry-wide availability",
        dimension="Scope of protection",
        playbook_ref="Section 2.3 -- Exceptions to Confidential Information",
        weight=1.4,
        base=Position("no industry-wide carve-out", 0, Tier.PREFERRED,
                      _p(B, "1.2", "Exclusions from Confidential Information")),
        proposed=Position(
            "carve-out for info 'generally available within the data analytics industry'",
            -35, Tier.HARD_LINE,
            _p(M, "1.2(b-1)", "is or becomes generally available within the data analytics "
                              "industry other than as a result of a disclosure ... in violation "
                              "of this Agreement")),
        instruction="REJECT -- far too broad; would gut protection for a significant "
                    "portion of the information we expect to receive.",
    ),
    DealPoint(
        id="DP-02",
        name="CI exception: independent development standard",
        dimension="Scope of protection",
        playbook_ref="Section 2.3 -- Exceptions to Confidential Information",
        weight=0.8,
        base=Position("independently developed 'without reference to, reliance upon, or use of'",
                      +10, Tier.PREFERRED,
                      _p(B, "1.2(d)", "without reference to, reliance upon, or use of any "
                                      "Confidential Information")),
        proposed=Position("independently developed 'without reference to'", 0, Tier.ACCEPTABLE,
                          _p(M, "1.2(d)", "without reference to, {-reliance upon, or use of-} "
                                          "any Confidential Information")),
        instruction="ACCEPT -- this is standard.",
    ),
    DealPoint(
        id="DP-03",
        name="CI exception: public availability (no fault)",
        dimension="Scope of protection",
        playbook_ref="Section 2.3 -- Exceptions to Confidential Information",
        weight=0.8,
        base=Position("publicly available through no fault of Receiving Party", 0, Tier.PREFERRED,
                      _p(B, "1.2(b)", "is or becomes generally available to the public other than "
                                      "as a result of a disclosure by the Receiving Party")),
        proposed=None,  # untouched by Corrigan; instruction is to confirm it survives
        instruction="RETAIN AS DRAFTED -- confirm it is not disturbed by the (b-1) edit.",
    ),
    DealPoint(
        id="DP-04",
        name="Representatives: affiliates included",
        dimension="Scope of protection",
        playbook_ref="Section 3 -- Definition of Representatives",
        weight=1.0,
        base=Position("Representatives excludes affiliates", 0, Tier.PREFERRED,
                      _p(B, "1.3", "directors, officers, members, managers, partners, employees, "
                                   "agents, and advisors")),
        proposed=Position("Representatives includes affiliates and their personnel",
                          -15, Tier.ESCALATION,
                          _p(M, "1.3", "and such Party's affiliates and such affiliates' "
                                       "respective directors, officers, ...")),
        instruction="ACCEPT ONLY IF paired with breach-responsibility language (DP-05).",
        couples_with=("DP-05",),
    ),
    DealPoint(
        id="DP-05",
        name="Representatives: breach responsibility for affiliates",
        dimension="Scope of protection",
        playbook_ref="Section 3.2 -- Acceptable Without Escalation",
        weight=1.2,
        base=Position("absent (not needed while affiliates excluded)", 0, Tier.PREFERRED),
        proposed=None,  # Corrigan did not offer it; we must draft it
        instruction="DRAFT AND INSERT -- Receiving Party remains responsible for any breach "
                    "by its affiliates and their personnel as if its own breach.",
        couples_with=("DP-04",),
    ),
    DealPoint(
        id="DP-06",
        name="Permitted Purpose scope",
        dimension="Scope of protection",
        playbook_ref="Section 4 -- Permitted Purpose",
        weight=1.0,
        base=Position("evaluation/negotiation/consummation of the Transaction", +5, Tier.PREFERRED,
                      _p(B, "1.4", "the evaluation, analysis, negotiation, and potential "
                                   "consummation of the Transaction")),
        proposed=Position("evaluating a potential acquisition by Velantis of the Data Analytics "
                          "Division of Corrigan", 0, Tier.ACCEPTABLE,
                          _p(M, "1.4", "evaluating a potential acquisition by Velantis of the "
                                       "Data Analytics Division of Corrigan")),
        instruction="ACCEPT the narrowing, but check the structural point: Corrigan DAD is an "
                    "unincorporated division, not a separate legal entity. Propose conforming "
                    "language if the reference is imprecise.",
    ),

    # ---- Use rights ------------------------------------------------------
    DealPoint(
        id="DP-07",
        name="Residuals clause",
        dimension="Use rights",
        playbook_ref="Section 6 -- Residuals Clause (MUST-HAVE)",
        weight=2.0,
        base=Position("full residuals clause (unaided memory)", +40, Tier.PREFERRED,
                      _p(B, "6.1-6.3", "Nothing in this Agreement shall restrict either Party "
                                       "from using Residuals for any purpose")),
        proposed=Position("Section 6 deleted in its entirety", 0, Tier.HARD_LINE,
                          _p(M, "Section 6", "{-## Section 6 -- Residuals-} (entire section struck)")),
        instruction="RESTORE -- must-have. Offer qualified version: residuals do not apply to "
                    "(a) trade secrets or (b) information marked 'Highly Confidential -- No "
                    "Residuals'. Acknowledge Corrigan's concerns re customer lists, proprietary "
                    "algorithms, employee compensation data.",
    ),

    DealPoint(
        id="DP-21",
        name="No License (collateral casualty of the Section 6 deletion)",
        dimension="Use rights",
        playbook_ref="Section 6 -- Residuals Clause (s.6.3 sits inside it)",
        weight=1.3,
        base=Position("no licence or right granted by implication or estoppel", +15,
                      Tier.PREFERRED,
                      _p(B, "6.3", "Nothing in this Section 6 shall be construed as granting, "
                                   "conveying, or conferring any license or right under any "
                                   "patent, copyright, trademark, trade secret, or other "
                                   "intellectual property right of the Disclosing Party, "
                                   "whether by implication, estoppel, or otherwise.")),
        proposed=Position("deleted with the rest of Section 6", 0, Tier.HARD_LINE,
                          _p(M, "6.3", "{-6.3 No License.-}{- Nothing in this Section 6 shall "
                                       "be construed as granting ... -}")),
        instruction="NOT SEPARATELY INSTRUCTED. Surfaced by the ContractNLI coverage audit "
                    "(nda-15). s.6.3 is an IP protection unrelated to residuals that was "
                    "deleted as collateral damage when Corrigan struck Section 6. Restoring "
                    "only the residuals definition would silently lose it.",
        couples_with=("DP-07",),
    ),

    # ---- Operational burden ---------------------------------------------
    DealPoint(
        id="DP-08",
        name="Standard of care",
        dimension="Operational burden",
        playbook_ref="Section 7 -- Standard of Care",
        weight=1.3,
        base=Position("reasonable care, no less than own standard", 0, Tier.PREFERRED,
                      _p(B, "2.2", "at least the same degree of care that the Receiving Party "
                                   "uses to protect its own ... but in no event less than a "
                                   "reasonable degree of care")),
        proposed=Position("highest degree of care", -45, Tier.HARD_LINE,
                          _p(M, "2.2", "using {+the highest degree of care+}")),
        instruction="REJECT -- above market. Revert to our standard.",
    ),
    DealPoint(
        id="DP-09",
        name="Compelled disclosure: cost allocation",
        dimension="Operational burden",
        playbook_ref="Section 8 -- Compelled Disclosure and Protective Orders",
        weight=0.9,
        base=Position("cooperation at Disclosing Party's cost; each bears own expenses",
                      0, Tier.PREFERRED,
                      _p(B, "4.2(b)", "reasonably cooperate with the Disclosing Party, at the "
                                      "Disclosing Party's sole cost and expense")),
        proposed=Position("Disclosing Party bears ALL costs incl. attorneys' fees",
                          -15, Tier.ESCALATION,
                          _p(M, "4.2(b)", "The Disclosing Party shall bear all costs and expenses "
                                          "of seeking a protective order ... including ... "
                                          "reasonable attorneys' fees and costs.")),
        instruction="REJECT. Counter: each party bears its own costs; Receiving Party uses "
                    "commercially reasonable efforts to cooperate.",
    ),
    DealPoint(
        id="DP-10",
        name="Return/destruction window",
        dimension="Operational burden",
        playbook_ref="Section 9 -- Return and Destruction of Materials",
        weight=1.0,
        base=Position("ten (10) business days", +10, Tier.PREFERRED,
                      _p(B, "5.1", "within ten (10) business days of receipt of such written request")),
        proposed=Position("five (5) business days", -20, Tier.HARD_LINE,
                          _p(M, "5.1", "within {+five (5) business days+}")),
        instruction="REJECT as impracticable given volume. Revert to 10 business days.",
    ),
    DealPoint(
        id="DP-11",
        name="Document-retention carve-out",
        dimension="Operational burden",
        playbook_ref="Section 9.3 -- Document-Retention Carve-Out",
        weight=1.1,
        base=Position("absent", 0, Tier.ACCEPTABLE),
        proposed=None,  # we must add it
        instruction="DRAFT AND INSERT -- bona fide retention of electronically archived copies on "
                    "automatic backup systems or per legal/regulatory retention requirements; "
                    "such copies remain subject to confidentiality for the full term.",
    ),

    # ---- Remedies --------------------------------------------------------
    DealPoint(
        id="DP-12",
        name="Non-solicitation period",
        dimension="Remedies and restrictions",
        playbook_ref="Section 10 -- Non-Solicitation of Employees",
        weight=1.0,
        base=Position("eighteen (18) months", 0, Tier.PREFERRED,
                      _p(B, "7.1", "ending on the date that is eighteen (18) months after the "
                                   "Effective Date")),
        proposed=Position("twenty-four (24) months", -10, Tier.ESCALATION,
                          _p(M, "6.1", "{+twenty-four (24) months+}")),
        instruction="REJECT -- retain 18 months.",
    ),
    DealPoint(
        id="DP-13",
        name="Non-solicitation: liquidated damages",
        dimension="Remedies and restrictions",
        playbook_ref="Section 10.4(b) -- Hard Line: Liquidated Damages",
        weight=1.5,
        base=Position("no liquidated damages", 0, Tier.PREFERRED),
        proposed=Position("$150,000 per breach as liquidated damages", -25, Tier.HARD_LINE,
                          _p(M, "6.3", "the sum of One Hundred Fifty Thousand Dollars ($150,000) "
                                       "per breach")),
        instruction="REJECT -- remove entirely.",
        couples_with=("DP-14",),
    ),
    DealPoint(
        id="DP-14",
        name="Non-solicitation: equitable remedy",
        dimension="Remedies and restrictions",
        playbook_ref="Section 10.4(a) -- Preferred Position",
        weight=1.0,
        base=Position("injunctive and equitable relief available", +10, Tier.PREFERRED,
                      _p(B, "8.2", "entitled to seek equitable relief, including ... injunctions")),
        proposed=None,
        instruction="CONFIRM the agreement provides injunctive/equitable relief as the remedy "
                    "for breach of non-solicitation, replacing the deleted liquidated damages.",
        couples_with=("DP-13",),
    ),
    DealPoint(
        id="DP-15",
        name="Bond requirement for injunctive relief",
        dimension="Remedies and restrictions",
        playbook_ref="Section 14 -- Equitable Relief and Bond",
        weight=1.4,
        base=Position("relief available without posting bond or security", +20, Tier.PREFERRED,
                      _p(B, "8.2", "without the requirement of posting any bond, surety, or other "
                                   "security")),
        proposed=Position("$500,000 bond required before seeking injunctive relief",
                          -35, Tier.HARD_LINE,
                          _p(M, "8.2", "{-without the requirement of posting any bond, surety, or "
                                       "other security-} (struck; $500,000 bond inserted)")),
        instruction="REJECT -- not market for M&A NDAs and inconsistent with Delaware Court of "
                    "Chancery practice. Restore 'without bond'.",
    ),

    # ---- Strategic freedom ----------------------------------------------
    DealPoint(
        id="DP-16",
        name="Standstill",
        dimension="Strategic freedom",
        playbook_ref="Section 11 -- Standstill",
        weight=2.2,
        base=Position("no standstill", 0, Tier.PREFERRED),
        proposed=Position("12-month one-way standstill with don't-ask-don't-waive",
                          -70, Tier.HARD_LINE,
                          _p(M, "Section 7", "For a period of twelve (12) months ... (g) request "
                                             "Corrigan ... to amend or waive any provision of this "
                                             "Section (a 'don't-ask-don't-waive' restriction)")),
        instruction="REJECT ENTIRELY in the redline -- consensual, buy-side-initiated process. "
                    "FALLBACK held in reserve, NOT offered in the first instance: 6-month "
                    "standstill with fall-away on a third-party bid and no DADW. Signal in the "
                    "cover note only as a concession available if Corrigan insists.",
    ),
    DealPoint(
        id="DP-17",
        name="No obligation to transact / negotiate",
        dimension="Strategic freedom",
        playbook_ref="Section 15 -- No Obligation to Transact",
        weight=1.0,
        base=Position("nothing obligates either party to consummate any transaction",
                      +5, Tier.PREFERRED,
                      _p(B, "Section 9", "No Obligation to Transact")),
        proposed=Position("neither party obligated to continue discussions or negotiate in good faith",
                          -20, Tier.HARD_LINE,
                          _p(M, "9.4", "Neither Party shall have any obligation to continue "
                                       "discussions or negotiate in good faith")),
        instruction="REJECT the good-faith negation; INSERT our preferred formulation "
                    "('Nothing herein shall obligate either party to consummate any transaction').",
    ),

    # ---- Duration --------------------------------------------------------
    DealPoint(
        id="DP-18",
        name="Confidentiality term",
        dimension="Duration",
        playbook_ref="Section 5 -- Confidentiality Term (HARD LINE: max 2 years)",
        weight=1.6,
        base=Position("two (2) years", +15, Tier.PREFERRED,
                      _p(B, "11.2", "for a period of two (2) years from the Effective Date")),
        proposed=Position("three (3) years", -15, Tier.HARD_LINE,
                          _p(M, "11.2", "{+three (3) years+}")),
        instruction="REJECT -- hard line, must not exceed 2 years. Revert.",
    ),

    # ---- Dispute posture -------------------------------------------------
    DealPoint(
        id="DP-19",
        name="Governing law",
        dimension="Dispute posture",
        playbook_ref="Section 12 -- Governing Law (HARD LINE: Delaware)",
        weight=1.5,
        base=Position("State of Delaware", +10, Tier.PREFERRED,
                      _p(B, "Section 12", "the laws of the State of Delaware")),
        proposed=Position("Commonwealth of Virginia", -15, Tier.HARD_LINE,
                          _p(M, "Section 12", "{+Commonwealth of Virginia+}")),
        instruction="REJECT -- non-negotiable. Restore Delaware.",
    ),
    DealPoint(
        id="DP-20",
        name="Dispute resolution forum",
        dimension="Dispute posture",
        playbook_ref="Section 13 -- Dispute Resolution and Jurisdiction",
        weight=1.7,
        base=Position("exclusive jurisdiction, Delaware Court of Chancery", +15, Tier.PREFERRED,
                      _p(B, "13.1", "exclusive jurisdiction of the Court of Chancery of the State "
                                    "of Delaware")),
        proposed=Position("binding JAMS arbitration, seat Richmond VA, jury waiver",
                          -30, Tier.HARD_LINE,
                          _p(M, "13.1-13.7", "finally and exclusively resolved by binding "
                                             "arbitration administered by JAMS ... seat of "
                                             "arbitration shall be Richmond, Virginia")),
        instruction="REJECT -- injunctive relief is the primary remedy and is better served by "
                    "court jurisdiction; Chancery practice is well suited. Restore Chancery "
                    "(fallback: D. Del. if Chancery declines).",
    ),
]


# ==========================================================================
# Facts and confidentiality boundary
# ==========================================================================

FACTS = {
    "project_code_name": "Project Lantern",
    "velantis_legal_name": "Velantis Technologies, Inc.",
    "velantis_jurisdiction": "Delaware corporation",
    "velantis_address": "4100 Colworth Boulevard, Suite 800, Wilmington, DE 19801",
    "velantis_ein": "84-3291076",
    "corrigan_legal_name": "Corrigan Systems Group, LLC",
    "corrigan_jurisdiction": "Virginia limited liability company",
    "corrigan_address": "770 Marbury Crossing, 3rd Floor, Richmond, VA 23219",
    "corrigan_ein": "54-6178234",
    "target_structure": "Corrigan DAD is an unincorporated division, not a separate legal entity",
    "markup_date": "19 May 2025",
    "counter_turn_internal_due": "22 May 2025",
    "counter_turn_delivery": "23 May 2025",
    "nda_execution_target": "30 May 2025",
    "diligence_commencement": "2 June 2025",
    "diligence_period": "90 days, through approximately 31 August 2025",
    "n_substantive_changes": "14",
    "n_conforming_changes": "6",
    "our_counsel": "David Orenbach, Kirkner Pratt LLP, 1501 Rodney Square, Wilmington, DE 19899",
    "their_counsel": "Nina Vasquez, Threlkeld & Aaronson PLLC, 200 Bankside Drive, Suite 1400, "
                     "Richmond, VA 23220",
    "cc_list": "Rachel Sung (VP & Associate GC, Velantis); Marcus Leeland (General Counsel, Corrigan)",
}

# Facts that exist in the world model but must never reach the counterparty.
# The model carries them so the drafting step can be checked against them.
CONFIDENTIAL = (
    "enterprise_value_215m",
    "implied_revenue_43m",
    "financial_advisor_haldane_ridgeway",
    "playbook_tiers_and_fallbacks",
    "internal_escalation_thresholds",
)

CONFIDENTIAL_VALUES = {
    "enterprise_value_215m": "approximately $215 million enterprise value for Corrigan DAD",
    "implied_revenue_43m": "approximately $43 million annual revenue at 5.0x EV/Revenue",
    "financial_advisor_haldane_ridgeway": "Haldane Ridgeway & Co.",
    "playbook_tiers_and_fallbacks": "playbook tier labels, escalation authority, fallback positions",
    "internal_escalation_thresholds": "Rachel Sung / General Counsel escalation thresholds",
}


# ==========================================================================
# State construction
# ==========================================================================

def _points() -> dict[str, DealPoint]:
    import copy
    return {dp.id: copy.deepcopy(dp) for dp in DEAL_POINTS}


def base_state() -> ContractState:
    """Velantis v.6.2 as circulated 12 May 2025. Nothing proposed against it yet."""
    pts = _points()
    for dp in pts.values():
        dp.proposed = None
        dp.current = dp.base
        dp.disposition = Disposition.RETAIN
    return ContractState(
        name="BASE -- Velantis Mutual NDA v.6.2",
        instrument=B,
        deal_points=pts,
        parties={"us": "Velantis Technologies, Inc.", "them": "Corrigan Systems Group, LLC"},
        facts=FACTS,
        confidential=CONFIDENTIAL,
    )


def corrigan_state() -> ContractState:
    """
    The INCOMING state: v.6.2 as Corrigan marked it up.
    Where Corrigan proposed a change, `current` is their language.
    Where they did not, `current` stays at our base.
    Everything they touched is UNRESOLVED -- we have not yet responded.
    """
    pts = _points()
    for dp in pts.values():
        if dp.proposed is not None:
            dp.current = dp.proposed
            dp.disposition = Disposition.UNRESOLVED
        else:
            dp.current = dp.base
            dp.disposition = Disposition.RETAIN
    return ContractState(
        name="PRE -- Corrigan first markup (19 May 2025)",
        instrument=M,
        deal_points=pts,
        parties={"us": "Velantis Technologies, Inc.", "them": "Corrigan Systems Group, LLC"},
        facts=FACTS,
        confidential=CONFIDENTIAL,
    )


# --------------------------------------------------------------------------
# The counter-turn, expressed as a list of ACTIONS on the incoming state.
# This is the transition the agent is being asked to produce.
# --------------------------------------------------------------------------

COUNTER_TURN_ACTIONS: list[Action] = [
    Action("DP-01", Disposition.REJECT,
           Position("no industry-wide carve-out", 0, Tier.PREFERRED),
           "Industry-wide availability is far broader than public availability and would "
           "exclude much of the information exchanged in diligence."),
    Action("DP-02", Disposition.ACCEPT,
           Position("independently developed 'without reference to'", 0, Tier.ACCEPTABLE),
           "Market-standard formulation; accepted."),
    Action("DP-03", Disposition.RETAIN,
           Position("publicly available through no fault of Receiving Party", 0, Tier.PREFERRED),
           "Retained as drafted."),
    Action("DP-04", Disposition.PARTIAL,
           Position("affiliates included, with breach responsibility", 0, Tier.ACCEPTABLE),
           "Accepted subject to the Receiving Party remaining responsible for affiliate breaches."),
    Action("DP-05", Disposition.COUNTER,
           Position("Receiving Party responsible for affiliate breach as if its own",
                    +10, Tier.PREFERRED),
           "Inserted as the condition of accepting the widened Representatives definition."),
    Action("DP-06", Disposition.PARTIAL,
           Position("acquisition of the Data Analytics Division (unincorporated division of "
                    "Corrigan Systems Group, LLC)", +5, Tier.ACCEPTABLE),
           "Narrowing accepted; conformed to reflect that the DAD is an unincorporated division "
           "and the transaction is an acquisition of a division/assets."),
    Action("DP-07", Disposition.PARTIAL,
           Position("residuals restored, qualified: excludes trade secrets and information "
                    "marked 'Highly Confidential -- No Residuals'", +25, Tier.PREFERRED),
           "Must-have clause restored, with a protective mechanism addressing Corrigan's "
           "concerns on customer lists, proprietary algorithms and compensation data."),
    Action("DP-21", Disposition.REJECT,
           Position("no licence or right granted by implication or estoppel", +15,
                    Tier.PREFERRED),
           "Section 6 restored in full, including s.6.3 No License, which protects against "
           "implied IP licences and is unrelated to the residuals concept Corrigan objected to."),
    Action("DP-08", Disposition.REJECT,
           Position("reasonable care, no less than own standard", 0, Tier.PREFERRED),
           "'Highest degree of care' is above market for an M&A NDA."),
    Action("DP-09", Disposition.COUNTER,
           Position("each party bears its own costs; Receiving Party uses commercially reasonable "
                    "efforts to cooperate", 0, Tier.PREFERRED),
           "Cost-shifting to the Disclosing Party rejected; each side bears its own costs."),
    Action("DP-10", Disposition.REJECT,
           Position("ten (10) business days", +10, Tier.PREFERRED),
           "Five business days is impracticable given the anticipated volume of materials."),
    Action("DP-11", Disposition.COUNTER,
           Position("bona fide document-retention carve-out for archived/backup copies, which "
                    "remain confidential for the full term", +15, Tier.ACCEPTABLE),
           "Added to make the return/destruction obligation operable."),
    Action("DP-12", Disposition.REJECT,
           Position("eighteen (18) months", 0, Tier.PREFERRED),
           "Retain the original 18-month period."),
    Action("DP-13", Disposition.REJECT,
           Position("no liquidated damages", 0, Tier.PREFERRED),
           "Liquidated damages are unnecessary and would create complications."),
    Action("DP-14", Disposition.COUNTER,
           Position("injunctive and equitable relief as the remedy for breach", +10, Tier.PREFERRED),
           "Equitable relief substituted as the remedy in place of liquidated damages."),
    Action("DP-15", Disposition.REJECT,
           Position("relief available without posting bond or security", +20, Tier.PREFERRED),
           "A bond requirement is not market for M&A NDAs and is inconsistent with Delaware "
           "Court of Chancery practice."),
    Action("DP-16", Disposition.REJECT,
           Position("no standstill", 0, Tier.PREFERRED),
           "A standstill is inappropriate in a consensual, buy-side-initiated process. Fallback "
           "(6 months, fall-away on third-party bid, no DADW) held in reserve, NOT offered here."),
    Action("DP-17", Disposition.COUNTER,
           Position("nothing herein shall obligate either party to consummate any transaction",
                    +5, Tier.PREFERRED),
           "Corrigan's formulation could be read to negate any implied duty; ours addresses the "
           "point more narrowly."),
    Action("DP-18", Disposition.REJECT,
           Position("two (2) years", +15, Tier.PREFERRED),
           "Confidentiality term must not exceed two years."),
    Action("DP-19", Disposition.REJECT,
           Position("State of Delaware", +10, Tier.PREFERRED),
           "Delaware governing law is non-negotiable."),
    Action("DP-20", Disposition.REJECT,
           Position("exclusive jurisdiction, Delaware Court of Chancery (D. Del. if Chancery "
                    "declines)", +15, Tier.PREFERRED),
           "Injunctive relief is the primary remedy contemplated and is better served by court "
           "jurisdiction; we are not willing to arbitrate."),
]


def counter_turn_state() -> ContractState:
    """The TARGET state: Corrigan's markup with our counter-turn applied."""
    s = corrigan_state().apply_all(COUNTER_TURN_ACTIONS)
    s.name = "POST -- Velantis counter-turn (23 May 2025)"
    s.instrument = "counter-turn-redline.docx"
    return s


# ==========================================================================
# Gating rules -- interaction effects, written by hand as discussed
# ==========================================================================

def _gate_affiliates_without_breach_language(state: ContractState):
    aff = state.deal_points["DP-04"]
    br = state.deal_points["DP-05"]
    if "affiliates included" in aff.current.label and br.disposition in (
        Disposition.UNRESOLVED, Disposition.RETAIN
    ):
        return ("DP-04 widens Representatives to affiliates but DP-05 breach-responsibility "
                "language is not in place. Per playbook 3.2 the widening is only acceptable "
                "with it. Treat DP-04 as ESCALATION until DP-05 is drafted.")
    return None


def _gate_residuals_must_have(state: ContractState):
    r = state.deal_points["DP-07"]
    if r.current.favourability <= 0:
        return ("DP-07 residuals is a MUST-HAVE (playbook Section 6). Any state in which the "
                "clause is absent is inadmissible regardless of how the other dimensions score.")
    return None


def _gate_standstill_fallback_leak(state: ContractState):
    s = state.deal_points["DP-16"]
    if "6-month" in s.current.label or "fall-away" in s.current.label:
        return ("DP-16 fallback has been conceded in the redline. Instruction is to reject "
                "outright and hold the fallback in reserve -- offering it now forfeits the "
                "concession for nothing.")
    return None


def _gate_ld_without_equitable_remedy(state: ContractState):
    ld = state.deal_points["DP-13"]
    eq = state.deal_points["DP-14"]
    if ld.disposition is Disposition.REJECT and eq.disposition in (
        Disposition.UNRESOLVED, Disposition.RETAIN
    ):
        return ("DP-13 liquidated damages removed but DP-14 equitable remedy not affirmatively "
                "confirmed. Removing the only stated remedy without substituting one leaves the "
                "non-solicitation covenant without a remedy.")
    return None


def _gate_forum_vs_injunction(state: ContractState):
    forum = state.deal_points["DP-20"]
    bond = state.deal_points["DP-15"]
    if "arbitration" in forum.current.label.lower() and bond.current.favourability > 0:
        return ("DP-20 sends disputes to arbitration while DP-15 preserves bond-free injunctive "
                "relief. Injunctive relief is materially weaker in arbitration -- these two deal "
                "points are coupled and DP-15's value is largely notional if DP-20 stands.")
    return None


def _gate_no_license_lost_with_residuals(state: ContractState):
    res = state.deal_points["DP-07"]
    lic = state.deal_points["DP-21"]
    if res.current.favourability > 0 and lic.current.favourability <= 0:
        return ("DP-07 residuals restored but DP-21 'No License' (v.6.2 s.6.3) is still "
                "deleted. s.6.3 sits inside Section 6 but protects against implied IP "
                "licences, which is unrelated to residuals. Restore the whole section, "
                "not just the residuals definition.")
    return None


GATING_RULES = [
    _gate_affiliates_without_breach_language,
    _gate_residuals_must_have,
    _gate_standstill_fallback_leak,
    _gate_ld_without_equitable_remedy,
    _gate_forum_vs_injunction,
    _gate_no_license_lost_with_residuals,
]
