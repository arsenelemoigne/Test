"""
The product of the world model: a compact, complete brief for the drafting LLM.

This is the whole point of the exercise. The raw task is ~190KB of documents
(form NDA + 47KB tracked-changes markup + 90KB playbook + two emails + a
timeline). The brief below is a few KB and is intended to carry everything an
agent needs to execute the counter-turn, in a form where OMISSION IS VISIBLE.

Design notes, following the evidence on structured context:
  * Structure here is LOSSLESS AND POINTER-BASED. Every deal point keeps its
    verbatim quote and locator; the brief never replaces the contract, it
    indexes it. Abstractive compression is what loses 3-55% on multi-hop.
  * The brief is INPUT, not a required output format. Do not ask the model to
    emit the schema before it reasons -- format-first is where small models
    lose accuracy to truncation.
  * Every required action is enumerated as a checklist item with an explicit
    "done when" test, because the dominant failure mode on long multi-part
    legal work is silent omission, not misreading.
"""

from __future__ import annotations

from .core import ContractState, Disposition, Tier, score
from .velantis_nda import CONFIDENTIAL_VALUES, GATING_RULES


def negotiation_brief(
    pre: ContractState,
    verbose_quotes: bool = True,
    include_instructions: bool = True,
) -> str:
    """
    The context block handed to the drafting LLM alongside the documents.

    `include_instructions=False` produces the INSTRUCTION-BLIND world model:
    deal points, playbook tiers, exposures, couplings and the confidentiality
    boundary, derived from the contract + form + playbook ONLY. The engagement
    email's per-issue directions are withheld.

    This split is load-bearing for the experiment. A world model built per
    (contract, instruction) pair is not a representation -- it is a precomputed
    answer, and comparing it to raw text measures the builder, not the
    representation. The blind brief is reusable across any instruction on this
    contract; the instructed brief is not.
    """
    rep = score(pre, GATING_RULES)
    L: list[str] = []

    L.append("=" * 78)
    L.append("NEGOTIATION STATE BRIEF -- Project Lantern NDA, counter-turn")
    L.append("=" * 78)
    L.append("")
    L.append("POSTURE")
    L.append(f"  We are:        {pre.parties['us']} ({pre.facts['velantis_jurisdiction']})")
    L.append(f"  Counterparty:  {pre.parties['them']} ({pre.facts['corrigan_jurisdiction']})")
    L.append("  Side:          buy-side, acquiror; we initiated the approach")
    L.append(f"  Target:        {pre.facts['target_structure']}")
    L.append(f"  Code name:     {pre.facts['project_code_name']}")
    L.append(f"  Incoming:      {pre.facts['n_substantive_changes']} substantive changes, "
             f"{pre.facts['n_conforming_changes']} conforming/typographical edits "
             f"({pre.facts['markup_date']})")
    L.append("")
    L.append("DATES THAT MUST APPEAR OR DRIVE URGENCY")
    L.append(f"  Internal review draft due   {pre.facts['counter_turn_internal_due']}")
    L.append(f"  Counter-turn to counsel     {pre.facts['counter_turn_delivery']}")
    L.append(f"  NDA execution target        {pre.facts['nda_execution_target']}")
    L.append(f"  Diligence commences         {pre.facts['diligence_commencement']}")
    L.append(f"  Diligence period            {pre.facts['diligence_period']}")
    L.append("")

    L.append("-" * 78)
    L.append("BALANCE IF WE SIMPLY ACCEPTED THEIR MARKUP")
    L.append("-" * 78)
    sign = "+" if rep.overall > 0 else ""
    L.append(f"  Overall: {sign}{rep.overall:.1f}   (0 = market standard for an M&A NDA)")
    for d in sorted(rep.dimensions, key=lambda x: x.score):
        s = "+" if d.score > 0 else ""
        L.append(f"    {d.dimension:<26s} {s}{d.score:6.1f}")
    L.append("")
    if rep.hard_line_violations:
        L.append(f"  {len(rep.hard_line_violations)} HARD-LINE violations -- this state cannot be signed.")
    L.append("")

    L.append("-" * 78)
    L.append("DEAL POINTS -- every one needs a disposition before this turn is complete")
    L.append("-" * 78)
    L.append("")

    for dim, points in pre.by_dimension().items():
        L.append(f"### {dim}")
        for dp in points:
            flag = {
                Tier.HARD_LINE: "HARD LINE",
                Tier.ESCALATION: "ESCALATE",
                Tier.ACCEPTABLE: "ok",
                Tier.PREFERRED: "ok",
            }[dp.current.tier]
            L.append(f"  {dp.id}  {dp.name}")
            L.append(f"        playbook : {dp.playbook_ref}")
            L.append(f"        our form : {dp.base.label}")
            if dp.proposed is not None:
                L.append(f"        they ask : {dp.proposed.label}   [{flag}]")
                L.append(f"        exposure : {dp.exposure:+d} if accepted as drafted")
                if verbose_quotes and dp.proposed.provenance:
                    L.append(f"        cite     : {dp.proposed.provenance}")
            else:
                L.append("        they ask : (untouched)")
            if dp.instruction and include_instructions:
                L.append(f"        INSTRUCT : {dp.instruction}")
            if dp.couples_with:
                L.append(f"        couples  : {', '.join(dp.couples_with)}")
            L.append("")
        L.append("")

    L.append("-" * 78)
    L.append("COUPLING RULES ACTIVE ON THIS STATE")
    L.append("-" * 78)
    for n in rep.gating_notes:
        L.append(f"  ! {n}")
    L.append("")

    L.append("-" * 78)
    L.append("CONFIDENTIALITY BOUNDARY -- known to us, must NOT reach the counterparty")
    L.append("-" * 78)
    for k in pre.confidential:
        L.append(f"  X  {CONFIDENTIAL_VALUES[k]}")
    L.append("")
    L.append("  The cover note is addressed to opposing counsel. Nothing above may appear in it,")
    L.append("  and no playbook tier label, escalation threshold or reserved fallback may be")
    L.append("  described as such.")
    L.append("")

    return "\n".join(L)


# --------------------------------------------------------------------------
# Completion checklist -- derived from the state, NOT from the rubric
# --------------------------------------------------------------------------

def completion_checklist(pre: ContractState) -> list[tuple[str, str, str]]:
    """
    Returns (id, deliverable, requirement).

    Derived mechanically from: one item per touched deal point, plus one per
    explicit instruction in the engagement email, plus the transmittal
    requirements, plus the confidentiality negatives. The rubric was not
    consulted -- any overlap with it is the hypothesis being tested.
    """
    items: list[tuple[str, str, str]] = []

    # 1. one redline action per deal point that needs one
    for dp in pre.deal_points.values():
        if dp.proposed is None and not dp.instruction:
            continue
        if dp.proposed is None and dp.instruction:
            items.append((f"RL-{dp.id}", "counter-turn-redline.docx",
                          f"{dp.name}: {dp.instruction}"))
        else:
            items.append((f"RL-{dp.id}", "counter-turn-redline.docx",
                          f"{dp.name}: respond to '{dp.proposed.label}' -- {dp.instruction}"))

    # 2. mechanical redline requirements
    items += [
        ("RL-FMT", "counter-turn-redline.docx",
         "Produced in Microsoft Word Track Changes against the Corrigan first markup."),
        ("RL-CONF", "counter-turn-redline.docx",
         f"Accept the {pre.facts['n_conforming_changes']} conforming/typographical edits "
         "(effective date, party names/addresses, code name, section renumbering, notice blocks)."),
        ("RL-NAME1", "counter-turn-redline.docx",
         f"Correct legal name and jurisdiction: {pre.facts['velantis_legal_name']}, "
         f"{pre.facts['velantis_jurisdiction']}."),
        ("RL-NAME2", "counter-turn-redline.docx",
         f"Correct legal name and jurisdiction: {pre.facts['corrigan_legal_name']}, "
         f"{pre.facts['corrigan_jurisdiction']}."),
        ("RL-CODE", "counter-turn-redline.docx",
         f"Project code name '{pre.facts['project_code_name']}' used consistently."),
        ("RL-RENUM", "counter-turn-redline.docx",
         "Restoring Section 6 (Residuals) and deleting the inserted Standstill section requires "
         "conforming section renumbering and cross-reference fixes, including the survival clause."),
    ]

    # 3. cover note -- addressing and logistics
    items += [
        ("CN-FROM", "cover-note.docx", f"From: {pre.facts['our_counsel']}."),
        ("CN-TO", "cover-note.docx", f"To: {pre.facts['their_counsel']}."),
        ("CN-CC", "cover-note.docx", f"cc: {pre.facts['cc_list']}."),
        ("CN-ATT", "cover-note.docx",
         "States that the counter-turn redline is attached in Word Track Changes format."),
        ("CN-CALL", "cover-note.docx",
         "Offers availability for a call to work through remaining issues."),
        ("CN-EXEC", "cover-note.docx",
         f"References the NDA execution target of {pre.facts['nda_execution_target']}."),
        ("CN-DIL", "cover-note.docx",
         f"References diligence commencing {pre.facts['diligence_commencement']}."),
        ("CN-URG", "cover-note.docx", "Conveys appropriate urgency on timing."),
        ("CN-COUNT", "cover-note.docx",
         f"Correctly characterises the markup as {pre.facts['n_substantive_changes']} substantive "
         f"changes and {pre.facts['n_conforming_changes']} conforming/typographical edits."),
        ("CN-ALL", "cover-note.docx",
         f"Summarises our position on all {pre.facts['n_substantive_changes']} substantive changes."),
    ]

    # 4. cover note -- rationales the instruction explicitly asks us to give
    rationale_points = ["DP-08", "DP-15", "DP-16", "DP-20", "DP-07"]
    for pid in rationale_points:
        dp = pre.deal_points[pid]
        items.append((f"CN-WHY-{pid}", "cover-note.docx",
                      f"Gives the rationale for our position on {dp.name}."))

    # 5. cover note -- the standstill has two distinct requirements
    items += [
        ("CN-SS-WHY", "cover-note.docx",
         "Standstill rejection rationale: consensual, buy-side-initiated process."),
        ("CN-SS-FB", "cover-note.docx",
         "Standstill fallback framed as a concession available only if Corrigan insists -- "
         "not offered as a current position."),
        ("CN-RES-ACK", "cover-note.docx",
         "Acknowledges Corrigan's concerns re customer lists, proprietary algorithms and "
         "employee compensation data when explaining the qualified residuals clause."),
    ]

    # 6. negatives
    for k in pre.confidential:
        items.append((f"CN-NOT-{k[:12]}", "cover-note.docx",
                      f"Does NOT disclose: {CONFIDENTIAL_VALUES[k]}."))

    return items


def render_checklist(pre: ContractState) -> str:
    items = completion_checklist(pre)
    L = ["=" * 78,
         f"COMPLETION CHECKLIST -- {len(items)} items, all must hold",
         "=" * 78, ""]
    current = None
    for cid, deliverable, req in items:
        if deliverable != current:
            current = deliverable
            L.append(f"\n--- {deliverable} ---")
        L.append(f"  [ ] {cid:<16s} {req}")
    return "\n".join(L)
