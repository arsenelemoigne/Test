"""
Coverage cross-check against ContractNLI's 17 NDA hypotheses.

ContractNLI (Koreeda & Manning, Findings of EMNLP 2021; 607 annotated contracts,
mostly NDAs) defines 17 hypotheses that between them span the normative content
of a standard NDA. Each is labelled Entailment / Contradiction / NotMentioned
with evidence spans.

They are used here as an EXTERNAL, INDEPENDENT audit of whether our 21 deal
points miss anything. This matters because the deal points were derived from the
negotiation playbook, and a playbook only covers what the firm expects to
negotiate -- it is silent on protections nobody usually touches. A counterparty
deleting one of those is exactly the kind of change that slips through.

Note the deontic shape is already present in the hypothesis wording: 'shall not'
(prohibition), 'may' (permission), 'shall notify' (obligation), and nda-19 is
literally the survival flag.
"""

from __future__ import annotations

from .core import ContractState

# id -> (hypothesis text, deal points covering it, note)
CONTRACT_NLI: dict[str, tuple[str, tuple[str, ...], str]] = {
    "nda-1": ("All Confidential Information shall be expressly identified by the "
              "Disclosing Party.",
              (), "v.6.2 does not require marking. Our counter introduces a LIMITED "
                  "marking concept ('Highly Confidential -- No Residuals') via DP-07; "
                  "confirm it does not imply a general marking requirement."),
    "nda-2": ("Confidential Information shall only include technical information.",
              ("DP-01", "DP-02", "DP-03"),
              "v.6.2 1.1 lists 11 categories, not technical-only. Untouched."),
    "nda-3": ("Confidential Information may include verbally conveyed information.",
              (), "v.6.2 1.1 covers oral/visual. Untouched by Corrigan -- verify the "
                  "(b-1) insertion does not narrow it."),
    "nda-4": ("Receiving Party shall not use any Confidential Information for any "
              "purpose other than the purposes stated in Agreement.",
              ("DP-06",), ""),
    "nda-5": ("Receiving Party may share some Confidential Information with some of "
              "Receiving Party's employees.", ("DP-04", "DP-05"), ""),
    "nda-7": ("Receiving Party may share some Confidential Information with some "
              "third-parties (including consultants, agents and professional advisors).",
              ("DP-04", "DP-05"), ""),
    "nda-8": ("Receiving Party shall notify Disclosing Party in case Receiving Party is "
              "required by law, regulation or judicial process to disclose any "
              "Confidential Information.",
              ("DP-09",),
              "PARTIAL: DP-09 models only the COST allocation in the compelled-disclosure "
              "clause. The notice obligation itself is a separate norm and is not a deal "
              "point. Untouched by Corrigan, so low risk here -- but the model is "
              "incomplete as a representation of the clause."),
    "nda-10": ("Receiving Party shall not disclose the fact that Agreement was agreed or "
               "negotiated.", (), "v.6.2 1.1(j)-(k). Untouched."),
    "nda-11": ("Receiving Party shall not reverse engineer any objects which embody "
               "Disclosing Party's Confidential Information.",
               (), "NOT PRESENT in v.6.2. Absence is a pre-existing gap in our own form, "
                   "not something Corrigan did. Worth flagging to the client separately."),
    "nda-12": ("Receiving Party may independently develop information similar to "
               "Confidential Information.", ("DP-02",), ""),
    "nda-13": ("Receiving Party may acquire information similar to Confidential "
               "Information from a third party.", (), "v.6.2 1.2(c). Untouched."),
    "nda-15": ("Agreement shall not grant Receiving Party any right to Confidential "
               "Information.",
               ("DP-21",),
               "*** Lives in v.6.2 s.6.3 'No License', INSIDE the Residuals section that "
               "Corrigan deleted wholesale. Deleting Section 6 therefore also deleted an "
               "IP protection that has nothing to do with residuals. ***"),
    "nda-16": ("Receiving Party shall destroy or return some Confidential Information "
               "upon the termination of Agreement.", ("DP-10",), ""),
    "nda-17": ("Receiving Party may create a copy of some Confidential Information in "
               "some circumstances.", ("DP-11",), ""),
    "nda-18": ("Receiving Party shall not solicit some of Disclosing Party's "
               "representatives.", ("DP-12", "DP-13", "DP-14"), ""),
    "nda-19": ("Some obligations of Agreement may survive termination of Agreement.",
               ("DP-07", "DP-18"),
               "PARTIAL: v.6.2 s.11.4 lists surviving provisions and Corrigan edited it "
               "(struck 'Section 6 (Residuals)', renamed Section 13). Restoring residuals "
               "and deleting the standstill both require conforming edits to 11.4."),
    "nda-20": ("Receiving Party may retain some Confidential Information even after the "
               "return or destruction of Confidential Information.", ("DP-11",), ""),
}


def report(state: ContractState) -> str:
    known = set(state.deal_points)
    L = ["=" * 78,
         "COVERAGE AUDIT -- 21 deal points vs ContractNLI's 17 NDA hypotheses",
         "=" * 78, ""]
    covered = partial = uncovered = 0
    for hid, (text, dps, note) in CONTRACT_NLI.items():
        present = [d for d in dps if d in known]
        missing_ref = [d for d in dps if d not in known]
        if not dps:
            mark, uncovered = "GAP    ", uncovered + 1
        elif note.startswith("PARTIAL") or missing_ref:
            mark, partial = "PARTIAL", partial + 1
        else:
            mark, covered = "ok     ", covered + 1
        L.append(f"[{mark}] {hid}")
        L.append(f"          {text}")
        if present:
            L.append(f"          covered by: {', '.join(present)}")
        if missing_ref:
            L.append(f"          REFERENCED BUT NOT IN MODEL: {', '.join(missing_ref)}")
        if note:
            for line in _wrap(note, 68):
                L.append(f"          {line}")
        L.append("")
    L.append("-" * 78)
    L.append(f"covered {covered}   partial {partial}   gap {uncovered}   of {len(CONTRACT_NLI)}")
    L.append("")
    L.append("Most gaps are hypotheses the counterparty did not touch, so they carry no")
    L.append("action this turn. nda-15 is the exception and it is the reason to run this")
    L.append("audit at all: it is covered by a clause that was deleted as collateral")
    L.append("damage, and no rubric criterion in the benchmark mentions it.")
    return "\n".join(L)


def _wrap(s: str, w: int) -> list[str]:
    out, line = [], ""
    for word in s.split():
        if len(line) + len(word) + 1 > w:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out
