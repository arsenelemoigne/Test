"""
A worked example of the drift valuation, hand-filled.

`python -m wm.run drift --demo` runs it and spends nothing. The changes are the
real ones from the Veridian/Halcyon licence markup; THE NUMBERS ARE HAND-SET and
illustrative. Read the shape, not the digits - in particular, read what happens
to D07: a midpoint cost of 60k, ranked eighth, carrying the second-largest
uncertainty in the document.
"""

from .drift import IssueDelta as D, Interaction as I, Drift

DELTAS = [
    D("D01", "Named User -> Authorized User", "parameter", 0, -612_000,
      "Definitional swap re-prices the per-user fee: $528/yr x ~1,160 extra seats."),
    D("D02", "Fee escalator: CPI floor 3% -> lesser of CPI or 2%", "parameter",
      0, -148_000, "Revenue growth below cost-of-service growth over a 5-year term."),
    D("D03", "Liability cap 12mo trailing -> 2x full-term", "parameter", 0, -410_000,
      "Expected cost of the added exposure band, probability-weighted."),
    D("D04", "Warranty 90 days -> 18 months", "parameter", 0, -96_000,
      "Overlapping warranty periods across the 10-14 month release cycle."),
    D("D05", "IP indemnity: sole-remedy clause deleted", "structural", 0, -205_000,
      "Removes the cap on our exposure route; we lose the election."),
    D("D06", "Burden of proof on SLA credits shifted to us", "structural", 0, -140_000,
      "We must now prove the outage did NOT occur; the presumption runs against us."),
    D("D07", "'commercially reasonable efforts' replaces the fixed uptime figure",
      "vague", 0, -60_000, "Replaces a measurable commitment with an adjudicable one.",
      True, -15_000, -240_000, "us"),
    D("D08", "'material' breach undefined for chronic-failure termination", "vague",
      0, -35_000, "Threshold for their exit right becomes a question for a tribunal.",
      True, -5_000, -180_000, "us"),
    D("D09", "Non-solicit 1yr -> 2yr, mutual", "structural", 0, 18_000,
      "Mutual and longer: on balance slightly favours us as the larger employer."),
]

INTERACTIONS = [
    I("D03", "D05", -95_000, "The cap and the sole-remedy clause were one "
                             "protection; losing both compounds rather than adds."),
    I("D01", "D02", -40_000, "More users under a lower escalator compounds the "
                             "revenue loss."),
    I("D06", "D07", -70_000, "A vague standard you also carry the burden on is "
                             "the worst case."),
]


def drift() -> Drift:
    return Drift(DELTAS, INTERACTIONS)
