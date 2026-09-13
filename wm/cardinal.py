"""
The answer to Cox (2008), and the three things it forces.

THE PROBLEM

Tony Cox, "What's Wrong with Risk Matrices?", Risk Analysis 28(2) 497-512 (2008),
proves - formally, not as an opinion - that risk scores built by multiplying
ordinal frequency and severity categories:

  * can rate a quantitatively SMALLER risk higher than a larger one;
  * are "worse than useless" where frequency and severity are NEGATIVELY
    correlated, i.e. they produce worse-than-random orderings;
  * can correctly and unambiguously order fewer than 10% of randomly chosen
    pairs of hazards.

`valuation.Assessment.severity` is exposure x likelihood on two 0-10 ordinal
scales. That is a risk matrix. Cox's result applies to it directly.

And the negative-correlation case is not hypothetical here - it is the structure
of a liability regime. High-frequency, low-severity claims sit UNDER the cap;
low-frequency, catastrophic claims sit in the CARVE-OUTS. Frequency and severity
are negatively correlated by construction in exactly the clause cluster that
matters most. That is Cox's worse-than-useless regime, on purpose.

THREE RESPONSES, IN ORDER OF HOW MUCH THEY HELP

1. GO CARDINAL. Denominate in money and probability, not points. exposure_eur x
   p_year = an expected annual cost in currency. Cox's theorem is about ordinal
   categories; it has nothing to say about an expected value in euros. Every
   domain where clause numbers survive adversarial scrutiny does this -
   increased limits factors in insurance, basis points in covenant pricing,
   rate-on-line in R&W, the 8-20% bid premium measured in construction. Every
   domain where they do not survive, does not.

2. STOP DECOMPOSING NON-MODULAR CLUSTERS. Henry Smith, "Modularity in Contracts"
   (104 Mich. L. Rev. 1175, 2006): boilerplate is engineered to be separable so
   it can be ported without unforeseen interactions. Where that modularity holds,
   clause-by-clause scoring is approximately valid. Where it fails - cap,
   carve-outs, indemnity, insurance and price are one deliberately non-separable
   interface - it is invalid. English law agrees: UCTA s.11(4) requires the court
   to weigh insurance availability, and Goodlife Foods v Hall Fire [2018] EWCA
   Civ 1371 valued an exclusion as a joint function of price, insurance, an
   offered priced alternative and bargaining power. So: find the clusters and
   report them as ONE number, never as members.

3. DEMOTE THE COMPOSITE. Aggregating across clauses is the single step that
   imports Cox, the arbitrary-weighting problem and the second-best problem at
   once, and buys almost nothing a ranked list of the three largest exposures
   does not. Lipsey & Lancaster's theory of the second best is the formal version:
   improving one clause while others are away from optimum can leave the package
   worse. "Raise every clause's score" is the piecemeal-optimisation fallacy.

WHAT THIS MODULE DOES NOT FIX

Cardinal denomination fixes the arithmetic. It does not fix the inputs. Lawyers
are measured at 11-15 percentage points overconfident, worst at high confidence
(Goodman-Delahunty et al., Psych. Pub. Pol. & L. 16(2), 2010). An expected-value
calculation launders that bias into false precision. Track calibration or treat
every figure as an order of magnitude.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
import math
import re
from dataclasses import dataclass, asdict


# ---------------------------------------------------------------------------
# Cardinal assessment: money and probability, not points
# ---------------------------------------------------------------------------

@dataclass
class Exposure:
    """One clause, denominated."""
    id: str
    name: str
    category: str

    loss_low: float        # currency, 10th percentile loss GIVEN the clause bites
    loss_mid: float        # median
    loss_high: float       # 90th percentile - the tail is the point
    p_year: float          # probability it bites in a given contract year
    favours: str           # "us" | "them" | "neutral"
    basis: str
    confident: bool

    @property
    def expected(self) -> float:
        """Expected annual cost. Log-ish weighting of the three points, so the
        tail is not averaged away."""
        return self.p_year * (self.loss_low + 4 * self.loss_mid + self.loss_high) / 6

    @property
    def tail(self) -> float:
        """The 90th-percentile loss, undiscounted by probability. This is the
        number that ends a company, and the one an expected value hides."""
        return self.loss_high

    @property
    def signed(self) -> float:
        return {"us": 1.0, "them": -1.0, "neutral": 0.0}[self.favours] * self.expected


def money(x: float) -> str:
    a = abs(x)
    if a >= 1_000_000:
        return f"{'-' if x < 0 else ''}{a/1_000_000:.2f}M"
    if a >= 1_000:
        return f"{'-' if x < 0 else ''}{a/1_000:.0f}k"
    return f"{x:,.0f}"


# ---------------------------------------------------------------------------
# 1. Cox diagnostic: is the ordinal model in its failure regime?
# ---------------------------------------------------------------------------

def cox_diagnostic(assessments) -> str:
    """
    Run this on any ORDINAL scoring before believing it.

    Cox's worse-than-useless condition is a negative correlation between
    frequency and severity across the items being compared. This measures it,
    and also reports how many pairs the ordinal score can actually order.
    """
    xs = [a.exposure for a in assessments]
    ys = [a.likelihood for a in assessments]
    n = len(xs)
    if n < 3:
        return "too few clauses to diagnose"

    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs) / n)
    sy = math.sqrt(sum((y - my) ** 2 for y in ys) / n)
    r = cov / (sx * sy) if sx and sy else 0.0

    # how many pairs does the composite score order strictly?
    sev = [a.severity for a in assessments]
    pairs = ties = 0
    for i in range(n):
        for j in range(i + 1, n):
            pairs += 1
            if abs(sev[i] - sev[j]) < 0.05:
                ties += 1
    resolvable = (pairs - ties) / pairs if pairs else 0.0

    L = ["COX DIAGNOSTIC on the ordinal exposure x likelihood score",
         "(Cox, Risk Analysis 28(2), 2008)", "",
         f"  correlation of exposure with likelihood : {r:+.2f}",
         f"  clause pairs the score orders strictly  : {resolvable:.0%} "
         f"({pairs - ties}/{pairs})", ""]
    if r < -0.2:
        L += ["  *** NEGATIVE CORRELATION. This is Cox's worse-than-useless regime.",
              "  The ordinal score can order these clauses WORSE THAN RANDOM.",
              "  Do not rank on it. Use the cardinal (expected-cost) view instead. ***", ""]
    elif r < 0.2:
        L += ["  Correlation near zero: Cox's pathology is not triggered, but the",
              "  ordinal score is still not a cardinal quantity. Do not add it up.", ""]
    else:
        L += ["  Positive correlation: the ordinal score is behaving, for ranking.",
              "  It is still ordinal - no arithmetic on it beyond ordering.", ""]
    if resolvable < 0.6:
        L.append(f"  Only {resolvable:.0%} of pairs are strictly ordered - the scale is too")
        L.append("  coarse for the number of clauses. Widen it or compare fewer.")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 2. Modularity: find the clusters that must not be decomposed
# ---------------------------------------------------------------------------

def packages(assessments, couplings) -> list[list[str]]:
    """
    Connected components of the coupling graph.

    Each component is a non-modular cluster in Smith's sense: its members'
    values are not separable, so a per-member score is not meaningful and must
    not be reported as if it were. Singletons are modular and may be scored
    individually.
    """
    ids = [a.id for a in assessments]
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for c in couplings:
        s = getattr(c, "source", None) or c[0]
        t = getattr(c, "target", None) or c[1]
        if s in parent and t in parent:
            parent[find(s)] = find(t)

    groups: dict[str, list[str]] = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=len, reverse=True)


def modularity_report(assessments, couplings) -> str:
    by_id = {a.id: a for a in assessments}
    groups = packages(assessments, couplings)
    clusters = [g for g in groups if len(g) > 1]
    singles = [g[0] for g in groups if len(g) == 1]

    L = ["MODULARITY: which clauses may be scored separately, and which may not",
         "(Smith, Modularity in Contracts, 104 Mich. L. Rev. 1175)", "",
         f"  {len(clusters)} non-modular packages covering "
         f"{sum(len(g) for g in clusters)} clauses",
         f"  {len(singles)} clauses are modular and may be scored on their own", ""]
    for g in clusters:
        members = [by_id[i] for i in g if i in by_id]
        tot = sum(m.signed for m in members)
        L.append(f"  PACKAGE ({len(g)} clauses)  combined value {tot:+.1f}")
        for m in members:
            L.append(f"      {m.id:<8}{m.name[:56]}")
        L.append("      ^ report the package figure. The per-clause numbers above are")
        L.append("        not separately meaningful and must not be quoted alone.")
        L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 3. What to report instead of a composite
# ---------------------------------------------------------------------------

def top_exposures(exposures: list[Exposure], k: int = 3) -> str:
    """
    The recommended deliverable. A ranked list of the largest exposures in
    currency, with the tail shown separately from the expected value, and no
    composite at all.
    """
    # A zero is not a small exposure, it is an absent answer. Ranking them puts
    # definitional clauses - effective date, party names - at the top of a risk
    # report, which is how you can tell the elicitation failed.
    priced = [e for e in exposures if e.expected > 0 or e.tail > 0]
    if not priced:
        return ("LARGEST EXPOSURES\n" + "=" * 72 + "\n\n"
                f"NOTHING IS PRICED. All {len(exposures)} figures came back at "
                "zero loss and zero\nprobability, which means the elicitation "
                "failed, not that the contract is\nsafe. Do not read anything "
                "into the ordering of zeros. Re-run, and check\nthe batch "
                "warnings above.")

    adverse = sorted((e for e in priced if e.favours == "them"),
                     key=lambda e: -e.expected)[:k]
    by_tail = sorted(priced, key=lambda e: -e.tail)[:k]

    L = ["LARGEST EXPOSURES", "=" * 72, "",
         f"{len(priced)} of {len(exposures)} decisions carry a priced exposure; "
         f"the rest are zero.", "",
         "By expected annual cost:", ""]
    for e in adverse:
        L.append(f"  {e.name[:44]:<46}{money(e.expected):>12}/yr   "
                 f"p={e.p_year:.0%}{'' if e.confident else '  [low confidence]'}")
        L.append(f"      {e.basis[:66]}")
    L += ["", "By tail loss, ignoring probability - the ones that end the company:", ""]
    for e in by_tail:
        L.append(f"  {e.name[:44]:<46}{money(e.tail):>12} at p90")
    L += ["", "No composite score is reported. Aggregating across clauses imports",
          "Cox's ordering pathology, the arbitrary-weighting problem and the",
          "second-best problem at once, and adds nothing this list does not give."]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# The prompt: money bands, not points
# ---------------------------------------------------------------------------

CARDINAL_PROMPT = """For each contract decision below, estimate what it would COST
in money, for {party}.

Do not score anything out of ten. Give currency figures and a probability. If you
cannot, say so via "confident": false rather than inventing a number.

For each decision:
  "loss_low"   currency  the loss at roughly the 10th percentile, GIVEN the clause
                         bites at all
  "loss_mid"   currency  the median loss given it bites
  "loss_high"  currency  the loss at roughly the 90th percentile. The tail is the
                         point - do not compress it toward the median.
  "p_year"     0-1       probability it bites in a given contract year
  "favours"    "us" | "them" | "neutral"
  "basis"      one sentence naming what drives the figure
  "confident"  true | false - false where the figure depends on facts you do not
               have (deal size, the party's revenue, sector claim rates)

Deal context you may assume: {context}

A clause that is catastrophic but rare gets a large loss_high and a small p_year.
Do not average those into a middling number.

Return ONLY a JSON array of objects with those keys plus "id".

=== DECISIONS ===
{decisions}"""


def _json_array(raw: str) -> list[dict]:
    m = re.search(r"\[.*\]", raw or "", re.S)
    if not m:
        raise ValueError("no JSON array in model output")
    return json.loads(m.group(0))


def assess_cardinal(decisions, llm, party: str, context: str,
                    batch: int = 20, workers: int = 6) -> list[Exposure]:
    """
    Elicit money figures in BATCHES.

    One call for a hundred-odd decisions does not work: the model answers the
    first dozen properly and then fills the rest with zeros, and because every
    unparseable row used to be dropped silently, the result looked like a
    contract with no exposure anywhere. Small batches, run concurrently, and
    every decision that does not come back is named.
    """
    by_id = {d.id: d for d in decisions}
    chunks = [decisions[i:i + batch] for i in range(0, len(decisions), batch)]

    def one(chunk):
        payload = "\n".join(
            f"{d.id} [{d.category}, s.{d.section}] {d.name}: {d.value[:200]}"
            for d in chunk)
        try:
            return _json_array(llm(CARDINAL_PROMPT.format(
                party=party, context=context, decisions=payload), max_tokens=8000))
        except Exception as e:                          # noqa: BLE001
            print(f"  batch of {len(chunk)} failed: {str(e)[:90]}", flush=True)
            return []

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(one, chunks):
            rows.extend(r)

    out: list[Exposure] = []
    for r in rows:
        d = by_id.get(r.get("id"))
        if d is None:
            continue
        try:
            out.append(Exposure(
                id=d.id, name=d.name, category=d.category,
                loss_low=float(r["loss_low"]), loss_mid=float(r["loss_mid"]),
                loss_high=float(r["loss_high"]), p_year=float(r["p_year"]),
                favours=r.get("favours", "neutral"), basis=r.get("basis", ""),
                confident=bool(r.get("confident", False))))
        except (KeyError, ValueError, TypeError):
            continue

    got = {e.id for e in out}
    missing = [d.id for d in decisions if d.id not in got]
    if missing:
        print(f"  WARNING {len(missing)}/{len(decisions)} decisions came back "
              f"unusable and are NOT in the figures below: "
              f"{', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}",
              flush=True)
    return out


def dump(exposures: list[Exposure]) -> str:
    return json.dumps([asdict(e) for e in exposures], indent=2)
