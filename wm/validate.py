"""
Can we trust the encoding?

There is no ground truth for what a clause is "worth". Nobody publishes correct
answers, and two good lawyers will disagree. So "is each clause value correct?"
is not answerable as asked. Three things ARE measurable, and together they cover
most of what you would actually want to know:

  1. GROUNDEDNESS   does every recorded value point at text that really exists in
                    the contract, and does that text support the value?
                    -> verifiable exactly. This is the legal-soundness question,
                       and it is the one that matters most: a score resting on a
                       clause that is not in the document is not a wrong score,
                       it is a fabrication.

  2. STABILITY      does the same model give the same clause the same score on
                    repeated runs? -> measurable, gives a noise floor. A score
                    whose run-to-run spread exceeds the spread between clauses is
                    not measuring anything.

  3. AGREEMENT      do two different models agree? -> measurable. This is the
                    machine analogue of inter-rater reliability, which is the
                    standard lawyers are themselves held to (contract annotation
                    agreement in published work sits around kappa 0.6-0.7, so
                    that is the bar, not 1.0).

What none of these establish is that the numbers are RIGHT. They establish that
the numbers are grounded, reproducible and not idiosyncratic to one model. That
is the honest ceiling without a validated external standard - which would mean
lawyer-scored gold data you do not have yet.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 1. Groundedness
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())).strip()


@dataclass
class Grounding:
    id: str
    name: str
    quote: str
    found: bool
    coverage: float     # longest matched run / quote length, 0-1


def check_quotes(decisions, document: str, min_coverage: float = 0.6) -> list[Grounding]:
    """
    Does each recorded value quote text that is actually in the contract?

    Exact matching is too strict (the model paraphrases whitespace and tracked-
    change markers), so this measures the longest contiguous word run from the
    quote that appears in the document.
    """
    doc = _norm(document)
    out = []
    for d in decisions:
        q = _norm(getattr(d, "value", "") or getattr(d, "quote", ""))
        words = q.split()
        best = 0
        # longest run of consecutive quote-words present verbatim in the document
        for start in range(len(words)):
            for end in range(len(words), start + best, -1):
                if " ".join(words[start:end]) in doc:
                    best = max(best, end - start)
                    break
        cov = best / len(words) if words else 0.0
        out.append(Grounding(d.id, d.name, q[:100], cov >= min_coverage, round(cov, 2)))
    return out


def grounding_report(groundings: list[Grounding]) -> str:
    ok = sum(1 for g in groundings if g.found)
    L = [f"GROUNDEDNESS: {ok}/{len(groundings)} values trace to text in the document",
         f"(threshold: {int(0.6*100)}% of the quoted words appear verbatim)", ""]
    bad = sorted((g for g in groundings if not g.found), key=lambda g: g.coverage)
    if bad:
        L.append(f"{len(bad)} ungrounded - these are the ones to distrust:")
        for g in bad[:20]:
            L.append(f"  {g.id:<8}{g.name[:40]:<42}coverage {g.coverage:.2f}")
            L.append(f"          “{g.quote[:80]}”")
    else:
        L.append("every value traces to the document.")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 2. Stability across repeated runs of the same model
# ---------------------------------------------------------------------------

@dataclass
class Stability:
    id: str
    name: str
    mean: float
    sd: float
    spread: float       # max - min
    flips: int          # how many times "favours" changed


def stability(runs: list[list]) -> list[Stability]:
    """runs: several assessment lists from the same model, same prompt."""
    by_id: dict[str, list] = {}
    for r in runs:
        for a in r:
            by_id.setdefault(a.id, []).append(a)
    out = []
    for cid, items in by_id.items():
        sev = [a.severity for a in items]
        dirs = {a.favours for a in items}
        out.append(Stability(
            cid, items[0].name,
            round(statistics.mean(sev), 2),
            round(statistics.pstdev(sev), 2) if len(sev) > 1 else 0.0,
            round(max(sev) - min(sev), 2),
            len(dirs) - 1))
    return sorted(out, key=lambda s: -s.sd)


def stability_report(stabs: list[Stability]) -> str:
    if not stabs:
        return "no runs to compare"
    between = statistics.pstdev([s.mean for s in stabs]) if len(stabs) > 1 else 0.0
    within = statistics.mean([s.sd for s in stabs])
    L = ["STABILITY across repeated runs of the same model", "",
         f"  spread BETWEEN clauses (signal) : {between:.2f}",
         f"  spread WITHIN a clause  (noise) : {within:.2f}",
         f"  signal-to-noise                  : "
         f"{(between / within):.1f}x" if within else "  noise is zero", ""]
    if within and between / within < 2:
        L.append("  WARNING: noise is close to signal. These scores do not separate")
        L.append("  clauses reliably. Average more runs, or the scale is too fine.")
    flips = [s for s in stabs if s.flips]
    if flips:
        L.append(f"  {len(flips)} clauses changed DIRECTION between runs - worse than noise,")
        L.append("  the model disagrees with itself about who the clause favours:")
        for s in flips[:10]:
            L.append(f"    {s.id:<8}{s.name[:46]}")
    L += ["", "least stable clauses:", ""]
    for s in stabs[:10]:
        L.append(f"  {s.id:<8}{s.name[:40]:<42}mean {s.mean:>5.2f}  sd {s.sd:>5.2f}"
                 f"  spread {s.spread:>5.2f}")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 3. Agreement between two different models
# ---------------------------------------------------------------------------

def agreement(a_runs: list, b_runs: list) -> str:
    """Spearman rank correlation plus direction agreement, over shared clauses."""
    a = {x.id: x for x in a_runs}
    b = {x.id: x for x in b_runs}
    shared = sorted(set(a) & set(b))
    if len(shared) < 3:
        return f"only {len(shared)} shared clauses - cannot compare"

    xs = [a[i].severity for i in shared]
    ys = [b[i].severity for i in shared]

    def ranks(v):
        order = sorted(range(len(v)), key=lambda k: v[k])
        r = [0.0] * len(v)
        for pos, k in enumerate(order):
            r[k] = float(pos)
        return r

    rx, ry = ranks(xs), ranks(ys)
    n = len(shared)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    rho = 1 - (6 * d2) / (n * (n * n - 1)) if n > 1 else 0.0

    same_dir = sum(1 for i in shared if a[i].favours == b[i].favours)
    both_top = set(sorted(shared, key=lambda i: -a[i].severity)[:10]) & \
               set(sorted(shared, key=lambda i: -b[i].severity)[:10])

    L = [f"AGREEMENT between two models over {n} shared clauses", "",
         f"  rank correlation (Spearman rho) : {rho:+.2f}",
         f"  same direction (who it favours) : {same_dir}/{n} ({same_dir/n:.0%})",
         f"  top-10 overlap                  : {len(both_top)}/10", ""]
    if rho < 0.5:
        L.append("  rho below 0.5: the two models are not measuring the same thing.")
        L.append("  Do not present a single number as if it were objective.")
    elif rho < 0.7:
        L.append("  rho 0.5-0.7: comparable to published lawyer-lawyer agreement on")
        L.append("  contract annotation. Usable for ranking, not for absolute values.")
    else:
        L.append("  rho above 0.7: stable ordering. Still ordinal, not currency.")
    disagree = sorted(shared, key=lambda i: -abs(a[i].severity - b[i].severity))[:8]
    L += ["", "biggest disagreements:", ""]
    for i in disagree:
        L.append(f"  {i:<8}{a[i].name[:36]:<38}{a[i].severity:>6.1f} vs {b[i].severity:>6.1f}"
                 f"   ({a[i].favours} / {b[i].favours})")
    return "\n".join(L)
