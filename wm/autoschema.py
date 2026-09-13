"""
Full-contract encoding with no playbook.

What changed from `encoder.py`:

  encoder.py   15 slots, hand-written by a human who had read the negotiation
               memo. Covers what the counterparty happened to change.
  autoschema   every decision the contract makes, enumerated by a model from
               the contract ALONE. No memo, no human, no knowledge of what the
               counterparty did.

This is the version that scales, because nothing in it is specific to this deal
or this client. Give it any contract and it produces a schema.

THE HARD PART IS THE WEIGHT, NOT THE SCHEMA.

Enumerating decision points is easy - a model does it well. Deciding how much
each one MATTERS is the whole problem, and there are only four places a weight
can come from:

  1. generic legal priors        model knows uncapped indemnity is dangerous
                                 -> available today, no data, THIS FILE
  2. firm policy (a playbook)    "our ceiling is $3.5M and counsel cannot waive"
                                 -> what carden-negotiation-authority-memo.txt is
  3. market baseline             deviation from comparable executed contracts
                                 -> needs a corpus
  4. expected cost               P(trigger) x severity, in currency
                                 -> needs claims history

(1) is the only one that works with zero inputs, and it is the weakest: it
captures what is dangerous IN GENERAL, not what is unacceptable TO YOU. A model
will tell you 36 months is a long retention. It cannot tell you that your policy
caps at 18 and that no one below the CPO can waive it.

`compare_to_memo()` measures exactly that gap on this contract, where both a
model-assigned weight and a real firm playbook exist.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict

# ---------------------------------------------------------------------------
# Step 1: enumerate every decision the contract makes
# ---------------------------------------------------------------------------

SCHEMA_PROMPT = """You are building a structured index of a commercial contract.

List EVERY decision this contract makes. A decision is any point where the drafter
chose something that could have been chosen differently: a number, a duration, a
threshold, an allocation of risk, a right granted or withheld, an obligation and
who bears it, a carve-out, a governing choice.

Include boilerplate (notices, severability, assignment, counterparts) - it is part
of the contract and will simply carry a low weight later.

Do NOT evaluate anything. Do not say whether a term is good or bad. Just enumerate
what is decided and where.

Return ONLY a JSON array, one object per decision:
  {{"id": "D001",
    "name": "short name, e.g. 'Liability cap - aggregate amount'",
    "section": "the section number where it lives",
    "category": "one of: money, time, data, risk_allocation, rights, process, boilerplate",
    "question": "the question you would ask a contract to read this value out of it",
    "value": "what THIS contract says, with the operative words quoted"}}

Aim for completeness. A contract of this length typically has 60-150 decisions.

=== CONTRACT ===
{document}"""


# ---------------------------------------------------------------------------
# Step 2: weight each decision, from generic priors only
# ---------------------------------------------------------------------------

WEIGHT_PROMPT = """For each contract decision below, assess its importance to {party}.

You have NO access to {party}'s internal policies, risk appetite or negotiation
authority. Judge only from general commercial and legal experience: what is market
standard, what creates real exposure, what is cosmetic.

For each decision return:
  "impact"   0-10  how much this term moves {party}'s overall position.
                   0 = cosmetic boilerplate. 10 = would change whether to sign.
  "favours"  "us" | "them" | "neutral"   who the current value favours
  "basis"    one short sentence: WHY this impact, in market or exposure terms
  "confident" true | false   false if this depends on facts you do not have
                             (deal size, sector practice, the party's risk appetite)

Be honest with "confident". Marking everything true is worse than useless.

Return ONLY a JSON array: [{{"id": "...", "impact": N, "favours": "...",
"basis": "...", "confident": true}}]

=== DECISIONS ===
{decisions}"""


@dataclass
class Decision:
    id: str
    name: str
    section: str
    category: str
    question: str
    value: str


@dataclass
class Valued:
    id: str
    name: str
    section: str
    category: str
    impact: int              # 0-10, model-assigned
    favours: str             # us | them | neutral
    basis: str
    confident: bool

    @property
    def signed(self) -> int:
        """Impact signed toward us. This is the per-clause contribution."""
        return {"us": 1, "them": -1, "neutral": 0}[self.favours] * self.impact


def _json_array(raw: str) -> list[dict]:
    m = re.search(r"\[.*\]", raw or "", re.S)
    if not m:
        raise ValueError("no JSON array in model output")
    return json.loads(m.group(0))


def build_schema(document: str, llm) -> list[Decision]:
    """Contract -> every decision it makes. No memo, no human."""
    rows = _json_array(llm(SCHEMA_PROMPT.format(document=document), max_tokens=32000))
    out = []
    for r in rows:
        try:
            out.append(Decision(
                id=r["id"], name=r["name"], section=str(r.get("section", "")),
                category=r.get("category", "unknown"),
                question=r.get("question", ""), value=str(r.get("value", "")),
            ))
        except KeyError:
            continue
    return out


def weigh(decisions: list[Decision], llm, party: str = "the disclosing party") -> list[Valued]:
    """Assign an impact to each decision using generic priors only."""
    payload = "\n".join(
        f'{d.id} [{d.category}, s.{d.section}] {d.name}: {d.value[:220]}' for d in decisions)
    rows = _json_array(llm(WEIGHT_PROMPT.format(party=party, decisions=payload),
                           max_tokens=32000))
    by_id = {d.id: d for d in decisions}
    out = []
    for r in rows:
        d = by_id.get(r.get("id"))
        if d is None:
            continue
        out.append(Valued(
            id=d.id, name=d.name, section=d.section, category=d.category,
            impact=int(r.get("impact", 0)), favours=r.get("favours", "neutral"),
            basis=r.get("basis", ""), confident=bool(r.get("confident", False)),
        ))
    return out


# ---------------------------------------------------------------------------
# Step 3: the scalar, and each clause's contribution to it
# ---------------------------------------------------------------------------

def contract_value(valued: list[Valued]) -> float:
    """
    One number for the whole contract, signed toward us.

    Deliberately a plain sum of signed impacts. It is NOT a currency amount and
    should never be presented as one - it is an ordinal index, only meaningful
    when comparing two versions of the SAME contract. Comparing the scalar across
    different contracts is meaningless.
    """
    return float(sum(v.signed for v in valued))


def delta(before: list[Valued], after: list[Valued]) -> list[tuple[str, str, int]]:
    """Per-clause change in contribution: (id, name, delta). This is the answer to
    'how much did this markup move the contract'."""
    a = {v.id: v for v in before}
    out = []
    for v in after:
        prev = a.get(v.id)
        if prev is None:
            out.append((v.id, f"{v.name} (new)", v.signed))
        elif prev.signed != v.signed:
            out.append((v.id, v.name, v.signed - prev.signed))
    for v in before:
        if not any(x.id == v.id for x in after):
            out.append((v.id, f"{v.name} (removed)", -v.signed))
    return sorted(out, key=lambda t: abs(t[2]), reverse=True)


# ---------------------------------------------------------------------------
# Step 4: does the model's own weighting match a real firm playbook?
# ---------------------------------------------------------------------------

# Model slot -> the memo's tier for it, from gravity.py. Only the 15 slots the
# hand-built schema covers can be compared; the rest of the auto-schema has no
# memo counterpart, which is itself the finding.
MEMO_TIER = {
    "Liability cap": 4, "Data retention period": 4, "Deletion obligation": 4,
    "Re-identification": 4, "Breach notification": 4, "Subprocessor": 4,
    "Cross-border transfer": 4, "Governing law": 4, "Venue": 4,
    "Audit": 3, "Residuals": 3, "Data use": 3,
    "Deletion certification": 2, "Term": 2, "renewal": 2,
}


def compare_to_memo(valued: list[Valued]) -> str:
    """
    The experiment that matters for the product.

    If the model's own impact scores rank the same issues at the top as the
    firm's playbook does, you do not need the playbook and this works for any
    client on day one. If they diverge - especially on the walk-aways - then
    producing the playbook IS the product.
    """
    def tier_for(name: str) -> int | None:
        for k, t in MEMO_TIER.items():
            if k.lower() in name.lower():
                return t
        return None

    rows = [(v, tier_for(v.name)) for v in valued]
    matched = [(v, t) for v, t in rows if t is not None]
    if not matched:
        return "no overlap between the auto-schema and the memo-covered issues"

    L = ["MODEL-ASSIGNED IMPACT vs THE FIRM'S ACTUAL PLAYBOOK TIER",
         "(model saw only the contract; the memo was never shown to it)",
         "=" * 82, "",
         f"{'issue':<44}{'model impact':>13}{'memo tier':>11}{'agree?':>9}", "-" * 82]

    # memo tier 4 = walk-away. Treat model impact >=8 as its equivalent.
    agree = 0
    for v, t in sorted(matched, key=lambda r: -r[0].impact):
        model_says_critical = v.impact >= 8
        memo_says_critical = t == 4
        ok = model_says_critical == memo_says_critical
        agree += ok
        L.append(f"{v.name[:42]:<44}{v.impact:>13}{t:>11}{('yes' if ok else 'NO'):>9}")
    L += ["", f"agreement on what is critical: {agree}/{len(matched)}", ""]
    unconfident = [v for v, _ in matched if not v.confident]
    if unconfident:
        L.append(f"model flagged {len(unconfident)} of these as low-confidence "
                 f"(depends on facts it does not have):")
        for v in unconfident:
            L.append(f"  {v.name[:60]}")
    return "\n".join(L)


def render(valued: list[Valued], top: int = 25) -> str:
    total = contract_value(valued)
    by_cat: dict[str, int] = {}
    for v in valued:
        by_cat[v.category] = by_cat.get(v.category, 0) + v.signed

    L = [f"FULL-CONTRACT VALUE: {total:+.0f}",
         "(ordinal index signed toward us; comparable only between versions of "
         "THIS contract)", "", "by category:"]
    for c, s in sorted(by_cat.items(), key=lambda kv: kv[1]):
        L.append(f"    {c:<18}{s:>+6}")
    L += ["", f"the {top} decisions carrying the most weight:", "",
          f"{'':<8}{'decision':<46}{'impact':>7}{'favours':>9}{'conf':>6}"]
    for v in sorted(valued, key=lambda x: -x.impact)[:top]:
        L.append(f"{v.id:<8}{v.name[:44]:<46}{v.impact:>7}{v.favours:>9}"
                 f"{('y' if v.confident else 'n'):>6}")
    n_low = sum(1 for v in valued if not v.confident)
    L += ["", f"{len(valued)} decisions encoded; {n_low} scored with low confidence.",
          "Low-confidence rows are where a firm playbook would actually change the answer."]
    return "\n".join(L)


def dump(valued: list[Valued]) -> str:
    return json.dumps([asdict(v) for v in valued], indent=2)
