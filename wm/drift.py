"""
How far has their markup moved the deal, and what would each reversion buy?

No mandate. No memo. No playbook. In practice you have two things: the paper
you sent, and the paper that came back. Everything else is inference.

    v(their markup as it stands)  ->  v(with issue X reverted)  ->  ...  ->  v(our template)

So the baseline is OUR TEMPLATE, not "no contract". That is a different
counterfactual from bridge.py and a better-supported one: it is the
favourable/unfavourable contract measurement that ASC 805 and IFRS 3 already
require for acquired contracts - the present value of (cash flows under the
actual terms minus cash flows under the reference terms). Auditors sign that
number. Nobody audits "what if there were no contract".

The total is the cost of their markup. The Shapley attribution says which of
their changes account for it, correctly, even where changes interact - and the
ladder says which reversions to spend negotiating capital on, in order.

WHAT CAN AND CANNOT BE ENCODED

Three kinds of term, and conflating them is what makes contract-as-code
projects fail:

  PARAMETER   a number, a date, a named party, a period. Encodes directly.
              "18 months", "$3,500,000", "Delaware", "72 hours".

  STRUCTURAL  who must do what, in what order, on what condition, with what
              consequence. NOT a number, and not vague either - it is logic.
              Burden of proof, conditions precedent, notice-and-cure, survival,
              carve-outs from a cap, deemed consent on silence. This is the
              largest category and the most under-encoded one; see Governatori
              on defeasible deontic logic and Symboleo on contract automata.

  VAGUE       "reasonable", "material", "substantial", "commercially reasonable
              efforts". These resist encoding BY DESIGN - Scott & Triantis,
              Anticipating Litigation in Contract Design, 115 Yale L.J. 814
              (2006): vague terms are deliberately chosen options on ex-post
              adjudication, and resolving them to a number destroys the thing
              the parties were buying.

The mistake is trying to pin a number to the third category. You do not encode
what "reasonable" means. You encode TWO things about it that are perfectly
determinate:

  1. the DISTRIBUTION of outcomes it admits (p10/p50/p90), not a point, and
  2. WHO BEARS the uncertainty - which party must persuade a tribunal, under
     what standard, and what happens if they fail.

(2) is the one people miss. "Switching the burden of proof" is not vague at
all. It is a structural term with four determinate fields: who must prove,
what standard applies, what presumption operates in the meantime, and what
follows on failure to discharge it. It is among the most encodable things in a
contract, and among the most valuable, because it decides who pays to find out
what "reasonable" meant.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict


KINDS = ("parameter", "structural", "vague")


@dataclass
class IssueDelta:
    """One disputed issue, valued both ways."""
    id: str
    name: str
    kind: str                   # parameter | structural | vague
    ours: float                 # value/yr to us under OUR template wording
    theirs: float               # value/yr to us under THEIR markup
    basis: str = ""
    confident: bool = True

    # vague terms only: the spread, and who has to live with it
    p10: float | None = None
    p90: float | None = None
    who_bears: str = ""         # "us" | "them" | "shared"

    @property
    def cost_of_their_change(self) -> float:
        """What their version costs us, before interaction."""
        return self.ours - self.theirs

    @property
    def spread(self) -> float:
        """How much is unresolved. For a vague term this is the real number:
        a wide spread borne by us is worse than a narrow one borne by them,
        whatever the midpoint says."""
        if self.p10 is None or self.p90 is None:
            return 0.0
        return abs(self.p90 - self.p10)


@dataclass
class Interaction:
    a: str
    b: str
    joint: float
    note: str = ""


class Drift:
    """v(S) = the deal's value with the issues in S reverted to OUR wording and
    everything else left as THEY marked it."""

    def __init__(self, deltas: list[IssueDelta], interactions: list[Interaction] | None = None):
        self.issues = {d.id: d for d in deltas}
        self.interactions = interactions or []

    def v(self, S) -> float:
        S = set(S)
        total = sum(d.ours if d.id in S else d.theirs for d in self.issues.values())
        for it in self.interactions:
            if it.a in S and it.b in S:
                total += it.joint
        return total

    @property
    def as_marked(self) -> float:
        """Their markup, untouched. THE baseline."""
        return self.v(set())

    @property
    def as_drafted(self) -> float:
        """Our template, had they signed it."""
        return self.v(set(self.issues))

    @property
    def drift(self) -> float:
        """What their markup cost us. The number you actually want."""
        return self.as_drafted - self.as_marked


def shapley(d: Drift) -> list[tuple[str, str, float, float]]:
    """Exact, in closed form: the model is 2-additive (see bridge.py).

    Returns (id, name, shapley, naive) sorted worst-first.
    """
    contrib = {i: d.issues[i].cost_of_their_change for i in d.issues}
    for it in d.interactions:
        if it.a in contrib and it.b in contrib:
            contrib[it.a] += it.joint / 2.0
            contrib[it.b] += it.joint / 2.0
    out = [(i, d.issues[i].name, contrib[i], d.issues[i].cost_of_their_change)
           for i in d.issues]
    return sorted(out, key=lambda t: -t[2])


def _m(x: float) -> str:
    a = abs(x)
    s = "-" if x < 0 else ""
    if a >= 1e6:
        return f"{s}{a/1e6:.2f}M"
    if a >= 1e3:
        return f"{s}{a/1e3:.0f}k"
    return f"{s}{a:.0f}"


def report(d: Drift, top: int = 20) -> str:
    attrs = shapley(d)
    L = ["WHAT THEIR MARKUP COST US", "=" * 76, "",
         f"{'':<44}{'contribution':>14}{'naive':>10}{'kind':>9}",
         "-" * 76,
         f"{'THEIR MARKUP AS IT STANDS':<44}{_m(d.as_marked):>14}"]
    for iid, name, sh, naive in attrs[:top]:
        it = d.issues[iid]
        flag = "  <-- entangled" if abs(sh - naive) > max(1.0, abs(naive) * 0.15) else ""
        L.append(f"  {iid} {name[:38]:<40}{_m(sh):>14}{_m(naive):>10}"
                 f"{it.kind[:8]:>9}{flag}")
    L += [f"{'OUR TEMPLATE':<44}{_m(d.as_drafted):>14}", "",
          f"  their markup moved the deal {_m(-d.drift)}/yr against us."
          if d.drift > 0 else
          f"  their markup moved the deal {_m(-d.drift)}/yr IN OUR FAVOUR.",
          f"  sum of contributions {_m(sum(a[2] for a in attrs))}  vs  "
          f"drift {_m(d.drift)}"]

    vague = [i for i in d.issues.values() if i.kind == "vague" and i.spread]
    if vague:
        L += ["", "VAGUE TERMS - no point value, only a spread and a bearer:", ""]
        for i in sorted(vague, key=lambda x: -x.spread)[:10]:
            L.append(f"  {i.id} {i.name[:34]:<36} spread {_m(i.spread):>8}  "
                     f"borne by {i.who_bears or '?'}")
        L += ["", "  A wide spread you bear is worse than a narrow one they bear,",
              "  whatever the midpoint says. This is the column to negotiate on -",
              "  not what 'reasonable' means, but who pays to find out."]

    low = [i.id for i in d.issues.values() if not i.confident]
    if low:
        L += ["", f"{len(low)} figures are low confidence: {', '.join(low[:12])}"]
    return "\n".join(L)


# --- elicitation, from the two documents alone -----------------------------

DRIFT_PROMPT = """You have our template wording and their markup of it. There is no
playbook and no mandate - infer everything from the two documents.

For each substantive change THEY made, value BOTH states for US, per year, in
currency. Signed: negative means the state costs us money.

  "ours"    the expected annual value to us of OUR original wording
  "theirs"  the expected annual value to us of THEIR replacement
  "kind"    one of:
              "parameter"  - a number, date, period or named party
              "structural" - who must do what, on what condition, with what
                             consequence: burden of proof, conditions precedent,
                             notice and cure, survival, carve-outs, deemed
                             consent. Logic, not arithmetic.
              "vague"      - reasonable, material, substantial, commercially
                             reasonable efforts, and the like
  "basis"   one sentence naming what drives the figure

For "vague" ONLY, also give:
  "p10", "p90"   the value to us at the favourable and adverse ends of what a
                 tribunal could plausibly decide. Do NOT collapse this to a
                 point - the spread IS the term.
  "who_bears"    "us", "them" or "shared": which party must persuade a tribunal
                 that its reading is right, and therefore carries the cost and
                 the risk of finding out.

Do not resolve a vague term to a number. A deliberately vague term is an option
on later adjudication, and pinning it to a point destroys what it is.

"confident": false wherever the figure depends on facts you do not have.

Deal context: {context}
We are: {party}

Return ONLY a JSON array:
[{{"id":"D01","name":"...","kind":"parameter","ours":0,"theirs":0,"basis":"...",
   "confident":true,"p10":null,"p90":null,"who_bears":""}}]

=== OUR TEMPLATE ===
{template}

=== THEIR MARKUP ===
{markup}"""


def load(raw: str) -> tuple[list[IssueDelta], list[Interaction]]:
    d = json.loads(raw)
    rows = d["deltas"] if isinstance(d, dict) else d
    inter = [Interaction(**i) for i in d.get("interactions", [])] if isinstance(d, dict) else []
    out = []
    for r in rows:
        r = {k: v for k, v in r.items() if k in IssueDelta.__dataclass_fields__}
        if r.get("kind") not in KINDS:
            r["kind"] = "parameter"
        out.append(IssueDelta(**r))
    return out, inter


def dump(deltas: list[IssueDelta], interactions: list[Interaction]) -> str:
    return json.dumps({"deltas": [asdict(d) for d in deltas],
                       "interactions": [asdict(i) for i in interactions]}, indent=2)
