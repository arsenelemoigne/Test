"""
Build the issue list and its authority limits FROM THE TASK, not by hand.

wm/abstraction.py holds fifteen issues and their checks written by hand for one
DSA. That does not port. Here the same structure is read out of the client's own
negotiation memo by a model, once, and cached - which is also what the project
set out to test: that the mandate can be extracted rather than supplied.

The limits are machine-checkable, not prose. A model that says "the cap must not
exceed $3.5M" gives us {"kind": "max_money", "value": 3500000}, and the check
that follows is deterministic arithmetic. The model reads the memo; it does not
score the answer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict

from . import attribution
from .abstraction import (Violation, quantities, money_amounts, Disposition,
                          normalise_percent)


@dataclass
class GenIssue:
    id: str
    name: str
    question: str
    our_position: str = ""
    their_position: str = ""
    their_quote: str = ""
    section: str = ""
    limits: list[dict] = field(default_factory=list)

    def as_issue(self):
        """Adapt to abstraction.Issue so the prompt builder and renderer,
        written for the hand-built list, work unchanged."""
        from .abstraction import Issue
        bits = []
        for l in self.limits:
            k = l.get("kind")
            if k in ("max_quantity", "min_quantity"):
                bits.append(f"{'max' if k[:3] == 'max' else 'min'} "
                            f"{l.get('value')} {l.get('unit')}s")
            elif k == "max_money":
                bits.append(f"max ${float(l.get('value', 0)):,.0f}")
            elif k == "forbid_phrase":
                bits.append("must not use: " + ", ".join(l.get("phrases", [])))
            elif k == "require_phrase":
                bits.append("must state: " + ", ".join(l.get("phrases", [])))
            elif k == "forbid_accept":
                bits.append("may not be accepted")
            elif k in ("require_money", "require_quantity"):
                bits.append("a specific figure is required")
        return Issue(id=self.id, name=self.name, section=self.section,
                     ours=self.our_position, theirs=self.their_position,
                     theirs_quote=self.their_quote, authority=self.our_position,
                     hard_limit="; ".join(bits) or None)


# --- the deterministic part ------------------------------------------------

def _n(v: float) -> str:
    """99.0 -> '99', 99.9 -> '99.9'. Un controle qui ecrit '99.0 percents' se lit
    comme un bug et fait douter du reste."""
    return f"{v:g}"


def _u(unit: str, v: float) -> str:
    return unit if unit == "percent" or abs(v) == 1 else f"{unit}s"


def _dit(texte: str, phrase: str) -> bool:
    """La phrase est-elle presente, aux notations pres ?

    "50%" et "50 percent" sont la meme exigence. Les comparer litteralement
    faisait dire au controle "50% early-termination fee required" a propos d'un
    texte qui accordait exactement cela, ecrit en toutes lettres.
    """
    a, b = normalise_percent(texte or "").lower(), normalise_percent(phrase or "").lower()
    return bool(b) and b in a


def check_generic(decisions, issues: list[GenIssue]) -> list[Violation]:
    """Apply each issue's limits to its decision. No model, no judgement.

    Reads ONLY the counter, never the rationale: the rationale is where the
    agent says what it rejected, and reading it scores the other side's
    position as ours. For the same reason it reads only the part of the counter
    that states OUR position - see attribution.py - because a counter-proposal
    normally names both positions in one sentence.
    """
    by_id = {d.issue_id: d for d in decisions}
    out: list[Violation] = []
    for issue in issues:
        d = by_id.get(issue.id)
        if d is None:
            out.append(Violation(issue.id, f"no decision recorded for {issue.id} ({issue.name})"))
            continue
        # Le texte sur lequel les limites s'appliquent est la part du counter
        # qui enonce NOTRE position. Lire la phrase entiere imputait a notre
        # client le chiffre qu'il venait de refuser : sur quatre formulations
        # courantes, trois declenchaient une violation imaginaire.
        text = attribution.ours(d.counter or "").lower()

        for lim in issue.limits:
            kind = lim.get("kind")
            msg = lim.get("message") or f"{issue.name}: {kind} limit breached"
            try:
                if kind == "max_quantity":
                    got = quantities(text, lim["unit"])
                    if got and max(got) > float(lim["value"]):
                        out.append(Violation(issue.id,
                                             f"{_n(max(got))} {_u(lim['unit'], max(got))} "
                                             f"exceeds the {_n(float(lim['value']))}-"
                                             f"{lim['unit']} maximum"))
                elif kind == "min_quantity":
                    got = quantities(text, lim["unit"])
                    if got and min(got) < float(lim["value"]):
                        out.append(Violation(issue.id,
                                             f"{_n(min(got))} {_u(lim['unit'], min(got))} "
                                             f"is below the {_n(float(lim['value']))}-"
                                             f"{lim['unit']} minimum"))
                elif kind == "require_quantity":
                    if not quantities(text, lim["unit"]):
                        out.append(Violation(issue.id, msg))
                elif kind == "max_money":
                    got = money_amounts(text)
                    if got and max(got) > float(lim["value"]):
                        out.append(Violation(issue.id,
                                             f"${max(got):,.0f} exceeds the "
                                             f"${float(lim['value']):,.0f} ceiling"))
                elif kind == "require_money":
                    if not money_amounts(text):
                        out.append(Violation(issue.id, msg))
                elif kind == "forbid_phrase":
                    hit = [p for p in lim["phrases"] if _dit(text, p)]
                    if hit:
                        out.append(Violation(issue.id, f"{msg} ({hit[0]!r})"))
                elif kind == "require_phrase":
                    if not any(_dit(text, p) for p in lim["phrases"]):
                        out.append(Violation(issue.id, msg))
                elif kind == "forbid_accept":
                    if d.disposition is Disposition.ACCEPT:
                        out.append(Violation(issue.id, msg))
            except (KeyError, TypeError, ValueError):
                # a malformed limit must not silently pass as compliance
                out.append(Violation(issue.id, f"UNCHECKABLE limit {lim!r}"))
    return out


# --- the elicited part -----------------------------------------------------

ISSUE_PROMPT = """You are reading one side's internal negotiation mandate and the
counterparty's markup of the contract. Produce the list of ISSUES IN DISPUTE and,
for each, the HARD LIMITS the mandate imposes.

An issue is something the counterparty changed (or refused to change) that our
side must now respond to. Twelve to twenty-five issues is the usual range. Do not
list matters the markup leaves alone.

For each issue give:
  "id"              I01, I02, ... in the order they appear in the contract
  "name"            five words or fewer
  "question"        the decision to be made, as a question
  "section"         the contract section number, if there is one
  "our_position"    what the mandate authorises, in one sentence
  "their_position"  what the markup proposes, in one sentence
  "their_quote"     the markup's own words, verbatim, including any {{+...+}} and
                    {{-...-}} tracked-change markers. Quote, never paraphrase.
  "limits"          machine-checkable constraints from the mandate (see below)

LIMITS. Only state a limit the mandate actually imposes. An issue with no hard
limit gets an empty list - that is normal and better than an invented number.
Each limit is one of:

  {{"kind":"max_quantity","unit":"month","value":18,"message":"..."}}
  {{"kind":"min_quantity","unit":"day","value":90,"message":"..."}}
  {{"kind":"require_quantity","unit":"hour","message":"no notification window stated"}}
  {{"kind":"max_money","value":3500000,"message":"..."}}
  {{"kind":"require_money","message":"no fixed dollar figure proposed"}}
  {{"kind":"forbid_phrase","phrases":["multiple of","times the fees"],"message":"..."}}
  {{"kind":"require_phrase","phrases":["delaware"],"message":"..."}}
  {{"kind":"forbid_accept","message":"accepting this is outside authority"}}

"unit" is a singular noun that appears next to the number in drafting: month,
day, hour, year, percent. These limits are applied to the counter text alone by
exact arithmetic, so a limit you cannot state numerically should be omitted
rather than approximated.

=== THE MANDATE ===
{memo}

=== THE COUNTERPARTY'S MARKUP ===
{markup}

Return ONLY a JSON array of issue objects."""


def _json_array(raw: str) -> list[dict]:
    m = re.search(r"\[.*\]", raw or "", re.S)
    if not m:
        raise ValueError("no JSON array in model output")
    return json.loads(m.group(0))


VALID = {"max_quantity", "min_quantity", "require_quantity", "max_money",
         "require_money", "forbid_phrase", "require_phrase", "forbid_accept"}


def build_issues(memo: str, markup: str, llm, max_tokens: int = 16000) -> list[GenIssue]:
    rows = _json_array(llm(ISSUE_PROMPT.format(memo=memo, markup=markup),
                           max_tokens=max_tokens))
    out, dropped = [], 0
    for r in rows:
        try:
            lims = []
            for lim in r.get("limits", []) or []:
                if lim.get("kind") in VALID:
                    lims.append(lim)
                else:
                    dropped += 1
            out.append(GenIssue(
                id=r["id"], name=r["name"], question=r.get("question", ""),
                our_position=r.get("our_position", ""),
                their_position=r.get("their_position", ""),
                their_quote=r.get("their_quote", ""),
                section=str(r.get("section", "")), limits=lims))
        except KeyError:
            dropped += 1
    if dropped:
        print(f"  {dropped} malformed issue/limit entries dropped", flush=True)
    return out


def dump(issues: list[GenIssue]) -> str:
    return json.dumps([asdict(i) for i in issues], indent=2)


def load(raw: str) -> list[GenIssue]:
    return [GenIssue(**r) for r in json.loads(raw)]


BOUNDED = {"max_quantity", "min_quantity", "max_money"}
PRESENCE = {"require_quantity": "unit", "require_money": None}


def silent_limits(issues: list[GenIssue]) -> list[tuple[str, str, str]]:
    """Bounded limits that a counter can evade by stating no number at all.

    max_quantity(month, 18) fires on "thirty-six (36) months" and stays silent
    on "two times the total fees paid and payable over the full initial Term" -
    which is the formulation the mandate actually calls unacceptable. A ceiling
    with no floor under it only catches the wrong magnitude, never the wrong
    shape.
    """
    out = []
    for i in issues:
        kinds = {l.get("kind") for l in i.limits}
        for l in i.limits:
            k = l.get("kind")
            if k not in BOUNDED:
                continue
            if k == "max_money":
                if "require_money" not in kinds:
                    out.append((i.id, i.name, "max_money without require_money"))
            else:
                unit = l.get("unit")
                has = any(o.get("kind") == "require_quantity" and o.get("unit") == unit
                          for o in i.limits)
                if not has:
                    out.append((i.id, i.name, f"{k}({unit}) without "
                                              f"require_quantity({unit})"))
    return out


def tighten(issues: list[GenIssue]) -> int:
    """Add the missing presence requirement beside each bounded limit."""
    n = 0
    for i in issues:
        add = []
        for l in list(i.limits):
            k = l.get("kind")
            if k == "max_money" and not any(o.get("kind") == "require_money"
                                            for o in i.limits):
                add.append({"kind": "require_money",
                            "message": f"{i.name}: no figure proposed"})
            elif k in ("max_quantity", "min_quantity"):
                unit = l.get("unit")
                if not any(o.get("kind") == "require_quantity" and o.get("unit") == unit
                           for o in i.limits + add):
                    add.append({"kind": "require_quantity", "unit": unit,
                                "message": f"{i.name}: no {unit} figure stated"})
        i.limits.extend(add)
        n += len(add)
    return n


def summary(issues: list[GenIssue]) -> str:
    n_lim = sum(len(i.limits) for i in issues)
    unlimited = [i.id for i in issues if not i.limits]
    L = [f"{len(issues)} issues, {n_lim} machine-checkable limits", ""]
    for i in issues:
        kinds = ", ".join(sorted({l["kind"] for l in i.limits})) or "-"
        L.append(f"  {i.id}  {i.name[:38]:<40}{len(i.limits):>2} limits  {kinds}")
    if unlimited:
        L += ["", f"{len(unlimited)} issues carry NO hard limit and cannot be checked "
                  f"automatically:", "  " + ", ".join(unlimited),
              "  They are still graded by the rubric; they just do not move the",
              "  violation column. Do not read a zero there as full compliance."]

    silent = silent_limits(issues)
    if silent:
        L += ["", f"{len(silent)} limits are SILENT when the counter states no number:"]
        for iid, name, why in silent[:12]:
            L.append(f"  {iid}  {name[:34]:<36} {why}")
        L += ["  A ceiling catches the wrong magnitude, never the wrong shape - a",
              "  counter that adopts the counterparty's formulation without a figure",
              "  passes. `python -m wm.run issues --tighten` adds the matching",
              "  presence requirement to each. Free, no model call."]
    return "\n".join(L)


def _qty(v: float, unit: str) -> str:
    """Write a quantity the way drafting does, so the checker's own regexes
    match it. "3 percents" is not a phrase any contract contains."""
    u = (unit or "").lower()
    if u in ("percent", "pct", "%"):
        return f"{v:g} percent ({v:g}%)"
    plural = "" if (v == 1 or u.endswith("s")) else "s"
    return f"{v:g} {u}{plural}"


def compliant_counter(issue: GenIssue) -> str:
    """Build a counter that satisfies every limit on this issue.

    Used by the self-test. It turns "does the plumbing run" into a real
    question: can a position satisfying all of an issue's stated limits be
    written at all, and does the checker then accept it? Limits that
    contradict each other - a ninety-day minimum under a thirty-day maximum -
    produce a counter that cannot pass, and the self-test says so.
    """
    bounds: dict[str, dict] = {}
    parts: list[str] = []
    money = None

    for l in issue.limits:
        k, unit = l.get("kind"), l.get("unit")
        if k == "max_quantity":
            bounds.setdefault(unit, {})["max"] = float(l["value"])
        elif k == "min_quantity":
            bounds.setdefault(unit, {})["min"] = float(l["value"])
        elif k == "require_quantity":
            bounds.setdefault(unit, {})
        elif k == "max_money":
            money = min(float(l["value"]), money if money is not None else float("inf"))
        elif k == "require_money":
            money = money if money is not None else 1_000_000.0
        elif k == "require_phrase":
            ph = l.get("phrases") or []
            if ph:
                parts.append(str(ph[0]))

    for unit, b in bounds.items():
        lo, hi = b.get("min"), b.get("max")
        if lo is not None and hi is not None:
            v = hi if lo <= hi else hi          # unsatisfiable; emit the ceiling
        elif hi is not None:
            v = hi
        elif lo is not None:
            v = lo
        else:
            v = 30
        parts.append(_qty(v, unit))

    if money is not None:
        parts.append(f"${money:,.0f}")
    if not parts:
        parts.append("Position reverted to our template.")
    return "; ".join(parts) + "."


def unsatisfiable(issues: list[GenIssue]) -> list[tuple[str, str, str]]:
    """Issues whose limits cannot all be met at once."""
    out = []
    for i in issues:
        per: dict[str, dict] = {}
        for l in i.limits:
            if l.get("kind") == "max_quantity":
                per.setdefault(l.get("unit"), {})["max"] = float(l["value"])
            elif l.get("kind") == "min_quantity":
                per.setdefault(l.get("unit"), {})["min"] = float(l["value"])
        for unit, b in per.items():
            if "min" in b and "max" in b and b["min"] > b["max"]:
                out.append((i.id, i.name,
                            f"min {b['min']:g} > max {b['max']:g} {unit}s"))
    return out


# --- provenance ------------------------------------------------------------

def audit(issues: list[GenIssue], memo: str) -> str:
    """Does every limit's number actually appear in the mandate it came from?

    The limits were read out of the memo BY A MODEL. That is one model call
    standing between the client's stated authority and a metric reported as
    deterministic. This checks the one thing that can be checked without
    judgement: a numeric limit whose value appears nowhere in the source is
    fabricated, and a require_phrase whose phrase appears nowhere is too.

    It cannot tell you a number was read in the right DIRECTION - an 18 that is
    a floor recorded as a ceiling passes this test. For that, read the quoted
    line.
    """
    import re as _re
    low = memo.lower()
    from .abstraction import _WORDS
    words = {v: k for k, v in _WORDS.items()}
    lines = memo.splitlines()

    def _show(ln: str, i: int) -> str:
        return ln[max(0, i - 70):i + 95].strip()

    def find_phrase(needle: str) -> str | None:
        for ln in lines:
            j = ln.lower().find(needle)
            if j >= 0:
                return _show(ln, j)
        return None

    def find_quantity(v, unit: str) -> str | None:
        """The number must sit NEXT TO its unit.

        A bare search for "12" matches "June 12, 2025" and reports a warranty
        limit as sourced. The checker only ever reads a number adjacent to its
        unit, so the audit must look for the same thing.
        """
        try:
            iv = int(float(v))
        except (TypeError, ValueError):
            return None
        u = _re.escape((unit or "").lower().rstrip("s"))
        alts = [_re.escape(f"{iv:g}")]
        if iv in words:
            alts.append(_re.escape(words[iv]))
        # "18 months", "18-month", "eighteen (18) month", "eighteen months"
        pat = _re.compile(rf"(?:{'|'.join(alts)})\s*(?:\(\d+\))?\s*[-\u2013 ]?\s*{u}",
                          _re.I)
        for ln in lines:
            m = pat.search(ln)
            if m:
                return _show(ln, m.start())
        return None

    def find_money(v) -> str | None:
        try:
            iv = int(float(v))
        except (TypeError, ValueError):
            return None
        for cand in (f"{iv:,}", f"{iv}", f"{iv/1e6:g} million"):
            hit = find_phrase(cand.lower())
            if hit:
                return hit
        return None

    rows, missing = [], []
    for i in issues:
        for l in i.limits:
            k = l.get("kind")
            if k in ("max_quantity", "min_quantity", "max_money"):
                v = l.get("value")
                hit = (find_money(v) if k == "max_money"
                       else find_quantity(v, l.get("unit", "")))
                unit = "" if k == "max_money" else f" {l.get('unit','')}"
                rows.append((i.id, i.name, f"{k}={v}{unit}", hit))
                if hit is None:
                    missing.append((i.id, i.name, f"{k}={v}"))
            elif k == "require_phrase":
                for ph in l.get("phrases", []):
                    hit = find_phrase(str(ph).lower())
                    rows.append((i.id, i.name, f'require "{ph}"', hit))
                    if hit is None:
                        missing.append((i.id, i.name, f'require "{ph}"'))
            elif k == "forbid_phrase":
                for ph in l.get("phrases", []):
                    hit = find_phrase(str(ph).lower())
                    rows.append((i.id, i.name, f'forbid "{ph}"', hit))

    # Une unite qui n'apparait nulle part dans le document ne peut jamais etre
    # trouvee a cote d'un nombre : la limite est muette, silencieusement.
    units = {l.get("unit") for i in issues for l in i.limits if l.get("unit")}
    absent = sorted(u for u in units if u and u.lower() not in low
                    and (u.lower().rstrip("s") not in low))

    checkable = [r for r in rows if r[2].startswith(("max", "min", "require"))]
    L = [f"LIMIT AUDIT - {len(checkable)} limits carry a value or phrase that must",
         f"appear in the mandate. {len(checkable) - len(missing)} do; "
         f"{len(missing)} do not.", "=" * 76, ""]
    for iid, name, what, hit in rows:
        if hit:
            L.append(f"  OK    {iid} {what[:26]:<28} \u2026{hit[:78]}\u2026")
        else:
            L.append(f"  ????  {iid} {what[:26]:<28} NOT FOUND IN THE MEMO")
    if missing:
        L += ["", f"{len(missing)} limits cite something the memo does not contain.",
              "Each is either a paraphrase, a unit the memo states differently, or",
              "a number the model invented. Read those sections yourself before",
              "quoting any figure that depends on them."]
    if absent:
        L += ["", f"UNITES ABSENTES DU DOCUMENT : {', '.join(absent)}.",
              "Une limite dont l'unite ne figure pas dans le texte ne peut jamais",
              "se declencher : le checker cherche un nombre COLLE a son unite. Ces",
              "limites sont muettes, et un zero dans la colonne violations ne veut",
              "alors rien dire."]
    L += ["", "This proves a number is PRESENT, not that it was read in the right",
          "direction. A floor recorded as a ceiling passes. Read the quoted line."]
    return "\n".join(L)
