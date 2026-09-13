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

from .abstraction import Violation, quantities, money_amounts, Disposition


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

def check_generic(decisions, issues: list[GenIssue]) -> list[Violation]:
    """Apply each issue's limits to its decision. No model, no judgement.

    Reads ONLY the counter, never the rationale: the rationale is where the
    agent says what it rejected, and reading it scores the other side's
    position as ours.
    """
    by_id = {d.issue_id: d for d in decisions}
    out: list[Violation] = []
    for issue in issues:
        d = by_id.get(issue.id)
        if d is None:
            out.append(Violation(issue.id, f"no decision recorded for {issue.id} ({issue.name})"))
            continue
        text = (d.counter or "").lower()

        for lim in issue.limits:
            kind = lim.get("kind")
            msg = lim.get("message") or f"{issue.name}: {kind} limit breached"
            try:
                if kind == "max_quantity":
                    got = quantities(text, lim["unit"])
                    if got and max(got) > float(lim["value"]):
                        out.append(Violation(issue.id,
                                             f"{max(got)} {lim['unit']}s exceeds the "
                                             f"{lim['value']}-{lim['unit']} maximum"))
                elif kind == "min_quantity":
                    got = quantities(text, lim["unit"])
                    if got and min(got) < float(lim["value"]):
                        out.append(Violation(issue.id,
                                             f"{min(got)} {lim['unit']}s is below the "
                                             f"{lim['value']}-{lim['unit']} minimum"))
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
                    hit = [p for p in lim["phrases"] if p.lower() in text]
                    if hit:
                        out.append(Violation(issue.id, f"{msg} ({hit[0]!r})"))
                elif kind == "require_phrase":
                    if not any(p.lower() in text for p in lim["phrases"]):
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
  "their_quote"     the markup's own words, verbatim, including any {+...+} and
                    {-...-} tracked-change markers. Quote, never paraphrase.
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
    return "\n".join(L)
