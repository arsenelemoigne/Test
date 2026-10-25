"""Step 0 feasibility check for the PTAB/IPR predictor (PROJECT_NOTES.md §10).

Runs the §10 checklist against the live USPTO Open Data Portal (ODP) PTAB
Trials API and writes a report that fills the blanks in §3 and §7.

    export USPTO_API_KEY=...            # or a Colab secret named USPTO_API_KEY
    python step0_feasibility.py         # -> step0_out/REPORT.md + step0_out/raw/*.json
    python step0_feasibility.py --out /content/drive/MyDrive/ptab/step0

Needs only `requests`. Makes roughly 60-80 API calls, paced at ~2 per second.
The key is read from the environment and is never printed or written to disk.

Facts already checked before having a key (25 Oct 2026):
- Routes live under /api/v1/patent/trials/... (they answer 401 without a key);
  /api/v1/patent/proceedings/... does not exist (403 "Missing Authentication Token").
- Responses (from patent-dev/uspto-odp example captures, June 2026):
  proceedings -> {"count", "patentTrialProceedingDataBag": [...]}
  documents and decisions -> {"count", "patentTrialDocumentDataBag": [...]}
  each record has trialNumber, trialMetaData, patentOwnerData, regularPetitionerData,
  documentData (and decisionData for decisions).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time
from datetime import date
from pathlib import Path

import requests

BASE = "https://api.uspto.gov/api/v1/patent/trials"

# Regimes from PROJECT_NOTES.md §7.
R1_START = "2025-03-26"  # interim discretionary process (Acting Director Stewart)
R2_START = "2025-10-17"  # Director Squires decides institution
TODAY = date.today().isoformat()

TEST_CASES = ["IPR2025-00318", "IPR2025-00363", "IPR2025-00877", "IPR2025-00486"]
OLDEST_CASE = "IPR2012-00001"

# Titles that may mark the discretionary stage. Used to spot DD briefs, Director
# decisions and summary notices; the report lists every hit so you can judge them.
DD_PATTERN = re.compile(
    r"discretion|\bDD\b|director|summary notice|settled expectation|fintiv|314\(a\)|325\(d\)",
    re.I,
)


class ODPClient:
    """Thin ODP client: auth header, pacing, retries, raw-response capture."""

    def __init__(self, key: str, raw_dir: Path | None = None, session=None, min_interval=0.5):
        self.s = session or requests.Session()
        self.s.headers.update({"X-API-Key": key, "Accept": "application/json"})
        self.raw_dir = raw_dir
        self.min_interval = min_interval
        self._last = 0.0
        self.calls = 0
        self.rate_headers: dict[str, str] = {}

    def _pace(self):
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def request(self, path: str, params: dict | None = None, tag: str | None = None, raw=False):
        url = path if path.startswith("http") else f"{BASE}{path}"
        for attempt in range(5):
            self._pace()
            self.calls += 1
            r = self.s.get(url, params=params, timeout=60, stream=raw)
            self.rate_headers.update(
                {k: v for k, v in r.headers.items() if "ratelimit" in k.lower() or k.lower() == "retry-after"}
            )
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(float(r.headers.get("Retry-After") or 2 ** attempt * 2))
                continue
            break
        if raw:
            return r
        try:
            body = r.json()
        except ValueError:
            body = {"_non_json": r.text[:500]}
        if self.raw_dir and tag:
            (self.raw_dir / f"{tag}.json").write_text(json.dumps({"status": r.status_code, "url": r.url, "body": body}, indent=1))
        return r.status_code, body

    # --- endpoints -------------------------------------------------------
    def proceeding(self, trial, tag=None):
        return self.request(f"/proceedings/{trial}", tag=tag)

    def documents(self, trial, tag=None):
        return self.request(f"/{trial}/documents", tag=tag)

    def decisions(self, trial, tag=None):
        return self.request(f"/{trial}/decisions", tag=tag)

    def document(self, doc_id, tag=None):
        return self.request(f"/documents/{doc_id}", tag=tag)

    def search(self, kind, q=None, range_filter=None, facets=None, limit=1, offset=0, sort=None, tag=None):
        """kind: proceedings | documents | decisions."""
        params = {"limit": limit, "offset": offset}
        if q:
            params["q"] = q
        if range_filter:
            params["rangeFilters"] = range_filter
        if facets:
            params["facets"] = ",".join(facets)
        if sort:
            params["sort"] = sort
        return self.request(f"/{kind}/search", params=params, tag=tag)


# --- response helpers ----------------------------------------------------
def records(body) -> list[dict]:
    for k in ("patentTrialProceedingDataBag", "patentTrialDocumentDataBag", "patentTrialDecisionDataBag", "patentTrialData"):
        if isinstance(body, dict) and isinstance(body.get(k), list):
            return body[k]
    return []


def count(body):
    return body.get("count") if isinstance(body, dict) else None


def get(rec: dict, dotted: str):
    cur = rec
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def facet_values(body, field: str) -> dict[str, int] | None:
    """Find facet counts for `field` wherever the API puts them; None if absent."""
    leaf = field.split(".")[-1]
    for key, val in (body.items() if isinstance(body, dict) else []):
        if "facet" not in key.lower():
            continue
        groups = val if isinstance(val, list) else [val]
        for g in groups:
            if not isinstance(g, dict):
                continue
            for name, items in g.items():
                if name in (field, leaf) and isinstance(items, list):
                    return {str(i.get("name", i.get("value"))): i.get("quantity", i.get("count")) for i in items if isinstance(i, dict)}
            if g.get("field") in (field, leaf) or g.get("name") in (field, leaf):
                items = g.get("values") or g.get("buckets") or g.get("items") or []
                return {str(i.get("name", i.get("value"))): i.get("quantity", i.get("count")) for i in items if isinstance(i, dict)}
    return None


def is_board_or_party_paper(rec):
    dd = rec.get("documentData", {})
    return (dd.get("documentCategory") or "").lower() != "exhibit"


def doc_line(rec):
    dd = rec.get("documentData", {})
    return (
        f"Paper {dd.get('documentNumber', '?'):>4} | {dd.get('documentFilingDate', '?')} | "
        f"{dd.get('filingPartyCategory', '?'):<12} | {dd.get('documentCategory', '?'):<9} | "
        f"{dd.get('documentTypeDescriptionText', '?')} | {(dd.get('documentTitleText') or '').strip()[:90]}"
    )


# --- the checklist -------------------------------------------------------
class Step0:
    def __init__(self, client: ODPClient):
        self.c = client
        self.out: list[str] = []
        self.verdicts: dict[str, str] = {}
        self._docs: dict[str, tuple[int, list[dict], int | None]] = {}

    def trial_documents(self, trial, tag=None):
        """All documents of a trial. Falls back to paging the documents search
        if the per-trial endpoint returns fewer records than its count."""
        if trial not in self._docs:
            status, body = self.c.documents(trial, tag=tag)
            recs, total = records(body), count(body)
            if status == 200 and isinstance(total, int) and len(recs) < total:
                recs = []
                for off in range(0, total, 100):
                    _, b = self.c.search("documents", q=f"trialNumber:{trial}", limit=100, offset=off)
                    page = records(b)
                    recs += page
                    if len(page) < 100:
                        break
            self._docs[trial] = (status, recs, total)
        return self._docs[trial]

    def h(self, title):
        self.out += ["", f"## {title}", ""]

    def p(self, line=""):
        self.out.append(line)

    def verdict(self, item, ok: bool | None, note):
        mark = {True: "PASS", False: "FAIL", None: "CHECK"}[ok]
        self.verdicts[item] = f"{mark} — {note}"

    def run(self):
        if not self.check_key():
            return
        self.check_counts()
        self.check_values()
        self.check_test_cases()
        self.check_squires_case()
        self.check_text()
        self.check_history()

    # 1. key works
    def check_key(self):
        self.h("1. API key works")
        status, body = self.c.proceeding(TEST_CASES[0], tag="01_key_test")
        recs = records(body)
        self.p(f"GET /proceedings/{TEST_CASES[0]} -> HTTP {status}, {len(recs)} record(s)")
        if status != 200:
            self.p(f"Body: `{json.dumps(body)[:300]}`")
            self.verdict("API key works", False, f"HTTP {status}; see raw/01_key_test.json")
            return False
        if recs:
            self.p("```json\n" + json.dumps(recs[0], indent=1)[:1500] + "\n```")
        self.verdict("API key works", True, f"HTTP 200 on {TEST_CASES[0]}")
        return True

    # 2. N at node [0], by filing year and by regime
    def check_counts(self):
        self.h("2. IPR petitions per filing year (node [0])")
        q = "trialMetaData.trialTypeCode:IPR"
        _, body = self.c.search("proceedings", q=q, tag="02_ipr_total")
        total = count(body)
        self.p(f"All IPR proceedings: **{total}**")
        self.p("")
        self.p("| Petition filing year | IPR petitions |")
        self.p("|---|---|")
        by_year = {}
        for y in range(2012, date.today().year + 1):
            _, b = self.c.search("proceedings", q=q, range_filter=f"trialMetaData.petitionFilingDate {y}-01-01:{y}-12-31", tag=f"02_year_{y}")
            by_year[y] = count(b)
            self.p(f"| {y} | {by_year[y]} |")
        summed = sum(v for v in by_year.values() if isinstance(v, int))
        self.p(f"\nSum of years: {summed} (vs total {total}; a gap means records without a filing date).")

        self.p("\n**By regime** (two cuts; the notes define regimes by who decides, so the institution-date cut is the closer one):\n")
        self.p("| Regime | by petition filing date | by institution decision date |")
        self.p("|---|---|---|")
        spans = {"R0": ("2012-01-01", "2025-03-25"), "R1": (R1_START, "2025-10-16"), "R2": (R2_START, TODAY)}
        regime_counts = {}
        for name, (a, b_) in spans.items():
            _, b1 = self.c.search("proceedings", q=q, range_filter=f"trialMetaData.petitionFilingDate {a}:{b_}", tag=f"02_{name}_filed")
            _, b2 = self.c.search("proceedings", q=q, range_filter=f"trialMetaData.institutionDecisionDate {a}:{b_}", tag=f"02_{name}_decided")
            regime_counts[name] = (count(b1), count(b2))
            self.p(f"| {name} ({a} → {b_}) | {count(b1)} | {count(b2)} |")
        ok = isinstance(total, int) and total > 1000 and any(isinstance(v, int) and v > 0 for v in by_year.values())
        self.verdict("IPR counts by year / regime", ok, f"total {total}; R2 decided {regime_counts['R2'][1]}")

    # 3. distinct values that decide how labels are built
    def check_values(self):
        self.h("3. Distinct values of status / outcome / document-type fields")
        q_ipr = "trialMetaData.trialTypeCode:IPR"
        plan = [
            ("proceedings", ["trialMetaData.trialStatusCategory", "trialMetaData.trialTypeCode"]),
            ("decisions", ["decisionData.trialOutcomeCategory", "decisionData.decisionTypeCategory",
                           "documentData.documentCategory", "documentData.documentTypeDescriptionText"]),
            ("documents", ["documentData.documentCategory", "documentData.documentTypeDescriptionText",
                           "documentData.filingPartyCategory", "trialDocumentCategory"]),
        ]
        facets_worked = False
        for kind, fields in plan:
            _, body = self.c.search(kind, q=q_ipr, facets=fields, limit=1, tag=f"03_facets_{kind}")
            for f in fields:
                vals = facet_values(body, f)
                if vals is None:
                    vals = self.sample_values(kind, q_ipr, f)
                    src = "sampled 500 recent records (facets not returned)"
                else:
                    facets_worked = True
                    src = "facet counts over all IPR records"
                self.p(f"**{kind} · `{f}`** — {src}\n")
                for k, v in sorted(vals.items(), key=lambda kv: -(kv[1] or 0))[:60]:
                    self.p(f"- `{k}`: {v}")
                self.p("")
        self.p("Facet support: " + ("yes" if facets_worked else "no — values above are from samples, so rare values may be missing"))

        # Is "discretionary denial" visible as its own value anywhere?
        self.p("\n**Free-text hunt for the discretionary stage** (documents search):\n")
        hits_total, dd_titles = 0, 0
        for term in ['"discretionary denial"', '"Director Discretionary Decision"', '"summary notice"', '"settled expectations"']:
            _, body = self.c.search("documents", q=f"{q_ipr} AND {term}", limit=10, tag=f"03_hunt_{re.sub(r'[^a-z]+', '_', term.lower()).strip('_')}")
            n = count(body)
            hits_total += n or 0
            self.p(f"- {term}: {n} documents")
            for rec in records(body)[:5]:
                line = doc_line(rec)
                dd_titles += bool(DD_PATTERN.search(line))
                self.p(f"  - {rec.get('trialNumber')} · {line}")
        self.verdict("Distinct label values readable", None,
                     "read §3 of the report; decide whether DD vs merits denial can be labelled from the values/titles")
        self.verdict("Discretionary stage findable by text", dd_titles > 0 if hits_total else False,
                     f"{hits_total} hits; {dd_titles} of the sampled titles look like DD papers")

    def sample_values(self, kind, q, field, pages=5, size=100):
        counter = collections.Counter()
        for i in range(pages):
            _, body = self.c.search(kind, q=q, limit=size, offset=i * size, tag=None)
            recs = records(body)
            for r in recs:
                v = get(r, field)
                if v is not None:
                    counter[str(v)] += 1
            if len(recs) < size:
                break
        return dict(counter)

    # 4. the known test cases
    def check_test_cases(self):
        self.h("4. Test cases: DD briefs and Director decisions in the record")
        found_any = False
        for trial in TEST_CASES:
            status, recs, total = self.trial_documents(trial, tag=f"04_docs_{trial}")
            papers = sorted((r for r in recs if is_board_or_party_paper(r)),
                            key=lambda r: int(str(r.get("documentData", {}).get("documentNumber") or 0) or 0)
                            if str(r.get("documentData", {}).get("documentNumber") or "").isdigit() else 10**6)
            self.p(f"### {trial} — HTTP {status}, {len(recs)}/{total} documents ({len(papers)} non-exhibit papers)\n")
            self.p("```")
            for r in papers:
                dd = r.get("documentData", {})
                flag = " <== DD?" if DD_PATTERN.search(f"{dd.get('documentTitleText')} {dd.get('documentTypeDescriptionText')}") else ""
                found_any |= bool(flag)
                self.p(doc_line(r) + flag)
            self.p("```")
            _, dec = self.c.decisions(trial, tag=f"04_decisions_{trial}")
            for r in records(dec):
                d = r.get("decisionData", {})
                self.p(f"- decision: {d.get('decisionIssueDate')} · outcome `{d.get('trialOutcomeCategory')}` · type `{d.get('decisionTypeCategory')}` · {(r.get('documentData', {}).get('documentTitleText') or '').strip()[:90]}")
            if recs:
                tm = recs[0].get("trialMetaData", {})
                self.p(f"- status `{tm.get('trialStatusCategory')}`, filed {tm.get('petitionFilingDate')}, instituted/decided {tm.get('institutionDecisionDate')}")
            self.p("")
        self.verdict("DD briefs / Director decisions in test cases", found_any,
                     "titles flagged '<== DD?' in §4" if found_any else "no DD-looking titles; check §4 by hand")

    # 5. a Squires-era case
    def check_squires_case(self):
        self.h("5. Squires-era cases (institution decided on/after 31 Oct 2025)")
        _, body = self.c.search("proceedings", q="trialMetaData.trialTypeCode:IPR",
                                range_filter=f"trialMetaData.institutionDecisionDate 2025-10-31:{TODAY}",
                                sort="trialMetaData.institutionDecisionDate desc", limit=5, tag="05_squires_cases")
        trials = [r.get("trialNumber") for r in records(body)][:3]
        self.p(f"{count(body)} IPRs with an institution decision date since 31 Oct 2025. Inspecting: {', '.join(trials) or 'none'}\n")
        notice_seen = False
        for trial in trials:
            _, docs, _ = self.trial_documents(trial, tag=f"05_docs_{trial}")
            board = [r for r in docs if (r.get("documentData", {}).get("filingPartyCategory") or "").upper() == "BOARD"]
            self.p(f"**{trial}** — board papers:")
            self.p("```")
            for r in board:
                line = doc_line(r)
                if re.search(r"notice|director|discretion", line, re.I):
                    line += " <=="
                    notice_seen = True
                self.p(line)
            self.p("```")
        self.verdict("Summary notice in a Squires-era record", notice_seen if trials else None,
                     "see §5" if trials else "no Squires-era case found by institution date")

    # 6. text availability
    def check_text(self):
        self.h("6. Petition text: documentOCRText or PDF download")
        petition = None
        for trial in TEST_CASES:
            for r in self.trial_documents(trial)[1]:
                dd = r.get("documentData", {})
                if (dd.get("documentCategory") or "").upper() == "PETITION" or str(dd.get("documentTypeDescriptionText", "")).lower().startswith("petition"):
                    petition = r
                    break
            if petition:
                break
        if not petition:
            self.p("No petition found among the test cases.")
            self.verdict("Petition text available", False, "no petition document located")
            return
        dd = petition["documentData"]
        self.p(f"Petition: {petition.get('trialNumber')} · {doc_line(petition)}")
        ocr_list = len(dd.get("documentOCRText") or "")
        _, single = self.c.document(dd.get("documentIdentifier"), tag="06_petition_document")
        srec = (records(single) or [{}])[0]
        ocr_single = len(get(srec, "documentData.documentOCRText") or "")
        self.p(f"- documentOCRText length: {ocr_list} in the case listing, {ocr_single} via /documents/{{id}}")
        pdf_ok = False
        uri = dd.get("fileDownloadURI")
        if uri:
            r = self.c.request(uri, raw=True)
            head = next(r.iter_content(2048), b"") if r.status_code == 200 else b""
            pdf_ok = head.startswith(b"%PDF")
            self.p(f"- PDF download: HTTP {r.status_code}, content-type {r.headers.get('Content-Type')}, "
                   f"size {r.headers.get('Content-Length', '?')} bytes, starts with %PDF: {pdf_ok}")
            r.close()
        ok = ocr_single > 1000 or ocr_list > 1000 or pdf_ok
        self.verdict("Petition text available", ok,
                     "OCR text present" if max(ocr_list, ocr_single) > 1000 else ("PDF only (needs pdfplumber/OCR)" if pdf_ok else "neither OCR text nor PDF"))

    # 7. history coverage
    def check_history(self):
        self.h("7. History coverage")
        status, body = self.c.proceeding(OLDEST_CASE, tag="07_oldest")
        exists = status == 200 and bool(records(body))
        self.p(f"{OLDEST_CASE}: HTTP {status}, exists: {exists}")
        _, b = self.c.search("proceedings", q="trialMetaData.trialTypeCode:IPR", sort="trialMetaData.petitionFilingDate asc", limit=3, tag="07_earliest")
        for r in records(b):
            self.p(f"- earliest by filing date: {r.get('trialNumber')} filed {get(r, 'trialMetaData.petitionFilingDate')}")
        _, d = self.c.search("decisions", q="trialMetaData.trialTypeCode:IPR", sort="decisionData.decisionIssueDate asc", limit=3, tag="07_earliest_decisions")
        for r in records(d):
            self.p(f"- earliest decision: {r.get('trialNumber')} issued {get(r, 'decisionData.decisionIssueDate')} · {get(r, 'decisionData.trialOutcomeCategory')}")
        self.verdict("History back to 2012", exists, f"{OLDEST_CASE} {'found' if exists else 'not found'}")

    # report
    def report(self) -> str:
        head = [
            "# Step 0 feasibility report — PTAB/IPR predictor",
            "",
            f"Run on {TODAY} · {self.c.calls} API calls · rate-limit headers seen: {self.c.rate_headers or 'none'}",
            "",
            "## Checklist (PROJECT_NOTES.md §10)",
            "",
            "| Item | Result |",
            "|---|---|",
            *[f"| {k} | {v} |" for k, v in self.verdicts.items()],
            "",
            "GO if the key works, counts are plausible, petition text is reachable, and the discretionary stage",
            "can be labelled from document titles or outcome values. Read §3–§5 before deciding the last one.",
        ]
        return "\n".join(head + self.out) + "\n"


def load_key() -> str | None:
    key = os.environ.get("USPTO_API_KEY")
    if key:
        return key.strip()
    try:
        from google.colab import userdata  # type: ignore
        return userdata.get("USPTO_API_KEY")
    except Exception:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="step0_out", help="output folder (use a Drive path in Colab)")
    args = ap.parse_args(argv)

    key = load_key()
    if not key:
        sys.exit("No key. Set USPTO_API_KEY (or a Colab secret of that name). See PROJECT_NOTES.md §5.")

    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    step = Step0(ODPClient(key, raw_dir=out / "raw"))
    try:
        step.run()
    finally:
        (out / "REPORT.md").write_text(step.report())
    print(f"Report: {out / 'REPORT.md'}  ({step.c.calls} calls)")
    for k, v in step.verdicts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
