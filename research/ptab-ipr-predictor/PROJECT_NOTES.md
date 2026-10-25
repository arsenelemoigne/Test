# PTAB / IPR Outcome Predictor — Project Notes

Summary of the research discussion (25 Oct 2026). To be reviewed against the real API data.

## About this document

**What's inside:**
- **§0** How we got here: Catala, the tree + classifiers idea, the candidate domains
- **§1** Why patent disputes and IPRs exist
- **§2** What changed at the PTAB in 2025–2026
- **§3** Data availability: what is confirmed and what is not
- **§4** Competitors found
- **§5** How to get the USPTO API key
- **§6** The objective (two prediction moments: at filing / after briefing)
- **§7** Procedural tree with blanks for the counts
- **§8** Architecture
- **§9** Plan, effort and risks
- **§10** Checklist for the first live session with the API
- **§11** Sources

**Where to find it:** repository `arsenelemoigne/test`, branch `ccr-ee17e0a5-gy4nxo`, file `research/ptab-ipr-predictor/PROJECT_NOTES.md`.

```bash
git fetch origin ccr-ee17e0a5-gy4nxo
git checkout ccr-ee17e0a5-gy4nxo
```

---

## 0. How we got here (short history of the discussion)

1. **Catala** (Inria): a programming language for writing law as code. The law text and the code sit side by side ("literate programming"), and it is built on *prioritized default logic*: a base rule plus exceptions, mirroring how statutes are written. It is used for French family benefits (the CNAF is partnering with Inria to replace its COBOL engine), and it found a bug in the official implementation.
   - Vague terms ("material", "substantial") are **not interpreted** by Catala. They become **explicit inputs** (`input ... content boolean`) that a human (or an LLM / classifier) must fill in.
   - Prior work on LLMs + Catala: Lorenzo, Pietromatera & Holzenberger, *Translating Tax Law to Code with LLMs* (NLLP 2025).
2. **Idea:** encode the procedure as a deterministic tree, put a probabilistic classifier at each judgement node, and propagate the probabilities to get the overall outcome and expected loss.
   - Related existing work: litigation decision-tree analysis (human-estimated probabilities); Pre/Dicta and Theo Ai (outcome and settlement prediction); academic work on predicting SCOTUS and ECHR outcomes; Bayesian networks for evidence.
   - **Realistic product:** a hybrid that combines deterministic procedural rules, trained classifiers for high-volume procedural nodes, LLM-assisted estimates for merits nodes, and sensitivity analysis.
3. **Candidate domains with real data:** US PTAB inter partes review (IPR) (chosen), French "rupture brutale des relations commerciales établies" (Judilibre open data), US securities class actions (FJC IDB + CourtListener), EPO oppositions.
   - Legal constraint in France: Art. 33 of loi 2019-222 forbids judge profiling. Never use judge identity as a feature.
4. **Chosen:** IPR institution / discretionary-denial prediction.

---

## 1. Background: why patent disputes and IPRs exist

- A patent is a 20-year monopoly. The owner sues infringers for damages or an injunction.
- An accused infringer defends with **non-infringement** or **invalidity**.
- An **IPR** is the invalidity attack filed at the USPTO (the PTAB) instead of in court. It is cheaper, faster and decided by technical judges, and is usually filed within 1 year of being sued.
- There are few cases (a few thousand US patent suits per year; historically about 1,000–1,500 IPR petitions per year) but very high stakes (millions in fees, sometimes hundreds of millions in damages). The market is **few buyers who pay a lot**: litigation funders, insurers, defensive aggregators (RPX, Unified Patents), big-tech in-house IP teams, and large law firms.

## 2. What changed in 2025–2026 (affects the data)

- **26 Mar 2025:** the "Interim Processes for PTAB Workload Management" memo split institution into two stages: **discretionary considerations** (decided by the Director) and **merits**. The patent owner files a discretionary-denial brief (DD Req., due about 2 months after the Notice of Filing Date) and the petitioner files an opposition (DD Opp., about 1 month later). New factor: "settled expectations" (patents about 6+ years old).
- **17 Oct 2025:** Director Squires takes over all institution decisions and issues mostly **summary notices without reasons**.
- **31 Oct 2025:** the first summary notice denies 13 IPRs. From about Nov 2025, notices list cases by category: discretionarily denied / referred to merits / instituted / denied on merits.
- **Impact:** institution rates fell from about 58–68% (FY2021–24) to roughly 25–40% (2026). Filings collapsed (about 51 petitions in Q1 2026). About 61% of petitions are discretionarily denied; petitions that are referred to the merits are instituted at about 68–75%.
- **Pending:** a final rule converting the discretionary factors into fixed regulatory bars (under White House review since 22 Jul 2026), and 5 Supreme Court petitions challenging the discretionary denials. **The rules may move again.** If the factors become fixed bars, the classifier node becomes a rule node.

## 3. Data availability: status of verification

Status labels: SECONDARY = third-party or press source only; NOT VERIFIED = must be checked with the live API.

| Element | Status | Detail |
|---|---|---|
| ODP PTAB Trials API exists, is free, needs an API key | SECONDARY | Released in ODP 3.0 (21 Nov 2025); legacy API v2 retired 6 Jan 2026 |
| Endpoints: proceedings, documents, decisions (search / by ID / by trial / download) | SECONDARY (patent-dev/uspto-odp, 19 endpoints) | e.g. `/api/v1/patent/trials/{trialNumber}/documents` |
| Fields | SECONDARY (from spec) | `trialMetaData.trialStatusCategory`, `institutionDecisionDate`, `petitionFilingDate`, `decisionData.trialOutcomeCategory`, `decisionData.decisionTypeCategory`, `documentData.documentCategory`, `documentTitleText`, `fileDownloadURI`, `documentOCRText`, `filingPartyCategory` |
| **Schema trap** | SECONDARY | Live decision search returns its array under `patentTrialDocumentDataBag`, not `patentTrialDecisionDataBag`; `downloadURI` → `fileDownloadURI`; `documentTypeCategory` → `documentTypeDescriptionText` |
| Enumerated status / outcome values | NOT FOUND | Must be read from the live API |
| A field separating "discretionary denial" from "merits denial" | NOT FOUND | Infer from document titles or the summary-notice categories |
| DD briefs in each case record | SECONDARY (strong) | IPR2025-00318: DD Req. = Paper 6, DD Opp. = Paper 8 |
| Stewart-era Director decisions in each record | SECONDARY (strong) | IPR2025-00363 Paper 10 (iRhythm); IPR2025-00877 Paper 10 "Director Discretionary Decision Refer" (Docket Alarm) |
| Squires summary notices in each record | SECONDARY (inferred) | Amended notices appear as Paper 12 / Paper 9 (different numbers per case, so entered per case) |
| History back to 2012 | NOT VERIFIED | |
| Full text in `documentOCRText` | NOT VERIFIED | |
| Rate limits / bulk download | NOT VERIFIED | 6 MB response cap per the spec; CSV/JSON `/search/download` |

**Key advantage:** the inputs (petition, preliminary response, DD briefs) are filed **before** the decision, so there is no outcome leakage (unlike most court-decision datasets).

## 4. Competitive landscape (as of 25 Oct 2026)

- **No public per-petition predictor found.** Analytics tools give historical statistics only: Lex Machina, Docket Navigator, Docket Alarm, Unified Patents, RPX, Patexia.
- **Academic:** Sokhansanj & Rosen 2022 (Applied Sciences, doi 10.3390/app12073656) predicted institution from the patent owner's preliminary response (XGBoost about 90% accuracy, old regime); a 2019 metadata study; PILOT-Bench (arXiv 2601.04758, patent legal-reasoning benchmark).
- **Caveat:** private models at funders and firms would be invisible, and the incumbents hold the data. The edge must be speed and rigour (calibration, the hybrid tree), not data access.
- **To validate:** ask 5–10 US patent litigators or funders whether they use any per-petition predictor, and what they would pay.

---

## 5. Getting the USPTO API key

1. Create a MyUSPTO account at https://my.uspto.gov
2. Verify your identity via **ID.me** (passport + selfie). ⚠️ Unconfirmed whether non-US residents are accepted. Fallback: a US-based partner holds the key.
3. Go to https://data.uspto.gov, sign in, and request an API key.
4. Test it:
   ```bash
   curl -H "X-API-Key: YOUR_KEY" "https://api.uspto.gov/api/v1/patent/trials/proceedings/IPR2025-00318"
   ```
   (Checked 25 Oct 2026 without a key: `/patent/trials/...` routes answer 401, so they exist; `/patent/proceedings/...` answers 403 "Missing Authentication Token", so it does not.)
5. Keys unused for 90 days are deleted. Store the key in Colab **Secrets**, never in the notebook.

**Without a US passport or SSN (non-US residents):**
- ID.me has a route for people outside the US: identity is verified **on a video call** with a foreign passport (e.g. a French passport). No SSN is needed on that route. Choose the international / video-call verification path in ID.me.
- Since 18 Aug 2026 the USPTO profile requires 4 extra fields. Keys whose profile was incomplete on 8 Oct 2026 were deactivated, so fill in the whole profile.
- Keys are personal: the USPTO refuses organisational exemptions. Don't borrow someone else's key; have a collaborator run the scripts with their own key instead.
- Fallbacks if verification fails:
  - ODP "data products" downloadable without a key (check whether PTAB is included)
  - Google BigQuery `patents-public-data.uspto_ptab` (trials metadata; may be outdated; no documents table confirmed) for historical counts from Colab
  - the public PTAB website, to manually check the test cases in §10
  - paid docket services (Docket Alarm, Lex Machina, Docket Navigator)
  - Unified Patents Portal (free account)
- Read the ODP terms of use before downloading at scale (registration was introduced partly to curb AI scraping).

---

## 6. Objective

**Given a filing (an IPR petition), how likely is it to:**
- (a) be **discretionarily denied**?
- (b) be **instituted**?
- (c) later: end with claims cancelled / be affirmed on appeal?

Two prediction moments:

| Moment | Known | Use case | Model |
|---|---|---|---|
| **T0: at filing** | petition + metadata | "Should I file?" | Model A |
| **T1: after briefing** | + DD brief, DD opposition, preliminary response | "What happens next?" | Model B |

---

## 7. Procedural tree (fill in the counts from the real data)

Regimes: **R0** = before 26 Mar 2025 (panels) · **R1** = 26 Mar–17 Oct 2025 (Acting Director Stewart) · **R2** = since 17 Oct 2025 (Director Squires)

```
[0] PETITION FILED (35 USC 311-312)                    N(R0)=___ N(R1)=___ N(R2)=___
 │   inputs: petition, expert declaration, exhibits
[1] FILING DATE ACCORDED                     RULE      N=___  (defective: ___)
 ├─[2] SETTLED/WITHDRAWN PRE-INSTITUTION (§317)  CLASSIFIER   N=___ (yes: ___)
[3] STATUTORY BARS                           RULE
 │   §315(b) >1 yr after service of complaint; §315(a)(1) prior DJ action; joinder §315(c)
 │                                                     N=___  (barred: ___)
[4] DISCRETIONARY STAGE (§314(a), §325(d))   CLASSIFIER #1   [R1, R2; R0 embedded in panel decision]
 │   new inputs: DD brief (~+2 mo), DD opposition (~+3 mo)
 │   factors: parallel litigation/trial date (Fintiv), settled expectations (patent age),
 │            serial/parallel petitions (General Plastic), art already considered (§325(d))
 │   ├─ DISCRETIONARILY DENIED                         N(R1)=___ N(R2)=___
 │   └─ REFERRED TO MERITS                             N(R1)=___ N(R2)=___
[5] MERITS INSTITUTION ("reasonable likelihood")  CLASSIFIER #2
 │   new input: patent owner preliminary response (~+3 mo)
 │   ├─ DENIED ON MERITS                    N(R0)=___ N(R1)=___ N(R2)=___
 │   └─ INSTITUTED                          N(R0)=___ N(R1)=___ N(R2)=___
[6] REHEARING / DIRECTOR REVIEW              (rare; ignore in v1)   N=___
[7] TRIAL (~12 months)
 ├─[7a] SETTLED DURING TRIAL                 CLASSIFIER   N=___
[8] FINAL WRITTEN DECISION (§316(a)(11))     CLASSIFIER #3 (v2)
 │   all unpatentable N=___ | some N=___ | none N=___
[9] FEDERAL CIRCUIT APPEAL (§319)            CLASSIFIER #4 (v3)
     affirmed / reversed / vacated  N=___

P(instituted | filing) = P(not settled) × P(not barred) × P(not DD | regime) × P(merits OK)
```

**v1 scope:** nodes [2]–[5]. Nodes [7]–[9] later, mostly from R0 data.

---

## 8. Architecture

```
INGESTION      ODP client (Python): proceedings/search, {trial}/documents, decisions/search
               pagination, retries, rate limiting, schema fix (patentTrialDocumentDataBag)
     ▼
RAW STORE      JSON metadata + PDFs on Google Drive / GCS (never Colab disk)
     ▼
PROCESSING     1. text: documentOCRText, else PDF → text (pdfplumber / OCR)
               2. normalise doc types (petition / POPR / DD brief / DD opp / DD decision / notice / institution decision)
               3. CASE TABLE: 1 row = 1 petition (Parquet + DuckDB)
               4. LABELS per node (decision titles, outcome fields, summary-notice lists)
               5. TIME CUT-OFF: only docs filed before the prediction moment (T0/T1), to prevent leakage
               6. FEATURES: patent age, claims challenged, # grounds, # parallel petitions,
                  petitioner type, tech centre, regime, parallel litigation*, texts
     ▼
MODELS         per node: base rate → metadata XGBoost → TF-IDF + XGBoost
               → long-doc legal transformer / LLM embeddings / LLM with reasoning (small R2 data)
               + calibration layer (isotonic / Platt)
     ▼
TREE ENGINE    rules + node probabilities → P(DD), P(instituted), later P(FWD)
               + sensitivity analysis + explanations (top factors, quoted brief passages)
     ▼
EVALUATION     time split (train old → test recent), per regime R0/R1/R2,
               accuracy, Brier score, calibration plots, vs base rate and baselines
     ▼
DEMO           Streamlit / Gradio
```

\* Parallel litigation is **not in PTAB data**. v1: extract it from the briefs' text. v2: link to court dockets on CourtListener.

**Stack:** Python, requests, pandas, duckdb, pdfplumber, scikit-learn, xgboost, Hugging Face transformers, streamlit. Runs in Google Colab (CPU for steps 0–3; L4/A100 GPU only for fine-tuning; store data on Drive).

---

## 9. Plan and effort

| Step | Content | Time |
|---|---|---|
| 0 | Feasibility check with the live API (checklist below), then GO / NO-GO | 1–2 days |
| 1 | Data pipeline (download everything to Drive) | 1–2 weeks |
| 2 | Labelling (+ manual check of about 100 cases) | 1 week |
| 3 | Baselines (base rate, metadata, TF-IDF) | 1 week |
| 4 | Language models | 2–3 weeks |
| 5 | Rigorous evaluation (time split, calibration) | 1 week |
| 6 | Hybrid tree + sensitivity analysis | 1 week |
| 7 | Demo + write-up | 1 week |

Total: about 8–12 weeks part-time. Difficulty: medium (the hard parts are clean labels and honest evaluation).

**Risks:** the API may not expose the DD documents or labels cleanly; there is little data in R2 and the rules keep moving; decisions without reasons; R0 data does not reflect today's rules (use it for backtesting the method).

---

## 10. Checklist for the first live session (fills the blanks in §7)

Automated in `step0_feasibility.py`. Run it once the key exists:

```bash
export USPTO_API_KEY=...   # in Colab: a secret named USPTO_API_KEY
python step0_feasibility.py --out step0_out
```

It writes `step0_out/REPORT.md` (a PASS/FAIL/CHECK table, then the details per item) and saves every raw response to `step0_out/raw/`. It makes about 60–80 calls.

- [ ] API key works (the test call returns JSON)
- [ ] Total IPR count by petition-filing year → N at node [0] per regime
- [ ] Distinct values of `trialStatusCategory`, `trialOutcomeCategory`, `decisionTypeCategory`, document type and title fields (**this decides how labels are built**)
- [ ] Test cases IPR2025-00318, -00363, -00877, -00486: DD briefs and Director decisions present? Exact titles?
- [ ] One Squires-era case: is the summary notice in its document list?
- [ ] Is `documentOCRText` filled in for a petition? If not, does the PDF download with the key?
- [ ] Does `IPR2012-00001` exist (history coverage)?

---

## 11. Key sources

- Catala paper: https://arxiv.org/abs/2103.03198v2
- LLM → Catala benchmark (NLLP 2025): https://aclanthology.org/2025.nllp-1.4/
- CNAF × Inria Catala: https://www.lemondeinformatique.fr/actualites/lire-la-cnaf-contribue-au-projet-open-source-catala-de-l-inria-100453.html
- USPTO PTAB data tools: https://www.uspto.gov/patents/ptab/ptab-it-systems
- ODP PTAB API: https://data.uspto.gov/apis/ptab-trials/search-proceedings
- ODP Go client (specs, schema quirks): https://github.com/patent-dev/uspto-odp
- Interim discretionary process: https://www.uspto.gov/patents/ptab/interim-director-discretionary-process
- Microsoft v. ParTec Director decision: https://www.finnegan.com/a/web/2saNKJffvTkbz1ZkgPYzN6/microsoft-v-partec.pdf
- Docket Alarm IPR2025-00877 Paper 10: https://www.docketalarm.com/cases/PTAB/IPR2025-00877/Google_LLC_v._Cellular_South_Inc/10-17-2025-Board/Director_Discretionary_Decision_Refer-10-Director_Discretionary_Decision_Refer/
- Summary notices (Finnegan): https://www.finnegan.com/en/insights/blogs/at-the-ptab-blog/updates-on-director-summary-notices-for-decisions-on-institution.html
- Institution rate collapse: https://ipwatchdog.com/2026/04/08/uspto-stats-show-ipr-institution-rate-plummeted/
- RPX institution data: https://www.rpxcorp.com/data-byte/ptab-institution-rates-dropped-in-2025-amid-discretionary-denial-expansion/
- Unified Patents Q1 2026 report: https://www.unifiedpatents.com/insights/2026/4/14/patent-dispute-report-q1-2026
- Sokhansanj & Rosen 2022: https://doi.org/10.3390/app12073656
- Theo Ai: https://legaltechnology.com/litigation-outcome-prediction-startup-theo-ai-raises-4-2m-seed-funding/
- Supreme Court petitions on discretionary denials: https://blog.patrickiplaw.com/the-director-s-ipr-denials-are-now-before-the-supreme-court
