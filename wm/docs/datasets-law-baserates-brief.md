# Public data, law modules, base rates — research brief (13 Sept 2026)

(S) = search snippet only; huggingface, arxiv, atticusprojectai.org, stanfordnlp.github.io,
crosby.ai, micro1.ai were blocked by the sandbox proxy.

## A. Datasets — human labels for the typed fields
Y = labelled span/value; (h) = NLI hypothesis / retrieval relevance; T = topic label only; rubric = graded in a pass/fail rubric.

| Field | CUAD | LegalBench cuad_* | ACORD | LEDGAR/LexGLUE | ContractNLI | MAUD | Harvey LAB | RedlineBench |
|---|---|---|---|---|---|---|---|---|
| Term / expiration | Y | Y | (h) | T | – | – | rubric | rubric |
| Renewal | Y | Y | – | T | – | – | rubric | rubric |
| Notice period | Y | Y | – | T | – | – | rubric | rubric |
| Fees / minimum commitment | Y (min. commitment) | Y | – | T | – | – | rubric | rubric |
| Price escalator | – | – | – | – | – | – | ? | ? |
| Cap on liability | Y | Y | (h) | T | – | – | rubric | rubric |
| Uncapped liability | Y | Y | (h) | – | – | – | rubric | rubric |
| Consequential-damages waiver | – | – | (h) | T | – | – | rubric | rubric |
| Warranty duration | Y | Y | – | T | – | – | – | – |
| Governing law | Y | Y | (h) | T | – | – | – | – |
| Audit rights | Y | Y | – | T | – | – | – | – |
| Change of control | Y | Y | – | T | – | Y | – | – |
| Termination for convenience | Y | Y | – | T | – | – | rubric | rubric |
| Liquidated damages | Y | Y | (h) | T | – | – | – | – |
| Indemnification | – | – | (h) | T | – | – | rubric | rubric |
| Source-code escrow | Y | Y | – | T | – | – | – | – |
| Non-solicit | Y | Y | – | T | (h) | – | – | – |
| MFN | Y | Y | – | T | – | – | – | – |

CUAD covers 14/18; the 4 gaps (escalator, consequential-damages waiver, indemnification, fees) need our own overlay labels.

| Dataset | Content | Licence | SaaS / licence agreements? | Note |
|---|---|---|---|---|
| CUAD (github.com/TheAtticusProject/cuad) | 510 EDGAR contracts, 13k+ expert spans, 41 categories | CC BY 4.0 | Yes (License, Hosting, Service, Maintenance, Outsourcing, Reseller) | **Extraction testbed** |
| LegalBench (HazyResearch) | 162 tasks; 41 cuad_*, 46 maud_*, 17 contract_nli_* | per task | via CUAD | leaderboard row, small samples |
| ACORD (ACL 2025) | 114 queries, 126k query–clause pairs, 1–5 relevance | CC BY 4.0 | yes | retrieval, not extraction |
| LEDGAR / LexGLUE | 60,540 EDGAR contracts, 846k provisions | CC BY 4.0 | mixed | noisy topic labels |
| MAUD | 152 merger agreements, 92 deal points | CC BY 4.0 | no | method precedent |
| ContractNLI | 607 NDAs, 17 hypotheses | CC BY 4.0 (HF mirror says NC-SA, S) | no | — |
| SEC EDGAR Ex-10 | all material contracts since 1993 | public domain | thousands | unlabelled |
| Harvey LAB (github.com/harveyai/harvey-labs) | ~1,660 tasks, 28 practice folders; synthetic matters + attorney rubrics | **MIT** | `contracts/commercial-vendor-customer` (40 tasks: saas-api-subscription-{first-turn-redline, subsequent-turn-redline, counterparty-paper-review, first-draft, playbook-escalation, term-negotiation}, MSA/SLA/VSA redlines); `contracts/ip-licensing` (14–15 tasks incl. license-agreement-first/subsequent-turn-redline, technology-cross-license) | ~13 licence/SaaS redline tasks; documents synthetic |
| **RedlineBench** (crosbylegal, June 2026) (S) | 3 simulated SaaS/professional-services MSA negotiations × 4 attorney turns; rubrics: legal correctness 25.7 %, commercial context 33.4 %, negotiation quality 17 %, acceptance prediction 10.2 %, deal-closing 13.7 % | CC BY 4.0 | **yes — template → marked-up state turn by turn** | closest public "template vs negotiated"; tiny |
| Better Call CLAUSE (EACL 2026) (S) | 7,500+ perturbed contracts, 10 anomaly categories | ? | via CUAD | synthetic discrepancies |
| Common Paper 2026 SaaS Benchmark | 1,000+ companies' CSAs (report, no raw text) | report | yes | **priors**: 99 % have a fee-multiple cap, 96 % at 1×, 2× supercaps 2.7 %, 5× 1.7 %; SLA present in 39 % (99.5 % monthly most common) |
| Bonterms / Common Paper CSA | standard SaaS templates | CC BY 4.0 | yes | template half only |

Template-vs-negotiated pairs: **no real-world public corpus.** Candidates: RedlineBench (real attorneys, simulated deal), Harvey LAB redline tasks (synthetic, rubric-scored), EDGAR "Amended and Restated" pairs (real, post-signature, unlabelled).

French / EU: data.gouv.fr has no contract-text corpus (LEGI, JADE, DECP only). DECP = award metadata since 2018, no clause text. CCAG-TIC 2021 (arrêté 30 mars 2021) = default public IT/SaaS terms. **Judilibre API** (PISTE, free): Cass. since 30 Sept 2021; all 36 cours d'appel since 15 Apr 2022 (~180k/yr); tribunaux judiciaires rolling 2023–2025; tribunaux de commerce delayed — full-text on "clause limitative de responsabilité" / "faute lourde" / "obligation essentielle" is where French enforceability rates can be measured. Légifrance API (PISTE). Catala `french-law`: socio-fiscal only; contract-law modules would be new.

## B. Law modules
### French law
1. **Art. 1231-3** — only foreseeable damage, unless faute lourde / dolosive. `recoverable = foreseeable if not gross_or_wilful else all_direct`.
2. **Cap defeated by faute lourde** — Ch. mixte 22 avr. 2005, n° 02-18.326; Faurecia II, Com. 29 juin 2010, n° 09-11.841: faute lourde "ne peut résulter du seul manquement… fût-elle essentielle". High bar; placeholder P ≈ 0.1–0.2 until mined on Judilibre.
3. **Art. 1170 / Chronopost (Com. 22 oct. 1996, n° 93-18.632) / Faurecia II** — cap réputée non écrite only if it empties the essential obligation; Faurecia II **upheld** a fees-based negotiated cap. `cap_void = derisory AND not negotiated`.
4. **Consequential-damages exclusions** — valid B2B subject to 2–3 (and 1171 for adhesion contracts).
5. **Art. 1231-5** — penalty clauses moderable d'office if manifestement excessive. `penalty = clip(stipulated, k_low×loss, k_high×loss)`.
6. **Termination for convenience with fee** — valid if stipulated; fee = clause de dédit (not moderable; Civ. 3e 8 janv. 2026, n° 24-12.082 (S)) unless coercive/full price → requalified clause pénale (rule 5). Check L442-1 II rupture brutale.
7. **Art. L442-1 I 2° C. com.** déséquilibre significatif — Com. 13 mai 2026, n° 24-17.137: art. 1171 does not apply where L442-1 applies; needs *soumission*; ≈ 0 for negotiated enterprise SaaS, > 0 for adhesion paper.

### Delaware / New York / Texas
8. **DE** — caps and waivers enforced; cannot shield intentional fraud (ABRY Partners v. F&W, Del. Ch. 2006; Express Scripts v. Bracket, 248 A.3d 824 (Del. 2021)); gross negligence **may** be capped. Most vendor-favourable.
9. **NY** — void for gross negligence "smacking of intentional wrongdoing" (Kalisch-Jarcho, 58 N.Y.2d 377 (1983); Sommer v. Federal Signal, 79 N.Y.2d 540 (1992); Abacus v. ADT, 18 N.Y.3d 675 (2012)); deliberate non-performance for self-interest is not "wilful" (Metropolitan Life v. Noble Lowndes, 84 N.Y.2d 430 (1994)).
10. **TX** — Bombardier v. SPEP, 572 S.W.3d 213 (Tex. 2019): limitation bars punitive damages even vs fraud plaintiff affirming the contract; § 2.719 unconscionability; gross-negligence releases: appellate split (Van Voris v. Team Chop Shop, 402 S.W.3d 915); **fair notice / conspicuousness** (Dresser v. Page Petroleum, 853 S.W.2d 505 (Tex. 1993)).
11. Differences: fraud carve-out mandatory everywhere; gross-negligence carve-out mandatory NY, arguable TX, not DE; TX adds conspicuousness.
12. **Arbitration with injunctive carve-out** — enforced when explicit; ambiguity read pro-arbitration (FAA); DXP Enters. v. Goulds Pumps (S.D. Tex.); Delaware Chancery: carve-outs reach arbitrability only if "obviously broad and substantial"; DRAA § 5804. P ≈ 0.9 if claim-scoped, ≈ 0.5 if only "equitable relief".

## C. Base rates
| Quantity | Value | Source |
|---|---|---|
| Hyperscaler uptime 2025 | AWS 99.982 %, Azure 99.975 %, GCP 99.973 %; AWS ≥ 1 outage/month in 10 of 12 months | IncidentHub 2025 |
| Outage duration | median 53 min; 18 % > 4 h; 6 % > 24 h; 53 % of operators had an outage in 3 yrs | Uptime Institute 2025 |
| SLA prevalence | 39 % of SaaS CSAs; 99.5 % monthly most common | Common Paper 2026 |
| Data breach cost | 2025: global $4.44M, US $10.22M; PII $160/record, IP $178/record. 2024: $4.88M / $9.36M | IBM 2024, 2025 |
| Breach frequency per firm | not published by IBM — DBIR / cyber-insurer loss ratios needed | — |
| IP claims vs software vendors | NPE defendants +30 % YoY H1 2025; software ~30 % of suits; ~100 demand letters per suit; 20 % of VC-backed startups received an NPE demand | RPX 2025; Chien / SCU |
| IP defence cost | AIPLA 2025 median through appeal ~$0.6M (< $1M at risk) to ~$3.6M (> $25M); NPE tech settlements $0.5–45M, ~14 months (S) | AIPLA 2025; PatentPC |
| Enterprise SaaS early termination | GRR 86–90 % → 10–14 % gross churn (all B2B); enterprise (ACV > $100k) logo churn 1–2 %/yr | KeyBanc / Sapphire 2025 summaries |
| US contract litigation | median federal civil disposition 6.9 mo; to trial 35.6 mo (D. Del. 40.6); ~1 % reach trial; breach case with $250k at stake costs $91–145k through trial (Norton Rose 2024, S) | Hughes Hubbard; MBH |
| France commercial litigation | tribunaux de commerce 8.1 mo avg (2024), référé 2.6 mo; cours d'appel 13.7 mo avg; fees €2–15k first instance (low confidence); art. 700 €500–3,000 | RSJ 2024 ch. 4.4, 4.5 |

## Recommendation
Extraction benchmark: **CUAD** (14/18 fields labelled) + a CC BY overlay for the 4 missing fields.
Valuation / negotiation: no public ground truth. Proxies: **RedlineBench** (SaaS MSA + 4 attorney turns), **Harvey LAB** `saas-api-subscription-*` and `license-agreement-*` tasks (~13), Common Paper frequencies as priors. Treat these as *process* benchmarks (did the engine move the right clause the right way), not as validation of dollar values — those must be defended with the law modules, the base rates, and Judilibre-mined French enforceability rates.
