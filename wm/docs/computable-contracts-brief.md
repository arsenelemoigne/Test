# Computable contracts & contract valuation — research brief (13 Sept 2026)

Verified = read directly (mostly GitHub). Snippet = search-engine abstract only (arxiv, SSRN,
Springer, Yale L.J., OECD, Stanford, INRIA, catala-lang.org, accordproject.org were blocked by the
sandbox proxy).

## Foundations
- **Surden, "Computable Contracts", 46 UC Davis L. Rev. 629 (2012)** (snippet). Contractual
  obligations formulated for computer processability; "data-oriented contracting"; prima-facie
  compliance assessment. Surden's own limit: only standardised terms with legal and factual
  certainty compute; vague standards stay outside. Cite; the data/rules/residue split is the reusable idea.
- **Flood & Goodenough, "Contract as Automaton"**, OFR WP 15-04 (2015); AI & Law 30:391 (2022)
  (snippet). Loan agreement as a deterministic finite automaton; checks coherence and completeness
  against an event alphabet. No valuation, no code. Follow-up: Holmes & Beigi, weighted FSTs with
  costs/penalties/probabilities (arXiv 2302.00200, 2023) — the only legal-side sketch of weighted
  states; a report, no artefact.
- **Stanford CodeX Computable Contracts** (partly verified). `stanford-codex/compk` dormant
  (whitepaper 2018). Live output is the Insurance Initiative: policies as Epilog logic programs,
  "ContractNet" answering hypothetical questions — scenario *evaluation*, not valuation.

## Language ecosystem (licences verified on GitHub unless noted)
- **Accord Project** (Linux Foundation, Apache-2.0, JS/TS): Cicero templates bind text ↔ Concerto
  data model ↔ Ergo logic. `cicero` repo renamed `template-archive`; Concerto maintained; Ergo status
  unverified. Reuse: Concerto for clause/data binding.
- **Daml** (Digital Asset, Apache-2.0, active): rights/obligations with authorisation semantics.
  **Daml Finance Contingent Claims**: instrument = tree of Claims (Peyton Jones–Eber combinators),
  lifecycling, experimental valuation semantics, no Monte Carlo. Heavy ledger dependency.
- **L4** (legalese/l4-ide, Apache-2.0, Haskell+TS, very active 2026): deontic MUST/MAY/SHANT rules,
  DECIDE functions, evaluation traces, ladder diagrams, REST/MCP export. No valuation.
- **Symboleo** (uOttawa; Sharifi, Parvizimosaed, Amyot, Logrippo, Mylopoulos, SoSyM 2022):
  obligations + powers with statechart lifetimes, events, roles, assets; SymboleoPC model-checks
  LTL/CTL (nuXmv); Symboleo-IDE MIT, SymboleoAC-Web GPL-3.0; **Symboleo-LLM-Tool** (2026) for
  text→Symboleo. Best formal normative model; no economic layer.
- **Lexon** (dormant, licence unclear); **Ricardian contracts** (Grigg): hash-bound signed text +
  parameters, no execution logic — a provenance pattern only.

## Catala (verified repo; paper snippet)
Merigoux, Chataing, Protzenko, Proc. ACM PL 5 (ICFP) 2021, doi 10.1145/3473582. Apache-2.0, OCaml,
active, "research project… unstable". Literate programming: statute text interleaved with `scope`
declarations and definitions-under-conditions with `exception` blocks; semantics = prioritised
default logic; compiles to OCaml/Python/JS/C plus lawyer-readable PDF. `catala-examples` covers
allocations familiales, aides au logement, impôt sur le revenu, successions, SMIC, US IRC sections.
Nothing on Code civil arts. 1170 / 1231-x. The default/exception idiom fits caps, carve-outs, fee
schedules; obligations-over-time and party powers do not (no deontic or event layer).

## Rules as Code
OECD, Mohun & Roberts, *Cracking the Code* (2020) (snippet). France: **OpenFisca-France**
(AGPL-3.0, Python, dated YAML parameters + variable formulas + population microsimulation) is the
reference engine; mes-aides sits on it. AGPL matters if linked from proprietary code.

## Law and economics of contract value (snippet)
- Contingent-claim framing: Arrow–Debreu state-contingent claims; Triantis, "Financial Contract
  Design in the World of Venture Capital", 68 U. Chi. L. Rev. 305 (2001). **Peyton Jones, Eber &
  Seward, "Composing contracts", ICFP 2000** — compositional valuation semantics for
  combinator-built contracts (root of LexiFi, Daml claims, luphord/monte-carlo-contracts).
- **Scott & Triantis, "Anticipating Litigation in Contract Design", 115 Yale L.J. 814 (2006)**:
  precise vs vague terms trade front-end drafting cost against back-end litigation cost; vagueness is
  often deliberately efficient. Also 56 Case W. Res. L. Rev. 187 (2005); Triantis, 62 La. L. Rev. 1065.
- **Ayres & Gertner, 99 Yale L.J. 87 (1989)**: majoritarian vs penalty defaults — valuing silence vs
  a drafted clause.
- Numbers on clauses: Marotta-Wurgler, 4 JELS 677 (2007) — bias index over 647 EULAs, no price per
  term, no price premium for better terms. Pricing exists only in finance: Bradley & Roberts (QJF
  2015, covenants → yield spread); Chamon, Schumacher & Trebesch (J. Int'l Econ. 2018, governing-law
  premium on bonds); Choi, Gulati & Scott on CACs (100 B.U. L. Rev. 2020) and "Contractual
  Landmines" (41 Yale J. Reg. 307, 2024). **No study prices SaaS liability caps.**

## LLM → formal representations (2023–2026, snippet)
Text→Catala: NLLP@EMNLP 2025 benchmark (aclanthology 2025.nllp-1.4); "Closing the Loop: Formally
Verified Law as a Reward Signal" (arXiv 2606.23913); arXiv 2605.25186; arXiv 2606.16118.
Text→Symboleo: arXiv 2411.15898 + Symboleo-LLM-Tool. Text→deontic logic: arXiv 2506.08899;
arXiv 2507.02846 (GPT-4o → Python, 89 % unit-test pass on 13 US breach-notification statutes);
arXiv 2604.02276. ContractCheck (FOL/SMT consistency of SPAs), AI & Law 2025, arXiv 2504.18422.
Negotiation: Rawlsian Agents (2025/26), NegotiationGym (arXiv 2510.04368), AgenticPay (2602.06008),
ANAC 2025 — **none negotiates over a formalised contract with a valuation model.**

## Answers
- **Scenario / Monte-Carlo valuation over an executable contract?** Yes for financial contracts
  (LexiFi proprietary; luphord/monte-carlo-contracts MIT; Daml claims; ACTUS with a restrictive
  licence). **No for legal contracts.** No Catala/Symboleo/L4/Accord work attaches cash flows or
  party utilities.
- **Closest artefact to build on:** normative layer from Symboleo or L4; cash-flow layer from a
  Peyton Jones–Eber claim tree (luphord as reference implementation). Catala's default logic if the
  core is exception-heavy conditional definitions (caps, carve-outs) and French-law defaults.
- **Critics:** Surden (only certain terms compute); Scott & Triantis (encoding vagueness as precision
  destroys or fabricates value); Cummins & Clack 2022 (you encode an interpretation, not the text);
  Sklaroff, U. Pa. L. Rev. 2017 (rigidity cost); Lipshaw, "The Persistence of 'Dumb' Contracts";
  Cohney & Hoffman, 105 Minn. L. Rev. 319 (2020) (scripts inside a natural-language stack).

## Reuse / cite / avoid
Reuse: Symboleo semantics + LLM tool; L4 for rule evaluation and traces; Catala default-logic idiom;
luphord combinators as a valuation kernel; OpenFisca's dated-parameter pattern (AGPL).
Cite: Surden 2012; Flood & Goodenough 2022; Peyton Jones & Eber 2000; Merigoux et al. 2021; Sharifi
et al. 2022; Scott & Triantis 2006; Ayres & Gertner 1989; Marotta-Wurgler 2007; Bradley & Roberts
2015; NLLP 2025; arXiv 2411.15898; OECD 2020.
Avoid: Lexon; stanford-codex/compk; ACTUS core licence; Ergo/Cicero as a runtime; Ricardian
contracts beyond hash-binding; any claim that prior work prices vague SaaS terms.
