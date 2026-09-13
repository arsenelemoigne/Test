# Contract world model — Harvey LAB pilot

A world model of a contract, built against one real benchmark task, to test whether a
structured representation lets a smaller model match a frontier model on contract markup.

## The task

Harvey **LAB** (Legal Agent Benchmark), `harveyai/harvey-labs`, MIT licensed.
1,671 tasks, 24 practice areas + contracting, graded by rubric with **all-pass** scoring.

Chosen task — the simplest contract-markup task in the set:

```
tasks/contracts/commercial-vendor-customer/non-disclosure-agreement-first-turn-redline/
├── task.json                                    # title, instructions, 50 criteria
└── documents/
    ├── velantis-standard-nda-v6-2.docx          # our form (the base)
    ├── corrigan-first-markup-nda.docx           # their markup, Word Track Changes
    ├── velantis-nda-negotiation-playbook.docx   # 19 sections, 4 authority tiers each
    ├── email-sung-to-orenbach-review-request.eml # client instructions, 14 issues
    ├── corrigan-transmittal-email.eml
    └── project-lantern-timeline.xlsx
```

**Deliverables:** `counter-turn-redline.docx`, `cover-note.docx`
**Scoring:** 50 binary criteria, dual LLM judge (`claude-sonnet-4-6` + `gpt-5.5`), temperature 0.
Task scores 1.0 only if all 50 pass. Use `n_passed / n_criteria`, not `score`, for this
experiment — at 50 criteria, all-pass is too sparse to carry signal.

Everything in `task_extract/` is the task folder as-is, plus `.txt` conversions
(the `.docx` extractor preserves tracked changes as `{+insertions+}` / `{-deletions-}`).

## The world model

```
nda_world_model/
├── core.py          # generic engine: state, transition, value function
├── velantis_nda.py  # 20 deal points, 3 states, 5 gating rules
├── brief.py         # the context block handed to the drafting LLM
└── __main__.py      # CLI
```

State / transition / value, as a world model requires:

- **State** — `ContractState`: 20 typed `DealPoint`s across 7 balance dimensions.
  Each carries our form position, their proposed position, playbook tier, signed
  favourability, verbatim quote and locator.
- **Transition** — `apply(state, Action) -> state'`. A redline is an action. Pure.
- **Value** — `score(state) -> BalanceReport`: per-dimension favourability, an overall
  balance anchored at 0 = market standard, plus hard-line violations and gating rules.

Interaction effects are hand-written gating rules, not a weighted sum — e.g. affiliates
admitted to *Representatives* without breach-responsibility language, or liquidated
damages struck without substituting an equitable remedy.

### Usage

```bash
python -m nda_world_model pre          # balance of Corrigan's markup:  -20.2, 11 hard-line violations
python -m nda_world_model post         # balance of our counter-turn:   +7.9,  admissible
python -m nda_world_model diff         # 18 deal points moved, +28.1
python -m nda_world_model brief        # ~3.3k tokens of context for the drafting LLM
python -m nda_world_model brief-blind  # instruction-blind version (see below)
python -m nda_world_model checklist    # 48-item completion checklist
python -m nda_world_model whatif       # counterfactual rollouts + sensitivity ranking
python -m nda_world_model sizes        # 52k tokens raw -> 5.2k tokens (10x)
```

## Rubric blindness

The 50 rubric criteria were **not** consulted when choosing the deal points, dimensions,
weights, gating rules or checklist. Those come only from the form, the markup, the
playbook and the engagement email. Any overlap between the checklist and the rubric is
the hypothesis under test, not a design input.

`brief-blind` goes further and withholds the engagement email's per-issue directions too.
This matters: a world model built per *(contract, instruction)* pair is not a
representation, it is a precomputed answer. The blind brief is reusable across any
instruction on this contract; the instructed brief is not. Run both.

## Status

Pilot / demonstration. The favourability weights are illustrative and would be set by
negotiating lawyers in a real build. Nothing here has been scored against the benchmark yet.
