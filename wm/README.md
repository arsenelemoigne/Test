# Does a world model of a contract help an LLM negotiate it?

One task from Harvey LAB, four input conditions, two models, one judge.

## What Harvey LAB is

`harveyai/harvey-labs` — an open benchmark (MIT) of **1,671 real-shaped legal tasks**.
A task is a folder:

```
<practice-area>/<task-name>/[scenario-NN]/
├── task.json      title + instructions + a list of pass/fail criteria
└── documents/     the .docx / .eml / .xlsx an associate would actually be handed
```

An agent reads `documents/`, follows `instructions`, and writes the named output files.
An LLM judge then grades each criterion independently against its `match_criteria` text.
There is no gold answer — the criterion text *is* the standard.

LAB's headline score is **all-pass**: 1.0 only if every criterion passes, else 0.0.
That is too sparse to compare arms, so this experiment reports **criterion pass rate**
(`n_passed / n_criteria`) and shows all-pass alongside.

## The task used

`tasks/contracts/data-privacy-security/privacy-addendum-first-turn-redline/scenario-03`

Carden Analytics sent a Data Sharing Agreement; Luminos Insights marked it up. You act for
Carden: decide accept / reject / modify on each change and write a cover note. Inputs are
the initial draft, the markup (Word tracked changes), an internal negotiation authority
memo with hard limits, a privacy policy, a fee schedule and the transmittal email.
**29 criteria** — 20 on substantive positions, 9 on cover-note rationales.

## The five steps

**1. Load.** `.docx` → text, preserving tracked changes as `{+inserted+}` / `{-deleted-}`.
Without this the counterparty's changes are invisible. → `task/`

**2. Abstract.** The contract becomes **14 `Issue` objects**. One issue = one thing the
parties disagree about, carrying: our draft position, their proposed position, their
verbatim words, and what the authority memo permits. → `abstraction.py`

**3. Negotiate.** The model is asked for a JSON list of `Decision`s — one per issue, each
with a disposition, a concrete counter-position and a rationale. → `conditions.py`

**4. Render.** `render.py` turns that JSON into the two text deliverables. **Every
condition goes through this same renderer.** The model never writes the final prose.

**5. Judge.** Each of the 29 criteria graded pass/fail by an LLM judge that sees only the
deliverables. → `judge.py`

There is also a deterministic **authority check** (`abstraction.check`) that flags
decisions breaching a stated limit — a cap over $3.5M, retention over 18 months, a
72-hour breach window, non-Delaware law. A program does this perfectly; a language model
does not do it consistently. It is a guard rail, not a grade.

## Conditions

| | input to the model | ~tokens |
|---|---|---|
| **A0** | all task documents, verbatim | 63,300 |
| **A2** | **prose twin** — same facts as A4, written as flowing prose by the frontier model, no fields or IDs | ≈ A4 |
| **A4** | the 14 typed Issues | 2,840 |
| **A6** | A4 + the raw documents (deployable shape) | 65,700 |

**A2 is the point of the whole design.** A4 beating A0 proves nothing: a frontier model
read the contract for you, and the context got 22× shorter. Either explains a win without
"structure helps" being true. A2 holds information, author and length constant so the only
thing that varies is *form*. Run A2 vs A4 first; if they tie, this is a distillation
result, not a representation result.

## Running it

```bash
pip install anthropic openai
export ANTHROPIC_API_KEY=...

python -m wm.run inputs              # sizes; no API calls
python -m wm.run prose               # build the prose twin (one frontier call), cached
python -m wm.run trial A4 claude-opus-5 3
python -m wm.run all 3               # 4 conditions x 2 models x 3 seeds
python -m wm.run report
```

Open-weights models plug in through `llm.open_weights(model, base_url=...)` — any
OpenAI-compatible endpoint. Nothing is hardcoded to a provider.

## Reading the results

- **Primary:** A2 vs A4, same model, paired. This is the only comparison that isolates form.
- **Secondary:** A4 on the small model vs A0 on the frontier model — the headline claim.
- **Sanity:** A4 vs A6. If A6 wins clearly, the abstraction is losing information the model
  needed, and it should be deployed as a supplement, not a replacement.
- **Cost:** report tokens and dollars per arm including the world-model build. If the
  pipeline costs more than just calling the frontier model, "small matches frontier" is
  economically empty.

## Limits

One task has no statistical power. A 10-point effect needs roughly 40–70 paired task
instances; 5 points needs 120+ across ~60 contracts. LAB has ~60 redline tasks — enough,
if you generalise the abstraction step past this one contract. Treat what is here as a
working pipeline and a pilot, not evidence.
