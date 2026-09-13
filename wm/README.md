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

## Is this JEPA?

Partly, and the part that differs matters.

JEPA maps two views through **tied encoders** into a shared latent space, predicts one
latent from the other, and puts the loss **in latent space** rather than reconstructing
the input. The tied-encoder and latent-diff instincts transfer directly. Three things
do not:

1. **An LLM cannot reason in a learned latent space** — it consumes tokens. Handing it a
   vector needs a trained projector (the LLaVA pattern), which needs training data.
2. **JEPA is a training recipe.** I-JEPA needed ImageNet; V-JEPA millions of videos.
   There is no public corpus of paired (draft, markup) contracts to learn from.
3. **JEPA deliberately discards detail.** The deliverable here is "$3,500,000",
   "18 months", "Delaware". The numbers *are* the contract.

So `encoder.py` keeps JEPA's **shape** with a symbolic encoder:

| JEPA | here |
|---|---|
| tied encoder | `encode(document, llm)` — identical function, run on each document separately, seeing only that document |
| s_x, s_y | the draft's 14 slot values, the markup's 14 slot values |
| latent diff | `latent_diff(s_x, s_y)` — per-slot change = how much each markup weighs |
| energy | `abstraction.check` — authority violations |
| decoder | `render.py` — back to text for the judge |

### The measured reason a learned latent can't be the primary channel

`encoder.py` also runs a **lexical** channel (character-5-gram cosine over aligned
sections) — a real continuous similarity space. Run `python -m wm.run blind`. It shows
the problem directly:

**Legal materiality is anti-correlated with textual magnitude.** At a conventional
threshold of 0.93 the channel catches every structural move — deleted sections, inserted
clauses, rewrites — and misses the three highest-impact edits in the markup:

| edit | legal weight | similarity |
|---|---|---|
| 24 hours → 72 hours (breach notification) | large | ~0.99 |
| Delaware → Maryland (governing law) | large | ~0.98 |
| 2 years → 3 years (term) | large | ~0.99 |

Only at 0.995 do they surface, and then 35 sections are flagged instead of 17. No
threshold separates them, because the separation isn't lexical — it's legal. A slot that
asks "how many hours?" reads 72 regardless of how small the edit was.

**Where the continuous channel does earn its place: recall.** It sees everything, so it
catches material changes the slot schema has no slot for. On this markup it flags four,
and §12.1/§12.2 are real — term 2→3 years, renewal 1→2 years, non-renewal notice 90→180
days — and **none of them appear in the 29 rubric criteria**. That is the honest use of a
latent space here: detection, not reasoning.

## The five steps

**1. Load.** `.docx` → text, preserving tracked changes as `{+inserted+}` / `{-deleted-}`.
Without this the counterparty's changes are invisible. → `task/`

**2. Abstract.** Both documents go through the **same tied encoder** into **14 slots**;
the diff is the signal. A lexical channel runs alongside as a recall guard.
Each slot carries: our draft's value, their markup's value, both verbatim quotes, and
what the authority memo permits. → `encoder.py`, `abstraction.py`

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
| **A4** | the 15 typed Issues | 3,010 |
| **A4G** | A4 + the gravity table (tier per issue, from the authority memo) | 3,650 |
| **A6** | A4 + the raw documents (deployable shape) | 65,900 |

**A2 is the point of the whole design.** A4 beating A0 proves nothing: a frontier model
read the contract for you, and the context got 22× shorter. Either explains a win without
"structure helps" being true. A2 holds information, author and length constant so the only
thing that varies is *form*. Run A2 vs A4 first; if they tie, this is a distillation
result, not a representation result.

## Running it

```bash
pip install openai
export OPENROUTER_API_KEY=sk-or-...

python -m wm.run blind               # lexical change detection; no API calls
python -m wm.run inputs              # sizes; no API calls
python -m wm.run encode              # tied encoder on both documents
python -m wm.run prose               # build the prose twin (one frontier call), cached
python -m wm.run trial A4 anthropic/claude-opus-4.1 3
python -m wm.run all 3               # 4 conditions x 2 models x 3 seeds
python -m wm.run report
```

Every model goes through one OpenRouter client, so transport and parameters are
identical across arms and only the model id varies. Roster is in `llm.py`;
`llm.spend_report()` prints tokens per model.

## Reading the results

- **Primary:** A2 vs A4, same model, paired. The only comparison that isolates *form*
  — same information, same author, same length.
- **A4 vs A4G:** does telling the model which issues are walk-aways help? Kept separate
  because gravity is extra *information*, not just extra form; folding it into A4 would
  confound "structure helps" with "being told what matters helps".
- **Secondary:** A4 on the small model vs A0 on the frontier model — the headline claim.
- **Sanity:** A4 vs A6. If A6 wins clearly, the abstraction is losing information the model
  needed, and it should be deployed as a supplement, not a replacement.
- **Cost:** report tokens and dollars per arm including the world-model build. If the
  pipeline costs more than just calling the frontier model, "small matches frontier" is
  economically empty.

## Gravity

`python -m wm.run gravity` — no API calls.

Weights are not assigned by hand. The client's negotiation authority memo grades every
issue in its own headings (`2.5 Walk-Away`, `4.2 Authorized Range`,
`10.2 Escalation Requirement`, `14.3 Walk-Away`), so each tier is auditable against a
line the client wrote.

```
gravity = tier  x2 where the counterparty crossed the stated limit
```

Moving inside an authorised range is not grave even on a big issue; breaching a walk-away
is grave even on a small one. On this markup: 9 walk-aways at gravity 8, 4 escalations at
6, 2 authorised-range at 4 — total exposure 104 across 15 issues.

## Limits

One task has no statistical power. A 10-point effect needs roughly 40–70 paired task
instances; 5 points needs 120+ across ~60 contracts. LAB has ~60 redline tasks — enough,
if you generalise the abstraction step past this one contract. Treat what is here as a
working pipeline and a pilot, not evidence.
