# Handoff: test "stop when thinking stops paying" + "fading worked examples"

A second Claude session runs this track while the first one keeps working on RL, steering and
AdaptThink on its own Colab runtime. The two must not share a runtime or a branch.

## The challenge

BottleCapAI ThinkingCap Test: make Qwen3-0.6B use fewer tokens (thinking included) without
losing accuracy. The hard constraint is **accuracy change >= -3 points** against the raw model.
The model must load in plain Transformers and vLLM with no custom code, so a LoRA or changed
weights are fine and a custom decoding loop is not.

## Where things are

- Repo: `github.com/arsenelemoigne/Test`. Base branch: `ccr-f7e14615-o4qu8r`. Code lives in `thinkingcap/`.
- **Create your own branch** from it (for example `mvt-fading`) and push only there. In your
  copy of `thinking_probe_colab.ipynb`, change `B=` / the wget URLs to your branch.
- Read these first:
  - `EXPERIMENT_LOG.md`: every run, its hyperparameters and its numbers.
  - `IDEA.md`: the method and why the SFT rounds failed.
- Benchmark harness: the challenge's `evaluate.py` and `compare.py`, run from the ThinkingCap-Test
  clone the notebook sets up.
- Evaluation subset: `subset_questions.py`, the first 40 questions per benchmark (240 in total),
  `--samples 2`.
- **Run the raw-model evaluation again on your own runtime** before comparing. Don't reuse
  numbers from another machine.

## Tools that already exist

| File | What it does |
|---|---|
| `probe_curve.py` | Samples replies. Every 16 thinking tokens it closes the thinking with `</think> Final answer:` and reads p(correct answer) by teacher forcing, all cuts packed into one forward pass. This gives the gain curve p(correct) vs tokens spent, per reply. Outputs: `rollouts.jsonl`, `probes.jsonl`. |
| `prune_traces.py` | Cuts each correct trace at the **settle point** (p >= 0.9 and stays), plus `--margin` paragraphs. Deletes flat paragraphs. Keeps a cut only if `--check-samples` natural answers from the cut trace are correct. Writes `sft.jsonl`. |
| `sft.py` | LoRA r16 SFT on completions (fp32 LoRA weights, `--max-per-source`). |
| `subset_questions.py`, `patch_evaluate_t4.py` | Evaluation subset; vLLM attention backend patch (TRITON_ATTN on a T4). |

## What already failed (read before designing)

SFT on cut traces made the model **underthink**:
- Round 1: +39% fewer tokens, -6.3 pts.
- Round 2: +60% fewer tokens, -8.0 pts; SVAMP -24 pts.

The model learned "stop right after restating the question". Most of the cut traces came from
easy multiple-choice questions it knew at once. Any SFT variant has to beat that failure, so
report SVAMP separately.

## Experiment 1: Marginal Value Theorem cut (Charnov 1976)

Treat thinking as foraging for certainty. g(t) is the probe curve p(correct | first t thinking
tokens). The MVT says to leave the patch when the marginal gain equals the average rate,
including a travel cost c (the answer tokens that follow anyway):

    stop at the first t where  g'(t) <= g(t) / (t + c)    (smooth g' over a few cuts)

This differs from the existing settle rule, which uses a fixed 0.9 threshold. Steps:

1. Implement it as `prune_traces.py --rule mvt --travel C`, alongside the settle rule.
2. Report how far the MVT cut point is from the settle point (median tokens, and the share of
   traces cut earlier vs later).
3. SFT on the cut traces, then evaluate. Keep the natural-answer check and the per-source cap.
4. Sweep c: a larger travel cost means staying longer.

## Experiment 2: fading worked examples (Renkl; Stepwise Internalization, Deng et al. 2024)

Remove reasoning gradually instead of one brutal cut, so the model absorbs the skipped steps:

1. Round k: from each correct trace, delete about 10% more of the thinking: the paragraphs with
   the smallest probe gain first (or the earliest ones, as a variant).
2. Keep a trace only if natural answers from the shortened trace are still correct.
3. SFT round k **starting from round k-1's adapter**, for about 5 rounds.
4. Evaluate after each round and stop at the last round within -3 pts.

This is the natural fix for the SFT failure above. MVT can set the final target length.

## Practical notes

- Colab Pro, L4, `DTYPE=bfloat16`.
- Editing a cell that is queued doesn't change what runs; interrupt and rerun.
- Each `!` line of a running cell stops with one click on its run button.
- Log every run (hyperparameters, data counts, per-benchmark tokens saved and accuracy change)
  in a new section of `EXPERIMENT_LOG.md` **on your branch**.
- Keep explanations to the user simple.
