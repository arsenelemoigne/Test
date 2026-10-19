# ThinkingCap Test: thinking probe and trace pruning

Tools for the [BottleCapAI ThinkingCap Test](https://github.com/BottleCapAI/ThinkingCap-Test):
make Qwen3-0.6B think shorter without losing accuracy.

| File | What it does |
|---|---|
| `probe_curve.py` | Samples answers, then reads p(correct answer) every few thinking tokens by forcing `</think>` and scoring each candidate answer. Shows how much thinking comes after the answer is already settled. |
| `prune_traces.py` | Turns correct rollouts into shorter ones: cuts after p(correct) settles, tries deleting steps that add nothing, and keeps a trace only if the model still answers it correctly on its own. Writes SFT targets. |
| `sft.py` | LoRA fine-tune on those targets; the adapter loads into the test's `evaluate.py --adapter`. |
| `thinking_probe_colab.ipynb` | Runs the above on a Colab GPU. |
| `tests/test_pipeline.py` | Unit tests plus an end-to-end test on a tiny toy model (CPU, under a minute). |
| `docs/thinking-reduction-research.md` | Literature survey and method proposal. |
| `docs/toy_curve.png` | The probe curve on the toy model. |

## Run

The scripts import the test repo's `utils/`, so copy them into a ThinkingCap-Test checkout:

```bash
git clone https://github.com/BottleCapAI/ThinkingCap-Test && cd ThinkingCap-Test
uv pip install -r requirements.txt
cp /path/to/thinkingcap/*.py . && cp /path/to/thinkingcap/tests/test_pipeline.py tests/
python utils/train/prepare_data.py
python probe_curve.py --out runs/probe                                  # add --dtype float16 on a T4
python prune_traces.py --rollouts runs/probe/rollouts.jsonl --out runs/prune
python sft.py --data runs/prune/sft.jsonl --out adapters/pruned-sft
python evaluate.py --out runs/pruned-sft --adapter adapters/pruned-sft
python compare.py runs/base runs/pruned-sft
```

Tests: `python -m pytest tests/test_pipeline.py -q` (needs torch, transformers, peft, matplotlib, pytest).

## Status

- Tested only on a tiny toy model: the code is checked, the method is not yet measured on Qwen3-0.6B.
- The vLLM sampling step in `probe_curve.py` has not been run yet (it needs a GPU).
- Not built yet: the "select shortest correct" mode, conformal calibration of the cut threshold,
  and the RL stage with a Lagrangian accuracy constraint.
