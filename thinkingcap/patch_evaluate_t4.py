"""Let evaluate.py run on a T4: choose vLLM's attention backend with an environment variable.

vLLM picks FlashInfer on a T4 and its prefill kernel fails there ("BatchPrefillWithPagedKVCache
failed with error invalid argument"). This adds one optional argument to the engine in
evaluate.py, read from EVAL_ATTENTION_BACKEND; prompts, sampling and token caps are untouched,
so scores stay comparable as long as the baseline is run the same way.

    python patch_evaluate_t4.py
    EVAL_ATTENTION_BACKEND=TRITON_ATTN python evaluate.py --out runs/base
"""

from pathlib import Path

ANCHOR = "enable_lora=rank is not None, max_lora_rank=rank or 16)"
PATCHED = ('enable_lora=rank is not None, max_lora_rank=rank or 16,\n'
           '                **({"attention_config": {"backend": os.environ["EVAL_ATTENTION_BACKEND"]}}\n'
           '                   if os.environ.get("EVAL_ATTENTION_BACKEND") else {}))')

path = Path("evaluate.py")
source = path.read_text()
if "EVAL_ATTENTION_BACKEND" in source:
    print("evaluate.py is already patched")
elif source.count(ANCHOR) != 1:
    raise SystemExit("evaluate.py changed: the engine call to patch was not found")
else:
    path.write_text(source.replace(ANCHOR, PATCHED))
    print("patched evaluate.py")
