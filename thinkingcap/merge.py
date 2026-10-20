"""Merge several LoRA adapters into the base weights, each at its own strength.

W = W0 + sum_i w_i * dW_i, where dW_i is adapter i's update. Each adapter was trained on one kind of
data (tool calls, code, math...), so its update stays its own and can be turned down if it spills
onto the other kinds, instead of pooling all the shortened traces into one training run, where
the cuts add up. The result is a plain model directory (no adapter), which Transformers and vLLM
load without custom code:

    python merge.py --out models/merged --adapters adapters/grid/tool-mvt1024:1.0 adapters/grid/math-fade/r2:0.5
    python evaluate.py --model models/merged --out runs/merged
"""

import argparse
import shutil
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapters", nargs="+", required=True, help="path:weight, e.g. adapters/x:0.5")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtype", default="bfloat16", help="dtype of the saved weights")
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)  # sum in fp32
    for spec in args.adapters:
        path, weight = spec.rsplit(":", 1)
        with tempfile.TemporaryDirectory() as folder:
            scaled = Path(folder) / "adapter"
            shutil.copytree(path, scaled)
            weights = scaled / "adapter_model.safetensors"
            tensors = load_file(weights)
            # dW = scaling * B @ A, so scaling B scales the whole update
            save_file({k: v * float(weight) if "lora_B" in k else v for k, v in tensors.items()}, weights)
            model = PeftModel.from_pretrained(model, scaled).merge_and_unload()
        print(f"merged {path} at {float(weight):g}", flush=True)
    model.to(getattr(torch, args.dtype)).save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.model).save_pretrained(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
