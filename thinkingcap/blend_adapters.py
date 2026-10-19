"""Exact blend of two LoRA adapters: delta_W = w * B1 A1 + (1 - w) * B2 A2.

Averaging the A and B matrices separately would not give that (the product mixes cross terms).
Stacking them does: B = [sqrt(w) B1, sqrt(1-w) B2] and A = [sqrt(w) A1; sqrt(1-w) A2] multiply
to exactly the weighted sum, as a plain LoRA of rank r1 + r2. lora_alpha is set so the new
rank keeps the inputs' scale (alpha / r), which must be equal in both.

Use it to trade length for accuracy without retraining: blending a shorter but less accurate
adapter with a safer one lands in between.

    python blend_adapters.py adapters/rl_kl3 adapters/rl_kl --weight 0.5 --out adapters/blend50
"""

import argparse
import json
import math
import shutil
from pathlib import Path

WEIGHTS = "adapter_model.safetensors"
CONFIG = "adapter_config.json"


def blend(first, second, weight):
    """State dicts of two LoRAs with the same modules -> state dict of the stacked blend."""
    import torch
    out = {}
    for key, tensor in first.items():
        other = second[key]
        if "lora_A" in key:
            out[key] = torch.cat([math.sqrt(weight) * tensor, math.sqrt(1 - weight) * other], dim=0)
        elif "lora_B" in key:
            out[key] = torch.cat([math.sqrt(weight) * tensor, math.sqrt(1 - weight) * other], dim=1)
        else:
            raise SystemExit(f"cannot blend non-LoRA weight {key}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--weight", type=float, required=True, help="share of the first adapter, 0..1")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from safetensors.torch import load_file, save_file

    configs = [json.loads((Path(a) / CONFIG).read_text()) for a in (args.first, args.second)]
    scales = [c["lora_alpha"] / c["r"] for c in configs]
    if abs(scales[0] - scales[1]) > 1e-9 or configs[0]["target_modules"] != configs[1]["target_modules"]:
        raise SystemExit("adapters differ in scale or target modules")
    weights = blend(load_file(Path(args.first) / WEIGHTS), load_file(Path(args.second) / WEIGHTS), args.weight)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in weights.items()}, out / WEIGHTS)
    config = dict(configs[0])
    config["r"] = configs[0]["r"] + configs[1]["r"]
    config["lora_alpha"] = scales[0] * config["r"]
    (out / CONFIG).write_text(json.dumps(config, indent=1))
    for extra in Path(args.first).iterdir():
        if extra.name not in (WEIGHTS, CONFIG) and extra.is_file():
            shutil.copy(extra, out / extra.name)
    print(f"{out}: rank {config['r']}, {args.weight:g} x {args.first} + {1 - args.weight:g} x {args.second}")


if __name__ == "__main__":
    main()
