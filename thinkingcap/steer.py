"""Steering vector for concise thinking, stored in the weights as a plain bias (no custom code).

ASC-style: sample --group replies to each question with the model (optionally with a LoRA),
pair each question's shortest correct reply with its longest correct one (at least --ratio
times longer), and take the mean difference of the residual stream after layer --layer over
the thinking tokens: v = mean(concise) - mean(verbose).

The vector goes into the weights through Qwen3's `attention_bias` flag. A bias b on v_proj adds
the same vector to every value, so, since attention weights sum to one, every position's
attention output gains W_o R b (R repeats each key-value head over its query heads). Solving
W_o R b = strength * v puts exactly strength * v into the residual stream at that layer, at
every position, in a model that standard Transformers and vLLM load as is. All other biases are
zero. A LoRA trained on the base model still applies on top (--adapter at evaluation).

    python steer.py --out steered/raw --strengths 0.5,1,2
    python steer.py --adapter adapters/rl_kl --out steered/rl --strengths 0.5,1,2
    python rl_train.py --eval-only --model steered/raw/s1 --data data/val.jsonl --out runs/held/steer1
"""

import argparse
import json
import random
from pathlib import Path

from rl_train import Sampler, balanced_order, is_correct, read_rows


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter", help="LoRA active while sampling and reading activations")
    parser.add_argument("--data", default="data/train.jsonl")
    parser.add_argument("--sources", default="gsm8k,skywork,arc_easy,commonsense_qa")
    parser.add_argument("--out", required=True)
    parser.add_argument("--questions", type=int, default=192)
    parser.add_argument("--group", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--ratio", type=float, default=1.5, help="verbose reply at least this much longer")
    parser.add_argument("--layer", type=int, help="decoder layer that receives the vector (default: middle)")
    parser.add_argument("--strengths", default="0.5,1,2")
    parser.add_argument("--vector", help="reuse a saved vector.pt (skips sampling); needs --layer")
    parser.add_argument("--slice-seed", type=int, default=99, help="question slice, apart from the RL runs'")
    parser.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    parser.add_argument("--attention-backend", default=None)
    parser.add_argument("--gpu-memory", type=float, default=0.5)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def concise_verbose_pairs(groups, ratio):
    """(shortest correct, longest correct) per group, when the longer is at least `ratio` times longer."""
    pairs = []
    for prompt, replies, correct in groups:
        right = sorted((r for r, ok in zip(replies, correct) if ok), key=len)
        if len(right) >= 2 and len(right[-1]) >= ratio * len(right[0]):
            pairs.append((prompt, right[0], right[-1]))
    return pairs


def mean_activation(model, prompt, reply, layer, close_id, device):
    """Residual stream after `layer`, averaged over the reply's thinking tokens."""
    import torch
    thinking = reply.index(close_id) if close_id in reply else len(reply)
    ids = torch.tensor([prompt + reply[:max(1, thinking)]], device=device)
    with torch.no_grad():
        hidden = model(input_ids=ids, output_hidden_states=True).hidden_states[layer + 1]
    return hidden[0, len(prompt):].float().mean(0)


def value_bias(attention, vector):
    """b with W_o R b = vector: a v_proj bias that adds `vector` to the attention output."""
    import torch
    weight = attention.o_proj.weight.float()
    head_dim = attention.head_dim
    kv_heads = attention.v_proj.weight.shape[0] // head_dim
    heads = weight.shape[1] // head_dim
    repeat = torch.zeros(heads * head_dim, kv_heads * head_dim, device=weight.device)
    for head in range(heads):
        kv = head // (heads // kv_heads)
        repeat[head * head_dim:(head + 1) * head_dim, kv * head_dim:(kv + 1) * head_dim] = torch.eye(head_dim)
    system = weight @ repeat
    return torch.linalg.lstsq(system, vector.float().to(weight.device)[:, None]).solution[:, 0]


def steered_model(model_name, layer, vector, dtype):
    """The base model with attention biases switched on: zero everywhere but `layer`'s v_proj."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(model_name)
    config.attention_bias = True
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config, dtype=dtype)
    with torch.no_grad():
        for block in model.model.layers:
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                getattr(block.self_attn, name).bias.zero_()
        attention = model.model.layers[layer].self_attn
        attention.v_proj.bias.copy_(value_bias(attention, vector).to(attention.v_proj.bias.dtype))
    return model


def main():
    args = parse_arguments()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype) if device == "cuda" else torch.float32
    if args.vector:
        vector = torch.load(args.vector)
        summary = {"vector": args.vector, "layer": args.layer, "vector_norm": float(vector.norm()), "strengths": {}}
        return write_strengths(args, tokenizer, args.layer, vector, dtype, out, summary)
    eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
    close_id = tokenizer.convert_tokens_to_ids("</think>")
    keep = set(args.sources.split(","))
    questions = balanced_order([r for r in read_rows(args.data) if r["source"] in keep],
                               random.Random(args.slice_seed))[:args.questions]
    prompts = [tokenizer.apply_chat_template(q["prompt"], tokenize=True, add_generation_prompt=True,
                                             enable_thinking=True) for q in questions]

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    replies = Sampler(args, tokenizer).sample(prompts, args.adapter, 0, model)
    model.eval()
    groups = [(prompt, group, [bool(r) and r[-1] == eos and is_correct(
        tokenizer.decode(r, skip_special_tokens=True), q) for r in group])
        for q, prompt, group in zip(questions, prompts, replies)]
    pairs = concise_verbose_pairs(groups, args.ratio)
    if not pairs:
        raise SystemExit("no concise/verbose pairs: raise --questions or lower --ratio")

    layers = model.config.num_hidden_layers
    layer = args.layer if args.layer is not None else layers // 2
    diffs = [mean_activation(model, p, short, layer, close_id, device)
             - mean_activation(model, p, long, layer, close_id, device) for p, short, long in pairs]
    vector = torch.stack(diffs).mean(0)
    torch.save(vector.cpu(), out / "vector.pt")
    summary = {"pairs": len(pairs), "layer": layer, "vector_norm": float(vector.norm()),
               "concise_tokens": sum(len(s) for _, s, _ in pairs) / len(pairs),
               "verbose_tokens": sum(len(v) for _, _, v in pairs) / len(pairs),
               "adapter": args.adapter, "strengths": {}}
    del model
    write_strengths(args, tokenizer, layer, vector, dtype, out, summary)


def write_strengths(args, tokenizer, layer, vector, dtype, out, summary):
    for strength in [float(s) for s in args.strengths.split(",")]:
        steered = steered_model(args.model, layer, strength * vector, dtype)
        target = out / f"s{strength:g}"
        steered.save_pretrained(target)
        tokenizer.save_pretrained(target)
        summary["strengths"][f"{strength:g}"] = str(target)
        del steered
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
