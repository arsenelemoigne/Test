"""Tests for chirurgical.py: the pure helpers, then probe + candidates + mech end to end on a tiny random Qwen3 (CPU,
no vLLM; the generation phases are checked by the Colab smoke run)."""

import json
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

import chirurgical as ch


def test_helpers():
    assert ch.repeat_share(list(range(20))) == 0.0
    assert ch.repeat_share([1, 2, 3, 4] * 10, n=4) > 0.8
    ids = list(range(40)) + list(range(10, 50))
    assert ch.loop_start(ids, n=8) == 40  # the stretch 10..17 starts again at index 40
    assert ch.loop_start(list(range(100)), n=8) is None
    assert ch.novelty(list(range(50)), list(range(100, 120))) == 1.0
    assert ch.novelty(list(range(50)), list(range(10, 30))) < 0.4  # only the 7 grams across the cut are new
    # copy score: query token 5 at t=6 appeared at j=1, so the target is j+1 = 2
    ids = [9, 5, 7, 8, 1, 2, 5]
    attn = torch.zeros(2, 7, 7)
    attn[0, 6, 2] = 0.9
    attn[1, 6, 0] = 0.9
    s = ch.copy_score(ids, attn, [6])
    assert abs(s[0] - 0.9) < 1e-6 and abs(s[1]) < 1e-6
    # direction removal: masked columns lose (alpha) of their component along u, others unchanged
    W = torch.randn(8, 5)
    u = torch.randn(8)
    u = u / u.norm()
    mask = torch.tensor([1.0, 0, 1, 0, 0])
    V = ch.remove_direction(W, u, mask, 1.0)
    assert torch.allclose(u @ V[:, 0], torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(V[:, 1], W[:, 1])
    # rounding: a 10% change survives bfloat16, a 1e-5 one does not
    W0 = torch.randn(64, 64)
    big = ch.survival(W0, W0 * 1.1, torch.bfloat16)
    tiny = ch.survival(W0, W0 + 1e-5, torch.bfloat16)
    assert big["entries_kept"] > 0.9 and big["norm_kept"] > 0.9, big
    assert tiny["entries_kept"] < 0.1, tiny
    st = ch.paired_stats([0, 1, 0, 1], [1, 1, 0, 0])
    assert st["diff_pp"] == 0 and st["wrong_to_right"] == 1 and st["right_to_wrong"] == 1
    print("helpers ok")


def test_pipeline():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    rng = random.Random(0)
    tmp = Path(tempfile.mkdtemp())
    cfg = Qwen3Config(vocab_size=120, hidden_size=64, intermediate_size=96, num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, max_position_embeddings=512)
    Qwen3ForCausalLM(cfg).save_pretrained(tmp / "model")
    close_id = 7
    pairs, normal = [], []
    for k in range(12):
        prefix = [rng.randrange(10, 120) for _ in range(30)]
        loop = (prefix[-6:] * 3)[:12]
        prod = [rng.randrange(10, 120) for _ in range(12)]
        pairs.append({"id": f"q{k}", "part": "svamp", "prompt_ids": [1, 2, 3], "prefix": prefix, "loop": loop,
                      "productive": prod, "split": "find" if k < 6 else "check"})
        out = [rng.randrange(10, 120) for _ in range(20)] + [close_id] + [rng.randrange(10, 120) for _ in range(5)]
        normal.append({"id": f"n{k}", "part": "mbpp", "prompt_ids": [1, 2, 3], "output_ids": out})
    out = tmp / "run"
    out.mkdir()
    ch.write_jsonl(out / "pairs.jsonl", pairs)
    ch.write_jsonl(out / "normal.jsonl", normal)
    args = SimpleNamespace(out=str(out), model=str(tmp / "model"), attn_pairs=4, attn_len=128, heads=2, layers=2,
                           columns=8, doses=[1, 6], seed=0, check_pairs=6, check_normal=6, normal_len=16)
    ch.probe(args)
    data = torch.load(out / "probe.pt")
    assert len(data["u"]) == 4 and len(data["stability"]) == 4 and data["copy"]["loop"].shape == (4, 4)
    # the loop branches repeat the prefix's last tokens: a random model's copy score cannot be checked for sign,
    # but it must be a finite attention share
    assert torch.isfinite(data["copy"]["loop"]).all() and (data["copy"]["loop"] >= 0).all()
    model = ch.load_hf(args.model)
    names = [n for n, _ in ch.candidates(args, data, model)]
    assert len(names) == 2 * 2 + 2 + 1, names  # 2 heads x {0, 0.5}, 2 doses, 1 random control
    tok = SimpleNamespace(convert_tokens_to_ids=lambda t: close_id)
    ch.mech(args, tok)
    res = json.loads((out / "mech.json").read_text())
    assert "original" in res and all("pass" in v for k, v in res.items() if k != "original")
    # a head zeroed must change the model; restoring must bring the original back exactly (no leak between candidates)
    assert any(abs(v["loop"]) > 0 for k, v in res.items() if k.startswith("head-"))
    print("pipeline ok:", {k: round(v["loop"], 4) for k, v in res.items() if k != "original"})


if __name__ == "__main__":
    test_helpers()
    test_pipeline()
