"""Tests for astra_harness.py without spending anything: a scripted stand-in for Astra, the REAL kernel process on a
tiny random Qwen3 (CPU), a stand-in forward tool. Checks the block loop, the refusals, the pass / code-run counting,
the final answer, and the HARD spending stop."""
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import astra_harness as ah

SNAP = next(Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*"))


def tiny_model(d):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=151936, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=1, head_dim=16, tie_word_embeddings=True)
    Qwen3ForCausalLM(cfg).save_pretrained(d)
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        shutil.copy(SNAP / f, Path(d) / f)


def make_args(tmp, **kw):
    a = dict(provider="openrouter", worker="", out=str(tmp / "run"), work=str(tmp / "work"), model_dir=str(tmp / "model"), practice=str(tmp / "practice.jsonl"),
             practice_n=3, passes=2, code_runs=6, cap_usd=1.0, max_reply=1000, handoff=100000, timeout=60, scratch=str(tmp / "s"))
    a.update(kw)
    return SimpleNamespace(**a)


def scripted(replies, cost):
    seq = iter(replies)
    return lambda messages: (next(seq), cost, {"prompt_tokens": 0})


def test_session():
    tmp = Path(tempfile.mkdtemp())
    tiny_model(tmp / "model")
    (tmp / "practice.jsonl").write_text('{"id": "p1", "type": "svamp"}\n')
    py = lambda c: f"```python\n{c}\n```"  # noqa: E731
    fw = lambda r: "```forward\n" + json.dumps(r) + "\n```"  # noqa: E731
    replies = [
        py("w = load('model.layers.0.mlp.down_proj.weight')\nw.shape"),
        py("save_edit('model.embed_tokens.weight', load('model.embed_tokens.weight'))"),
        py("save_edit('model.layers.0.mlp.down_proj.weight', 2 * w)"),
        py("v = w.clone(); v[:, :4] = 0\nsave_edit('model.layers.0.mlp.down_proj.weight', v)"),
        py("import subprocess\nsubprocess.run(['ls'])"),
        fw({"type": "generate", "model": "both", "problems": ["p1"]}),
        fw({"type": "bad"}),
        fw({"type": "teacher_forced", "texts": ["hi"]}),
        "Final answer: I zeroed four columns of layer 0's down_proj.",
    ]
    calls = []

    def forward(req, k):
        if req["type"] == "bad":
            raise ValueError("unknown type")
        calls.append(k)
        return {"ok": k}
    st = ah.run_main(make_args(tmp), llm=scripted(replies, 0.01), forward=forward)
    turns = sorted((tmp / "run" / "turns").iterdir())
    out = lambda i: (turns[i + 1] / "prompt.txt").read_text()  # noqa: E731  (the reply to turn i is the next prompt)
    assert "torch.Size([32, 64])" in out(0), out(0)
    assert "out of scope" in out(1) and "constant" in out(2) and "saved model.layers.0" in out(3), (out(1), out(2), out(3))
    assert "Refused, not run" in out(4)
    assert "Forward pass 1/2" in out(5) and "not counted" in out(6) and "last forward pass" in out(7)
    assert st["finished"] and st["passes"] == 2 and st["code_runs"] == 4 and calls == [1, 2], st
    assert (tmp / "work" / "edited" / "model.layers.0.mlp.down_proj.weight.pt").exists()
    assert (tmp / "run" / "final_answer.md").read_text().startswith("Final answer")
    print("session ok:", st)


def test_spending_stop():
    tmp = Path(tempfile.mkdtemp())
    tiny_model(tmp / "model")
    (tmp / "practice.jsonl").write_text('{"id": "p1", "type": "svamp"}\n')
    replies = ["```python\n1 + 1\n```"] * 50
    st = ah.run_main(make_args(tmp, cap_usd=0.2), llm=scripted(replies, 0.06), forward=lambda r, k: {})
    assert st["spent_usd"] <= 0.2 + 1e-9, st
    assert st["turn"] < 10, st
    print("spending stop ok: stopped after", st["turn"], "calls at $", round(st["spent_usd"], 2), "with a $0.20 cap")


def test_key_is_never_passed_on():
    import os
    os.environ["OPENROUTER_API_KEY"] = "sk-test-not-real"
    tmp = Path(tempfile.mkdtemp())
    tiny_model(tmp / "model")
    (tmp / "practice.jsonl").write_text('{"id": "p1", "type": "svamp"}\n')
    replies = ["```python\nimport os\nos.environ.get('OPENROUTER_API_KEY')\n```", "```python\nimport sys\nlen(sys.argv)\n```", "done"]
    st = ah.run_main(make_args(tmp, passes=0), llm=scripted(replies, 0.001), forward=lambda r, k: {})
    assert "OPENROUTER_API_KEY" not in os.environ, "the key must be removed from the driver's environment"
    turns = sorted((tmp / "run" / "turns").iterdir())
    assert "Refused" in (turns[1] / "prompt.txt").read_text()
    for t in turns:
        for f in t.iterdir():
            assert "sk-test" not in f.read_text(), f
    print("key ok: removed from the environment, never in a log or a reply")


def test_blocks():
    one = "```python\nnotes = 'use ```python blocks'\nx = 1\n```"
    assert ah.find_blocks(one) == [("python", "notes = 'use ```python blocks'\nx = 1")], ah.find_blocks(one)
    two = "```python\na=1\n```\ntext\n```forward\n{}\n```"
    assert [b[0] for b in ah.find_blocks(two)] == ["python", "forward"]
    assert ah.find_blocks("no code") == []
    print("blocks ok")


def test_token_cost():
    assert abs(ah.token_cost({"prompt_tokens": 100000, "completion_tokens": 4000}) - (1.0 + 0.2)) < 1e-9
    assert ah.token_cost({"prompt_tokens": 300000, "completion_tokens": 0}) == 300000 * 20e-6
    print("token cost ok")


if __name__ == "__main__":
    test_blocks()
    test_token_cost()
    test_session()
    test_spending_stop()
    test_key_is_never_passed_on()
