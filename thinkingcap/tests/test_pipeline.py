"""Tests for prune_traces.py and sft.py, runnable without a GPU or Qwen3's weights.

The unit tests check the pruning logic directly. The end-to-end test trains a tiny Qwen3-shaped
model on a toy task where the answer is only knowable from one "key" paragraph of the thinking,
surrounded by filler before it and re-checking after it. The pruner should keep the key
paragraph and drop the rest, and sft.py should train on the result.

    python -m pytest tests -q        # from the directory holding prune_traces.py
"""

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from probe_curve import gold_probability  # noqa: E402
from prune_traces import drop_flat, flat_steps, settle_index, split_steps, thinking_text  # noqa: E402

FILLER = ["Let me read the question again.", "Okay, I need to find the number.",
          "Hmm, this looks like a lookup.", "Let me think about it carefully."]
CHECKS = ["Wait, let me double check that.", "Yes, that seems right.", "Let me verify once more."]
NUMBERS = list(range(10, 50))


# ---------------------------------------------------------------- unit tests

def test_split_steps_keeps_separators_and_drops_blank_parts():
    assert split_steps("\nA one.\n\nB two.\n\nC three.") == ["\nA one.\n\n", "B two.\n\n", "C three."]
    assert "".join(split_steps("x\n\ny\n\n")) == "x\n\ny\n\n"


def test_thinking_text_matches_qwen_layout():
    assert thinking_text(["\nA.\n\n", "B.\n\n"]) == "<think>\nA.\n\nB.\n</think>\n\n"


def test_settle_index_requires_staying_above():
    assert settle_index([0.1, 0.95, 0.5, 0.92, 0.97], 0.9) == 3
    assert settle_index([0.95, 0.96], 0.9) == 0
    assert settle_index([0.1, 0.95, 0.5], 0.9) is None


def test_flat_steps_measures_gain_over_best_earlier_point():
    curve = [0.1, 0.1, 0.6, 0.5, 0.95]  # step 0 flat, 1 raises, 2 drops, 3 raises
    assert flat_steps(curve, range(4), 0.02) == [0, 2]


def test_drop_flat_finds_every_removable_step_with_few_checks():
    essential = {3, 11}
    calls = []

    def accept(kept):
        calls.append(kept)
        return essential <= set(kept)

    kept = drop_flat(range(16), list(range(16)), accept)
    assert kept == sorted(essential)
    assert len(calls) < 16  # group testing beats one check per step


def test_drop_flat_keeps_everything_when_nothing_can_go():
    assert drop_flat([0, 1, 2], [0, 1, 2], lambda kept: len(kept) == 3) == [0, 1, 2]


def test_gold_probability_is_raw_for_math_and_a_share_for_choices():
    assert gold_probability([0.02], "math") == 0.02  # a lone candidate must not read 100%
    assert gold_probability([0.2, 0.2, 0.1], "multiple_choice") == 0.4


# ---------------------------------------------------------------- end to end

def toy_thinking(rng, number, before=2, after=2):
    before = rng.sample(FILLER, before)
    after = rng.sample(CHECKS, after)
    return "\n" + "\n\n".join(before + [f"So the number is {number}."] + after) + "\n"


def toy_question(number):
    # Nothing in the question may hint at the number: only the key paragraph knows it.
    return "Find the number. Put the final answer on the last line, like: Final answer: 7"


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    """A tiny Qwen3 that copies the number from the key paragraph, plus rollouts in probe_curve's format."""
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    work = tmp_path_factory.mktemp("toy")
    rng = random.Random(0)
    texts = []
    for _ in range(600):
        number = rng.choice(NUMBERS)
        texts.append(f"<|im_start|>user\n{toy_question(number)}<|im_end|>\n<|im_start|>assistant\n"
                     # varied lengths, so the model can answer from a shortened trace like a real one
                     f"<think>{toy_thinking(rng, number, rng.randint(0, 2), rng.randint(0, 2))}</think>\n\nFinal answer: {number}<|im_end|>")
    specials = ["<|im_end|>", "<|im_start|>", "<think>", "</think>"]
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(texts, trainers.BpeTrainer(
        vocab_size=600, special_tokens=specials, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    hf = PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token="<|im_end|>", pad_token="<|im_end|>")
    hf.save_pretrained(work / "model")

    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=len(hf), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512,
        eos_token_id=hf.eos_token_id, pad_token_id=hf.eos_token_id))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    batches = [hf(texts[i:i + 16], return_tensors="pt", padding=True) for i in range(0, len(texts), 16)]
    for _ in range(30):
        for batch in batches:
            labels = batch["input_ids"].masked_fill(batch["attention_mask"] == 0, -100)
            loss = model(**batch, labels=labels).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    model.save_pretrained(work / "model")

    rollouts = []
    for qid in range(6):
        number = rng.choice(NUMBERS)
        prompt = hf.encode(f"<|im_start|>user\n{toy_question(number)}<|im_end|>\n<|im_start|>assistant\n")
        for sample, final in enumerate([number, number + 3, number + 7]):  # wrong finals become candidates
            thinking = hf.encode("<think>" + toy_thinking(rng, number))
            output = thinking + hf.encode(f"</think>\n\nFinal answer: {number}<|im_end|>")
            rollouts.append({"id": f"toy-{qid}", "source": "toy", "task": "math", "answer": str(number),
                             "question": toy_question(number), "sample": sample, "prompt_ids": prompt,
                             "thinking_ids": thinking, "output_ids": output, "output_tokens": len(output),
                             "closed_thinking": True, "truncated": False, "correct": True,
                             "final": str(final)})
    (work / "rollouts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rollouts))
    return work, loss.item()


def run(*arguments, cwd):
    result = subprocess.run([sys.executable, *arguments], cwd=cwd, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_toy_model_learned_the_task(toy):
    _, final_loss = toy
    assert final_loss < 0.6


def test_prune_keeps_the_key_step_and_drops_filler(toy):
    work, _ = toy
    out = work / "prune"
    run(str(ROOT / "prune_traces.py"), "--model", str(work / "model"), "--rollouts", str(work / "rollouts.jsonl"),
        "--out", str(out), "--dtype", "float32", "--threshold", "0.8", "--check-samples", "2", cwd=ROOT)
    targets = [json.loads(line) for line in (out / "sft.jsonl").read_text().splitlines()]
    log = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    print((out / "summary.txt").read_text())

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(work / "model")
    pruned = [t for t in targets if t["kind"] in ("pruned", "truncated")]
    assert len(pruned) >= 4, log  # most questions get a shortened target
    for target in pruned:
        text = tokenizer.decode(target["completion_ids"])
        assert f"So the number is {target['answer']}." in text  # the step that carries the answer stays
        assert not any(check in text for check in CHECKS)       # re-checking after the answer goes
        assert text.rstrip().endswith(f"Final answer: {target['answer']}<|im_end|>")
        assert len(target["completion_ids"]) < target["original_tokens"]
    assert any(t["kind"] == "pruned" and not any(f in tokenizer.decode(t["completion_ids"]) for f in FILLER)
               for t in pruned)  # and some filler before it is gone too


def test_sft_trains_and_saves_a_loadable_adapter(toy):
    work, _ = toy
    data = work / "prune" / "sft.jsonl"
    if not data.exists():
        pytest.skip("needs the pruning test's output")
    adapter = work / "adapter"
    stdout = run(str(ROOT / "sft.py"), "--model", str(work / "model"), "--data", str(data), "--out", str(adapter),
                 "--epochs", "3", "--accum", "2", "--lr", "1e-3", "--rank", "4", "--log-every", "1", cwd=ROOT)
    losses = [float(line.split("loss ")[1].split()[0]) for line in stdout.splitlines() if line.startswith("step ")]
    assert losses and all(loss == loss for loss in losses)  # finite
    assert (adapter / "adapter_model.safetensors").exists()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(work / "model"), adapter)


def test_probe_curve_scores_the_toy_rollouts(toy):
    import shutil
    work, _ = toy
    out = work / "probe"
    out.mkdir()
    shutil.copy(work / "rollouts.jsonl", out / "rollouts.jsonl")  # skips the vLLM phase
    stdout = run(str(ROOT / "probe_curve.py"), "--model", str(work / "model"), "--out", str(out),
                 "--stride", "2", "--dtype", "float32", cwd=ROOT)
    print(stdout)
    results = [json.loads(line) for line in (out / "probes.jsonl").read_text().splitlines()]
    assert (out / "curve.png").exists()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(work / "model")
    rollouts = {(r["id"], r["sample"]): r for r in map(json.loads, (work / "rollouts.jsonl").read_text().splitlines())}
    for result in results:
        thinking = rollouts[(result["id"], result["sample"])]["thinking_ids"]
        # the first cut that contains the number itself
        key = next(k for k in range(len(thinking) + 1)
                   if f"is {result['answer']}" in tokenizer.decode(thinking[:k]))
        before = [p["p_gold"] for p in result["probes"] if p["k"] < key]
        after = [p["p_gold"] for p in result["probes"] if p["k"] >= key]
        assert sum(before) / len(before) < 0.1  # the number is unknown before the key paragraph (a guess may spike)
        assert max(after) > 0.8   # and known right after it


def test_packed_probe_matches_scoring_each_cut_from_scratch():
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from probe_curve import probe_scores, score
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=50, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, max_position_embeddings=128)).eval()
    rng = random.Random(0)
    prompt, thinking = [rng.randrange(50) for _ in range(7)], [rng.randrange(50) for _ in range(20)]
    closings = {0: [3, 4, 5], 6: [4, 5], 13: [3, 4, 5], 20: [4, 5]}
    suffixes = [[7], [8, 9, 1], [10, 11]]  # unequal lengths exercise the padding
    rows = [(cut, index, prompt + thinking[:cut] + closing, suffix)
            for cut, closing in closings.items() for index, suffix in enumerate(suffixes)]
    scratch = score(model, rows, 10_000, "cpu")
    for budget in (6144, 7):  # one forward pass, then a few
        packed = probe_scores(model, prompt, thinking, closings, suffixes, "cpu", budget)
        assert packed.keys() == scratch.keys()
        for key in packed:
            assert abs(packed[key] - scratch[key]) < 1e-4, (budget, key)


def test_length_rewards_follow_the_shortest_correct_reply():
    from rl_train import group_advantages, length_rewards
    rewards = length_rewards([100, 50, 80, 20], [True, True, False, False], alpha=0.5)
    assert rewards[:2] == [0.5, 1.0] and rewards[2] < 0 and rewards[3] < 0  # L* = 50
    assert max(rewards[2:]) < min(rewards[:2])  # a wrong reply never beats a correct one
    assert length_rewards([10, 30], [False, False], 0.5) == [-0.25, -0.25]  # no correct: L* = mean
    advantages = group_advantages(rewards)
    assert advantages[1] > advantages[0] > max(advantages[2:])
    assert group_advantages([1.0, 1.0]) == [0.0, 0.0]


def test_redundant_tokens_and_their_extra_penalty():
    from rl_train import confidence_redundant, settle_redundant, token_advantages
    logps = [-0.1, -3.0, -0.2, -2.5, -0.1, -9.0]  # the last token is the answer, outside the thinking
    assert confidence_redundant(logps, 5, 0.4) == {1, 3}
    assert settle_redundant({1: 0.2, 3: 0.95, 5: 0.97}, 6, 0.9) == {3, 4, 5}
    assert settle_redundant({1: 0.95, 3: 0.2}, 6, 0.9) == set()
    assert token_advantages(1.0, 3, {2}, 0.25) == [1.0, 1.0, 0.75]


def test_rl_train_runs_on_the_toy_model(toy):
    import shutil
    work, _ = toy
    model = work / "rl-model"
    shutil.copytree(work / "model", model, dirs_exist_ok=True)
    config = json.loads((model / "tokenizer_config.json").read_text())
    config["chat_template"] = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
                               "{% endfor %}<|im_start|>assistant\n")
    (model / "tokenizer_config.json").write_text(json.dumps(config))
    rows = [{"prompt": [{"role": "user", "content": toy_question(n)}], "answer": str(n), "task": "math",
             "source": "toy"} for n in NUMBERS[:6]]
    (work / "rl-data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for mode in ("confidence", "settle", "tokenprobe-rs", "tokenprobe-kl"):
        out = work / f"rl-{mode}"
        run(str(ROOT / "rl_train.py"), "--model", str(model), "--data", str(work / "rl-data.jsonl"),
            "--sources", "toy", "--out", str(out), "--steps", "2", "--questions", "2", "--group", "3",
            "--max-tokens", "40", "--backend", "hf", "--redundant", mode, "--dtype", "float32", cwd=ROOT)
        log = [json.loads(line) for line in (out / "rl_log.jsonl").read_text().splitlines()]
        assert [e["step"] for e in log] == [1, 2] and all(0 <= e["accuracy"] <= 1 for e in log)
        assert (out / "adapter_config.json").exists()


def test_tokenprobe_core_windows_budget_and_coefficients():
    from rl_train import core_positions, kl_coefficients, norm_lp, rs_advantages
    z = norm_lp([-1.0, -1.0, -3.0, -3.0])
    assert abs(sum(z)) < 1e-6 and z[0] > 0 > z[2]
    scores = [0, 0, 5, 5, 0, -1, 4, 4, -2, 0]
    assert core_positions(scores, 2, 2) == {2, 3, 6, 7}  # the two best non-overlapping windows
    assert core_positions(scores, 2, 1) == {2, 3}
    adv = rs_advantages(1.0, [1.0, -1.0, 0.5, -0.5], {0, 2}, budget=2, beta=1.0, alpha=0.1, lam=0.2)
    assert adv == [1.1, 0.9, 1.05, 0.9]  # over budget by 2/4: redundant tokens pay 0.2 * 0.5
    assert rs_advantages(1.0, [1.0, -1.0], {0}, budget=5, beta=1.0, alpha=0.1, lam=0.2)[1] == 1.0
    betas, gammas = kl_coefficients(4, {0, 1}, budget=2, beta=0.01, lam_decay=0.4)
    assert betas == [0.01, 0.01, 0.0, 0.0] and gammas == [0.0, 0.0, 0.2, 0.2]
    assert kl_coefficients(2, {0}, budget=3, beta=0.01, lam_decay=0.4) == ([0.01, 0.01], [0.0, 0.0])


def test_balanced_order_interleaves_sources():
    from rl_train import balanced_order
    rows = [{"source": "a", "i": i} for i in range(5)] + [{"source": "b", "i": i} for i in range(2)]
    order = balanced_order(rows, random.Random(0))
    assert [r["source"] for r in order] == ["a", "b", "a", "b", "a", "a", "a"]
    assert sorted(r["i"] for r in order if r["source"] == "a") == list(range(5))
    assert [r["i"] for r in balanced_order(rows, random.Random(1))] != [r["i"] for r in order]


def test_batched_logps_match_one_row_at_a_time():
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from rl_train import batch_logps, token_batches
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=50, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, max_position_embeddings=128)).eval()
    rows = [([1, 2, 3], [4, 5, 6, 7]), ([8, 9], [10]), ([11, 12, 13, 14], [15, 16])]
    with torch.no_grad():
        together = batch_logps(model, rows, "cpu")
        alone = [batch_logps(model, [row], "cpu")[0] for row in rows]
    for a, b in zip(together, alone):
        assert torch.allclose(a, b, atol=1e-5)
    jobs = [([0] * n, [0] * n) for n in (1, 2, 3, 4, 5)]
    batches = token_batches(jobs, budget=12)
    assert sum(len(b) for b in batches) == 5
    assert all(max(len(p) + len(r) for p, r in b) * len(b) <= 12 or len(b) == 1 for b in batches)
