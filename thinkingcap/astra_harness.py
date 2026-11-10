"""Astra weight-space programming harness (3 Nov, user): reproduce "Sparks of Weight-Space Programming" (Takehi et al., MIT,
2 Nov 2026, Exp 8) on Qwen3-0.6B, with OUR goal: fewer thinking tokens at equal accuracy. The frontier model does the
diagnosis and the fix itself; we only provide the tools of the paper's harness and score its final edit.

Three parts, as in the paper (logs: github.com/RikiyaT/lm_understand-logs):
    kernel   a persistent Python process over the weights (CPU, no GPU, no internet): NAMES, load, load_original,
             save_edit (refuses the embedding / output layer, the final norm, non-finite tensors and plain whole-tensor
             scaling), drop_edit, edited, tok. One ```python block per turn; the last expression is shown; output cut
             at 10,000 characters; --code-runs at most. It runs with a scrubbed environment (no API key in it).
    forward  the forward-pass tool, --passes per run, each a fresh GPU process: "generate" (original / edited / both,
             practice questions, the evaluation's sampling and per-type caps, graded with typed.py: accuracy, thinking
             tokens, cut share, failures, and for "both" the paired difference +- SE) or "teacher_forced" (log-probs,
             top-k, hidden states, MLP down_proj inputs, attention). A request that fails does not count.
    driver   the conversation through OpenRouter (model --llm, default openai/gpt-6-astra). HARD SPENDING STOP: the
             real cost of every call is read from OpenRouter's usage accounting; before each call the worst case of the
             next one (whole context at the uncached long-context price + --max-reply tokens at the output price) is
             added, and the run ends (final answer, no code) once spent + worst case would pass --cap-usd. The
             conversation restarts with the task, /work/notes.md and the edit list when the context passes
             --handoff tokens (the paper restarted every 40 turns; this also keeps the price below the long-context tier).

    The API key is read from the environment variable OPENROUTER_API_KEY (Colab: set it from the Secrets panel in a
    notebook cell before the plan cell; never in a file, a plan line or the repo) and removed from the environment at once.

    python astra_harness.py run --name a1 --passes 10 --cap-usd 8   (track.py astra wraps it and scores the result)
"""

import argparse
import ast
import contextlib
import io
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

PROTECTED = {"model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"}
# gpt-6-astra on OpenRouter, 3 Nov 2026 (USD per token), prompts under 272k tokens: uncached input 10e-6 (cached 1e-6),
# output 50e-6; above 272k both rise (20e-6 / 75e-6). --handoff keeps every prompt far below 272k, so the worst case of a
# call is its whole context uncached at 10e-6 plus --max-reply tokens at 50e-6.
LONG_TIER = 272000
PRICE_IN_WORST, PRICE_OUT_WORST = 10e-6, 50e-6
# Only real imports and calls (4 Nov: "pty" matched inside "empty" and 4 harmless blocks were refused)
FORBIDDEN = re.compile(r"(?m)^\s*(?:import|from)\s+(?:socket|urllib|requests|http|subprocess|pty|ctypes|multiprocessing|"
                       r"shutil\.rmtree)\b|\bos\.(?:system|popen|environ|getenv|exec\w*|spawn\w*|fork|kill)\b|"
                       r"\bgetenv\s*\(|/proc/|\.backward\s*\(|\btorch\.optim\b|\btorch\.autograd\b|"
                       r"requires_grad_\s*\(\s*True|__import__\s*\(|\bimportlib\b")
BLOCK = re.compile(r"```(python|forward)\s*\n(.*?)```", re.S)
OPEN = re.compile(r"^```(python|forward)\s*$")
CLOSE = re.compile(r"^```\s*$")


def find_blocks(text):
    """The code blocks of a reply: a fence must open at the start of a line (```python / ```forward) and close on a line
    of its own (the LAST such line before the next opening fence), so backticks inside the code's strings do not split it
    (3 Nov: Astra's notes text held ``` and two turns were refused)."""
    text = re.sub(r"```\s*```(python|forward)", r"```\n```\1", text)  # a closing and an opening fence on one line (4 Nov)
    lines = text.split("\n")
    opens = [i for i, l in enumerate(lines) if OPEN.match(l.strip())]
    blocks = []
    for j, i in enumerate(opens):
        stop = opens[j + 1] if j + 1 < len(opens) else len(lines)
        closes = [c for c in range(i + 1, stop) if CLOSE.match(lines[c].strip())]
        if closes:
            blocks.append((OPEN.match(lines[i].strip()).group(1), "\n".join(lines[i + 1:closes[-1]])))
    return blocks


# ---------------- kernel (runs in its own process) ----------------

def kernel_main(args):
    import torch
    from safetensors import safe_open
    torch.set_grad_enabled(False)
    model_dir, work = Path(args.model_dir), Path(args.work)
    edits = work / "edited"
    edits.mkdir(parents=True, exist_ok=True)
    files = sorted(model_dir.glob("*.safetensors"))
    where = {}
    for f in files:
        with safe_open(f, "pt") as h:
            for k in h.keys():
                where[k] = f

    def load_original(name):
        with safe_open(where[name], "pt") as h:
            return h.get_tensor(name).float()

    def load(name):
        f = edits / f"{name}.pt"
        return torch.load(f).float() if f.exists() else load_original(name)

    def save_edit(name, tensor):
        if name not in where:
            raise ValueError(f"unknown tensor {name}")
        if name in PROTECTED:
            raise ValueError(f"{name} is out of scope (embedding / output layer / final norm)")
        o = load_original(name)
        t = torch.as_tensor(tensor).float()
        if tuple(t.shape) != tuple(o.shape):
            raise ValueError(f"shape {tuple(t.shape)} != {tuple(o.shape)}")
        if not torch.isfinite(t).all():
            raise ValueError("the tensor has non-finite values")
        c = float((t * o).sum() / (o * o).sum().clamp_min(1e-30))
        if torch.allclose(t, c * o, rtol=1e-5, atol=1e-8) and abs(c - 1) > 1e-6:
            raise ValueError("an edit that only multiplies the whole tensor by a constant is out of scope")
        torch.save(t, edits / f"{name}.pt")
        return f"saved {name} (relative change {float((t - o).norm() / o.norm()):.4%})"

    def drop_edit(name):
        f = edits / f"{name}.pt"
        if f.exists():
            f.unlink()
        return f"dropped {name}"

    def edited():
        return sorted(p.name[:-3] for p in edits.glob("*.pt"))
    from transformers import AutoTokenizer
    env = {"NAMES": sorted(where), "load": load, "load_original": load_original, "save_edit": save_edit,
           "drop_edit": drop_edit, "edited": edited, "tok": AutoTokenizer.from_pretrained(str(model_dir)),
           "torch": torch, "__name__": "__kernel__"}
    os.chdir(work)

    def alarm(signum, frame):
        raise TimeoutError(f"the execution passed {args.timeout} s")
    signal.signal(signal.SIGALRM, alarm)
    for line in sys.stdin:
        code = json.loads(line)["code"]
        buf = io.StringIO()
        signal.alarm(args.timeout)
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                tree = ast.parse(code)
                last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
                exec(compile(tree, "<block>", "exec"), env)
                if last is not None:
                    value = eval(compile(ast.Expression(last.value), "<block>", "eval"), env)
                    if value is not None:
                        print(repr(value))
        except BaseException:  # noqa: BLE001  (the model's code may raise anything; report it, keep the kernel)
            buf.write(traceback.format_exc(limit=4))
        finally:
            signal.alarm(0)
        out = buf.getvalue().encode("ascii", "replace").decode()
        if len(out) > args.out_chars:
            out = out[:args.out_chars] + f"\n[output cut at {args.out_chars:,} characters]"
        sys.stdout.write(json.dumps({"out": out}) + "\n")
        sys.stdout.flush()


class Kernel:
    """The kernel process, with a scrubbed environment (no API key, no tokens of any kind)."""

    def __init__(self, args):
        self.args = args
        self.start()

    def start(self):
        env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "PYTHONPATH") if k in os.environ}
        env["HF_HUB_OFFLINE"] = "1"
        self.proc = subprocess.Popen([sys.executable, __file__, "kernel", "--model-dir", str(self.args.model_dir),
                                      "--work", str(self.args.work), "--timeout", str(self.args.timeout),
                                      "--out-chars", str(getattr(self.args, "out_chars", 10000))],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env,
                                     cwd=str(Path(__file__).parent))

    def run(self, code):
        try:
            self.proc.stdin.write(json.dumps({"code": code}) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("no reply")
            return json.loads(line)["out"]
        except Exception as e:  # noqa: BLE001
            self.proc.kill()
            self.start()
            return f"[the kernel was restarted ({e}); its variables are lost]"


# ---------------- forward-pass tool ----------------

def stable_seed(text):
    return int.from_bytes(text.encode()[:8].ljust(8, b"\0"), "little") % (2 ** 31) ^ (len(text) * 7919)


def edited_model_dir(args):
    """Original weights + the kernel's current edits, saved for vLLM / HF (bfloat16, as the evaluation loads it)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    edits = sorted((Path(args.work) / "edited").glob("*.pt"))
    out = Path(args.scratch) / "edited"
    stamp = json.dumps([(p.name, p.stat().st_mtime) for p in edits])
    if (out / "stamp.json").exists() and (out / "stamp.json").read_text() == stamp:
        return out
    model = AutoModelForCausalLM.from_pretrained(str(args.model_dir), dtype=torch.float32)
    sd = model.state_dict()
    for p in edits:
        sd[p.name[:-3]].copy_(torch.load(p))
    model.to(torch.bfloat16).save_pretrained(out)
    AutoTokenizer.from_pretrained(str(args.model_dir)).save_pretrained(out)
    (out / "stamp.json").write_text(stamp)
    return out


# The TRAIN tool's menu (user, 4 Nov: every knob we have; 15 GPU-hours): name -> (track.py lupin flag, type, allowed)
RECIPES = ("long2short", "dast", "adar1", "rightwrong", "four", "simright")
PAIR_MENU = {
    "min_solve": ("--lp-min-solve", float, (0.0, 1.0)), "easy_ratio": ("--lp-easy-ratio", float, (0.0, 5.0)),
    "shorten_cap": ("--lp-shorten-cap", float, (0.0, 5.0)), "answers": ("--lp-answers", int, (2, 10)),
    "long_ratio": ("--long-ratio", float, (1.0, 5.0)), "add_hard": ("--add-hard", float, (0.0, 1.0)),
    "add_right_below": ("--add-right-below", float, (0.0, 1.0)), "right_band": ("--right-band", float, (0.0, 1.0)),
    "margin_drop": ("--margin-drop", float, (0.0, 0.9)), "drop_similar": ("--drop-similar", float, (0.0, 1.0)),
    "budget": ("--budget", bool, None), "margin_any": ("--margin-any", bool, None), "add_right": ("--add-right", bool, None),
}
DPO_MENU = {
    "target_steps": ("--target-steps", int, (10, 339)), "lr": ("--lp-lr", float, (1e-7, 1e-3)),
    "beta": ("--lp-beta", float, (0.01, 2.0)), "nll": ("--lp-nll", float, (0.0, 5.0)),
    "schedule": ("--lp-lr-schedule", str, ("cosine", "constant", "linear")), "bida_exact": (None, bool, None),
}
SECONDS_PER_STEP = 33.0  # bidax5q: 339 steps in ~3.1 h on an L4 (the budget check before a run)


def lupin_command(req):
    """The track.py lupin command for a train request (validated against the menu), its steps and adapter name."""
    name = str(req.get("name", ""))
    if not re.fullmatch(r"[a-z0-9]{2,16}", name):
        raise ValueError('"name": 2-16 lowercase letters or digits')
    recipe = req.get("recipe", "long2short")
    if recipe not in RECIPES:
        raise ValueError(f'"recipe" must be one of {RECIPES}')
    start = req.get("start", "raw")
    if start not in ("raw", "v1"):
        raise ValueError('"start" must be "raw" or "v1"')
    lupin, extra = [], []
    for key, value in (req.get("pairs") or {}).items():
        if key not in PAIR_MENU:
            raise ValueError(f"unknown pair option {key!r}; allowed: {sorted(PAIR_MENU)}")
        flag, kind, allowed = PAIR_MENU[key]
        if kind is bool:
            if value:
                extra.append(flag)
            continue
        value = kind(value)
        if allowed and not allowed[0] <= value <= allowed[1]:
            raise ValueError(f"{key} must be in {allowed}")
        (lupin if flag.startswith("--lp-") else extra).extend([flag, value])
    dpo, steps, exact = [], 339, True
    for key, value in (req.get("dpo") or {}).items():
        if key not in DPO_MENU:
            raise ValueError(f"unknown dpo option {key!r}; allowed: {sorted(DPO_MENU)}")
        flag, kind, allowed = DPO_MENU[key]
        if key == "bida_exact":
            exact = bool(value)
            continue
        value = kind(value)
        if (kind is str and value not in allowed) or (kind is not str and allowed and not allowed[0] <= value <= allowed[1]):
            raise ValueError(f"{key} must be in {allowed}")
        if key == "target_steps":
            steps = value
        else:
            dpo.extend([flag, value])
    cmd = ["lupin", "--name", f"astra-{name}", "--lp-recipe", recipe, "--lp-no-match", "--seeds", 0, "--lp-save", 10,
           "--dpo-steps", 9999, "--lp-pool", "dp13", "--lp-extra-q", "nofit1,tools1,code1,nofit2,nofit4,math1,math2",
           "--lp-add-q", "nofit4,math1,math2", "--lp-answers-pool", "dp10"]
    for flag, default in (("--lp-min-solve", 0), ("--lp-answers", 4)):
        if flag not in lupin:
            cmd += [flag, default]
    for flag, default in (("--lp-nll", 0), ("--lp-lr-schedule", "cosine")):
        if flag not in dpo:
            cmd += [flag, default]
    cmd += lupin + dpo
    if extra:
        cmd.append("--lp-extra=" + ",".join(str(x) for x in extra))
    cmd.append("--lp-dpo=" + ",".join((["--bida-exact"] if exact else []) + [f"--target-steps={steps}"]))
    if start == "v1":
        cmd += ["--lp-init", "adapters/lpc-bidax5q-s0"]
    return [str(c) for c in cmd], steps, f"astra-{name}"


def model_for(args, label):
    """The weights a forward pass runs: original (the start: raw or v1), edited (+ the kernel's edits), or
    checkpoint:<step-N|final> = one of v1's training checkpoints (data/v1_training/checkpoints), merged on the RAW Qwen."""
    if label == "original":
        return str(args.model_dir)
    if label == "edited":
        return str(edited_model_dir(args))
    if label.startswith("checkpoint:") and getattr(args, "data", ""):
        ck = Path(args.data) / "v1_training" / "checkpoints" / label.split(":", 1)[1]
        if not (ck / "adapter_config.json").exists():
            raise ValueError(f"no checkpoint {label}")
        out = Path(args.scratch) / ("ckpt-" + ck.name)
        if not (out / "config.json").exists():
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
            base = AutoModelForCausalLM.from_pretrained(args.raw_model, dtype=torch.float32)
            PeftModel.from_pretrained(base, str(ck)).merge_and_unload().to(torch.bfloat16).save_pretrained(out)
            AutoTokenizer.from_pretrained(args.raw_model).save_pretrained(out)
        return str(out)
    if label.startswith("trained:"):  # an adapter the TRAIN tool made (merged on the raw Qwen; a v1 start is in the adapter)
        bits = label.split(":")
        ad = Path(getattr(args, "repo", ".")) / "adapters" / f"astra-{bits[1]}-s0"
        if len(bits) > 2:
            ad = ad / bits[2]
        if not (ad / "adapter_config.json").exists():
            raise ValueError(f"no trained model {label}")
        out = Path(args.scratch) / ("trained-" + "-".join(bits[1:]))
        if not (out / "config.json").exists():
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
            base = AutoModelForCausalLM.from_pretrained(args.raw_model, dtype=torch.float32)
            PeftModel.from_pretrained(base, str(ad)).merge_and_unload().to(torch.bfloat16).save_pretrained(out)
            AutoTokenizer.from_pretrained(args.raw_model).save_pretrained(out)
        return str(out)
    raise ValueError('"model" must be "original", "edited", "both", "checkpoint:<step-N|final>" or "trained:<name>[:step-N]"')


def forward_main(args):
    """One forward pass in a fresh process: request JSON in, summary JSON out (files in /work)."""
    req = json.loads(Path(args.request).read_text())
    k = args.k
    work = Path(args.work)
    if req.get("type") == "generate":
        summary = run_generate(args, req, k, work)
    elif req.get("type") == "teacher_forced":
        summary = run_teacher_forced(args, req, k, work)
    else:
        raise ValueError('"type" must be "generate" or "teacher_forced"')
    Path(args.result).write_text(json.dumps(summary))


def practice_rows(args):
    return {r["id"]: r for r in (json.loads(l) for l in Path(args.practice).read_text().splitlines() if l.strip())}


def run_generate(args, req, k, work):
    import statistics
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    import typed
    from utils.eval.recipe import SAMPLING, token_cap
    rows = practice_rows(args)
    items = list(req.get("problems") or [])
    bad = [p for p in items if p not in rows]
    if bad:
        raise ValueError(f"unknown practice ids: {bad[:5]}")
    n = int(req.get("n", 1))
    which = req.get("model", "original")
    if isinstance(req.get("compare"), list):
        if len(req["compare"]) != 2:
            raise ValueError('"compare" takes exactly two models')
        which = "compare"
    limit = 512 if which not in ("both", "compare") else 256
    if not items or len(items) * n > limit:
        raise ValueError(f"items x n must be 1..{limit}")
    cap = int(req.get("max_tokens", 8192))
    tok = AutoTokenizer.from_pretrained(str(args.model_dir))
    close_id = tok.convert_tokens_to_ids("</think>")
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    models = {"both": ["original", "edited"], "compare": list(req.get("compare") or [])}.get(which, [which])
    results = {}
    for label in models:
        path = model_for(args, label)
        llm = LLM(model=path, dtype="bfloat16", seed=0, gpu_memory_utilization=0.85, max_model_len=8192 + 2048,
                  generation_config="vllm")
        prompts, params, keys = [], [], []
        for p in items:
            r = rows[p]
            ids = tok.apply_chat_template(r["prompt"], tokenize=True, add_generation_prompt=True)
            ids = list(ids["input_ids"]) if hasattr(ids, "keys") else list(ids)
            for s in range(n):
                prompts.append({"prompt_token_ids": ids})
                params.append(SamplingParams(max_tokens=min(cap, token_cap(r["benchmark"])), seed=stable_seed(f"{p}#{s}"),
                                             stop_token_ids=[eos], **SAMPLING))
                keys.append((p, s))
        outs = llm.generate(prompts, params)
        del llm
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        recs = []
        for (p, s), o in zip(keys, outs):
            ids = list(o.outputs[0].token_ids)
            cut = o.outputs[0].finish_reason == "length"
            text = typed.full_text(tok, ids)
            right = (not cut) and bool(typed.is_correct(rows[p], text))
            think = ids.index(close_id) if close_id in ids else len(ids)
            fail = "right" if right else "cut_at_max_tokens" if cut else "no_finished_thinking" if close_id not in ids else "wrong_answer"
            recs.append({"id": p, "sample": s, "type": rows[p]["type"], "right": right, "thinking_tokens": think,
                         "tokens": len(ids), "cut": cut, "failure": fail, "text": text})
        name = f"gen_{k}.jsonl" if which not in ("both", "compare") else f"gen_{k}_{label}.jsonl"
        name = name.replace(":", "-")
        (work / name).write_text("".join(json.dumps(r) + "\n" for r in recs))
        results[label] = recs

    def describe(recs):
        by = {}
        for t in sorted({r["type"] for r in recs}):
            rs = [r for r in recs if r["type"] == t]
            by[t] = {"n": len(rs), "accuracy": round(sum(r["right"] for r in rs) / len(rs), 4),
                     "mean_thinking_tokens": round(statistics.fmean(r["thinking_tokens"] for r in rs), 1),
                     "cut_share": round(sum(r["cut"] for r in rs) / len(rs), 4)}
        fails = {}
        for r in recs:
            fails[r["failure"]] = fails.get(r["failure"], 0) + 1
        return {"n": len(recs), "accuracy": round(sum(r["right"] for r in recs) / len(recs), 4),
                "mean_thinking_tokens": round(statistics.fmean(r["thinking_tokens"] for r in recs), 1),
                "mean_tokens": round(statistics.fmean(r["tokens"] for r in recs), 1),
                "cut_share": round(sum(r["cut"] for r in recs) / len(recs), 4), "failures": fails, "by_type": by}
    summary = {label: describe(recs) for label, recs in results.items()}
    if which in ("both", "compare"):
        a, b = results[models[0]], results[models[1]]
        d = [int(y["right"]) - int(x["right"]) for x, y in zip(a, b)]
        mean = sum(d) / len(d)
        sd = statistics.pstdev(d) if len(d) > 1 else 0.0
        both_done = [(x, y) for x, y in zip(a, b) if not x["cut"] and not y["cut"]]
        summary["paired"] = {
            "accuracy_diff_pp": round(100 * mean, 2), "se_pp": round(100 * sd / math.sqrt(len(d)), 2),
            "wrong_to_right": sum(x == 0 and y == 1 for x, y in zip([r["right"] for r in a], [r["right"] for r in b])),
            "right_to_wrong": sum(x == 1 and y == 0 for x, y in zip([r["right"] for r in a], [r["right"] for r in b])),
            "thinking_saved_pct": round(100 * (1 - summary[models[1]]["mean_thinking_tokens"] /
                                               max(1e-9, summary[models[0]]["mean_thinking_tokens"])), 2),
            "thinking_saved_pct_on_answers_both_finished": round(100 * (1 - statistics.fmean(y["thinking_tokens"] for _, y in both_done) /
                                                                        max(1e-9, statistics.fmean(x["thinking_tokens"] for x, _ in both_done))), 2) if both_done else None}
    return summary


def run_teacher_forced(args, req, k, work, device="cuda", dtype=None):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(args.model_dir))
    texts = req.get("texts")
    if texts is None:
        texts = [tok.apply_chat_template(c, tokenize=False, enable_thinking=bool(req.get("enable_thinking", True)))
                 for c in req.get("chats", [])]
    ids = [tok.encode(t, add_special_tokens=False) for t in texts]
    if not ids or len(ids) > 256 or max(map(len, ids)) > 4096 or sum(map(len, ids)) > 131072:
        raise ValueError("1..256 texts, each <= 4,096 tokens, <= 131,072 tokens in total")
    ret = req.get("return", {})
    attn = ret.get("attn_layers") or []
    if attn and (len(ids) > 2 or max(map(len, ids)) > 2048):
        raise ValueError("attn_layers: at most 2 texts of at most 2,048 tokens")
    path = model_for(args, req.get("model", "original"))
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype or torch.float32,
                                                 **({"attn_implementation": "eager"} if attn else {})).to(device).eval()
    layers = model.model.layers
    acts = {}
    hooks = [layers[i].mlp.down_proj.register_forward_pre_hook(lambda m, inp, i=i: acts.__setitem__(i, inp[0].detach()))
             for i in (ret.get("mlp_act") or [])]
    out = work / f"fwd_{k}"
    out.mkdir(parents=True, exist_ok=True)
    topk = min(int(ret.get("topk", 0)), 50)
    meta = {"texts": len(ids), "tokens": [len(x) for x in ids], "files": []}
    for j, x in enumerate(ids):
        t = torch.tensor([x], device=device)
        with torch.no_grad():
            o = model(t, output_hidden_states=bool(ret.get("hidden_states")), output_attentions=bool(attn))
        lp = torch.log_softmax(o.logits[0].float(), -1)
        if ret.get("token_logprobs"):
            np.save(out / f"t{j}_token_logprobs.npy", lp[:-1].gather(1, t[0, 1:, None])[:, 0].cpu().numpy())
        if topk:
            v, i = lp.topk(topk, -1)
            np.save(out / f"t{j}_topk_ids.npy", i.cpu().numpy())
            np.save(out / f"t{j}_topk_logprobs.npy", v.cpu().numpy())
        for l in ret.get("hidden_states") or []:
            np.save(out / f"t{j}_hidden_{l}.npy", o.hidden_states[l][0].float().cpu().numpy().astype("float16"))
        for l in ret.get("mlp_act") or []:
            np.save(out / f"t{j}_mlp_act_{l}.npy", acts[l][0].float().cpu().numpy().astype("float16"))
        for l in attn:
            np.save(out / f"t{j}_attn_{l}.npy", o.attentions[l][0].float().cpu().numpy().astype("float16"))
        np.save(out / f"t{j}_ids.npy", np.array(x))
    for h in hooks:
        h.remove()
    meta["files"] = sorted(p.name for p in out.glob("*.npy"))[:200]
    (out / "meta.json").write_text(json.dumps(meta))
    return {"folder": str(out), "texts": len(ids), "tokens": sum(map(len, ids)), "files_example": meta["files"][:12]}


# ---------------- driver ----------------

def task_prompt(args):
    text = (Path(__file__).parent / ({"consult": "astra_prompt_consult.txt", "consult-dpo": "astra_prompt_consult_dpo.txt",
                                         "consult-merge": "astra_prompt_consult_merge.txt",
                                         "consult-fresh": "astra_prompt_consult_fresh.txt",
                                         "consult-review": "astra_prompt_consult_review.txt"}.get(
                                         getattr(args, "task", "edit"), "astra_prompt.txt"))).read_text()
    note = ("\n\nTHIS IS A SHORT TEST SESSION ON A LAPTOP CPU: the practice set is 12 easy math word problems (graded by the "
            "final number); a generation pass takes at most 8 generations per model of at most 768 tokens; a teacher-forced "
            "pass at most 8 texts of about 1,000 tokens. Keep your work small; the point is to check that the tools and the "
            "data folder work: open every file of the data folder once, try every kind of forward pass once (including a "
            "checkpoint and a compare), make one small edit, and report anything missing or broken."
            if getattr(args, "cpu", False) else "")
    for k, v in {"{PASSES}": args.passes, "{CODE_RUNS}": args.code_runs, "{PRACTICE_N}": args.practice_n,
                 "{MODEL_DIR}": args.model_dir, "{PRACTICE}": args.practice, "{WORK}": args.work,
                 "{DATA}": getattr(args, "data", "") or ""}.items():
        text = text.replace(k, str(v))
    if getattr(args, "data", "") and not getattr(args, "task", "edit").startswith("consult"):
        extra = (Path(__file__).parent / "astra_prompt_data.txt").read_text()
        start = " (in this run you edit v1's weights, not these)" if args.goal == "on-top" else " (the weights you edit)"
        for k, v in {"{DATA}": args.data, "{PRACTICE}": args.practice, "{GOAL}": GOALS[args.goal], "{START_NOTE}": start}.items():
            extra = extra.replace(k, str(v))
        text += extra
    if getattr(args, "train_hours", 0):
        extra = (Path(__file__).parent / "astra_prompt_research.txt").read_text()
        for k, v in {"{TRAIN_HOURS}": args.train_hours, "{TRAIN_USED}": "0 h at the start", "{DATA}": args.data,
                     "{RECIPES}": list(RECIPES), "{PAIR_KEYS}": sorted(PAIR_MENU), "{DPO_KEYS}": sorted(DPO_MENU)}.items():
            extra = extra.replace(k, str(v))
        text += extra
    return text + note


GOALS = {  # (3 Nov, user) the two data experiments
    "vs-dpo": ("You start from the unedited model, the same starting point v1 had. Beat v1 by editing weights only: more "
               "thinking tokens saved than v1 at no worse accuracy, or v1's saving at better accuracy. v1's answers show "
               "what a shorter-thinking version of this model does: use them to find what changes inside the network."),
    "on-top": ("The weights you edit ARE v1's: the raw model + preference training (DPO) = v1, which thinks about 47% "
               "fewer tokens than the raw model at about -0.5 accuracy points. OBJECTIVE: more than 50% fewer thinking "
               "tokens than the RAW model at less than 1 point of accuracy lost against the RAW model, so v1 must think "
               "shorter still without losing more accuracy. v1 still loses tokens to answers cut at the limit and still "
               "hesitates and re-checks; raw.jsonl shows the original model's habits, v1.jsonl v1's own answers."),
}


class Budget:
    """Real spend from OpenRouter's usage accounting; worst-case estimate before each call."""

    def __init__(self, cap_usd, spent=0.0):
        self.cap, self.spent = cap_usd, spent

    price_in, price_out = PRICE_IN_WORST, PRICE_OUT_WORST  # Astra's list prices; Argon (google) sets its own
    google = False
    last_cached = 0

    def worst_next(self, context_tokens, max_reply):
        if self.google:  # Argon: the tier doubles above 200k; what the last call had cached costs the cached price
            k = 2 if context_tokens > 200000 else 1
            cached = min(self.last_cached, context_tokens)
            return k * ((context_tokens - cached) * 2e-6 + cached * 0.1e-6 + max_reply * 10e-6)
        return context_tokens * self.price_in + max_reply * self.price_out

    def allows(self, context_tokens, max_reply):
        return self.spent + self.worst_next(context_tokens, max_reply) <= self.cap


# Gemini 4 Argon (Google AI Studio key; OpenRouter's list on 3 Nov: $2/M in, $10/M out, thinking billed as output);
# above 200k prompt tokens Google charges more: counted at 2x as the worst case
def google_cost(usage):
    """OpenAI-compatible usage (no cache info): every prompt token at the full price; thinking only in total_tokens."""
    prompt = usage.get("prompt_tokens", 0)
    out = max(usage.get("completion_tokens", 0), usage.get("total_tokens", 0) - prompt)
    long = prompt > 200000
    return prompt * (4e-6 if long else 2e-6) + out * (20e-6 if long else 10e-6)


def google_native_cost(meta):
    """Google's native usageMetadata (4 Nov: implicit caching is real and reported, 93% of a repeated 29k prompt):
    uncached prompt $2/M, cached $0.10/M, output and thinking $10/M; all doubled above 200k prompt tokens (worst case)."""
    prompt = meta.get("promptTokenCount", 0)
    cached = meta.get("cachedContentTokenCount", 0)
    out = meta.get("candidatesTokenCount", 0) + meta.get("thoughtsTokenCount", 0)
    k = 2 if prompt > 200000 else 1
    return k * ((prompt - cached) * 2e-6 + cached * 0.1e-6 + out * 10e-6)


def call_google_native(args, key, messages):
    """Gemini via Google's native generateContent: system -> systemInstruction, assistant -> model; thought parts are
    returned inside <thought> tags so split_thoughts keeps them out of the conversation."""
    import urllib.request
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                for m in messages if m["role"] != "system"]
    body = {"contents": contents, "generationConfig": {"maxOutputTokens": args.max_reply,
                                                       "thinkingConfig": {"includeThoughts": True, "thinkingLevel": args.effort}}}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    model = args.llm.split("/")[-1]
    req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                                 data=json.dumps(body).encode(), headers={"x-goog-api-key": key, "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                d = json.loads(r.read())
            parts = (d.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
            thoughts = "".join(f"<thought>{p.get('text', '')}</thought>" for p in parts if p.get("thought"))
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            meta = d.get("usageMetadata", {})
            return thoughts + text, google_native_cost(meta), meta
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise
            print(f"astra: API error ({e}); retrying", flush=True)
            time.sleep(30 * (attempt + 1))


def token_cost(usage):
    """Spend from token counts at gpt-6-astra's public list price with NO cache discount (Delos workers report tokens only)."""
    long = usage.get("prompt_tokens", 0) > LONG_TIER
    return usage.get("prompt_tokens", 0) * (20e-6 if long else 10e-6) + usage.get("completion_tokens", 0) * (75e-6 if long else 50e-6)


def call_llm(args, key, messages):
    import urllib.request
    if args.provider == "google":
        return call_google_native(args, key, messages)
    if args.provider == "google":  # 3 Nov (user): Gemini 4 Argon on the user's own Google AI Studio key
        url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        body = {"model": args.llm.split("/")[-1], "messages": messages, "max_tokens": args.max_reply,
                "extra_body": {"google": {"thinking_config": {"include_thoughts": True, "thinking_level": args.effort}}}}
    elif args.provider == "delos":  # 3 Nov (user): a Delos worker's raw, stateless, tool-free model endpoint
        url = f"https://app.delos.so/api/v1/workers/{args.worker}/chat/completions"
        body = {"model": args.llm.split("/")[-1], "messages": messages, "max_tokens": args.max_reply,
                "reasoning_effort": args.effort}
    else:
        url = "https://openrouter.ai/api/v1/chat/completions"
        body = {"model": args.llm, "messages": messages, "max_tokens": args.max_reply, "usage": {"include": True},
                "reasoning": {"effort": args.effort}}
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                data = json.loads(r.read())
            text = data["choices"][0]["message"].get("content") or ""
            u = data.get("usage", {})
            cost = u.get("cost")
            if cost is None:  # no accounting returned: full list price, nothing counted as cached
                cost = google_cost(u) if args.provider == "google" else token_cost(u)
            return text, float(cost), u
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise
            print(f"astra: API error ({e}); retrying", flush=True)
            time.sleep(30 * (attempt + 1))


THOUGHT = re.compile(r"<thought>(.*?)</thought>", re.S)


def split_thoughts(text):
    """Argon's thought summaries (<thought>...</thought>) apart from its reply: we keep them in thoughts.md for the record
    and leave them out of the conversation (as the paper's runs: the next turn sees the reply, not the reasoning)."""
    thoughts = "\n\n".join(t.strip() for t in THOUGHT.findall(text))
    return thoughts, THOUGHT.sub("", text).strip()


def approx_tokens(messages):
    return sum(len(m["content"]) for m in messages) // 3 + 200  # ~3 characters per token, generous


def run_main(args, llm=None, forward=None):
    """The session. `llm(messages) -> (text, cost, usage)` and `forward(request, k) -> summary` are replaceable (tests)."""
    keys = {"openrouter": "OPENROUTER_API_KEY", "delos": "DELOS_API_KEY", "google": "GOOGLE_API_KEY"}
    key = os.environ.pop(keys[args.provider], None)
    for other in keys.values():  # never left where the kernel or a forward pass could see it
        os.environ.pop(other, None)
    if args.provider != "google":  # Astra's prices are only modelled below its 272k tier
        assert args.handoff + args.max_reply + 20000 < LONG_TIER, "the spending stop assumes prompts stay below the long-context tier"
    if llm is None:
        if not key:
            raise SystemExit(f"{keys[args.provider]} is not set (Colab: Secrets panel + the key cell); nothing was sent")
        llm = lambda messages: call_llm(args, key, messages)  # noqa: E731
    run = Path(args.out)
    (run / "turns").mkdir(parents=True, exist_ok=True)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    state_f = run / "state.json"
    st = json.loads(state_f.read_text()) if state_f.exists() else {"turn": 0, "passes": 0, "code_runs": 0, "spent_usd": 0.0,
                                                                  "conversations": 1, "finished": False}
    for k in ("no_block", "refused", "forward_failed", "kernel_restarts", "repeated_block", "code_errors"):
        st.setdefault(k, 0)
    last_block = [None]
    if st["finished"]:
        print("astra: the session is already finished", flush=True)
        return st
    budget = Budget(args.cap_usd, st["spent_usd"])
    if args.provider == "google":  # Argon: exact cost from the native usage (cached / uncached / thinking), any context size
        budget.google = True
    kernel = Kernel(args)
    if forward is None:
        forward = lambda request, k: run_forward_process(args, request, k)  # noqa: E731
    system = {"role": "system", "content": "Answer the user's request."}
    msgs_f = run / "messages.json"
    messages = json.loads(msgs_f.read_text()) if msgs_f.exists() else [system, {"role": "user", "content": task_prompt(args)}]
    final_sent = False

    def save():
        st["spent_usd"] = budget.spent
        state_f.write_text(json.dumps(st, indent=1))
        msgs_f.write_text(json.dumps(messages))
    while True:
        out_of_passes = st["passes"] >= args.passes
        if not budget.allows(approx_tokens(messages), args.max_reply):
            if final_sent or not budget.allows(approx_tokens(messages[:2]) + 4000, 4000):
                print(f"astra: SPENDING STOP at ${budget.spent:.2f} (cap ${args.cap_usd}); edits in /work/edited stand", flush=True)
                break
            # one last short call, in a fresh small conversation, for the final answer
            notes = (work / "notes.md").read_text()[-6000:] if (work / "notes.md").exists() else "(no notes)"
            # (4 Nov) also its last forward-pass results, so the final report can state what the edit achieved
            results = [m["content"][:2500] for m in messages if m["role"] == "user" and m["content"].startswith("Forward pass")][-3:]
            messages = [system, {"role": "user", "content": "The budget for this session is used up. Your notes:\n" + notes +
                                 "\n\nYour last forward-pass results:\n" + ("\n---\n".join(results) or "(none)") +
                                 f"\n\nCurrent edits: {sorted(p.name[:-3] for p in (work / 'edited').glob('*.pt'))}\n"
                                 "Reply WITHOUT a code block: what you changed, the evidence, and why you expect it to help."}]
            final_sent = True
        if approx_tokens(messages) > args.handoff and not final_sent:  # restart the conversation, as the paper did
            notes = (work / "notes.md").read_text()[-8000:] if (work / "notes.md").exists() else "(no notes yet)"
            messages = [system, {"role": "user", "content": task_prompt(args) + "\n\nHANDOFF. The conversation was restarted. "
                                 f"Forward passes used: {st['passes']}/{args.passes}; code runs used: {st['code_runs']}/{args.code_runs}. "
                                 f"Current edits: {sorted(p.name[:-3] for p in (work / 'edited').glob('*.pt'))}. "
                                 "The kernel's variables are kept. Your notes (/work/notes.md):\n" + notes}]
            st["conversations"] += 1
        st["turn"] += 1
        t = run / "turns" / f"t{st['turn']:03d}"
        t.mkdir(exist_ok=True)
        (t / "prompt.txt").write_text(messages[-1]["content"])
        text, cost, usage = llm(messages)
        budget.spent += cost
        budget.last_cached = usage.get("cachedContentTokenCount", 0) if isinstance(usage, dict) else 0
        thoughts, text = split_thoughts(text)
        if thoughts:
            (t / "thoughts.md").write_text(thoughts)
        (t / "answer.md").write_text(text)
        (t / "run.json").write_text(json.dumps({"cost_usd": cost, "spent_usd": budget.spent, "usage": usage}))
        flags = {k: st[k] for k in ("no_block", "refused", "forward_failed", "kernel_restarts", "repeated_block", "code_errors") if st[k]}
        print(f"astra turn {st['turn']}: ${cost:.3f} (total ${budget.spent:.2f} / ${args.cap_usd}), passes {st['passes']}/{args.passes}, "
              f"code runs {st['code_runs']}" + (f", flags {flags}" if flags else ""), flush=True)
        messages.append({"role": "assistant", "content": text})
        if final_sent or (out_of_passes and not find_blocks(text)):
            (run / "final_answer.md").write_text(text)
            st["finished"] = True
            save()
            break
        blocks = find_blocks(text)
        if blocks and blocks[0] == last_block[0]:
            st["repeated_block"] += 1  # the same action as last turn: possibly stuck
        last_block[0] = blocks[0] if blocks else None
        ignored = ""
        if len(blocks) > 1:  # (4 Nov) run the first, say so, instead of wasting the turn
            st["extra_blocks"] = st.get("extra_blocks", 0) + 1
            ignored = (f"\n\n[Only your first block was run; {len(blocks) - 1} more block(s) in that reply were ignored. "
                       "Send exactly one block per reply.]")
            blocks = blocks[:1]
        if not blocks:
            st["no_block"] += 1
            reply = "Reply with exactly one ```python or ```forward block (or, once the forward passes are used, with no block)."
        elif blocks[0][0] == "python":
            code = blocks[0][1]
            if st["code_runs"] >= args.code_runs:
                reply = "The code-run limit is reached: no more Python blocks; use the forward passes left, then answer."
            elif FORBIDDEN.search(code):
                st["refused"] += 1
                reply = ("Refused, not run: the block uses something the rules exclude (training / autograd, internet, "
                         "processes or the environment). Rewrite it without that.")
            else:
                st["code_runs"] += 1
                reply = kernel.run(code)
                if reply.startswith("[the kernel was restarted"):
                    st["kernel_restarts"] += 1
                elif "Traceback (most recent call last)" in reply:
                    st["code_errors"] += 1
        else:
            if out_of_passes:
                reply = "All forward passes are used: make your final choice and reply without a code block."
            else:
                k = st["passes"] + 1
                try:
                    req = json.loads(blocks[0][1])
                    if req.get("type") == "train":
                        if not getattr(args, "train_hours", 0):
                            raise ValueError("training is not available in this session (weight edits only)")
                        summary = run_train(args, req, st)
                    else:
                        summary = forward(req, k)
                    st["passes"] = k
                    reply = f"Forward pass {k}/{args.passes} done:\n" + json.dumps(summary, indent=1)[:getattr(args, "out_chars", 10000)]
                except Exception as e:  # noqa: BLE001  (a failed request does not count)
                    st["forward_failed"] += 1
                    reply = f"The forward request failed and was not counted: {str(e)[:2000]}"
        reply += ignored
        if st["passes"] >= args.passes and not out_of_passes:
            reply += "\n\nThat was the last forward pass: make your final choice (the tensors in /work/edited are submitted) and reply without a code block."
        messages.append({"role": "user", "content": reply})
        if (work / "notes.md").exists():
            (run / "notes.md").write_text((work / "notes.md").read_text())
        save()
    save()
    return st


def cpu_forward(args, request, k):
    """--cpu (a Mac test): the same tool, small, on CPU with HF (bfloat16); math questions graded by the final number."""
    import statistics
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    work = Path(args.work)
    if request.get("type") == "teacher_forced":
        texts = request.get("texts") or []
        if not texts or len(texts) > 8 or any(len(t) > 4000 for t in texts):
            raise ValueError("CPU test: 1..8 texts of at most ~1,000 tokens")
        request = dict(request)
        request["return"] = {k2: v for k2, v in request.get("return", {}).items()}
        return run_teacher_forced(args, request, k, work, device="cpu", dtype=torch.bfloat16)
    if request.get("type") != "generate":
        raise ValueError('"type" must be "generate" or "teacher_forced"')
    rows = practice_rows(args)
    items, n, which = list(request.get("problems") or []), int(request.get("n", 1)), request.get("model", "original")
    if isinstance(request.get("compare"), list):
        which = "compare"
    if not items or any(p not in rows for p in items) or len(items) * n > args.cpu_items:
        raise ValueError(f"CPU test: known practice ids, problems x n <= {args.cpu_items} (per model)")
    cap = min(int(request.get("max_tokens", args.cpu_tokens)), args.cpu_tokens)
    tok = AutoTokenizer.from_pretrained(str(args.model_dir))
    tok.padding_side = "left"
    close_id = tok.convert_tokens_to_ids("</think>")
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    models = {"both": ["original", "edited"], "compare": list(request.get("compare") or [])}.get(which, [which])
    results = {}
    for label in models:
        path = model_for(args, label)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).eval()
        keys = [(p, s2) for p in items for s2 in range(n)]
        texts = [tok.apply_chat_template(rows[p]["prompt"], tokenize=False, add_generation_prompt=True) for p, _ in keys]
        enc = tok(texts, return_tensors="pt", padding=True)
        torch.manual_seed(0)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=cap, do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
                                 eos_token_id=eos, pad_token_id=tok.pad_token_id or eos)
        del model
        recs = []
        for (p, s2), g in zip(keys, gen[:, enc["input_ids"].shape[1]:].tolist()):
            ids = g[:g.index(eos) + 1] if eos in g else g
            cut = eos not in g
            text = tok.decode(ids, skip_special_tokens=False)
            nums = re.findall(r"-?\d+(?:\.\d+)?", text.split("</think>")[-1].replace(",", ""))
            right = (not cut) and bool(nums) and abs(float(nums[-1]) - float(rows[p]["answer"])) < 1e-6
            think = ids.index(close_id) if close_id in ids else len(ids)
            recs.append({"id": p, "sample": s2, "type": rows[p]["type"], "right": right, "thinking_tokens": think,
                         "tokens": len(ids), "cut": cut, "text": text})
        (work / (f"gen_{k}.jsonl" if which not in ("both", "compare") else f"gen_{k}_{label}.jsonl".replace(":", "-"))).write_text(
            "".join(json.dumps(r) + "\n" for r in recs))
        results[label] = recs
    summary = {label: {"n": len(r), "accuracy": round(sum(x["right"] for x in r) / len(r), 3),
                       "mean_thinking_tokens": round(statistics.fmean(x["thinking_tokens"] for x in r), 1),
                       "cut_share": round(sum(x["cut"] for x in r) / len(r), 3)} for label, r in results.items()}
    if which in ("both", "compare"):
        a, b = results[models[0]], results[models[1]]
        summary["paired"] = {"accuracy_diff_pp": round(100 * statistics.fmean(int(y["right"]) - int(x["right"]) for x, y in zip(a, b)), 1),
                             "thinking_saved_pct": round(100 * (1 - summary[models[1]]["mean_thinking_tokens"] /
                                                                max(1e-9, summary[models[0]]["mean_thinking_tokens"])), 1)}
    return summary


def run_train(args, request, st):
    """A TRAIN request: track.py lupin with the validated options, timed against the GPU-hour budget."""
    cmd, steps, adapter = lupin_command(request)
    left = args.train_hours * 3600 - st.get("train_seconds", 0.0)
    need = steps * SECONDS_PER_STEP
    if need > left:
        raise ValueError(f"not enough training budget: this run needs ~{need / 3600:.1f} h, {left / 3600:.1f} h left")
    env = {k: v for k, v in os.environ.items() if "KEY" not in k and "TOKEN" not in k}
    began = time.time()
    p = subprocess.run([sys.executable, "track.py", *cmd], cwd=str(getattr(args, "repo", ".")), env=env,
                       capture_output=True, text=True, timeout=8 * 3600)
    st["train_seconds"] = st.get("train_seconds", 0.0) + (time.time() - began)
    out = p.stdout + p.stderr
    if p.returncode != 0:
        raise RuntimeError("training failed: " + out[-1500:])
    pairs = [l for l in out.splitlines() if re.match(r"\s*\d+ pairs, recipe|\s+\w+ / \w+: ", l)][:40]
    log = [l for l in out.splitlines() if l.startswith("step ") and "pairs ordered right" in l]
    return {"trained": adapter.replace("astra-", "trained:", 1), "steps": steps,
            "hours_used_total": round(st["train_seconds"] / 3600, 2),
            "hours_left": round(args.train_hours - st["train_seconds"] / 3600, 2),
            "pairs": pairs, "training_log": log[::max(1, len(log) // 20)] + log[-1:]}


def run_forward_process(args, request, k):
    if getattr(args, "cpu", False):
        return cpu_forward(args, request, k)
    req_f = Path(args.work) / f"request_{k}.json"
    res_f = Path(args.work) / f"result_{k}.json"
    req_f.write_text(json.dumps(request))
    env = {k2: v for k2, v in os.environ.items() if "KEY" not in k2 and "TOKEN" not in k2}
    p = subprocess.run([sys.executable, __file__, "forward", "--request", str(req_f), "--result", str(res_f), "--k", str(k),
                        "--model-dir", str(args.model_dir), "--work", str(args.work), "--practice", str(args.practice),
                        "--scratch", str(args.scratch), "--data", str(getattr(args, "data", "") or ""),
                        "--raw-model", str(getattr(args, "raw_model", "Qwen/Qwen3-0.6B"))], env=env, capture_output=True, text=True, timeout=3 * 3600)
    if p.returncode != 0 or not res_f.exists():
        raise RuntimeError((p.stderr or p.stdout)[-1500:])
    return json.loads(res_f.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("mode", choices=("run", "kernel", "forward", "smoke"))
    p.add_argument("--name", default="a1")
    p.add_argument("--out")
    p.add_argument("--work")
    p.add_argument("--model-dir")
    p.add_argument("--practice")
    p.add_argument("--practice-n", type=int, default=0)
    p.add_argument("--scratch", default="/content/astra-models")
    p.add_argument("--llm", default="openai/gpt-6-astra")
    p.add_argument("--provider", default="openrouter", choices=("openrouter", "delos", "google"))
    p.add_argument("--worker", default="", help="delos: the worker id (billing follows it)")
    p.add_argument("--effort", default="high", help="reasoning effort (the paper: max; high keeps the budget)")
    p.add_argument("--passes", type=int, default=10)
    p.add_argument("--code-runs", type=int, default=120)
    p.add_argument("--cap-usd", type=float, default=8.0, help="HARD spending stop (user, 3 Nov: EUR 8; USD 8 is below it)")
    p.add_argument("--max-reply", type=int, default=8000, help="tokens per reply, reasoning included")
    p.add_argument("--handoff", type=int, default=60000, help="restart the conversation above this many context tokens")
    p.add_argument("--out-chars", type=int, default=10000, help="a code run's or a forward summary's output is cut here (the paper: 10,000)")
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--train-hours", type=float, default=0.0, help="research mode: the TRAIN tool's GPU-hour budget (0 = off)")
    p.add_argument("--repo", default=".", help="where track.py and adapters/ live (the TRAIN tool)")
    p.add_argument("--raw-model", default="Qwen/Qwen3-0.6B", help="the raw base the training checkpoints were trained on")
    p.add_argument("--data", default="", help="a folder of existing answers for Astra (raw.jsonl, v1.jsonl, summary.json)")
    p.add_argument("--goal", default="vs-dpo", choices=("vs-dpo", "on-top"))
    p.add_argument("--task", default="edit", choices=("edit", "consult", "consult-dpo", "consult-merge", "consult-fresh", "consult-review"),
                   help="consult (4 Nov, user): strategy analysis only, no forward passes, our work never mentioned")
    p.add_argument("--cpu", action="store_true", help="a small CPU test session (Mac): HF forward passes, 12 math questions")
    p.add_argument("--cpu-items", type=int, default=8, help="--cpu: generations per model per pass")
    p.add_argument("--cpu-tokens", type=int, default=768, help="--cpu: max new tokens")
    p.add_argument("--request")
    p.add_argument("--result")
    p.add_argument("--k", type=int, default=0)
    args = p.parse_args()
    if args.mode == "kernel":
        return kernel_main(args)
    if args.mode == "forward":
        sys.path.insert(0, str(Path(__file__).parent))
        return forward_main(args)
    if args.mode == "smoke":  # the GPU tools on Colab, no API call: one paired generation, one teacher-forced read
        Path(args.work).mkdir(parents=True, exist_ok=True)
        (Path(args.work) / "edited").mkdir(exist_ok=True)
        ids = [json.loads(l)["id"] for l in Path(args.practice).read_text().splitlines()[:6]]
        print("smoke generate:", json.dumps(run_forward_process(args, {"type": "generate", "model": "both", "problems": ids,
                                                                        "max_tokens": 1024}, 1))[:3000], flush=True)
        print("smoke teacher_forced:", json.dumps(run_forward_process(args, {"type": "teacher_forced", "model": "original",
              "texts": ["<think>\nTwo plus two is four.\n</think>\n\n4"], "return": {"token_logprobs": True, "topk": 5,
              "hidden_states": [0, 14, 28], "mlp_act": [14], "attn_layers": [14]}}, 2)), flush=True)
        k = Kernel(args)
        print("smoke kernel:", k.run("len(NAMES), load('model.layers.14.mlp.down_proj.weight').shape"), flush=True)
        return
    run_main(args)


if __name__ == "__main__":
    main()
