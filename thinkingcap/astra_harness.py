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
FORBIDDEN = re.compile(r"\.backward\s*\(|torch\.optim|autograd|requires_grad_\s*\(\s*True|/proc\b|environ|getenv|"
                       r"\bsocket\b|urllib|requests|http\.client|subprocess|os\.system|os\.popen|pty\b|ctypes")
BLOCK = re.compile(r"```(python|forward)\s*\n(.*?)```", re.S)


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
        if len(out) > 10000:
            out = out[:10000] + "\n[output cut at 10,000 characters]"
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
                                      "--work", str(self.args.work), "--timeout", str(self.args.timeout)],
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
    limit = 256 if which == "both" else 512
    if not items or len(items) * n > limit:
        raise ValueError(f"items x n must be 1..{limit}")
    cap = int(req.get("max_tokens", 8192))
    tok = AutoTokenizer.from_pretrained(str(args.model_dir))
    close_id = tok.convert_tokens_to_ids("</think>")
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    models = {"original": [("original", str(args.model_dir))], "edited": [("edited", str(edited_model_dir(args)))],
              "both": [("original", str(args.model_dir)), ("edited", str(edited_model_dir(args)))]}[which]
    results = {}
    for label, path in models:
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
        name = f"gen_{k}.jsonl" if which != "both" else f"gen_{k}_{label}.jsonl"
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
    if which == "both":
        a, b = results["original"], results["edited"]
        d = [int(y["right"]) - int(x["right"]) for x, y in zip(a, b)]
        mean = sum(d) / len(d)
        sd = statistics.pstdev(d) if len(d) > 1 else 0.0
        both_done = [(x, y) for x, y in zip(a, b) if not x["cut"] and not y["cut"]]
        summary["paired"] = {
            "accuracy_diff_pp": round(100 * mean, 2), "se_pp": round(100 * sd / math.sqrt(len(d)), 2),
            "wrong_to_right": sum(x == 0 and y == 1 for x, y in zip([r["right"] for r in a], [r["right"] for r in b])),
            "right_to_wrong": sum(x == 1 and y == 0 for x, y in zip([r["right"] for r in a], [r["right"] for r in b])),
            "thinking_saved_pct": round(100 * (1 - summary["edited"]["mean_thinking_tokens"] /
                                               max(1e-9, summary["original"]["mean_thinking_tokens"])), 2),
            "thinking_saved_pct_on_answers_both_finished": round(100 * (1 - statistics.fmean(y["thinking_tokens"] for _, y in both_done) /
                                                                        max(1e-9, statistics.fmean(x["thinking_tokens"] for x, _ in both_done))), 2) if both_done else None}
    return summary


def run_teacher_forced(args, req, k, work):
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
    path = str(edited_model_dir(args)) if req.get("model") == "edited" else str(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32,
                                                 **({"attn_implementation": "eager"} if attn else {})).to("cuda").eval()
    layers = model.model.layers
    acts = {}
    hooks = [layers[i].mlp.down_proj.register_forward_pre_hook(lambda m, inp, i=i: acts.__setitem__(i, inp[0].detach()))
             for i in (ret.get("mlp_act") or [])]
    out = work / f"fwd_{k}"
    out.mkdir(parents=True, exist_ok=True)
    topk = min(int(ret.get("topk", 0)), 50)
    meta = {"texts": len(ids), "tokens": [len(x) for x in ids], "files": []}
    for j, x in enumerate(ids):
        t = torch.tensor([x], device="cuda")
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
    text = (Path(__file__).parent / "astra_prompt.txt").read_text()
    for k, v in {"{PASSES}": args.passes, "{CODE_RUNS}": args.code_runs, "{PRACTICE_N}": args.practice_n,
                 "{MODEL_DIR}": args.model_dir, "{PRACTICE}": args.practice, "{WORK}": args.work}.items():
        text = text.replace(k, str(v))
    return text


class Budget:
    """Real spend from OpenRouter's usage accounting; worst-case estimate before each call."""

    def __init__(self, cap_usd, spent=0.0):
        self.cap, self.spent = cap_usd, spent

    def worst_next(self, context_tokens, max_reply):
        return context_tokens * PRICE_IN_WORST + max_reply * PRICE_OUT_WORST

    def allows(self, context_tokens, max_reply):
        return self.spent + self.worst_next(context_tokens, max_reply) <= self.cap


def call_llm(args, key, messages):
    import urllib.request
    body = {"model": args.llm, "messages": messages, "max_tokens": args.max_reply, "usage": {"include": True},
            "reasoning": {"effort": args.effort}}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                data = json.loads(r.read())
            text = data["choices"][0]["message"].get("content") or ""
            u = data.get("usage", {})
            cost = u.get("cost")
            if cost is None:  # no accounting returned: charge the worst case
                cost = u.get("prompt_tokens", 0) * PRICE_IN_WORST + u.get("completion_tokens", 0) * PRICE_OUT_WORST
            return text, float(cost), u
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise
            print(f"astra: API error ({e}); retrying", flush=True)
            time.sleep(30 * (attempt + 1))


def approx_tokens(messages):
    return sum(len(m["content"]) for m in messages) // 3 + 200  # ~3 characters per token, generous


def run_main(args, llm=None, forward=None):
    """The session. `llm(messages) -> (text, cost, usage)` and `forward(request, k) -> summary` are replaceable (tests)."""
    key = os.environ.pop("OPENROUTER_API_KEY", None)
    assert args.handoff + args.max_reply + 20000 < LONG_TIER, "the spending stop assumes prompts stay below the long-context tier"
    if llm is None:
        if not key:
            raise SystemExit("OPENROUTER_API_KEY is not set (Colab: Secrets panel + the key cell); nothing was sent")
        llm = lambda messages: call_llm(args, key, messages)  # noqa: E731
    run = Path(args.out)
    (run / "turns").mkdir(parents=True, exist_ok=True)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    state_f = run / "state.json"
    st = json.loads(state_f.read_text()) if state_f.exists() else {"turn": 0, "passes": 0, "code_runs": 0, "spent_usd": 0.0,
                                                                  "conversations": 1, "finished": False}
    if st["finished"]:
        print("astra: the session is already finished", flush=True)
        return st
    budget = Budget(args.cap_usd, st["spent_usd"])
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
            notes = (work / "notes.md").read_text()[-6000:] if (work / "notes.md").exists() else ""
            messages = [system, {"role": "user", "content": "The budget for this session is used up. Your notes:\n" + notes +
                                 f"\nCurrent edits: {sorted(p.name[:-3] for p in (work / 'edited').glob('*.pt'))}\n"
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
        (t / "answer.md").write_text(text)
        (t / "run.json").write_text(json.dumps({"cost_usd": cost, "spent_usd": budget.spent, "usage": usage}))
        print(f"astra turn {st['turn']}: ${cost:.3f} (total ${budget.spent:.2f} / ${args.cap_usd}), passes {st['passes']}/{args.passes}, "
              f"code runs {st['code_runs']}", flush=True)
        messages.append({"role": "assistant", "content": text})
        if final_sent or (out_of_passes and not BLOCK.search(text)):
            (run / "final_answer.md").write_text(text)
            st["finished"] = True
            save()
            break
        blocks = BLOCK.findall(text)
        if len(blocks) != 1:
            reply = "Reply with exactly one ```python or ```forward block (or, once the forward passes are used, with no block)."
        elif blocks[0][0] == "python":
            code = blocks[0][1]
            if st["code_runs"] >= args.code_runs:
                reply = "The code-run limit is reached: no more Python blocks; use the forward passes left, then answer."
            elif FORBIDDEN.search(code):
                reply = ("Refused, not run: the block uses something the rules exclude (training / autograd, internet, "
                         "processes or the environment). Rewrite it without that.")
            else:
                st["code_runs"] += 1
                reply = kernel.run(code)
        else:
            if out_of_passes:
                reply = "All forward passes are used: make your final choice and reply without a code block."
            else:
                k = st["passes"] + 1
                try:
                    summary = forward(json.loads(blocks[0][1]), k)
                    st["passes"] = k
                    reply = f"Forward pass {k}/{args.passes} done:\n" + json.dumps(summary, indent=1)[:10000]
                except Exception as e:  # noqa: BLE001  (a failed request does not count)
                    reply = f"The forward request failed and was not counted: {str(e)[:2000]}"
        if st["passes"] >= args.passes and not out_of_passes:
            reply += "\n\nThat was the last forward pass: make your final choice (the tensors in /work/edited are submitted) and reply without a code block."
        messages.append({"role": "user", "content": reply})
        save()
    save()
    return st


def run_forward_process(args, request, k):
    req_f = Path(args.work) / f"request_{k}.json"
    res_f = Path(args.work) / f"result_{k}.json"
    req_f.write_text(json.dumps(request))
    env = {k2: v for k2, v in os.environ.items() if "KEY" not in k2 and "TOKEN" not in k2}
    p = subprocess.run([sys.executable, __file__, "forward", "--request", str(req_f), "--result", str(res_f), "--k", str(k),
                        "--model-dir", str(args.model_dir), "--work", str(args.work), "--practice", str(args.practice),
                        "--scratch", str(args.scratch)], env=env, capture_output=True, text=True, timeout=3 * 3600)
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
    p.add_argument("--effort", default="high", help="reasoning effort (the paper: max; high keeps the budget)")
    p.add_argument("--passes", type=int, default=10)
    p.add_argument("--code-runs", type=int, default=120)
    p.add_argument("--cap-usd", type=float, default=8.0, help="HARD spending stop (user, 3 Nov: EUR 8; USD 8 is below it)")
    p.add_argument("--max-reply", type=int, default=8000, help="tokens per reply, reasoning included")
    p.add_argument("--handoff", type=int, default=60000, help="restart the conversation above this many context tokens")
    p.add_argument("--timeout", type=int, default=900)
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
