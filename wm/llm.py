"""
Model adapters.

Everything goes through one OpenRouter client so that the comparison is clean:
same transport, same parameters, same retry behaviour for every model. The only
thing that varies across arms is the model id and the prompt.

    export OPENROUTER_API_KEY=sk-or-...
"""

from __future__ import annotations

import os
import time

BASE_URL = "https://openrouter.ai/api/v1"

# Running totals, so the cost of each arm is reportable.
USAGE: list[dict] = []


def model(name: str, temperature: float = 0.0):
    """Returns callable(prompt, max_tokens) -> str, and records token usage."""
    from openai import OpenAI

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    client = OpenAI(base_url=BASE_URL, api_key=key)

    def call(prompt: str, max_tokens: int = 16000) -> str:
        last = None
        for attempt in range(4):
            try:
                r = client.chat.completions.create(
                    model=name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                u = getattr(r, "usage", None)
                USAGE.append({
                    "model": name,
                    "prompt_tokens": getattr(u, "prompt_tokens", None),
                    "completion_tokens": getattr(u, "completion_tokens", None),
                })
                return r.choices[0].message.content or ""
            except Exception as e:          # noqa: BLE001 - transport errors vary by provider
                last = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"{name} failed after 4 attempts: {last}")

    return call


def spend_report() -> str:
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0, 0])
    for u in USAGE:
        a = agg[u["model"]]
        a[0] += 1
        a[1] += u["prompt_tokens"] or 0
        a[2] += u["completion_tokens"] or 0
    L = [f"{'model':<40}{'calls':>7}{'in tok':>12}{'out tok':>12}"]
    for m, (c, i, o) in sorted(agg.items()):
        L.append(f"{m:<40}{c:>7}{i:>12,}{o:>12,}")
    return "\n".join(L)


# --- the roster -----------------------------------------------------------
# FRONTIER builds the encoder output and the prose twin, and is also an arm.
# SMALL is the model the hypothesis says structure can lift.
# JUDGE must not be either arm, and should come from a third family where possible.

# Override any of these without editing code:
#   export WM_FRONTIER=anthropic/claude-sonnet-4.5
#   export WM_SMALL=qwen/qwen3-32b
#   export WM_JUDGE=google/gemini-2.5-flash
# Check the exact slugs at https://openrouter.ai/models before running.
FRONTIER = os.environ.get("WM_FRONTIER", "anthropic/claude-opus-4.1")
SMALL = os.environ.get("WM_SMALL", "qwen/qwen-2.5-72b-instruct")
JUDGE = os.environ.get("WM_JUDGE", "google/gemini-2.5-pro")

ARMS = [FRONTIER, SMALL]


def preflight() -> None:
    """Confirm the key works and every model slug resolves, before spending."""
    print(f"frontier : {FRONTIER}")
    print(f"small    : {SMALL}")
    print(f"judge    : {JUDGE}")
    print()
    ok = True
    probe = ('Reply with exactly this JSON and nothing else: '
             '{"verdict": "pass", "reasoning": "probe"}')
    for role, name in (("frontier", FRONTIER), ("small", SMALL), ("judge", JUDGE)):
        try:
            # 2000 tokens, not 16: reasoning models spend their budget thinking and
            # return an empty string if the cap is small. An empty reply is a FAILURE.
            r = (model(name)(probe, max_tokens=2000) or "").strip()
            if not r:
                ok = False
                print(f"  {role:<9} {name:<40} FAILED: empty response "
                      f"(reasoning model exhausting max_tokens?)")
            elif '"verdict"' not in r:
                ok = False
                print(f"  {role:<9} {name:<40} FAILED: did not return the JSON "
                      f"-> {r[:70]!r}")
            else:
                print(f"  {role:<9} {name:<40} OK")
        except Exception as e:                      # noqa: BLE001
            ok = False
            print(f"  {role:<9} {name:<40} FAILED: {str(e)[:150]}")
    print()
    print("ready" if ok else "fix the failures above before running `all`")
