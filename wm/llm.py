"""Model adapters. Claude via the Anthropic SDK; an open-weights slot you fill in."""

from __future__ import annotations

import os


def claude(model: str = "claude-opus-5", effort: str = "high"):
    """Returns a callable(prompt, max_tokens) -> str."""
    import anthropic

    client = anthropic.Anthropic()

    def call(prompt: str, max_tokens: int = 32000) -> str:
        with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            thinking={"type": "adaptive"},
            output_config={"effort": effort},
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            msg = stream.get_final_message()
        if msg.stop_reason == "refusal":
            raise RuntimeError(f"refusal: {msg.stop_details}")
        return "".join(b.text for b in msg.content if b.type == "text")

    return call


def open_weights(model: str, base_url: str | None = None, api_key_env: str = "OPENAI_API_KEY"):
    """
    Slot for Qwen / Llama / whatever you self-host, via any OpenAI-compatible
    endpoint (vLLM, Together, Fireworks, DashScope).

    Deliberately not wired to a specific provider: point base_url at yours.
    """
    from openai import OpenAI

    client = OpenAI(
        base_url=base_url or os.environ.get("OPEN_WEIGHTS_BASE_URL"),
        api_key=os.environ.get(api_key_env, "EMPTY"),
    )

    def call(prompt: str, max_tokens: int = 32000) -> str:
        r = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.choices[0].message.content or ""

    return call


# Model roster for the experiment.
#   frontier  - builds the world model and the prose twin; also an arm
#   small     - the model the hypothesis says can match it given structure
#   judges    - two different families, neither of which is an arm
FRONTIER = "claude-opus-5"
SMALL_CLAUDE = "claude-haiku-4-5"
JUDGES = ["claude-sonnet-5", "gpt-5.5"]
