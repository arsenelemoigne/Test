"""Shorten the model's own correct rollouts into traces it still gets right: SFT targets.

For each correct rollout from probe_curve.py:

  1. Split the thinking into steps (paragraphs) and score p(gold) after every step, the same
     teacher-forced probe probe_curve.py uses.
  2. Cut everything after the step from which p(gold) stays above --threshold: the model
     already knew the answer.
  3. Before that point, try deleting the steps that did not raise p(gold) ("flat" steps).
     A deletion is kept only if p(gold) of the *whole edited trace* stays above the
     threshold, because a step can matter without raising p(gold) by itself (setting up a
     variable). Flat steps are tested in groups and the groups split on failure, so a
     trace with many removable steps costs a few forward passes, not one per step.
  4. Let the model answer naturally from the edited thinking (no forced "Final answer:")
     --check-samples times. The trace is kept only if every answer is graded correct.

Per question it keeps the shortest accepted completion (thinking + answer, as the evaluation
counts it). Questions whose p(gold) never settles keep their shortest correct original
rollout, so training does not teach "always stop early" on the hard ones.

    python probe_curve.py --out runs/probe            # phase 1 writes runs/probe/rollouts.jsonl
    python prune_traces.py --rollouts runs/probe/rollouts.jsonl --out runs/prune
    python sft.py --data runs/prune/sft.jsonl --out adapters/pruned-sft
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from probe_curve import candidates, gold_probability, read_jsonl, score, write_jsonl
from utils.eval.recipe import SAMPLING
from utils.train.rewards import is_task_correct

STEP_END = re.compile(r"(?<=\n\n)")


def main():
    args = parse_arguments()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pruner = Pruner(args)

    by_question = defaultdict(list)
    for rollout in read_jsonl(Path(args.rollouts)):
        by_question[rollout["id"]].append(rollout)

    targets, log = [], []
    for number, group in enumerate(by_question.values(), 1):
        accepted = []
        for rollout in group:
            if not (rollout["correct"] and rollout["closed_thinking"]):
                continue
            outcome = pruner.prune(rollout, group)
            log.append({"id": rollout["id"], "sample": rollout["sample"], "source": rollout["source"],
                        "original_tokens": rollout["output_tokens"]} | outcome["log"])
            accepted += outcome["targets"]
        if not accepted and args.keep_unsettled:
            accepted = [original_target(r) for r in group
                        if r["correct"] and not r["truncated"] and r["closed_thinking"]]
        if accepted:
            targets.append(min(accepted, key=lambda t: len(t["completion_ids"])))
        if number % 10 == 0 or number == len(by_question):
            print(f"pruned {number}/{len(by_question)} questions, {len(targets)} targets", flush=True)

    write_jsonl(out / "sft.jsonl", targets)
    write_jsonl(out / "log.jsonl", log)
    report(targets, log, by_question, out)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--rollouts", required=True, help="rollouts.jsonl from probe_curve.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--threshold", type=float, default=0.9, help="p(gold) a kept trace must reach")
    parser.add_argument("--flat", type=float, default=0.02,
                        help="a step whose p(gold) gain over the best earlier point is at most this is 'flat'")
    parser.add_argument("--min-steps", type=int, default=1, help="never prune the thinking below this")
    parser.add_argument("--check-samples", type=int, default=2,
                        help="natural answers that must all be correct; 0 skips the check")
    parser.add_argument("--answer-tokens", type=int, default=512, help="cap on a natural answer")
    parser.add_argument("--batch-tokens", type=int, default=32768)
    parser.add_argument("--no-keep-unsettled", dest="keep_unsettled", action="store_false")
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


class Pruner:
    def __init__(self, args):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=getattr(torch, args.dtype)).to(self.device).eval()
        self.end_of_turn = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.open_id = self.tokenizer.convert_tokens_to_ids("<think>")

    def prune(self, rollout, siblings):
        """{"targets": [...], "log": {...}} for one correct rollout."""
        args = self.args
        steps = split_steps(self.decode_thinking(rollout["thinking_ids"]))
        answers, terminated = candidates(rollout, siblings)
        suffixes = [self.encode(" " + a) + ([self.end_of_turn] if terminated else []) for a in answers]

        def p_gold(variants):
            return self.p_gold(rollout, [[steps[i] for i in kept] for kept in variants], suffixes)

        curve = p_gold([range(n) for n in range(len(steps) + 1)])
        settle = settle_index(curve, args.threshold)
        log = {"steps": len(steps), "settle_step": settle, "curve": [round(p, 4) for p in curve]}
        if settle is None:
            return {"targets": [], "log": log | {"status": "unsettled"}}

        truncated = list(range(min(len(steps), max(settle, args.min_steps))))
        flat = flat_steps(curve, truncated, args.flat)
        pruned = drop_flat(truncated, flat, lambda kept: len(kept) >= args.min_steps
                           and p_gold([kept])[0] >= args.threshold)
        log |= {"truncated_steps": len(truncated), "pruned_steps": len(pruned), "flat_steps": len(flat)}

        targets = []
        for kind, kept in (("pruned", pruned), ("truncated", truncated)):
            target = self.checked_target(rollout, [steps[i] for i in kept], kind)
            log[f"{kind}_check"] = target is not None
            if target is not None:
                targets.append(target)
            if kind == "pruned" and pruned == truncated:
                break
        return {"targets": targets, "log": log | {"status": "accepted" if targets else "failed_check"}}

    def decode_thinking(self, thinking_ids):
        if thinking_ids and thinking_ids[0] == self.open_id:
            thinking_ids = thinking_ids[1:]
        return self.tokenizer.decode(thinking_ids)

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def p_gold(self, rollout, variants, suffixes):
        """p(gold) after each variant's thinking, as probe_curve.py measures it."""
        import math

        rows = [(v, c, rollout["prompt_ids"] + self.encode(thinking_text(steps) + "Final answer:"), suffix)
                for v, steps in enumerate(variants) for c, suffix in enumerate(suffixes)]
        scores = score(self.model, rows, self.args.batch_tokens, self.device)
        result = []
        for v in range(len(variants)):
            raw = [math.exp(scores[(v, c)]) for c in range(len(suffixes))]
            result.append(gold_probability(raw, rollout["task"]))
        return result

    def checked_target(self, rollout, steps, kind):
        """The SFT row for this thinking, or None if the model's natural answers are not all correct."""
        import torch

        thinking = self.encode(thinking_text(steps))
        context = rollout["prompt_ids"] + thinking
        if self.args.check_samples == 0:
            answer = self.encode(f"Final answer: {rollout['answer']}") + [self.end_of_turn]
            return make_target(rollout, thinking, answer, kind, len(steps))
        inputs = torch.tensor([context], device=self.device)
        with torch.no_grad():
            output = self.model.generate(
                inputs, attention_mask=torch.ones_like(inputs), do_sample=True,
                num_return_sequences=self.args.check_samples, max_new_tokens=self.args.answer_tokens,
                eos_token_id=self.end_of_turn, pad_token_id=self.end_of_turn, **SAMPLING)
        answers = []
        for row in output[:, len(context):].tolist():
            if self.end_of_turn not in row:
                return None  # ran out of room: the evaluation would cut it off too
            answer = row[:row.index(self.end_of_turn) + 1]
            text = self.tokenizer.decode(thinking + answer, skip_special_tokens=True)
            if not is_task_correct(text, rollout["answer"], rollout["task"]):
                return None
            answers.append(answer)
        return make_target(rollout, thinking, min(answers, key=len), kind, len(steps))


def thinking_text(steps):
    """A thinking block in Qwen3's own layout: "<think>\\n...\\n</think>\\n\\n"."""
    return "<think>\n" + "".join(steps).strip() + "\n</think>\n\n"


def split_steps(text):
    """Paragraphs of the thinking, each keeping its trailing blank line."""
    return [part for part in STEP_END.split(text) if part.strip()]


def settle_index(curve, threshold):
    """Fewest steps after which p(gold) stays at or above the threshold, or None."""
    settle = None
    for steps, p in enumerate(curve):
        if p >= threshold:
            settle = steps if settle is None else settle
        else:
            settle = None
    return settle


def flat_steps(curve, indices, tolerance):
    """Steps that did not raise p(gold) above the best point before them.

    curve[n] is p(gold) after the first n steps, so step i moves it from curve[i] to curve[i + 1].
    """
    return [i for i in indices if curve[i + 1] - max(curve[:i + 1]) <= tolerance]


def drop_flat(kept, flat, accept):
    """Delete as many flat steps as `accept` allows, testing them in groups that split on failure."""
    kept = list(kept)

    def attempt(group):
        nonlocal kept
        trial = [i for i in kept if i not in group]
        if accept(trial):
            kept = trial
        elif len(group) > 1:
            half = len(group) // 2
            attempt(group[:half])
            attempt(group[half:])

    if flat:
        attempt(list(flat))
    return kept


def make_target(rollout, thinking, answer, kind, steps):
    return {"id": rollout["id"], "source": rollout["source"], "task": rollout["task"],
            "answer": rollout["answer"], "kind": kind, "steps": steps,
            "prompt_ids": rollout["prompt_ids"], "completion_ids": thinking + answer,
            "original_tokens": rollout["output_tokens"], "thinking_tokens": len(thinking)}


def original_target(rollout):
    return {"id": rollout["id"], "source": rollout["source"], "task": rollout["task"],
            "answer": rollout["answer"], "kind": "original", "steps": None,
            "prompt_ids": rollout["prompt_ids"], "completion_ids": rollout["output_ids"],
            "original_tokens": rollout["output_tokens"], "thinking_tokens": len(rollout["thinking_ids"])}


def report(targets, log, by_question, out):
    lines = [f"{'source':<16}{'questions':>10}{'targets':>9}{'pruned':>8}{'trunc':>7}{'orig':>6}"
             f"{'tokens before':>15}{'after':>8}{'saved':>8}"]
    sources = sorted({r[0]["source"] for r in by_question.values()})
    for source in sources + ["ALL"]:
        group = [t for t in targets if source in ("ALL", t["source"])]
        questions = sum(1 for r in by_question.values() if source in ("ALL", r[0]["source"]))
        kinds = Counter(t["kind"] for t in group)
        # "before": the mean full length of the question's correct rollouts, the honest baseline
        before = [sum(r["output_tokens"] for r in by_question[t["id"]] if r["correct"])
                  / max(1, sum(r["correct"] for r in by_question[t["id"]])) for t in group]
        after = [len(t["completion_ids"]) for t in group]
        saved = 1 - sum(after) / sum(before) if before else float("nan")
        lines.append(f"{source:<16}{questions:>10}{len(group):>9}{kinds['pruned']:>8}{kinds['truncated']:>7}"
                     f"{kinds['original']:>6}{avg(before):>15.0f}{avg(after):>8.0f}{saved:>8.0%}")
    statuses = Counter(entry["status"] for entry in log)
    lines += ["", "correct rollouts: " + ", ".join(f"{k} {v}" for k, v in sorted(statuses.items())),
              "pruned/trunc/orig: which variant became the question's target (shortest accepted)",
              "tokens: mean completion length (thinking + answer) before and after"]
    text = "\n".join(lines)
    print(text)
    (out / "summary.txt").write_text(text + "\n")


def avg(values):
    return sum(values) / len(values) if values else float("nan")


if __name__ == "__main__":
    main()
