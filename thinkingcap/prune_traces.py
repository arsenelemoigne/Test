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

--rule mvt replaces step 2's fixed threshold with the Marginal Value Theorem (Charnov 1976):
thinking is foraging for certainty, g(t) is p(gold) after t thinking tokens (its running
maximum, the gain secured so far), and the model should leave the patch at the first t where
the best gain rate still available, (g(t') - g(t)) / (t' - t) for any later t', drops to the
average rate g(t) / (t + c) at every later point. The travel cost c (--travel) is the answer tokens that follow
anyway: a larger c means staying longer. Step 3 then keeps a deletion if p(gold) stays at the
level the cut reached (or the threshold, whichever is lower).

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
        if group[0]["source"] in args.full_sources:  # e.g. math: the reasoning is needed to the end
            full = [original_target(r) for r in group if r["correct"] and not r["truncated"] and r["closed_thinking"]]
            if full:
                targets.append(min(full, key=lambda t: len(t["completion_ids"])))
            continue
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
    parser.add_argument("--rule", choices=("settle", "mvt"), default="settle",
                        help="settle: cut where p(gold) reaches --threshold and stays; "
                             "mvt: cut where the marginal gain drops to the average rate (Charnov)")
    parser.add_argument("--travel", type=float, default=256,
                        help="mvt travel cost c in tokens: the answer that follows anyway; larger stays longer")
    parser.add_argument("--adapter", help="LoRA to merge into --model before scoring and checking")
    parser.add_argument("--full-sources", type=lambda text: set(text.split(",")), default=set(),
                        help="datasets kept whole (shortest correct rollout, no cut), e.g. gsm8k")
    parser.add_argument("--threshold", type=float, default=0.9, help="p(gold) a kept trace must reach")
    parser.add_argument("--flat", type=float, default=0.02,
                        help="a step whose p(gold) gain over the best earlier point is at most this is 'flat'")
    parser.add_argument("--min-steps", type=int, default=1, help="never prune the thinking below this")
    parser.add_argument("--margin", type=int, default=0,
                        help="steps kept after the settle point: p(gold) can settle before a small model "
                             "has done the work its sampled answer needs")
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
        if getattr(args, "adapter", None):
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, args.adapter).merge_and_unload().eval()
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
        positions = step_positions([len(self.encode(step)) for step in steps])
        settle = settle_index(curve, args.threshold)
        mvt = mvt_index(curve, positions, args.travel)
        log = {"steps": len(steps), "settle_step": settle, "mvt_step": mvt, "positions": positions,
               "curve": [round(p, 4) for p in curve]}
        cut = mvt if args.rule == "mvt" else settle
        if cut is None:
            return {"targets": [], "log": log | {"status": "unsettled"}}

        truncated = list(range(min(len(steps), max(cut + args.margin, args.min_steps))))
        level = args.threshold if args.rule == "settle" else min(args.threshold, curve[len(truncated)])
        flat = flat_steps(curve, truncated, args.flat)
        pruned = drop_flat(truncated, flat, lambda kept: len(kept) >= args.min_steps
                           and p_gold([kept])[0] >= level)
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


def step_positions(lengths):
    """Thinking tokens spent after the first n steps, n = 0..len(lengths)."""
    positions = [0]
    for length in lengths:
        positions.append(positions[-1] + length)
    return positions


def mvt_index(curve, positions, travel):
    """Steps to keep under the Marginal Value Theorem: the first n where staying stops paying.

    g is the running maximum of p(gold) (the certainty secured so far, never lost by a dip).
    Stay while some later point of the trace pays better per token than the average rate so
    far, g(t) / (t + travel); leave at the first n where none does. Looking at every later
    point, not only the next steps, matters on math: p(gold) of the exact number stays near 0
    until the computation ends, then jumps, and a short look-ahead would leave at 0.
    Keeps everything when the gain never slows down.
    """
    gain, best = [], 0.0
    for p in curve:
        best = max(best, p)
        gain.append(best)
    last = len(curve) - 1
    for n in range(last):
        ahead = max((gain[m] - gain[n]) / max(1, positions[m] - positions[n]) for m in range(n + 1, last + 1))
        if ahead <= gain[n] / (positions[n] + travel):
            return n
    return last


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
    lines += ["", *cut_comparison(log)]
    statuses = Counter(entry["status"] for entry in log)
    lines += ["", "correct rollouts: " + ", ".join(f"{k} {v}" for k, v in sorted(statuses.items())),
              "pruned/trunc/orig: which variant became the question's target (shortest accepted)",
              "tokens: mean completion length (thinking + answer) before and after"]
    text = "\n".join(lines)
    print(text)
    (out / "summary.txt").write_text(text + "\n")


def cut_comparison(log):
    """Where the MVT cut falls against the settle cut, in thinking tokens, over the rollouts both define."""
    pairs = [(entry["positions"][entry["settle_step"]], entry["positions"][entry["mvt_step"]])
             for entry in log if entry.get("mvt_step") is not None and entry.get("settle_step") is not None]
    if not pairs:
        return ["settle vs mvt: no rollout has both cuts"]
    earlier = sum(m < s for s, m in pairs) / len(pairs)
    later = sum(m > s for s, m in pairs) / len(pairs)
    return [f"settle vs mvt cut over {len(pairs)} rollouts: median thinking tokens kept "
            f"settle {median([s for s, _ in pairs]):.0f}, mvt {median([m for _, m in pairs]):.0f}, "
            f"median (mvt - settle) {median([m - s for s, m in pairs]):+.0f}; "
            f"mvt earlier {earlier:.0%}, later {later:.0%}, same {1 - earlier - later:.0%}"]


def median(values):
    values = sorted(values)
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def avg(values):
    return sum(values) / len(values) if values else float("nan")


if __name__ == "__main__":
    main()
