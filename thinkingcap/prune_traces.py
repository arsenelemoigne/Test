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

--rule marginal is the marginal token analysis at the level of --unit: delete each unit alone
and measure the p(gold) it costs; units costing at most --flat are deleted (in groups that
split on failure, as in step 3) as long as the whole edited trace keeps p(gold) within --flat
of the full trace. No truncation.

--unit picks what a "step" is: paragraph (default), sentence, or tokensN (fixed batches of N
thinking tokens, e.g. tokens32).

Code, tool-call and instruction rollouts (typed.OWN_ANSWER) have no short gold answer: their
"p(gold)" is p(the rollout's own correct answer | thinking), a per-token geometric mean.

--checker vllm runs every natural-answer check of the run in one vLLM batch (check_vllm.py),
which long code and instruction answers need; hf checks one trace at a time.

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
from typed import OWN_ANSWER, full_text, is_correct
from utils.eval.recipe import SAMPLING

STEP_END = re.compile(r"(?<=\n\n)")
SENTENCE = re.compile(r"[^.!?\n]*(?:[.!?]+[\"')\]]*|\n+|$)[ \t]*")
GRADE_FIELDS = ("id", "task", "answer", "grader", "entry_point", "tests", "setup", "category", "function",
                "reference", "instruction_id_list", "kwargs", "original_prompt")


def main():
    args = parse_arguments()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pruner = Pruner(args)

    by_question = defaultdict(list)
    for rollout in read_jsonl(Path(args.rollouts)):
        if not args.sources or rollout["source"] in args.sources:
            by_question[rollout["id"]].append(rollout)
    if args.easy_tokens:  # keep the questions the raw model solves briefly: SVAMP-like difficulty
        by_question = defaultdict(list, {q: g for q, g in by_question.items()
                                         if any(r["correct"] and r["output_tokens"] <= args.easy_tokens for r in g)})

    targets, log, jobs, pending = [], [], [], {}
    for number, (qid, group) in enumerate(by_question.items(), 1):
        # a source (e.g. math: the reasoning is needed to the end) or a category (e.g. BFCL irrelevance)
        if args.full_sources & {group[0]["source"], group[0].get("category")}:
            full = [original_target(r) for r in group if r["correct"] and not r["truncated"] and r["closed_thinking"]]
            if full:
                targets.append(min(full, key=lambda t: len(t["completion_ids"])))
            continue
        pending[qid] = []
        for rollout in group:
            if not (rollout["correct"] and rollout["closed_thinking"]):
                continue
            outcome = pruner.prune(rollout, group)
            entry = {"id": rollout["id"], "sample": rollout["sample"], "source": rollout["source"],
                     "original_tokens": rollout["output_tokens"]} | outcome["log"]
            log.append(entry)
            for kind, units in outcome["candidates"]:
                pending[qid].append(len(jobs))
                jobs.append((rollout, units, kind, entry))
        if number % 10 == 0 or number == len(by_question):
            print(f"probed {number}/{len(by_question)} questions, {len(jobs)} traces to check", flush=True)

    checked = pruner.check_many([(r, u, k) for r, u, k, _ in jobs], out)
    for (rollout, units, kind, entry), target in zip(jobs, checked):
        entry[f"{kind}_check"] = target is not None
        entry["status"] = "accepted" if target is not None or entry.get("status") == "accepted" else "failed_check"
    for qid, indices in pending.items():
        accepted = [checked[i] for i in indices if checked[i] is not None]
        group = by_question[qid]
        if not accepted and args.keep_unsettled:
            accepted = [original_target(r) for r in group
                        if r["correct"] and not r["truncated"] and r["closed_thinking"]]
        if accepted:
            targets.append(min(accepted, key=lambda t: len(t["completion_ids"])))

    write_jsonl(out / "sft.jsonl", targets)
    write_jsonl(out / "log.jsonl", log)
    report(targets, log, by_question, out)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--rollouts", required=True, help="rollouts.jsonl from probe_curve.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--rule", choices=("settle", "mvt", "marginal"), default="settle",
                        help="settle: cut where p(gold) reaches --threshold and stays; "
                             "mvt: cut where the marginal gain drops to the average rate (Charnov)")
    parser.add_argument("--travel", type=float, default=256,
                        help="mvt travel cost c in tokens: the answer that follows anyway; larger stays longer")
    parser.add_argument("--adapter", help="LoRA to merge into --model before scoring and checking")
    parser.add_argument("--sources", type=lambda text: set(text.split(",")), default=set(),
                        help="only these datasets' rollouts (default: all)")
    parser.add_argument("--easy-tokens", type=int, default=0,
                        help="keep only questions with a correct rollout at most this long (0: all)")
    parser.add_argument("--unit", default="paragraph", help="paragraph, sentence, or tokensN (e.g. tokens32)")
    parser.add_argument("--checker", choices=("hf", "vllm"), default="hf")
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
        """{"candidates": [(kind, units)], "log": {...}} for one correct rollout; checks come later."""
        args = self.args
        steps = split_units(self.decode_thinking(rollout["thinking_ids"]), args.unit, self.tokenizer)
        suffixes = self.suffixes(rollout, siblings)

        def p_gold(variants):
            return self.p_gold(rollout, [[steps[i] for i in kept] for kept in variants], suffixes)

        curve = p_gold([range(n) for n in range(len(steps) + 1)])
        positions = step_positions([len(self.encode(step)) for step in steps])
        settle = settle_index(curve, args.threshold)
        mvt = mvt_index(curve, positions, args.travel)
        log = {"steps": len(steps), "settle_step": settle, "mvt_step": mvt, "positions": positions,
               "curve": [round(p, 4) for p in curve]}
        everything = list(range(len(steps)))

        if args.rule == "marginal":
            full = curve[-1]
            without = p_gold([[j for j in everything if j != i] for i in everything])
            costs = {i: full - p for i, p in zip(everything, without)}
            cheap = sorted((i for i in everything if costs[i] <= args.flat), key=costs.get)
            pruned = drop_flat(everything, cheap, lambda kept: len(kept) >= args.min_steps
                               and p_gold([kept])[0] >= full - args.flat)
            log |= {"costs": [round(costs[i], 4) for i in everything], "pruned_steps": len(pruned),
                    "flat_steps": len(cheap)}
            if pruned == everything:
                return {"candidates": [], "log": log | {"status": "nothing_to_delete"}}
            return {"candidates": [("pruned", [steps[i] for i in pruned])], "log": log | {"status": "pending"}}

        cut = mvt if args.rule == "mvt" else settle
        if cut is None:
            return {"candidates": [], "log": log | {"status": "unsettled"}}

        truncated = list(range(min(len(steps), max(cut + args.margin, args.min_steps))))
        level = args.threshold if args.rule == "settle" else min(args.threshold, curve[len(truncated)])
        flat = flat_steps(curve, truncated, args.flat)
        pruned = drop_flat(truncated, flat, lambda kept: len(kept) >= args.min_steps
                           and p_gold([kept])[0] >= level)
        log |= {"truncated_steps": len(truncated), "pruned_steps": len(pruned), "flat_steps": len(flat)}
        variants = [("pruned", pruned)] + ([("truncated", truncated)] if pruned != truncated else [])
        return {"candidates": [(kind, [steps[i] for i in kept]) for kind, kept in variants],
                "log": log | {"status": "pending"}}

    def suffixes(self, rollout, siblings):
        """Teacher-forced answers to score: the candidate answers, or None for an own-answer task."""
        if rollout["task"] in OWN_ANSWER:
            return None
        answers, terminated = candidates(rollout, siblings)
        return [self.encode(" " + a) + ([self.end_of_turn] if terminated else []) for a in answers]

    def own_answer(self, rollout):
        """The rollout's reply after its thinking, as token ids ending with the end-of-turn token."""
        if "_answer_ids" not in rollout:
            close = self.tokenizer.convert_tokens_to_ids("</think>")
            ids = rollout["output_ids"]
            tail = ids[ids.index(close) + 1:] if close in ids else ids
            text = self.tokenizer.decode(tail, skip_special_tokens=True).lstrip("\n")
            rollout["_answer_ids"] = self.encode(text) + [self.end_of_turn]
        return rollout["_answer_ids"]

    def answer_cap(self, rollout):
        return max(self.args.answer_tokens, 1024) if rollout["task"] in OWN_ANSWER else self.args.answer_tokens

    def check_many(self, jobs, out):
        """[(rollout, units, kind)] -> the SFT row for each, or None where a natural answer failed."""
        if self.args.check_samples == 0 or self.args.checker == "hf" or not jobs:
            return [self.checked_target(r, u, k) for r, u, k in jobs]
        import subprocess
        import sys

        import torch

        if torch.cuda.is_available():  # hand back the scoring passes' cached blocks before vLLM starts
            torch.cuda.empty_cache()
        thinking = [self.encode(thinking_text(units)) for _, units, _ in jobs]
        inputs, results = out / "check_inputs.jsonl", out / "check_results.jsonl"
        write_jsonl(inputs, [{"key": i, "prompt_ids": r["prompt_ids"], "thinking_ids": t,
                              "max_tokens": self.answer_cap(r), "row": {f: r[f] for f in GRADE_FIELDS if f in r}}
                             for i, ((r, _, _), t) in enumerate(zip(jobs, thinking))])
        command = [sys.executable, "check_vllm.py", "--model", self.args.model, "--inputs", str(inputs),
                   "--out", str(results), "--samples", str(self.args.check_samples), "--dtype", self.args.dtype]
        if getattr(self.args, "adapter", None):
            command += ["--adapter", self.args.adapter]
        subprocess.run(command, check=True)
        verdicts = {v["key"]: v["answer_ids"] for v in read_jsonl(results)}
        return [make_target(r, t, verdicts[i], k, len(u)) if verdicts.get(i) else None
                for i, ((r, u, k), t) in enumerate(zip(jobs, thinking))]

    def decode_thinking(self, thinking_ids):
        if thinking_ids and thinking_ids[0] == self.open_id:
            thinking_ids = thinking_ids[1:]
        return self.tokenizer.decode(thinking_ids)

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def p_gold(self, rollout, variants, suffixes):
        """p(gold) after each variant's thinking, as probe_curve.py measures it.

        suffixes None: an own-answer task, scored as the per-token geometric mean of p(own answer).
        """
        import math

        if suffixes is None:
            answer = self.own_answer(rollout)
            rows = [(v, 0, rollout["prompt_ids"] + self.encode(thinking_text(steps)), answer)
                    for v, steps in enumerate(variants)]
            scores = score(self.model, rows, self.args.batch_tokens, self.device)
            return [math.exp(scores[(v, 0)] / len(answer)) for v in range(len(variants))]

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
            answer = self.own_answer(rollout) if rollout["task"] in OWN_ANSWER \
                else self.encode(f"Final answer: {rollout['answer']}") + [self.end_of_turn]
            return make_target(rollout, thinking, answer, kind, len(steps))
        inputs = torch.tensor([context], device=self.device)
        with torch.no_grad():
            output = self.model.generate(
                inputs, attention_mask=torch.ones_like(inputs), do_sample=True,
                num_return_sequences=self.args.check_samples, max_new_tokens=self.answer_cap(rollout),
                eos_token_id=self.end_of_turn, pad_token_id=self.end_of_turn, **SAMPLING)
        answers = []
        for row in output[:, len(context):].tolist():
            if self.end_of_turn not in row:
                return None  # ran out of room: the evaluation would cut it off too
            answer = row[:row.index(self.end_of_turn) + 1]
            text = full_text(self.tokenizer, thinking + answer)
            if not is_correct(rollout, text):
                return None
            answers.append(answer)
        return make_target(rollout, thinking, min(answers, key=len), kind, len(steps))


def thinking_text(steps):
    """A thinking block in Qwen3's own layout: "<think>\\n...\\n</think>\\n\\n"."""
    return "<think>\n" + "".join(steps).strip() + "\n</think>\n\n"


def split_units(text, unit="paragraph", tokenizer=None):
    """The thinking cut into the pieces the rules work on: paragraphs, sentences or token batches."""
    if unit == "paragraph":
        return split_steps(text)
    if unit == "sentence":
        parts = []
        for part in SENTENCE.findall(text):
            if part.strip() or not parts:
                parts.append(part)
            else:
                parts[-1] += part  # a bare line break belongs to the sentence before it
        return [part for part in parts if part]
    if unit.startswith("tokens"):
        size, ids = int(unit[len("tokens"):]), tokenizer.encode(text, add_special_tokens=False)
        return [tokenizer.decode(ids[i:i + size]) for i in range(0, len(ids), size)]
    raise ValueError(f"unknown unit {unit!r}")


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
