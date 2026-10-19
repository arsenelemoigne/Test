"""Fading worked examples: shorten the model's own correct traces a little more every round.

One brutal cut (prune_traces.py) taught the model to stop right after restating the question.
Here the reasoning is removed gradually, as in fading worked examples (Renkl) and Stepwise
Internalization (Deng et al. 2024), so each round the model only has to absorb a few skipped
steps:

  1. Per question, take its shortest correct rollout and split the thinking into steps
     (paragraphs). Score p(gold) after every step once, with the base model (cached).
  2. Round k deletes steps until about k x --step of the thinking tokens are gone: the steps
     with the smallest probe gain first (--order gain), or the earliest ones (--order early).
     With --floor-travel C it never goes below the Marginal Value Theorem cut for travel cost
     C (prune_traces.mvt_index): MVT sets the final target length.
  3. Inside the steps that remain, compact the words (TokenSkip, Xia et al. 2025): a
     token classifier trained to tell filler from content (LLMLingua-2) scores every word,
     and round k keeps the best 1 - k x --word-step of them. Words with digits or math
     symbols always stay. "Okay, so 3 bags of 12 apples, that means 3 times 12 is 36"
     becomes something like "3 bags 12 apples 3 times 12 is 36".
     --word-scorer marginal scores each word with our own model instead (marginal token
     analysis): delete that one word from the whole trace and measure how much log p(gold)
     drops. A word the answer does not need scores about 0 and goes first.
  4. The current model (--adapter, round k-1's LoRA) answers naturally from the shortened
     thinking --check-samples times. The trace is kept only if every answer is correct;
     otherwise the question keeps its round k-1 trace (--previous), so hard questions stay
     long instead of dropping out and leaving only the easy ones. Before falling back it
     tries once more with round k-1's word compaction.

    python fade_traces.py --rollouts runs/probe/rollouts.jsonl --out runs/fade/r1 --round 1
    python sft.py --data runs/fade/r1/sft.jsonl --out adapters/fade-r1 --epochs 1
    python fade_traces.py --rollouts runs/probe/rollouts.jsonl --out runs/fade/r2 --round 2 \\
        --adapter adapters/fade-r1 --previous runs/fade/r1/sft.jsonl --cache runs/fade/r1/curves.jsonl
    python sft.py --data runs/fade/r2/sft.jsonl --out adapters/fade-r2 --epochs 1 --init-adapter adapters/fade-r1
"""

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from probe_curve import candidates, read_jsonl, score, write_jsonl
from prune_traces import Pruner, avg, mvt_index, original_target, split_steps, step_positions, thinking_text


def main():
    args = parse_arguments()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pruner = Pruner(SimpleNamespace(model=args.model, adapter=args.adapter, dtype=args.dtype,
                                    check_samples=args.check_samples, answer_tokens=args.answer_tokens,
                                    batch_tokens=args.batch_tokens))

    by_question = defaultdict(list)
    for rollout in read_jsonl(Path(args.rollouts)):
        by_question[rollout["id"]].append(rollout)
    chosen = {qid: min(ok, key=lambda r: r["output_tokens"]) for qid, group in by_question.items()
              if (ok := [r for r in group if r["correct"] and r["closed_thinking"] and not r["truncated"]])}
    previous = {row["id"]: row for row in read_jsonl(Path(args.previous))} if args.previous else {}
    cache = {row["id"]: row for row in read_jsonl(Path(args.cache))} if args.cache else {}
    scorer = None
    if args.word_step > 0:
        scorer = MarginalScorer(pruner, args.batch_tokens) if args.word_scorer == "marginal" \
            else WordScorer(args.word_model, pruner.device)
    keep_rate = max(args.min_keep, 1 - args.round * args.word_step)
    fallback_rate = max(args.min_keep, 1 - (args.round - 1) * args.word_step)

    curves, targets, log = [], [], []
    for number, (qid, rollout) in enumerate(chosen.items(), 1):
        steps = split_steps(pruner.decode_thinking(rollout["thinking_ids"]))
        if qid in cache:
            curve, positions = cache[qid]["curve"], cache[qid]["positions"]
            word_scores = cache[qid].get("word_scores")
        else:
            positions = step_positions([len(pruner.encode(step)) for step in steps])
            curve = base_curve(pruner, rollout, by_question[qid], steps)
            word_scores = None
        if scorer and word_scores is None:
            if isinstance(scorer, MarginalScorer):
                word_scores = scorer.trace_scores(rollout, by_question[qid], steps)
            else:
                word_scores = [scorer.scores(words_of(step)) for step in steps]
        curves.append({"id": qid, "curve": curve, "positions": positions, "word_scores": word_scores})

        floor = positions[mvt_index(curve, positions, args.floor_travel, args.smooth)] \
            if args.floor_travel is not None else 0
        kept = fade(curve, positions, args.round * args.step, args.order, floor, args.min_steps)
        earlier = previous.get(qid)
        entry = {"id": qid, "source": rollout["source"], "steps": len(steps), "kept_steps": len(kept),
                 "thinking_tokens": positions[-1], "floor": floor}

        def shortened(rate):
            if not scorer or rate >= 1:
                return [steps[i] for i in kept]
            return [compact(steps[i], word_scores[i], rate) for i in kept]

        rates = [keep_rate] + ([fallback_rate] if scorer and fallback_rate != keep_rate else [])
        if len(kept) == len(steps) and (not scorer or keep_rate >= 1):
            target, status = original_target(rollout), "nothing_to_delete"
        else:
            target, status = None, None
            for attempt, rate in enumerate(rates):
                target = pruner.checked_target(rollout, shortened(rate), "faded")
                if target is not None:
                    target |= {"word_keep": rate}
                    status = "faded" if attempt == 0 else "faded_fewer_words"
                    break
            if target is None:
                target = earlier or original_target(rollout)
                status = "kept_previous" if earlier else "kept_original"
        if target is not None:
            fresh = status in ("faded", "faded_fewer_words")
            targets.append(target | {"round": args.round if fresh else target.get("round", 0)})
        log.append(entry | {"status": status})
        if number % 20 == 0 or number == len(chosen):
            print(f"faded {number}/{len(chosen)} questions", flush=True)

    write_jsonl(out / "curves.jsonl", curves)
    write_jsonl(out / "sft.jsonl", targets)
    write_jsonl(out / "log.jsonl", log)
    report(targets, log, chosen, out, args)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter", help="the current learner (round k-1's LoRA): it scores the natural-answer check")
    parser.add_argument("--rollouts", required=True, help="rollouts.jsonl from probe_curve.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--round", type=int, required=True, help="k: delete about k x --step of the thinking")
    parser.add_argument("--step", type=float, default=0.1, help="share of the thinking removed per round")
    parser.add_argument("--order", choices=("gain", "early"), default="gain",
                        help="gain: smallest probe gain first; early: earliest steps first")
    parser.add_argument("--previous", help="round k-1's sft.jsonl: the fallback when a trace fails the check")
    parser.add_argument("--cache", help="an earlier round's curves.jsonl, to skip rescoring the base curves")
    parser.add_argument("--floor-travel", type=float, help="never shorten below the MVT cut for this travel cost")
    parser.add_argument("--word-step", type=float, default=0.1,
                        help="share of the words compacted away per round inside kept steps; 0 turns it off")
    parser.add_argument("--min-keep", type=float, default=0.3, help="never keep fewer than this share of words")
    parser.add_argument("--word-scorer", choices=("lingua", "marginal"), default="lingua",
                        help="lingua: LLMLingua-2's filler classifier; marginal: log p(gold) lost without the word")
    parser.add_argument("--word-model", default="microsoft/llmlingua-2-xlm-roberta-large-meetingbank")
    parser.add_argument("--smooth", type=int, default=2)
    parser.add_argument("--min-steps", type=int, default=1)
    parser.add_argument("--check-samples", type=int, default=2)
    parser.add_argument("--answer-tokens", type=int, default=512)
    parser.add_argument("--batch-tokens", type=int, default=32768)
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


def base_curve(pruner, rollout, siblings, steps):
    """p(gold) after each prefix of the steps, scored once (with --adapter merged, if given)."""
    answers, terminated = candidates(rollout, siblings)
    suffixes = [pruner.encode(" " + a) + ([pruner.end_of_turn] if terminated else []) for a in answers]
    curve = pruner.p_gold(rollout, [steps[:n] for n in range(len(steps) + 1)], suffixes)
    return [round(p, 4) for p in curve]


def fade(curve, positions, share, order, floor=0, min_steps=1):
    """Indices of the steps kept once about `share` of the thinking tokens are deleted.

    Steps go in order of their probe gain (curve[i + 1] - curve[i], smallest first, the later
    step first on ties) or from the start. A step is skipped if deleting it would take the
    thinking under `floor` tokens or under `min_steps` steps.
    """
    count = len(curve) - 1
    lengths = [positions[i + 1] - positions[i] for i in range(count)]
    total = positions[-1]
    if order == "gain":
        queue = sorted(range(count), key=lambda i: (curve[i + 1] - curve[i], -i))
    else:
        queue = list(range(count))
    kept, removed = set(range(count)), 0
    for i in queue:
        if removed >= share * total:
            break
        if len(kept) <= min_steps or total - removed - lengths[i] < floor:
            continue
        kept.discard(i)
        removed += lengths[i]
    return sorted(kept)


PROTECTED = re.compile(r"[0-9=+\-*/×÷^%$<>()]")


def words_of(step):
    """The words of a step, line by line, in reading order."""
    return [word for line in step.split("\n") for word in line.split()]


def compact(step, scores, rate):
    """The step with only the best-scored `rate` of its words; words with digits or math stay.

    Line breaks and the step's trailing blank line are kept, so the thinking keeps its layout.
    """
    lines = [line.split() for line in step.rstrip("\n").split("\n")]
    flat = [word for line in lines for word in line]
    if len(flat) != len(scores):
        return step
    droppable = sorted((score, i) for i, (word, score) in enumerate(zip(flat, scores))
                       if not PROTECTED.search(word))
    return without(step, {i for _, i in droppable[:int(round((1 - rate) * len(droppable)))]})


def without(step, dropped):
    """The step with the words at the `dropped` positions (words_of order) removed."""
    lines = [line.split() for line in step.rstrip("\n").split("\n")]
    out, index = [], 0
    for line in lines:
        kept = [word for word in line if (index := index + 1) - 1 not in dropped]
        if kept or not line:
            out.append(" ".join(kept))
    trailing = step[len(step.rstrip("\n")):]
    return "\n".join(out) + trailing


class MarginalScorer:
    """Our model's own marginal value of each word: log p(gold) lost when only that word is deleted.

    Every one-word deletion of the trace is scored by teacher forcing, batched like the probe.
    Words with digits or math symbols are never deleted, so they are not scored (kept at 99).
    """

    def __init__(self, pruner, batch_tokens):
        self.pruner, self.batch_tokens = pruner, batch_tokens

    def trace_scores(self, rollout, siblings, steps):
        pruner = self.pruner
        answers, terminated = candidates(rollout, siblings)
        suffix = pruner.encode(" " + answers[0]) + ([pruner.end_of_turn] if terminated else [])
        words = [words_of(step) for step in steps]
        variants = [(None, None)] + [(s, j) for s, ws in enumerate(words) for j, w in enumerate(ws)
                                     if not PROTECTED.search(w)]
        rows = []
        for v, (s, j) in enumerate(variants):
            edited = steps if s is None else steps[:s] + [without(steps[s], {j})] + steps[s + 1:]
            context = rollout["prompt_ids"] + pruner.encode(thinking_text(edited) + "Final answer:")
            rows.append((v, 0, context, suffix))
        logps = score(pruner.model, rows, self.batch_tokens, pruner.device)
        full = logps[(0, 0)]
        result = [[99.0] * len(ws) for ws in words]
        for v, (s, j) in enumerate(variants[1:], 1):
            result[s][j] = round(full - logps[(v, 0)], 4)
        return result


class WordScorer:
    """LLMLingua-2's token classifier: p(keep) per word, averaged over the word's sub-tokens."""

    def __init__(self, name, device):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.torch, self.device = torch, device
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForTokenClassification.from_pretrained(name).to(device).eval()

    def scores(self, words, window=200):
        result = []
        for start in range(0, len(words), window):
            chunk = words[start:start + window]
            encoded = self.tokenizer(chunk, is_split_into_words=True, truncation=True, max_length=512,
                                     return_tensors="pt")
            with self.torch.no_grad():
                keep = self.model(**encoded.to(self.device)).logits.softmax(-1)[0, :, 1].tolist()
            per_word = defaultdict(list)
            for position, word in enumerate(encoded.word_ids()):
                if word is not None:
                    per_word[word].append(keep[position])
            # a word cut off by truncation has no score: keep it
            result += [round(avg(per_word[j]), 4) if per_word[j] else 1.0 for j in range(len(chunk))]
        return result


def report(targets, log, chosen, out, args):
    lines = [f"round {args.round}: target {min(1.0, args.round * args.step):.0%} of the steps' tokens removed, "
             f"words kept inside steps {max(args.min_keep, 1 - args.round * args.word_step):.0%}, "
             f"order {args.order}, floor travel {args.floor_travel}", "",
             f"{'source':<16}{'questions':>10}{'faded':>7}{'fewer':>7}{'prev':>6}{'orig':>6}"
             f"{'tokens before':>15}{'after':>8}{'saved':>8}"]
    for source in sorted({e["source"] for e in log}) + ["ALL"]:
        entries = [e for e in log if source in ("ALL", e["source"])]
        group = [t for t in targets if source in ("ALL", t["source"])]
        status = Counter(e["status"] for e in entries)
        before = [chosen[t["id"]]["output_tokens"] for t in group]
        after = [len(t["completion_ids"]) for t in group]
        saved = 1 - sum(after) / sum(before) if before else float("nan")
        lines.append(f"{source:<16}{len(entries):>10}{status['faded']:>7}{status['faded_fewer_words']:>7}"
                     f"{status['kept_previous']:>6}{status['kept_original'] + status['nothing_to_delete']:>6}"
                     f"{avg(before):>15.0f}{avg(after):>8.0f}{saved:>8.0%}")
    lines += ["", "faded: shortened this round and passed the check; fewer: passed only with round k-1's",
              "word compaction; prev: failed, kept round k-1's trace; orig: failed with no earlier trace",
              "tokens: mean completion length (thinking + answer) of the question's rollout before and after"]
    text = "\n".join(lines)
    print(text)
    (out / "summary.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
