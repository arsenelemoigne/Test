"""Track A pairs (GAME_PLAN.md v3, 5 Nov): "be right" pairs from a TRAINED model's own answers, no shortening pressure.

Per question (the model's 4 answers), two kinds, each a right answer over a FINISHED wrong one (no cut rejections):
  guard  the shortest right answer > the shortest finished wrong answer
  right  a right answer > a finished wrong answer of similar length (max(Lc/Lr, Lr/Lc) <= --ratio), exactly half of each
         part's right pairs with the right answer strictly LONGER (a correctness signal without a length signal)
Quotas per part (--quota part=n), half guard / half right; <= 3 pairs per question, <= 2 right pairs per question; no
sample pair used twice. --kinds guard (9 Nov): guard pairs only, the whole quota (the user's own 21 Oct rule, bon_pairs'
guard; the similar-length "right" kind matches Lightning McQwen3's correctness DPO and is left out). A question gives at
most one guard pair, so a part needs at least its quota of mixed-outcome questions. Mixed-outcome questions are the only source (a pair needs a right and a wrong answer).
--think-only PARTS (9 Nov, user, from the rc1-soup-30 full-test diagnosis): for these parts (ifeval) "shortest" is the
shortest THINKING, never the shortest whole answer, so a pair never rewards a shorter final response (length constraints:
"at least 600 words"). --prefer-reread (same day): the chosen right answer is the shortest one that RE-CHECKS THE QUESTION
(what exactly is asked, the example / assert, the constraints, whether a tool fits; regex markers on its thinking) when one
exists within 1.5x the shortest right answer's thinking, else the shortest: the shortened model's lost questions were
right calculations of the wrong question, the re-check it had dropped.
--count-rules (10 Nov, user): IFEval questions with a COUNTING rule (word / sentence / paragraph counts, keyword frequency,
number of sections, bullets or placeholders) fail with responses that are too short (the full-test diagnostic): there the
chosen right answer must have a RESPONSE at least as long as the rejected one's and THINKING no longer than 1.2x the
rejected one's (the pair then differs in response length and correctness only: "write what is asked", not "think more");
when no right answer fits, the question gives no pair.
The output is ordered so that every --block consecutive pairs hold the intended part x kind mix (training reads it in
order with dpo.py --keep-order: a checkpoint at step 5 has seen a representative slice, not one part).

    python repair_pairs.py --rollouts rp/rollouts.jsonl --out pairs-rp --quota svamp=640,mbpp=400,bfcl_irrelevance=240,...
"""
import argparse
import json
import random
import re
from array import array
from collections import Counter, defaultdict
from pathlib import Path

from build_pool import part_of


REREAD = re.compile(r"\b(the (question|problem|task|user|prompt) (is )?(asks?|asking|says|wants|wanted|states|requires|mentions)|"
                    r"(re-?read|reading|read) (the|it) (question|problem|again|carefully)|let me re-?read|"
                    r"(they('re| are)|it('s| is)|we('re| are)) (being )?asked|asked (for|to)|"
                    r"(only|actually) (asks?|wants?)|wait,? (the|it) (question|problem|asks)|"
                    r"(check|compare|verify|test)\w* (it )?(against|with)? ?the (example|assert|test|sample|constraints?)|"
                    r"(none of the|no) (available )?(functions?|tools?) (fit|match|appl|is relevant|are relevant|can)|"
                    r"(does|do) (the|this|these|any) (functions?|tools?) (fit|match|apply|help)|"
                    r"the (constraints?|requirements?|instructions?) (are|is|say|require))", re.I)


COUNT_RULES = ("length_constraints:", "keywords:frequency", "keywords:letter_frequency", "detectable_format:number_",
               "detectable_format:multiple_sections", "detectable_content:number_placeholders",
               "change_case:capital_word_frequency")


def counting(r):
    return any(rule.startswith(COUNT_RULES) for rule in (r.get("instruction_id_list") or []))


def at_least(r):
    """(11 Nov, rpL) An IFEval question with a rule that asks for a MINIMUM amount of text: at least N words / sentences, a
    keyword or a letter at least N times, at least N placeholders or highlighted sections. The full-test records show these
    (not every counting rule) are where a shortened response fails."""
    for rule, kw in zip(r.get("instruction_id_list") or [], r.get("kwargs") or []):
        kw = kw or {}
        if rule in ("length_constraints:number_words", "length_constraints:number_sentences", "keywords:frequency"):
            if kw.get("relation") == "at least":
                return True
        elif rule == "keywords:letter_frequency":
            if kw.get("let_relation") == "at least":
                return True
        elif rule in ("detectable_content:number_placeholders", "detectable_format:number_highlighted_sections"):
            return True
    return False


def length_rule(r):
    """(11 Nov, rpM, consult10) An IFEval question where a SHORT response is what fails: at least N words / sentences, a keyword at
    least N times, N paragraphs (N >= 2). Narrower than at_least(): in rpL, placeholders / highlighted sections / letter
    frequency took two thirds of the counting pairs, and the true word-count questions fell from 37.5% to 19.6% right."""
    for rule, kw in zip(r.get("instruction_id_list") or [], r.get("kwargs") or []):
        kw = kw or {}
        if rule in ("length_constraints:number_words", "length_constraints:number_sentences", "keywords:frequency"):
            if kw.get("relation") == "at least":
                return True
        elif rule == "length_constraints:number_paragraphs":
            if (kw.get("num_paragraphs") or 0) >= 2:
                return True
    return False


def response_len(r):
    return len(r["output_ids"]) - len(r.get("thinking_ids") or [])


def thinking_len(r):  # (10 Nov, reviewer) an EMPTY thinking is 0 tokens, not the whole answer; only a missing field falls back
    ids = r.get("thinking_ids")
    return len(ids) if ids is not None else r["output_tokens"]


def finished(r):
    return r["closed_thinking"] and not r["truncated"]


def right(r):
    return bool(r["correct"]) and finished(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rollouts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quota", required=True, help="part=n,... (pairs per part, half guard / half right)")
    ap.add_argument("--ratio", type=float, default=1.25)
    ap.add_argument("--block", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--think-only", default="", help="parts (comma) whose shortness is the thinking only (ifeval)")
    ap.add_argument("--count-rules", action="store_true",
                    help="ifeval counting-rule questions: chosen response >= rejected's, chosen thinking <= 1.2x rejected's")
    ap.add_argument("--prefer-reread", action="store_true",
                    help="chosen = the shortest right answer that re-checks the question, within 1.5x the shortest's thinking")
    ap.add_argument("--shorten-min-right", type=float, default=0.75,
                    help="guard,shorten: a shorten pair only where this share of the answers is right")
    ap.add_argument("--shorten-ratio", type=float, default=1.5,
                    help="guard,shorten: the longest right answer must be >= this x the shortest")
    ap.add_argument("--mix-shares", default="shorten=0.25,right=0.60,guard=0.15",
                    help="--kinds mix (11 Nov, consult6 path 1): shares of shorten / equal-length right / capped guard per part")
    ap.add_argument("--floor-by-part", default="", help="(13 Nov) PART=ABS:RATIO,...: chosen answers think >= max(ABS, RATIO x "
                    "the ORIGINAL's median right thinking on that question (--base-think), else the group's)")
    ap.add_argument("--base-think", default="", help="(13 Nov) JSON {question id: the original's median right thinking tokens}")
    ap.add_argument("--mix-preset", default="", choices=("", "round1", "typed", "typed2", "round1b", "typed3", "round1c", "hold", "r2s1", "r2s2", "od3_arena"),
                    help="--kinds mix: per-type shares instead of --mix-shares (shorten / right / guard). round1 (path 1, from "
                         "the original): math + no-fit 60/40/0, code + one / several functions 70/30/0, IFEval's non-counting "
                         "half 50/50/0. typed (11 Nov, the IFEval + tool-call repair): math + code 0/80/20 (hold: no "
                         "'think less' pair), IFEval's non-counting half 60/40/0, tool calls 50/50/0")
    ap.add_argument("--resp-guard", type=float, default=0.0,
                    help="--kinds mix, --think-only parts (IFEval): a 'shorter' pair's chosen answer must have a RESPONSE at "
                         "least this x the rejected one's (0.9); 0 = off. The full-test records show IFEval's loss is "
                         "shortened responses ('at least N words'), not shortened thinking")
    ap.add_argument("--count-mode", default="all", choices=("all", "atleast", "length"),
                    help="--kinds mix: which IFEval questions get the counting-rule pairs (and no 'shorter' pair): all = every "
                         "counting rule; atleast (11 Nov, rpL) = only rules asking for a MINIMUM amount of text")
    ap.add_argument("--count-share", type=float, default=0.5, help="--kinds mix: the counting-rule pairs' share of IFEval's quota")
    ap.add_argument("--count-resp-cap", type=float, default=0.0,
                    help="--kinds mix: a counting-rule pair's chosen RESPONSE is at most this x the rejected one's + 25 tokens "
                         "(1.35); 0 = no cap. Without a cap the pairs taught 'write more' (consult9)")
    ap.add_argument("--count-resp-min", type=float, default=0.0,
                    help="--kinds mix (rpM): a counting-rule pair's chosen RESPONSE is at least this x the rejected one's + 15 "
                         "tokens (1.10): the pair teaches writing clearly MORE where a minimum length is asked; 0 = at least equal")
    ap.add_argument("--count-resp-pad", type=int, default=25, help="--kinds mix: the tokens added to --count-resp-cap (rpM: 60)")
    ap.add_argument("--no-shorten-counting", action="store_true",
                    help="--kinds mix (rpM): no 'shorter' pair on ANY IFEval question with a counting rule (they keep be-right pairs)")
    ap.add_argument("--resp-near-long", type=float, default=0.0,
                    help="--kinds mix, --think-only parts (rpM): a 'shorter' pair's chosen RESPONSE is at least this x the "
                         "rejected one's (0.95): with --resp-max 1.05 the response is held within 5%, only the thinking differs")
    ap.add_argument("--loop-response-only", default="",
                    help="--kinds mix (rpM): parts (comma) whose anti-loop pairs use only RESPONSE loops (thinking closed, cut at the cap)")
    ap.add_argument("--easy-math-prefix", default="sv-",
                    help="with a quota for svamp_hard: math questions whose id starts with this are 'svamp', the others 'svamp_hard'")
    ap.add_argument("--guard-wide", default="",
                    help="--kinds mix (rpM): parts (comma) whose 'don't stop early' guard is widened: the rejected wrong answer thinks "
                         "under 0.95x the question's median right thinking (was 0.5x), the chosen right answer 0.95-1.25x. The "
                         "full test showed the new math errors think 0.5-1.0x as long as a right answer (a skipped re-check)")
    ap.add_argument("--guard-wide-max", type=float, default=0.95,
                    help="(12 Nov, recipe 2 / consult16) with --guard-wide: the rejected wrong answer thinks under this x the "
                         "question's median right thinking (rpM used 0.95; consult16's plan 0.75)")
    ap.add_argument("--right-hold", default="",
                    help="--kinds mix (rpM): parts (comma) whose be-right pair takes a right answer thinking 0.95-1.25x as long as "
                         "the rejected wrong one (else the right answer nearest the median), never the shortest right answer")
    ap.add_argument("--resp-median", type=float, default=0.0,
                    help="--kinds mix, --think-only parts: a 'shorter' pair's chosen RESPONSE is at least this x the question's "
                         "median right response (0.85); 0 = use --resp-guard")
    ap.add_argument("--resp-max", type=float, default=0.0,
                    help="--kinds mix, --think-only parts: a 'shorter' pair's chosen RESPONSE is at most this x the rejected "
                         "one's (1.05): thinking shorter, response not longer; 0 = no cap")
    ap.add_argument("--shorten-second", default="",
                    help="--kinds mix (12 Nov, recipe 2): parts (comma or +) where a question may give a SECOND 'shorter' pair, "
                         "rejecting its second-longest right answer with another chosen answer (same thresholds); IFEval runs short")
    ap.add_argument("--right-resp-cap", type=float, default=0.0,
                    help="--kinds mix, --think-only parts: a be-right pair's chosen RESPONSE is at most this x the rejected "
                         "one's + 15 tokens (1.10); 0 = no cap")
    ap.add_argument("--kinds", default="guard,right", choices=("guard,right", "guard", "guard,shorten", "mix"),
                    help="guard,right: half / half (Track A); guard: only shortest right > shortest finished wrong")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    quota = {k: int(v) for k, v in (x.split("=") for x in args.quota.split(","))}
    groups = defaultdict(list)
    for path in args.rollouts:
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                for k in ("prompt_ids", "output_ids", "thinking_ids"):  # (11 Nov) compact: 4 bytes per token, not ~36
                    if isinstance(r.get(k), list):
                        r[k] = array("i", r[k])
                for k in ("question", "final", "original_prompt"):  # text copies the pair builder never reads
                    r.pop(k, None)
                groups[r["id"]].append(r)
    think_only = set(filter(None, re.split(r"[,+]", args.think_only)))
    decode = None
    if args.prefer_reread:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        decode = lambda r: tok.decode(list(r.get("thinking_ids") or r["output_ids"]), skip_special_tokens=True)  # noqa: E731
    rereads = Counter()
    floors = {}
    for spec in filter(None, re.split(r"[,+]", args.floor_by_part)):  # PART=ABS_TOKENS:RATIO
        part_f, val = spec.split("=")
        a_, r_ = val.split(":")
        floors[part_f] = (float(a_), float(r_))
    base_think = json.loads(Path(args.base_think).read_text()) if args.base_think else {}
    floor_drop = Counter()
    cand = defaultdict(lambda: {"guard": [], "right_long": [], "right_short": [], "shorten": [], "shorten_soft": [],
                                "right_cr": [], "capguard": [], "right_count": [], "loop": []})  # mix: right_cr = counting-rule OR compact re-read pairs
    loop_resp_only = set(filter(None, re.split(r"[,+]", args.loop_response_only)))
    guard_wide = set(filter(None, re.split(r"[,+]", args.guard_wide)))
    right_hold = set(filter(None, re.split(r"[,+]", args.right_hold)))
    shorten_second = set(filter(None, re.split(r"[,+]", args.shorten_second)))
    for qid, q in groups.items():
        part = part_of(q[0])
        # (11 Nov, rpM) with a quota for "svamp_hard", math questions are two parts: "svamp" = the SVAMP-like questions
        # answered for this run (their ids carry --easy-math-prefix), "svamp_hard" = every other math question (the
        # full-difficulty GSM8K of earlier draws): math pairs can be mostly SVAMP-like with a set share of hard questions
        if part == "svamp" and "svamp_hard" in quota and not str(qid).startswith(args.easy_math_prefix):
            part = "svamp_hard"
        if part not in quota:
            continue
        base_part = "svamp" if part == "svamp_hard" else part
        size = thinking_len if part in think_only else (lambda r: r["output_tokens"])
        rights = sorted((r for r in q if right(r)), key=size)
        wrongs = sorted((r for r in q if finished(r) and not right(r)), key=size)
        all_rights = rights
        if floors.get(base_part):  # (13 Nov, consult18) a chosen answer never thinks below the type's floor for this question
            abs_min, ratio_min = floors[base_part]
            bt = base_think.get(str(qid))
            if bt is None:
                bt = sorted(thinking_len(r) for r in rights)[len(rights) // 2] if rights else 0
            ftok = max(abs_min, ratio_min * bt)
            rights = [r for r in all_rights if thinking_len(r) >= ftok]
            floor_drop[part] += len(all_rights) - len(rights)
        if args.kinds == "mix":  # (11 Nov) path 1 rounds and the typed repair: shorten / compact be-right / capped guard
            is_cr = part == "ifeval" and (at_least(q[0]) if args.count_mode == "atleast" else
                                          length_rule(q[0]) if args.count_mode == "length" else counting(q[0]))
            no_shorten = is_cr or (args.no_shorten_counting and part == "ifeval" and counting(q[0]))
            # (11 Nov, rpL) an answer that RUNS AWAY (loops to the token cap, or never closes its thinking) loses to a
            # finished right answer of the same question. Before, an unfinished answer was never used at all, so no pair
            # said "do not loop" (runaways hold 26% of rpE-80's IFEval tokens on the full test)
            runaway = [r for r in q if not finished(r)]
            if base_part in loop_resp_only:  # (rpM) only loops in the RESPONSE (the thinking closed, then cut at the cap)
                runaway = [r for r in runaway if r["closed_thinking"]]
            if rights and runaway:
                cand[part]["loop"].append((qid, rights[0], runaway[0]))
            if (len(all_rights) >= 2 and len(all_rights) / len(q) >= args.shorten_min_right and not no_shorten and rights
                    and rights[0] is not all_rights[-1]):
                long, short = all_rights[-1], rights[0]
                if part in think_only and (args.resp_median > 0 or args.resp_max > 0):
                    # (11 Nov, rpL) thinking shorter, response neither clipped nor longer: at least --resp-median x the
                    # question's median right response, at most --resp-max x the rejected answer's response
                    rmed = sorted(response_len(r) for r in rights)[len(rights) // 2]
                    ok = [r for r in rights if r is not long and response_len(r) >= args.resp_median * rmed
                          and (args.resp_max <= 0 or response_len(r) <= args.resp_max * response_len(long))
                          and response_len(r) >= args.resp_near_long * response_len(long)]
                    short = ok[0] if ok else long
                elif part in think_only and args.resp_guard > 0:  # (11 Nov) never prefer a notably shorter RESPONSE
                    short = next(r for r in rights if response_len(r) >= args.resp_guard * response_len(long))
                if short is not long:
                    spread = size(long) / max(1, size(short))
                    if spread >= args.shorten_ratio:
                        cand[part]["shorten"].append((qid, short, long))
                    elif spread >= 1.20:  # (11 Nov, consult6 2nd review) backup supply if clear shorten pairs run short
                        cand[part]["shorten_soft"].append((qid, short, long))
                if base_part in shorten_second and len(rights) >= 3 and short is not long:
                    # (12 Nov) a second pair on two OTHER answers: rejected = the second-longest right answer, chosen = the
                    # first other answer passing the same response rules; the same ratio thresholds apply
                    long2 = rights[-2]
                    if part in think_only and (args.resp_median > 0 or args.resp_max > 0):
                        rmed = sorted(response_len(r) for r in rights)[len(rights) // 2]
                        ok2 = [r for r in rights[:-2] if r is not short and response_len(r) >= args.resp_median * rmed
                               and (args.resp_max <= 0 or response_len(r) <= args.resp_max * response_len(long2))
                               and response_len(r) >= args.resp_near_long * response_len(long2)]
                    else:
                        ok2 = [r for r in rights[:-2] if r is not short]
                    if ok2:
                        spread2 = size(long2) / max(1, size(ok2[0]))
                        if spread2 >= args.shorten_ratio:
                            cand[part]["shorten"].append((qid, ok2[0], long2))
                        elif spread2 >= 1.20:
                            cand[part]["shorten_soft"].append((qid, ok2[0], long2))
            if rights and wrongs:
                if is_cr:  # a counting rule: the right answer writes at least as much, thinks at most 1.2x
                    taken = set()  # (11 Nov) up to 2 pairs per question, on different answers: counting-rule pairs are the
                    for rej in wrongs[:2]:  # scarce kind (244 for a half-quota of 400 on the original's real answers)
                        need = args.count_resp_min * response_len(rej) + 15 if args.count_resp_min > 0 else response_len(rej)
                        fit = [r for r in rights if id(r) not in taken and response_len(r) >= need
                               and thinking_len(r) <= 1.2 * thinking_len(rej)
                               and (args.count_resp_cap <= 0
                                    or response_len(r) <= args.count_resp_cap * response_len(rej) + args.count_resp_pad)]
                        if fit:
                            pick = min(fit, key=thinking_len)
                            taken.add(id(pick))
                            cand[part]["right_count"].append((qid, pick, rej))
                else:  # (11 Nov, both reviews) rpE's proven compact "be right" pair: the shortest right answer that re-checks
                    # the question (within 1.5x the shortest right's thinking), else the shortest right, over the shortest
                    # finished wrong; replaces the equal-length pairs (D6 / runa2: equal-length right>wrong halves the saving)
                    pool_r = rights
                    tmed = sorted(thinking_len(r) for r in rights)[len(rights) // 2]  # the question's median right thinking
                    if part in think_only and args.right_resp_cap > 0:  # (11 Nov, rpL) be right without writing more
                        pool_r = [r for r in rights if response_len(r) <= args.right_resp_cap * response_len(wrongs[0]) + 15]
                    if base_part in right_hold:
                        # (rpM) hold the length: the right / wrong pair of this question whose thinking lengths are closest,
                        # kept only if the right answer thinks 0.95-1.25x as long as the wrong one (a pair about being
                        # right, with no "be shorter" signal: the shortest right answer over a long wrong one taught math to
                        # skip its re-check); among equally close pairs, a right answer that re-reads the question
                        best = None
                        for r in rights:
                            for w in wrongs:
                                ratio = thinking_len(r) / max(1, thinking_len(w))
                                if 0.95 <= ratio <= 1.25:
                                    key = (round(abs(ratio - 1.05), 2), 0 if decode and REREAD.search(decode(r)) else 1)
                                    if best is None or key < best[0]:
                                        best = (key, r, w)
                        if best:
                            cand[part]["right_cr"].append((qid, best[1], best[2]))
                    elif pool_r:
                        checked = [r for r in pool_r if decode and thinking_len(r) <= 1.5 * thinking_len(pool_r[0]) and REREAD.search(decode(r))]
                        cand[part]["right_cr"].append((qid, (checked or pool_r)[0], wrongs[0]))
                    # the capped "don't stop too early" guard (non-counting questions only: counting-rule questions belong
                    # to IFEval's counting half; consult8)
                    if base_part in guard_wide:  # (rpM) the 0.5-1.0x band: a calculation that skipped its re-check
                        capped = sorted((r for r in rights if 0.95 * tmed <= thinking_len(r) <= 1.25 * tmed), key=thinking_len)
                        # the rejected answer: the wrong one nearest 0.75x the median right thinking (a clear, bounded gap),
                        # listed last so that the line below takes it
                        early = sorted((w for w in wrongs if thinking_len(w) < args.guard_wide_max * tmed),
                                       key=lambda w: -abs(thinking_len(w) - 0.75 * tmed))
                    else:
                        capped = [r for r in rights if thinking_len(r) <= 1.15 * tmed]
                        early = sorted((w for w in wrongs if thinking_len(w) < 0.5 * tmed), key=thinking_len)
                    if capped and early:
                        pick = ([r for r in capped if decode and REREAD.search(decode(r))] or capped)[0]
                        cand[part]["capguard"].append((qid, pick, early[-1]))  # the LONGEST early stop: the smallest length gap
            continue
        if (args.kinds == "guard,shorten" and len(rights) >= 2 and len(rights) / len(q) >= args.shorten_min_right
                and not (part == "ifeval" and counting(q[0]))):  # (10 Nov, odpo) a counting rule needs its full response
            spread = size(rights[-1]) / max(1, size(rights[0]))
            if spread >= args.shorten_ratio:
                cand[part]["shorten"].append((qid, rights[0], rights[-1]))
            elif spread >= 1.25:  # backup supply: used only if the clear (>= 1.5x) pairs run short
                cand[part]["shorten_soft"].append((qid, rights[0], rights[-1]))
        if not rights or not wrongs:
            continue
        chosen = rights[0]
        if args.count_rules and part == "ifeval" and counting(q[0]):
            rej = wrongs[0]
            fit = [r for r in rights if response_len(r) >= response_len(rej) and thinking_len(r) <= 1.2 * thinking_len(rej)]
            rereads[("ifeval", "counting-rule questions")] += 1
            if not fit:
                rereads[("ifeval", "counting-rule questions without a fitting right answer (no pair)")] += 1
                continue
            chosen = min(fit, key=thinking_len)
            rereads[("ifeval", "counting-rule pairs")] += 1
            cand[part]["guard"].append((qid, chosen, rej))
            continue
        if decode:
            checked = [r for r in rights if thinking_len(r) <= 1.5 * thinking_len(rights[0]) and REREAD.search(decode(r))]
            chosen = checked[0] if checked else rights[0]
            rereads[(part, "chosen re-checks")] += bool(checked)
            rereads[(part, "rejected re-checks")] += bool(REREAD.search(decode(wrongs[0])))
            rereads[(part, "guard questions")] += 1
        cand[part]["guard"].append((qid, chosen, wrongs[0]))
        close = [(r, w) for r in rights for w in wrongs
                 if max(r["output_tokens"], w["output_tokens"]) <= args.ratio * min(r["output_tokens"], w["output_tokens"])]
        rng.shuffle(close)
        for r, w in close:
            cand[part]["right_long" if r["output_tokens"] > w["output_tokens"] else "right_short"].append((qid, r, w))
    pairs, counts = [], Counter()
    for part_f, n_f in floor_drop.items():
        counts[(part_f, "right answers below the type floor (never chosen)")] = n_f
    shortfall = {}  # (11 Nov) part -> pairs missing to reach the quota at the fixed mix (written to shortfall.json)
    per_q, right_q, used = Counter(), Counter(), set()

    def take(part, kind, pool, n):
        rng.shuffle(pool)
        got = []
        for qid, c, w in pool:
            if len(got) >= n:
                break
            key = (id(c), id(w))
            if key in used or per_q[qid] >= 3 or (kind != "guard" and right_q[qid] >= 2):
                continue
            used.add(key)
            per_q[qid] += 1
            right_q[qid] += kind != "guard"
            got.append({"id": qid, "source": c["source"], "part": part, "kind": kind, "prompt_ids": list(c["prompt_ids"]),
                        "chosen_ids": list(c["output_ids"]), "rejected_ids": list(w["output_ids"]),
                        "chosen_tokens": c["output_tokens"], "rejected_tokens": w["output_tokens"]})
        return got
    by_part = {}
    for part, n in quota.items():
        if args.kinds == "mix":
            base_part = "svamp" if part == "svamp_hard" else part
            sh_share = {k: float(v) for k, v in (x.split("=") for x in args.mix_shares.split(","))}
            if args.mix_preset == "round1":
                # (11 Nov, both reviews) shorten-dominant round 1: it must reach ~53-57% saved for round 2 to repair
                sh_share = ({"shorten": 0.60, "right": 0.40, "guard": 0.0} if part in ("svamp", "bfcl_irrelevance") else
                            {"shorten": 0.50, "right": 0.50, "guard": 0.0} if part == "ifeval" else
                            {"shorten": 0.70, "right": 0.30, "guard": 0.0})
            if args.mix_preset == "typed":
                # (11 Nov, the IFEval + tool-call repair) "think less" only where the full-test records show thinking is
                # worth little (IFEval without a counting rule, tool calls); math and code get NO "think less" pair, only
                # "be right" and "don't stop early" pairs (they hold the rest in place)
                sh_share = ({"shorten": 0.0, "right": 0.80, "guard": 0.20} if part in ("svamp", "mbpp") else
                            {"shorten": 0.60, "right": 0.40, "guard": 0.0} if part == "bfcl_irrelevance" else
                            {"shorten": 0.70, "right": 0.30, "guard": 0.0})
            count_share = args.count_share
            if args.mix_preset == "typed2":
                # (11 Nov, rpL = rpK + consult9's IFEval fixes + the anti-loop pair) tool calls as in "typed"; math and code
                # hold as in "typed", with a tenth of anti-loop pairs; IFEval: 20% counting-rule pairs (true "at least N" rules,
                # --count-mode atleast), 50% "shorter thinking", 15% be-right, 15% anti-loop
                sh_share = ({"shorten": 0.0, "right": 0.70, "guard": 0.20, "loop": 0.10} if part in ("svamp", "mbpp") else
                            {"shorten": 0.625, "right": 0.1875, "guard": 0.0, "loop": 0.1875} if part == "ifeval" else
                            {"shorten": 0.60, "right": 0.40, "guard": 0.0} if part == "bfcl_irrelevance" else
                            {"shorten": 0.70, "right": 0.30, "guard": 0.0})
                count_share = 0.20
            if args.mix_preset == "round1b":
                # (11 Nov, Path 1 redesigned) round 1 as launched (60 / 40 on math and no fit, 70 / 30 on code and tool calls)
                # with two changes: anti-loop pairs take 15 points of the be-right side on math and code (the original's
                # runaways hold 30% of its code tokens and 10% of its math tokens), and IFEval uses rpL's rules (20% counting
                # pairs on true "at least N" rules, 50% "shorter thinking", 15% be-right, 15% anti-loop)
                sh_share = ({"shorten": 0.60, "right": 0.25, "guard": 0.0, "loop": 0.15} if part == "svamp" else
                            {"shorten": 0.70, "right": 0.15, "guard": 0.0, "loop": 0.15} if part == "mbpp" else
                            {"shorten": 0.625, "right": 0.1875, "guard": 0.0, "loop": 0.1875} if part == "ifeval" else
                            {"shorten": 0.60, "right": 0.40, "guard": 0.0} if part == "bfcl_irrelevance" else
                            {"shorten": 0.70, "right": 0.30, "guard": 0.0})
                count_share = 0.20
            if args.mix_preset == "typed3":
                # (11 Nov, rpM, consult10 after rpK-20's full test and rpL) math: NO "shorter", NO anti-loop, half widened guards
                # and half length-held be-right; code as rpL; IFEval: 25% counting pairs (true length rules), 45% "shorter"
                # (questions without a counting rule, response held), 22.5% be-right, 7.5% response loops; several functions 50 / 50
                sh_share = ({"shorten": 0.0, "right": 0.50, "guard": 0.50, "loop": 0.0} if base_part == "svamp" else
                            {"shorten": 0.0, "right": 0.7125, "guard": 0.20, "loop": 0.0875} if part == "mbpp" else
                            {"shorten": 0.60, "right": 0.30, "guard": 0.0, "loop": 0.10} if part == "ifeval" else
                            {"shorten": 0.50, "right": 0.50, "guard": 0.0} if part == "bfcl_multiple" else
                            {"shorten": 0.60, "right": 0.40, "guard": 0.0} if part == "bfcl_irrelevance" else
                            {"shorten": 0.70, "right": 0.30, "guard": 0.0})
                count_share = 0.25
            if args.mix_preset == "round1c":
                # (11 Nov, Path 1's round 1 redesigned, consult10's edits to round1b) math 45% shorter / 35% be-right / 15% widened
                # guards / 5% anti-loop (the first round 1 over-cut math); code 70 / 13 / 10 / 7; IFEval as rpM with 20% counting
                # pairs; several functions 50 / 50; anti-loop pairs 144 instead of 360
                sh_share = ({"shorten": 0.45, "right": 0.35, "guard": 0.15, "loop": 0.05} if base_part == "svamp" else
                            {"shorten": 0.70, "right": 0.13, "guard": 0.10, "loop": 0.07} if part == "mbpp" else
                            {"shorten": 0.55, "right": 0.375, "guard": 0.0, "loop": 0.075} if part == "ifeval" else
                            {"shorten": 0.50, "right": 0.50, "guard": 0.0} if part == "bfcl_multiple" else
                            {"shorten": 0.60, "right": 0.40, "guard": 0.0} if part == "bfcl_irrelevance" else
                            {"shorten": 0.70, "right": 0.30, "guard": 0.0})
                count_share = 0.20
            if args.mix_preset == "od3_arena":
                # (13 Nov, consult18 od3) the online arena's rounds: mostly "shorter", floored by type (--floor-by-part), no
                # guards (shortening and protection are not mixed in one step); math 75 / 20 / 5 loops, code 75 / 17.5 / 7.5,
                # IFEval 20% counting pairs + 75 / 18.75 / 6.25, several 70 / 30, one function and no fit 75 / 25
                sh_share = ({"shorten": 0.75, "right": 0.20, "guard": 0.0, "loop": 0.05} if base_part == "svamp" else
                            {"shorten": 0.75, "right": 0.175, "guard": 0.0, "loop": 0.075} if part == "mbpp" else
                            {"shorten": 0.75, "right": 0.1875, "guard": 0.0, "loop": 0.0625} if part == "ifeval" else
                            {"shorten": 0.70, "right": 0.30, "guard": 0.0} if part == "bfcl_multiple" else
                            {"shorten": 0.75, "right": 0.25, "guard": 0.0})
                count_share = 0.20
            if args.mix_preset == "r2s2":
                # (13 Nov, recipe 2 step 2 as consult16 planned it) shorten again on the step-1 model's own answers, every type:
                # math 40 shorter / 30 be-right / 25 widened guards / 5 loops; code 45 / 35 / 15 / 5; IFEval 20% counting pairs,
                # the rest 56.25 shorter / 31.25 be-right / 12.5 response loops; one function 60 / 40, several 50 / 50, no fit 60 / 40
                sh_share = ({"shorten": 0.40, "right": 0.30, "guard": 0.25, "loop": 0.05} if base_part == "svamp" else
                            {"shorten": 0.45, "right": 0.35, "guard": 0.15, "loop": 0.05} if part == "mbpp" else
                            {"shorten": 0.5625, "right": 0.3125, "guard": 0.0, "loop": 0.125} if part == "ifeval" else
                            {"shorten": 0.50, "right": 0.50, "guard": 0.0} if part == "bfcl_multiple" else
                            {"shorten": 0.60, "right": 0.40, "guard": 0.0} if part == "bfcl_irrelevance" else
                            {"shorten": 0.60, "right": 0.40, "guard": 0.0})
                count_share = 0.20
            if args.mix_preset == "r2s1":
                # (12 Nov, recipe 2 step 1, consult16) moderate shortening from the original (target ~47% saved, like rpE-80):
                # math 40 shorter / 40 be-right / 15 widened guards / 5 loops; code 60 / 20 / 12.5 / 7.5; IFEval 20% counting
                # pairs (true length rules), the rest 50 shorter / 41.7 be-right / 8.3 response loops; one function 70 / 30,
                # several 50 / 50, no fit 60 / 40. Thinking-only shortness with the response held (--think-only all parts,
                # --resp-median 0.95 --resp-max 1.05): Path 1's --resp-guard 0.9 made IFEval responses longer (263 -> 345)
                # (12 Nov ~18:30, consult17 change A, user: yes) NO widened guards in step 1: from the original they reward
                # ~1,000-token right answers and cancel the shortening (p1cr1s0 stalled at 34% Dev, p1r1s0 without them 47.5%);
                # math 55 shorter / 40 be-right / 5 loops, code 70 / 22.5 / 7.5 (guards come back in steps 2-3, as in rpM)
                sh_share = ({"shorten": 0.55, "right": 0.40, "guard": 0.0, "loop": 0.05} if base_part == "svamp" else
                            {"shorten": 0.70, "right": 0.225, "guard": 0.0, "loop": 0.075} if part == "mbpp" else
                            {"shorten": 0.50, "right": 0.4167, "guard": 0.0, "loop": 0.0833} if part == "ifeval" else
                            {"shorten": 0.50, "right": 0.50, "guard": 0.0} if part == "bfcl_multiple" else
                            {"shorten": 0.60, "right": 0.40, "guard": 0.0} if part == "bfcl_irrelevance" else
                            {"shorten": 0.70, "right": 0.30, "guard": 0.0})
                count_share = 0.20
            if args.mix_preset == "hold":
                # (11 Nov, rpN, consult12) an ACCURACY-ONLY repair of an already shortened model: no "think less" pair on any
                # type; math: widened guards + length-matched be-right; code: be-right / guards / loops; IFEval: 40% length-rule
                # counting pairs, the rest be-right with 15% response loops; tool calls: be-right only
                sh_share = ({"shorten": 0.0, "right": 0.50, "guard": 0.50, "loop": 0.0} if base_part == "svamp" else
                            {"shorten": 0.0, "right": 0.7125, "guard": 0.20, "loop": 0.0875} if part == "mbpp" else
                            {"shorten": 0.0, "right": 0.85, "guard": 0.0, "loop": 0.15} if part == "ifeval" else
                            {"shorten": 0.0, "right": 1.0, "guard": 0.0})
                count_share = 0.40
            sh_share.setdefault("loop", 0.0)
            # (11 Nov, user) HARD RULE: the PROPORTION is fixed, the total is not. Pairs are taken, then every side is cut
            # back to the scarcest one, so the final file holds exactly the set mix whatever ran short (a short supply, a
            # question's pair limit, or the same pair proposed by two kinds); nothing is ever refilled by another kind.
            # IFEval: counting-rule questions never get "shorter" pairs; their be-right pairs hold a FIXED half of the quota
            # and the mix below holds the other half, 1 : 1 (before: a short counting half let the other half grow)
            want_cnt = int(round(count_share * n)) if cand[part]["right_count"] else 0
            cnt = take(part, "right", cand[part]["right_count"], want_cnt)
            n_full, n_mix = n, n - want_cnt
            s_sh, s_br = sh_share["shorten"], sh_share["right"] + sh_share["guard"] + sh_share["loop"]
            n_sh = int(round(s_sh * n_mix))
            sh = take(part, "shorten", cand[part]["shorten"], n_sh)
            sh += take(part, "shorten", cand[part]["shorten_soft"], n_sh - len(sh))
            n_br = n_mix - n_sh
            # (no call at all without a loop share: take() shuffles its pool, and an extra shuffle would change which pairs
            # the older presets pick: rpK's and round 1's files must stay reproducible)
            lp = take(part, "loop", cand[part]["loop"], min(n_br, int(round(sh_share["loop"] * n_mix)))) if sh_share["loop"] > 0 else []
            g = take(part, "guard", [c for c in cand[part]["capguard"] if c[0] not in {p["id"] for p in lp}],
                     min(n_br - len(lp), int(round(sh_share["guard"] * n_mix))))
            guarded = {p["id"] for p in g + lp}  # one be-right pair per question: a guarded / looping question gives no second one
            r = take(part, "right", [c for c in cand[part]["right_cr"] if c[0] not in guarded], n_br - len(g) - len(lp))
            g = g + lp  # from here the anti-loop pairs travel with the guards (both are the be-right side's rare kinds)
            total = float(n_mix)
            if s_sh > 0:
                total = min(total, len(sh) / s_sh)
            if s_br > 0:
                total = min(total, (len(g) + len(r)) / s_br)
            if want_cnt:
                total = min(total, len(cnt) * n_mix / want_cnt)
            keep_sh = min(len(sh), int(round(s_sh * total)))
            keep_br = min(len(g) + len(r), int(round(s_br * total)))
            sh = sh[:keep_sh]
            r = r[:max(0, keep_br - len(g))]
            g = g[:keep_br - len(r)]
            if want_cnt:
                cnt = cnt[:int(round(total * want_cnt / n_mix))]
            by_part[part] = sh + g + r + cnt
            if len(by_part[part]) < n_full:
                shortfall[part] = n_full - len(by_part[part])  # the REAL gap (what fill-until-full reads)
                counts[(part, "MIX KEPT: total cut to")] = f"{len(by_part[part])} of {n_full} (the scarcest kind limits it)"
            for kind, got in (("shorten", sh), ("guard (capped)", [x for x in g if x["kind"] == "guard"]),
                              ("anti-loop (finished right > runaway)", [x for x in g if x["kind"] == "loop"]),
                              ("be right (compact)", r), ("be right (counting rule)", cnt)):
                counts[(part, kind)] += len(got)
            counts[(part, "candidates anti-loop")] = len(cand[part]["loop"])
            counts[(part, "candidates shorten (clear + backup) / capped guard / be right / counting rule")] = (
                f"{len(cand[part]['shorten'])} + {len(cand[part]['shorten_soft'])} / {len(cand[part]['capguard'])} / "
                f"{len(cand[part]['right_cr'])} / {len(cand[part]['right_count'])}")
            counts[(part, "SHORT of quota")] = max(0, n_full - len(by_part[part]))
            continue
        if args.kinds == "guard,shorten":  # (10 Nov, odpo) half "be right", half "be shorter when sure"; no backfill
            g = take(part, "guard", cand[part]["guard"], n // 2)
            sh = take(part, "shorten", cand[part]["shorten"], n - n // 2)
            soft = take(part, "shorten", cand[part]["shorten_soft"], n - n // 2 - len(sh))
            counts[(part, "shorten from the 1.25-1.5x backup")] += len(soft)
            counts[(part, "shorten candidates (>= 1.5x / 1.25-1.5x)")] = f"{len(cand[part]['shorten'])} / {len(cand[part]['shorten_soft'])}"
            sh += soft
            by_part[part] = g + sh
            counts[(part, "guard")] += len(g)
            counts[(part, "shorten")] += len(sh)
            counts[(part, "SHORT of quota")] = max(0, n - len(by_part[part]))
            continue
        if args.kinds == "guard":
            by_part[part] = take(part, "guard", cand[part]["guard"], n)
            for p in by_part[part]:
                counts[(part, "guard")] += 1
            counts[(part, "SHORT of quota")] = max(0, n - len(by_part[part]))
            continue
        g = take(part, "guard", cand[part]["guard"], n // 2)
        r_long = take(part, "right", cand[part]["right_long"], (n - n // 2) // 2)
        r_short = take(part, "right", cand[part]["right_short"], n - n // 2 - len(r_long))
        if len(g) < n // 2:  # a short guard supply is filled with right pairs, never with another kind
            r_short += take(part, "right", cand[part]["right_short"], n // 2 - len(g))
        by_part[part] = g + r_long + r_short
        for p in by_part[part]:
            counts[(part, p["kind"] + ("" if p["kind"] == "guard" else
                                       " (right longer)" if p["chosen_tokens"] > p["rejected_tokens"] else " (right shorter)"))] += 1
        counts[(part, "SHORT of quota")] = max(0, n - len(by_part[part]))
    # stratified order: interleave the parts by their shares, kinds alternating inside each part
    # (11 Nov, review fix) kinds interleaved by their shares (credit scheme), shuffled inside each kind: any prefix of the
    # file holds every kind in proportion (before: kinds sorted by name, so with 3 kinds the "shorten" pairs all came last)
    queues = {}
    for part, ps in by_part.items():
        kinds = {}
        for pr in sorted(ps, key=lambda p: rng.random()):
            kinds.setdefault(pr["kind"], []).append(pr)
        tot = sum(len(v) for v in kinds.values())
        kshare = {k: len(v) / max(1, tot) for k, v in kinds.items()}
        kcredit, order = Counter(), []
        while any(kinds.values()):
            for k, v in kinds.items():
                kcredit[k] += kshare[k]
                while kcredit[k] >= 1 and v:
                    order.append(v.pop(0))
                    kcredit[k] -= 1
        queues[part] = order
    total = sum(len(q) for q in queues.values())
    share = {part: len(q) / max(1, total) for part, q in queues.items()}
    credit = Counter()
    while any(queues.values()):  # each round adds every part's share; a part emits a pair per whole unit of credit
        for part, q in queues.items():
            credit[part] += share[part]
            while credit[part] >= 1 and q:
                pairs.append(q.pop(0))
                credit[part] -= 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pairs.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pairs))
    (out / "shortfall.json").write_text(json.dumps(shortfall))
    lines = [f"{len(pairs)} pairs ({len(per_q)} questions)"]
    for (part, what), n in sorted(counts.items()):
        lines.append(f"  {part} / {what}: {n}")
    for part in quota:
        ps = by_part[part]
        if ps:
            med = lambda xs: sorted(xs)[len(xs) // 2]  # noqa: E731
            lines.append(f"  {part}: median chosen / rejected tokens {med([p['chosen_tokens'] for p in ps])} / "
                         f"{med([p['rejected_tokens'] for p in ps])}; chosen longer in "
                         f"{sum(p['chosen_tokens'] > p['rejected_tokens'] for p in ps) / len(ps):.0%}")
    for (part, what), n in sorted(rereads.items()):
        if what.startswith("counting-rule"):
            lines.append(f"  {part}: {what}: {n}")
    ifeval_q = [q for q in groups.values() if part_of(q[0]) == "ifeval"]
    if ifeval_q:
        lines.append(f"  ifeval questions answered: {len(ifeval_q)}, with a counting rule: {sum(counting(q[0]) for q in ifeval_q)}")
    for part in quota:
        if rereads[(part, "guard questions")]:
            n = rereads[(part, "guard questions")]
            lines.append(f"  {part}: re-check of the question in the chosen answer {rereads[(part, 'chosen re-checks')] / n:.0%}, "
                         f"in the rejected {rereads[(part, 'rejected re-checks')] / n:.0%} ({n} guard questions)")
    first = Counter(p["part"] for p in pairs[:args.block])
    lines.append(f"  first {args.block} pairs by part: {dict(first)}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
