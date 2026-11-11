"""Per-answer tables for IFEval (per-instruction verdicts with the official checkers) and BFCL (reply classes)."""
import json, random, sys, ast, re, pickle
from pathlib import Path
S = Path(sys.argv[1]); D = Path(sys.argv[2])
sys.path.insert(0, str(S / "tct"))
sys.path.insert(0, str(S / "tct/questions/vendor"))
import langdetect
langdetect.DetectorFactory.seed = 0
import nltk
for pk in ("punkt", "punkt_tab"):
    try: nltk.data.find(f"tokenizers/{pk}")
    except LookupError: nltk.download(pk, quiet=True)
from instruction_following_eval import instructions_registry
from utils.eval.graders import decode_calls
Q = {}
for l in open(S / "tct/questions/questions.jsonl"):
    q = json.loads(l); Q[q["id"]] = q

def ifeval_rules(row, reply):
    out = []
    for index, iid in enumerate(row["instruction_id_list"]):
        random.seed(f"{row['id']}:{iid}")
        ins = instructions_registry.INSTRUCTION_DICT[iid](iid)
        args = {k: v for k, v in (row["kwargs"][index] or {}).items() if v is not None}
        ins.build_description(**args)
        wants = (ins.get_instruction_args() or {}) if hasattr(ins, "get_instruction_args") else {}
        if "prompt" in wants:
            ins.build_description(prompt=row["original_prompt"])
        try: ok = bool(reply.strip()) and bool(ins.check_following(reply))
        except Exception: ok = False
        out.append(ok)
    return out

def call_class(reply):
    r = reply.strip()
    if not r: return "empty"
    try:
        calls = decode_calls(r)
    except Exception:
        return "prose"
    return "empty_list" if calls == [] else "call"

for name in sys.argv[3:]:
    rows = []
    bad = 0
    for l in open(D / f"{name}.jsonl"):
        r = json.loads(l)
        q = Q[r["id"]]
        ds = r["dataset"]
        if ds not in ("ifeval", "bfcl_irrelevance", "bfcl_simple_python", "bfcl_multiple"):
            rows.append({"id": r["id"], "ds": ds, "s": r["sample_index"], "ok": bool(r["correct"]) and not r["truncated"],
                         "out": r["output_tokens"], "think": r["reasoning_tokens"] or 0, "trunc": r["truncated"]})
            continue
        row = {"id": r["id"], "ds": ds, "s": r["sample_index"], "ok": bool(r["correct"]) and not r["truncated"],
               "out": r["output_tokens"], "think": r["reasoning_tokens"] or 0, "trunc": r["truncated"],
               "resp": r["response"], "text": r["text"]}
        if ds == "ifeval":
            row["rules"] = ifeval_rules(q, r["response"])
            if all(row["rules"]) != bool(r["correct"]): bad += 1
        else:
            row["cls"] = call_class(r["response"])
        rows.append(row)
    pickle.dump(rows, open(S / f"an/{name}.pkl", "wb"))
    print(name, len(rows), "ifeval regrade mismatches:", bad, flush=True)
