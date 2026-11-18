"""Build Argon consult31's pack (user 18 Nov: "> 60% at better than -1"; run once od5f rounds 19-21 have full tests).
CPU only, light. Strips competitor and BottleCap-method lines from everything copied (user rules).

python reviews/consult31_over60/build_pack.py --canvas <Main.dc.html> [--effprobe <dir with effprobe summaries>]
"""
import argparse, json, re, shutil
from pathlib import Path

R = Path(__file__).resolve().parent; ROOT = R.parents[1]
BAN = re.compile(r"bida|michal|herz1c|matousek|senkyr|mcqwen|lightning|khudgins|competitor|other team|bottlecap'?s? (own )?(method|recipe)|bottlecap method", re.I)

def clean(text):
    return "\n".join(l for l in text.splitlines() if not BAN.search(l)) + "\n"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--canvas", required=True); ap.add_argument("--effprobe", default="")
    ap.add_argument("--article", default=str(ROOT.parents[1] / "Personal/arsenelemoigne.eu/src/pages/posts/thinkingcap.astro"))
    a = ap.parse_args()
    P = R / "pack"; shutil.rmtree(P, ignore_errors=True); (P / "code").mkdir(parents=True)
    log = (ROOT / "EXPERIMENT_LOG.md").read_text()
    (P / "EXPERIMENT_LOG.md").write_text(clean(log))
    L = log.splitlines()
    (P / "NEW_RESULTS_od5f.md").write_text("# od5f: every log line since its launch (monitors, Dev, full tests; rounds 19-21 at the end)\n\n" +
        clean("\n".join(l for l in L if "od5f" in l)))
    (P / "THINKING_OFF.md").write_text("# Thinking switched off: the original and trained models, per type (log lines)\n\n" +
        clean("\n".join(l for l in L if re.search(r"nothink|thinking[ -]off|thinking OFF|ntothers", l, re.I))))
    for src, dst in (("reviews/consult28_online60/REPORT.md", "YOUR_CONSULT28_REPORT.md"), ("reviews/consult30_efficiency/REPORT.md", "YOUR_CONSULT30_REPORT.md"),
                     ("analysis/SMARTER_DATA.md", "SMARTER_DATA.md"), ("reviews/pertype_best_full.md", "pertype_best_full.md"),
                     ("reviews/consult28_online60/pack/od5c/REPRODUCE_od5c.md", "REPRODUCE_od5c.md")):
        if (ROOT / src).exists(): (P / dst).write_text(clean((ROOT / src).read_text()))
    for f in ("repair_pairs.py", "track.py", "dpo.py", "dev_score.py", "probe_curve.py", "budget_curves.py", "eff_allocate.py"):
        if (ROOT / f).exists(): (P / "code" / f).write_text(clean((ROOT / f).read_text()))
    s = Path(a.canvas).read_text(); i = s.index("const DATA = ") + 13; D, _ = json.JSONDecoder().raw_decode(s[i:])
    full = [{k: m[k] for k in ("key", "name", "what", "saved", "acc", "parts") if k in m} for m in D["models"] if m["kind"] == "full" and not BAN.search(m["name"] + m.get("what", ""))]
    json.dump({"note": "every full test (2,759 x 8). parts: v math, c code, f IFEval, s one function, m several functions, i no function fits; saved %, acc in points vs the original",
               "original_thinking_off": D["off"], "models": full}, open(P / "fulltests.json", "w"), indent=1)
    art = Path(a.article).read_text()
    if art.startswith("---"): art = art.split("---", 2)[2]
    art = re.sub(r"<script.*?</script>|<style.*?</style>", "", art, flags=re.S)
    art = re.sub(r"<(h[1-4]|p|li|tr|div)[^>]*>", "\n", art); art = re.sub(r"<[^>]+>", "", art)
    art = re.sub(r"\n\s*\n+", "\n\n", art).strip()
    (P / "ARTICLE_TEXT.md").write_text(clean(art))
    if a.effprobe and Path(a.effprobe).exists(): shutil.copytree(a.effprobe, P / "effprobe")
    (P / "README_PACK.md").write_text(f"""# Pack for consult31: > 60% saved at better than -1 (built from the lab notebook as of its last line)

- NEW_RESULTS_od5f.md: od5f (your consult28 plan) line by line: monitors per round (slice = fresh test-matched questions; the monitor reads
  ~2.5 pts optimistic on accuracy), Dev-512 scores, full tests per type, and the extension to round 21.
- fulltests.json: {len(full)} full tests of our models, per type, plus the original with thinking off.
- THINKING_OFF.md: thinking-off full tests (original; od5c-best +2.1, od5r-r16 +1.9, rpM-60 -0.3, rpL-80 -3.7 vs the original off).
- YOUR_CONSULT28_REPORT.md (od5f's design), YOUR_CONSULT30_REPORT.md (efficiency, KKT redistribution estimate).
- SMARTER_DATA.md: question-level movement (net drift, excess churn), Pass@8 and 8/8, trimming vs condensing, post-answer thinking,
  marginal token value, for od5c-r16 (54.8 / -0.78).
- ARTICLE_TEXT.md: the user's article as published (charts removed), for context.
- pertype_best_full.md, REPRODUCE_od5c.md, code/ (as run), EXPERIMENT_LOG.md (the whole notebook).
- effprobe/ (if present): forced early-stop curves and budget allocation on unseen slices 19-20, original and od5c-best.
""")
    print("pack:", sorted(p.name for p in P.iterdir()))

if __name__ == "__main__":
    main()
