#!/bin/zsh
# Argon consult31 (> 60% at better than -1). Rebuild the pack first (after od5f r19-21 full tests are in the log):
#   python3 reviews/consult32_od5h/build_pack.py --canvas reviews/consult32_od5h/canvas/Main.dc.html [--effprobe <dir>]
# Key from the Keychain only (never in files); memory guard kills any child above 2 GB RSS (8.6 GB Mac).
cd ~/"Claude Build"/Test-mvt-fading/thinkingcap || exit 1
R="$PWD/reviews/consult32_od5h"; rm -rf "$R/run" "$R/work"; mkdir -p "$R/run" "$R/work"
export OPENROUTER_API_KEY="$(security find-generic-password -s openrouter-api-key -w)"
PYTHONIOENCODING=utf-8 LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 nice -n 10 \
  python3 -u astra_harness.py run --name consult32 --task consult-od5h --data "$R/pack" --out "$R/run" --work "$R/work" \
  --model-dir ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --provider openrouter --llm google/gemini-4-argon --cap-usd 3 --code-runs 100 --stop-turn 110 --report-reserve 0.4 > "$R/session.log" 2>&1 &
PID=$!; unset OPENROUTER_API_KEY
while kill -0 $PID 2>/dev/null; do for p in $(pgrep -P $PID) $PID; do r=$(ps -o rss= -p $p 2>/dev/null | tr -d ' ')
  if [ -n "$r" ] && [ $r -gt 2000000 ]; then kill $p; echo "$(date +%T) killed pid $p at $((r/1024)) MB" >> "$R/memguard.log"; fi; done; sleep 3; done
tail -3 "$R/session.log"
