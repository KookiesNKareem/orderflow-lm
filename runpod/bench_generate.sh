#!/usr/bin/env bash
# Generation throughput on the GPU pod, right after training. Runs one batch
# of all five injection arms at each batch size and reports what a full
# 2,048-window symbol-day would cost. Then smoke-tests the week-0 analyses on
# the batch-64 output.
#
#   bash bench_generate.sh                       # seed-1337 checkpoint
#   CKPT=/workspace/checkpoints/oflm87m_seed2027.pt bash bench_generate.sh
set -euo pipefail

CKPT=${CKPT:-/workspace/checkpoints/oflm87m_seed1337.pt}
TAPE=${TAPE:-data/test_days/SOLUSDT_2026-08-01.bin}

export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR=/workspace/.uv-cache
cd /workspace/orderflow-lm

for B in 64 128 256; do
  OUT=/workspace/bench/b$B
  rm -rf "$OUT"
  t0=$(date +%s)
  uv run python scripts/pilot_inject.py --bin "$TAPE" --ckpt "$CKPT" --symbol SOLUSDT \
    --seeds "$B" --batch "$B" --new-tokens 2500 --pcrn --out "$OUT" > "$OUT.log" 2>&1
  dt=$(( $(date +%s) - t0 ))
  echo "batch $B: ${dt}s for 5 arms -> $(( 5 * B * 2500 / dt )) tok/s;" \
       "2,048 windows ~ $(( dt * 2048 / B / 60 )) min per symbol-day"
done

uv run python scripts/eval_context_split.py --pilot /workspace/bench/b64 --symbol SOLUSDT
uv run python scripts/eval_depth_response.py --pilot /workspace/bench/b64 --symbol SOLUSDT
