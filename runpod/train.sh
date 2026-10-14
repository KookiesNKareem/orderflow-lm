#!/usr/bin/env bash
# Train the 87M model on a GPU pod (H100 80GB recommended). Resumable: re-run
# the same command after a pod restart and it continues from <out>.last.pt.
#
# Architecture and optimizer match the paper: 12 layers, width 768, 12 heads,
# 2,048-token context, Muon at 2e-2, 65k tokens/step (32 x 2,048). The paper's
# selected checkpoint was step 22,000 of a run whose total length was not
# recorded; STEPS sets the cosine schedule and validation picks the best step.
#
#   SEED=1337 bash train.sh
#   SEED=2027 bash train.sh          # second seed, for seed-to-seed variation
#   HF_REPO=<user>/oflm bash train.sh   # also upload the best checkpoint (needs HF_TOKEN)
set -euo pipefail

SEED=${SEED:-1337}
STEPS=${STEPS:-25000}
OUT=/workspace/checkpoints/oflm87m_seed${SEED}.pt

export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR=/workspace/.uv-cache
cd /workspace/orderflow-lm
mkdir -p logs

uv run python scripts/train.py \
  --train data/tokens/all/train.bin --val data/tokens/all/val.bin \
  --steps "$STEPS" --batch 32 --block 2048 \
  --n-layer 12 --n-head 12 --d 768 \
  --optimizer muon --lr 2e-2 --bf16 \
  --eval-every 250 --ckpt-every 500 \
  --seed "$SEED" --out "$OUT" --resume 2>&1 | tee -a "logs/train_seed${SEED}.log"

sha256sum "$OUT" | tee "$OUT.sha256"
if [ -n "${HF_REPO:-}" ]; then
  uvx --from huggingface_hub huggingface-cli upload "$HF_REPO" "$OUT" "$(basename "$OUT")"
  uvx --from huggingface_hub huggingface-cli upload "$HF_REPO" "$OUT.sha256" "$(basename "$OUT").sha256"
fi
