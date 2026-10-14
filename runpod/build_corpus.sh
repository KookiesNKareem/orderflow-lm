#!/usr/bin/env bash
# Full corpus build on a CPU pod. Resumable: finished normalized days are
# skipped, so after a failure or a quota stop, just run it again.
#
#   Train: 2021-01-01 .. 2026-06-01   Val: 2026-07-01   Test: 2026-08/09/10-01
#
# Outputs (all on the volume, under the repo's data/):
#   normalized/<SYM>/<date>.parquet    reconstructed event tables
#   tokens/<SYM>/{train,val,test}.bin  whole-day chronological splits per symbol
#   tokens/all/{train,val}.bin         symbols concatenated, for train.py
#   test_days/<SYM>_<date>.bin         one file per held-out day, for injection runs
#
#   WORKERS=16 bash build_corpus.sh
set -euo pipefail

SYMBOLS=(SOLUSDT ETHUSDT BTCUSDT)  # download order: free-tier quota is ~44 days/24h
START=2021-01
END=2026-10
TRAIN_END=2026-06-01
VAL_END=2026-07-01
TEST_DAYS=(2026-07-01 2026-08-01 2026-09-01 2026-10-01)  # 07-01 is val; kept for paper comparison
# Tardis rate-limits parallel downloads; past ~16 workers extra cores mostly wait.
WORKERS=${WORKERS:-$(( $(nproc) < 16 ? $(nproc) : 16 ))}

export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR=/workspace/.uv-cache
cd /workspace/orderflow-lm
mkdir -p logs

echo "== normalize ($WORKERS workers) =="
uv run python scripts/build_corpus.py --symbols "${SYMBOLS[@]}" \
  --start "$START" --end "$END" --workers "$WORKERS" 2>&1 | tee -a logs/build_corpus.log

echo "== chronological splits =="
for SYM in "${SYMBOLS[@]}"; do
  uv run python scripts/make_chronological_splits.py --norm "data/normalized/$SYM" \
    --symbol "$SYM" --out "data/tokens/$SYM" --train-end "$TRAIN_END" --val-end "$VAL_END" \
    --workers "$WORKERS" 2>&1 | tee -a logs/splits.log
done

echo "== concatenate symbols =="
uv run python - "${SYMBOLS[@]}" <<'EOF'
import json, shutil, sys
from pathlib import Path
syms, out = sys.argv[1:], Path("data/tokens/all")
out.mkdir(parents=True, exist_ok=True)
for split in ("train", "val"):
    total, parts = 0, []
    with open(out / f"{split}.bin.part", "wb") as dst:
        for s in syms:
            src = Path(f"data/tokens/{s}/{split}.bin")
            with open(src, "rb") as f:
                shutil.copyfileobj(f, dst, length=16 << 20)
            meta = json.loads(Path(f"{src}.meta.json").read_text())
            total += meta["tokens"]
            parts.append({"symbol": s, "tokens": meta["tokens"], "dates": meta["dates"]})
    (out / f"{split}.bin.part").replace(out / f"{split}.bin")
    (out / f"{split}.bin.meta.json").write_text(json.dumps(
        {"vocab": meta["vocab"], "dtype": "uint16", "tokens": total, "parts": parts}, indent=2))
    print(f"{split}: {total:,} tokens")
EOF

echo "== per-day test files =="
uv run python scripts/make_test_days.py --norm data/normalized --out data/test_days \
  --dates "${TEST_DAYS[@]}" --symbols "${SYMBOLS[@]}"
du -sh data/normalized data/tokens data/test_days
