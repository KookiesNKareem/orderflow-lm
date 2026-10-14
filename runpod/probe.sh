#!/usr/bin/env bash
# Time the corpus build on its worst-case day before committing to the full run.
# Downloads one symbol-day from Tardis, reconstructs it, and tokenizes it,
# timing each stage separately; prints sizes and an upper bound for the full
# corpus (every day costed like this one).
#
#   bash probe.sh                       # BTCUSDT 2026-06-01, the largest kind of day
#   bash probe.sh ETHUSDT 2023-03-01
set -euo pipefail

SYM=${1:-BTCUSDT}
DATE=${2:-2026-06-01}
N_DAYS=${N_DAYS:-198}   # symbol-days in the training corpus (2021-01..2026-06, 3 symbols)
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR=/workspace/.uv-cache
cd /workspace/orderflow-lm

DIR=/workspace/probe/$SYM-$DATE
mkdir -p "$DIR"
Y=${DATE:0:4}; M=${DATE:5:2}; D=${DATE:8:2}

t0=$(date +%s)
for TYPE in incremental_book_L2 trades; do
  curl -fsSL --retry 5 --retry-delay 30 -o "$DIR/$TYPE.csv.gz" \
    "https://datasets.tardis.dev/v1/binance-futures/$TYPE/$Y/$M/$D/$SYM.csv.gz"
done
t1=$(date +%s)

uv run python - "$DIR" "$SYM" "$((t1 - t0))" "$N_DAYS" <<'EOF'
import json, os, resource, sys, time
sys.path.insert(0, "src")
import polars as pl
from oflm.reconstruct import reconstruct
from oflm.tokenizer import SYMBOLS, tokenize_day

d, sym, t_dl, n_days = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
l2, tr, norm = f"{d}/incremental_book_L2.csv.gz", f"{d}/trades.csv.gz", f"{d}/day.parquet"
t0 = time.time()
reconstruct(l2, tr, norm, 10**12)
t_rec = time.time() - t0
t0 = time.time()
toks = tokenize_day(pl.read_parquet(norm), SYMBOLS[sym])
t_tok = time.time() - t0
gb = lambda p: os.path.getsize(p) / 1e9
r = {
    "download_s": t_dl, "reconstruct_s": round(t_rec), "tokenize_s": round(t_tok),
    "raw_gb": round(gb(l2) + gb(tr), 2), "normalized_gb": round(gb(norm), 2),
    "tokens_M": round(len(toks) / 1e6, 1),
    "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1),
}
per_day = t_dl + t_rec + t_tok
r["upper_bound_cpu_hours"] = round(per_day * n_days / 3600, 1)
r["upper_bound_normalized_gb"] = round(r["normalized_gb"] * n_days)
print(json.dumps(r, indent=1))
json.dump(r, open(f"{d}/probe.json", "w"), indent=1)
EOF
rm -f "$DIR"/*.csv.gz "$DIR/day.parquet"
echo "Wall-clock for the full build is roughly upper_bound_cpu_hours / workers,"
echo "unless Tardis download throughput or quota is the binding constraint."
