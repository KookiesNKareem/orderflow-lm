# RunPod: rebuild the corpus and retrain

All state lives on a network volume mounted at `/workspace` (repo, venv, data,
checkpoints), so pods are disposable. Every step resumes when re-run.

| Step | Pod | Command | Expected |
|---|---|---|---|
| 0 | any | `bash setup.sh` | once per volume; re-run to pull updates |
| 1 | CPU, 16–32 vCPU | `bash probe.sh` | times one BTC day; prints an upper bound for step 2 |
| 2 | CPU, 16–32 vCPU | `bash build_corpus.sh` | hours; resumable after quota or failures |
| 3 | H100 80GB | `SEED=1337 bash train.sh` | ~1 h for 25k steps |
| 4 | same GPU pod | `bash bench_generate.sh` | generation throughput + analysis smoke test |
| 5 | same GPU pod | `SEED=2027 bash train.sh` | second seed |

Before step 2, check `probe.json`: `upper_bound_normalized_gb` sizes the volume
(add ~30 GB for tokens and headroom); `upper_bound_cpu_hours / workers` sizes the wait.

Splits: train 2021-01-01..2026-06-01, validation 2026-07-01, test 2026-08-01,
2026-09-01, 2026-10-01 (free first-of-month Tardis days, three symbols).

Checkpoints land in `/workspace/checkpoints/` with a `.sha256`, and `train.sh`
copies both to Google Drive (`gdrive:oflm/checkpoints` by default, override with
`GDRIVE_DEST`). Authorize Drive once per volume with `bash gdrive_auth.sh` before
the first training run.
