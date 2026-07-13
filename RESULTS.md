# Run log

Version lineage with configs and headline numbers, for the writeup.
Uniform-baseline loss: ln(384) = 5.951 (v0: ln(320) = 5.768).

| version | tokenization | data | model | optimizer | budget | best val | notes |
|---|---|---|---|---|---|---|---|
| v0 | no jump escape (silent seams) | BTC 2026, 7 days (233M tok) | 26M, block 1024 | AdamW 3e-4 | killed @ step 700 | 2.090 @ 500 | MPS smoke; stopped once >16-tick improvements found unrepresentable (0.8–3.2% of frames) |
| v0.1 | + jump escape (exact to 1,040 ticks) | BTC 2026, 7 days (247M tok) | 26M | AdamW 3e-4 | 2k steps = 16.4M tok (0.6 tok/param) | 1.986 | MPS, ~30 min; grammar learned (illegal mass 0.9%/step); realism: spread KS 0.254 vs 0.037 floor, trade:frame mix regressed to train marginal |
| A/B | v0.1 | same | 26M | AdamW 5e-4 vs Muon 4e-2 | 600 steps @ batch 64 = 39M tok each | 2.038 vs **1.974** | pod (RTX PRO 6000); Muon reached AdamW's 600-step loss by step ~200; +8% step cost at this scale |
| v1 | v0.1 | same | 26M | Muon 4e-2 | 8k steps = 524M tok (20 tok/param, Chinchilla) | 1.987 @ step 500, then val RISES to ~2.3 | **compute didn't help: non-stationarity wall.** Val (June) degrades as the model specializes to Jan–May. Feed cadence doubled at the train/val boundary (frames/day: Jan–May 1.2–1.5M, Jun–Jul 2.7–2.8M), so val is structurally different data. v0.1 (1.986) and v1 (1.987) hit the same wall from different budgets. |
| v2.0 | + symbol tokens, inferred ticks | BTC+ETH+SOL 2021–2026: 200 days, **11.75B tok** | 87M (d=768, L=12), block 2048 | Muon 4e-2 | killed @ 2.8k steps | diverged | **Muon LR cliff moved with width**: 4e-2 (optimal at d=512) slowly diverges at d=768 — train loss rose 1.89→2.96 after warmup. Echoes the optim-study cliff observation; tuned LR did not transfer across scale. |
| v2 | same | same | same | Muon 2e-2, warmup 500 | 26.5k steps = 1.74B tok (20 tok/param) | **1.375** @ step 22k | broke the ~1.99 wall by 0.61 nats on a strictly-future holdout (July 2026 first half). Multi-year corpus + walk-forward split + 3.3× params + 2× context; ~1.8h on pod at 265k tok/s. | multi-year regime diversity targets the conditioning failure. Split lesson from v1: train ≤2026-06 includes the June cadence change; val/test = first/second half of 2026-07-01 (129M tok each), strictly walk-forward. Pod: 266k tok/s, ~1.8h. |

## Realism scorecard history (vs real, KS distance; noise floor 0.037)

| version | val set | spread KS (floor) | depth KS | crossed/kframe (applier baseline) | frames/s gen:real | trades:frames gen (real) |
|---|---|---|---|---|---|---|
| v0.1 | June | 0.254 (0.037) — **fails 7× floor** | 0.154 | 21.5 (9.0) | 0.49 | 2.50 (0.85–1.6) |
| v1 | June | not run — checkpoint ≈ v0.1 (same 1.99 wall) | | | | |
| v2 | July (walk-forward) | **0.0325 (0.0479) — passes below the noise floor** | 0.334* | 13.4 (2.9) | 0.77 | 2.18 (1.19) |

\* depth KS is confounded in v2: val pools BTC/ETH/SOL whose contract sizes differ by orders of magnitude, so the distribution compares symbol *composition* as much as model quality. Needs per-symbol stratification before it means anything.

~~v2 realism read: generated spread distribution statistically indistinguishable from
real (pooled KS 0.0325 < 0.048 floor).~~ **CORRECTION from the stratified eval: the
pooled "pass" was a mixture artifact.** Pooling BTC/ETH/SOL lets cross-symbol
composition swamp within-symbol differences (SOL's spread is degenerately 1 tick).
Per-symbol, spread KS fails: BTC 0.346 (floor 0.038), ETH 0.303 (0.140), SOL trivially
passes. Methodology lesson for the writeup: never pool symbols in distributional evals.

## v2 stratified scorecard (July holdout, per symbol; 4×12k-token rollouts each)

| | BTC | ETH | SOL |
|---|---|---|---|
| frames/s gen (real) | 33.2 (36.4) | 39.7 (38.9) ✓ | 34.2 (32.9) ✓ |
| trades/s gen (real) | 33.4 (47.6) | 62.6 (66.5) ✓ | 24.7 (27.1) ✓ |
| spread p50/90/99 gen (real) | 1/2/3 (1/2/6) too calm | 1/4/4 (1/1/2) too wide | 1/1/1 (1/1/1) ✓ |
| spread KS (floor) | 0.346 (0.038) | 0.303 (0.140) | 0.0001 (0.0002) |
| depth p50 gen (real) | 14.5 (11.8) | 371 (197) | 3707 (2966) |
| crossed/kframe gen (baseline) | 12.5 (6.7) | 0.61 (0.09) | 0.34 (0.00) |
| illegal mass/step | 0.05% | 0.05% | 0.07% |

Timing/mix essentially solved per symbol (the pooled 2.18-vs-1.19 mix error was also
mostly composition). Grammar 15× better than v0.1 (0.9% → 0.05%). Books consistently
too deep. Spread tails: BTC under-disperses, ETH over-disperses.

## Deep eval (8×30k-token rollouts/symbol, 4×2M-token real segments; eval_out/*.json)

- **Sign-memory explains the impact overshoot.** ACF ~ l^-γ: real γ 0.45/0.48/0.61
  (BTC/ETH/SOL) vs generated 0.25/0.32/0.55. Generated flow too persistent everywhere,
  and the γ gap orders exactly with the impact overshoot (BTC worst → 164 vs 56 ticks
  @ lag 200; SOL matched → slight undershoot). Two independent metrics, one mechanism.
- **Vol clustering** matches at short lags, dies beyond a few hundred frames ≈ 10–13s
  ≈ the 2048-token context window. Context bounds volatility memory.
- **Tails**: r1 kurtosis BTC 514 gen vs 476 real (✓); ETH/SOL generate too thin
  (143 vs 558, 106 vs 329). Aggregational decay correct in both.
- **Signature plot**: BTC gen Var/Δ grows 6× from Δ=1 to 200 vs 2.4× real —
  super-diffusion (the trending failure in price space). ETH/SOL shapes near-real.
- **Stability**: median/p90 spread flat across full 2-min rollouts, all symbols.
  No book collapse or spread explosion. v0.1's degradation modes are gone.
- **Seed-set variance**: BTC spread KS 0.15 here vs 0.35 in the stratified run;
  ETH passed below floor here (0.017 vs 0.083). Report ranges, not points.

## Impact response function R(l) — first economic metric

R(l) = E[sign · (mid_{t+l} − mid_t)] in ticks, lags in frames (~26ms each):

| lag | 1 | 2 | 5 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|---|---|
| BTC real | 26.3 | 38.3 | 45.1 | 46.4 | 47.7 | 50.0 | 50.3 |
| BTC gen | 22.1 | 32.6 | 47.6 | 54.4 | 58.7 | 76.1 | 100.8 |
| ETH real | 10.3 | 12.0 | 12.6 | 13.2 | 13.9 | 14.1 | 14.1 |
| ETH gen | 9.8 | 14.0 | 17.8 | 13.8 | 15.6 | 14.9 | 16.8 |
| SOL real | 0.86 | 0.98 | 1.13 | 1.20 | 1.24 | 1.24 | 1.24 |
| SOL gen | 0.88 | 1.15 | 1.66 | 1.94 | 2.02 | 2.01 | 1.02 |

**The model reproduces the impact function's shape and magnitude at short horizons
unprompted** — nothing in training targets impact; it emerges from next-token
prediction. ETH tracks the full curve within noise. The systematic failure: generated
impact keeps rising where real impact plateaus (clearest on BTC: 101 vs 50 ticks at
lag 100) — rollouts over-trend after trades instead of mean-reverting. n(gen trades)
≈ 1.8–2.7k per symbol, so lag-100 gen values are noisy but the trend is consistent.
Size-conditioned impact at lag 20 reproduces the real (non-monotone) tercile ordering
on ETH and BTC. This is the tier-3 result in embryo: the simulator responds to order
flow like a real market at horizons up to ~0.5s, and over-responds beyond.
