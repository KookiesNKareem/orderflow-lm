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

## Injection pilot: causal response to exogenous trades (2026-07-19, SOL)

Go/no-go gate for the reactive backtester: does v2 react to a trade it didn't
generate? Design: 48 seeds (1532 tok, cut at event boundaries, July holdout),
five arms per seed sharing the context — marketable BUY/SELL at the touch
(offset 0) injected as 4 appended tokens, at median trade size (d1) and +3
buckets (d2), plus no-injection control. Arms generated under identical
per-batch torch seeds (CRN), 2500 new tokens each, so the paired
(R_buy − R_sell)/2 is a causal impact response. `scripts/pilot_inject.py`,
streams + json in `eval_out/pilot/`.

- **Sign**: paired response positive, bootstrap CI > 0, lags 1–30 (both doses;
  d2 through lag 50). Control drift ≈ 0. Grammar stable under injection
  (illegal mass 0.0006, unchanged); spread/crossed marginals match control.
- **Dose**: d2 > d1 at nearly all lags (l=50: +0.42 vs +0.07 ticks). Concave.
- **Magnitude, vs the right yardstick**: injected response ~0.2 ticks (frame
  clock) is 5× below real R(l) ≈ 1.15 — but R(l) is ~93% correlated-flow herd.
  Deconvolving real R with sign ACF C under the propagator model
  (`scripts/eval_propagator.py`, trade clock, 295k trades) gives a bare kernel
  G_real plateauing at +0.028–0.034 ticks (lags 16–24). Injected d1 there:
  +0.033–0.036 — **agreement within ~20%**, untrained and under intervention.
- **Induced herding**: net sign of the model's next generated trade after a
  buy vs sell injection is +0.48, decaying to +0.14 by lag 32 — a partial
  metaorder read into one exogenous trade (real conditional C(1) = 0.835,
  model endogenous C_gen(1) = 0.868). So injected response = bare + induced,
  sitting above G_real as theory requires: G_real ≤ inj < R_real, all measured.

Caveats: SOL is the best-scorecard symbol (BTC falsification below); square
64×64 deconvolution fits exactly, so rms is vacuous — per-segment
sd (±0.002–0.004) is the error bar; G_real slightly negative at lags 1–3
(near-collinear C at l=1 0.835 + bounce artifact); G_gen unstable past lag 16
(5.3k trades). Pilot infra: per-batch resume cache in `eval_out/pilot/parts_*`;
~65 min per symbol on MPS at ~155 tok/s, minutes on a pod.

## BTC falsification of the injection test (2026-07-20, pod)

Purpose: a pass is only informative if the test can fail. Ran the identical
48-seed pilot + propagator on BTC, the symbol with known deep-eval defects
(impact overshoot, excess sign persistence, super-diffusion). It should fail —
and does, discriminating a usable simulator from an unusable one.

- **No resolvable causal signal.** Injected d1 CIs span zero at every lag
  (l=8: −2.23 [−6.24, +0.02]). Where SOL's 48 seeds pinned the response to
  ±0.02 ticks, BTC's give ±6 — the ~0.2-tick bare impact is swamped by the
  model's own super-diffusive price wander.
- **G_gen diverges**: endogenous kernel blows to +14 ticks at lag 24 vs
  G_real +0.42, echoing the deep-eval impact overshoot.
- Honest limit: the wide CIs mean BTC's true causal response could be
  right-signed-but-buried rather than broken; 48 seeds can't tell. But a
  simulator whose own noise makes impact unmeasurable at a sample size where a
  good one is crisp cannot support agent evaluation. The instrument
  discriminates; that is the result. `eval_out/pilot/propagator_BTCUSDT.json`.

Infra note (pod): GPU is Blackwell sm_120; the preinstalled torch 2.4/cu124
lacks kernels — needs torch 2.8+cu128. Generation ~1,170 tok/s at batch 48
(≈7.5× MPS). K-segment runs OOM at batch 64 on 24 GB (full 2048 KV cache per
row); batch 32 + PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True fits.

## PnL-gap experiment (2026-07-20, SOL, pod)

The payoff test: how much does a replay backtest understate execution cost vs a
reactive sim that pushes back? A TWAP taker buys a fixed quantity as K=8
marketable child orders at the touch, evenly spaced (spacing 250). Three arms,
128 seeds, doses = size buckets {6,9,12,15} (≈ ⅛×, 1×, 8×, 64× the median
trade): naive replay (recorded future, no reaction), reactive (children
injected, market reacts), reactive control (no trading, counterfactual).
Impact = reactive − control shortfall, drift removed by CRN.
`scripts/pnl_gap.py`, `eval_out/pnl/pnlgap_SOLUSDT.json`.

| dose | rel size | reactive impact (ticks/unit) [CI] |
|---|---|---|
| 6  | 1×   | +0.281 [+0.15, +0.41] |
| 9  | 8×   | +0.261 [+0.14, +0.38] |
| 12 | 64×  | +0.410 [+0.27, +0.55] |
| 15 | 512× | +0.636 [+0.50, +0.78] |

- **Replay hides a real, size-growing cost.** Naive impact = 0 by construction;
  reactive charges +0.28→+0.64 ticks/unit, rising with size, all CIs clear of
  zero. Replay's zero-impact assumption is wrong and increasingly wrong with
  aggressiveness — the "backtests lie" failure mode, measured.
- **Shape is NOT square-root-through-origin** (initial R²=−0.38 was a
  misspecified no-intercept fit). A floor+√ fits: impact = 0.248 + 0.017·√size,
  R²=0.976. Log-log exponent β=0.14 (√ law = 0.5): far more concave than the
  textbook, with a ~0.25-tick spread-crossing floor paid even by tiny orders.
- Calibration note: generated no-trade baseline (ctrl shortfall 0.51 ticks)
  runs ~0.14 above the real recorded baseline (naive 0.37) — impact numbers are
  drift-removed so internally clean, but the sim's absolute spread/drift
  baseline sits slightly high.
- Limits: 4 coarse power-of-2 dose points (shape, not a precise coefficient);
  SOL only, K=8, single TWAP schedule; dose 6→9 flat within noise (both on the
  floor).

## Real-data validation of size scaling (2026-07-20, SOL, pod)

Resolves "is the too-flat curve a model defect or a real feature?" — never
validate against a textbook law, only against real data measured identically.
Size-conditioned response R(l=20 | size bucket), same estimator on 427k real
trades (val.bin) and the pilot's endogenous generated trades.
`scripts/eval_size_impact.py`, `eval_out/pnl/sizeimpact_SOLUSDT.json`.

| bucket | real R (ticks) | gen R | n_real | n_gen |
|---|---|---|---|---|
| 6  | 1.22 | 0.96 | 145k | 1706 |
| 9  | 1.08 | 0.81 | 88k  | 1227 |
| 12 | 1.23 | 0.61 | 104k | 1219 |
| 15 | 1.17 | 0.69 | 60k  | 739  |

- **The √-law worry was a false alarm.** Real SOL's per-trade response is FLAT
  in size (exponent β=+0.001): a 64×-larger real trade moves price no more at
  lag 20. So β=0.5 does not appear in this estimator on REAL data either — the
  √ law (metaorders vs daily volume) was the wrong benchmark for a second-scale
  per-trade experiment. The model not following √ is fine; reality doesn't here.
- **Model reproduces real size-behavior, mildly under-reactive.** Gen exponent
  β=−0.06 vs real +0.00 (slightly declining vs flat), level ~20–35% low. So the
  model is faithful in-distribution with a documented mild under-reaction at
  large size → the causal PnL curve if anything UNDER-states large-order impact.
- **Causal (injection, rising) vs conditional (endogenous, flat) scale
  oppositely, correctly.** Conditional flatness is selection (big real trades
  placed when liquid); injection removes selection so causal rises. The model
  reproducing the conditional flatness is evidence its mechanism is sound.
- Permanent limit: conditional ≠ causal. This validates fidelity to real
  size-behavior, not the causal magnitude directly — true causal validation
  needs real intervention data, which does not exist.

Summary of the arc: injection gate PASS on SOL (causal, dose-ordered, magnitude
within 20% of the deconvolved real kernel) → falsification PASS (BTC fails the
same test, so it discriminates) → PnL gap measured (replay hides +0.28→0.64
ticks/unit, size-growing) → real-data check (model's size scaling matches real,
mildly conservative; √-law deviation was a benchmark mismatch, not a defect).
Open: PnL gap is SOL/one-schedule; absolute causal magnitude unvalidatable
without real intervention data; model runs modestly low on impact level.
