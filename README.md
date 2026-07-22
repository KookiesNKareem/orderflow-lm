# orderflow-lm

Order-flow language modeling for reactive market simulation. The project tests
whether a learned limit-order-book simulator responds causally to an exogenous
trade, rather than only matching the distribution of historical market data.

The paper introduces paired buy/sell injections under positional common random
numbers and evaluates the response against the impact kernel implied by a
propagator model.

## Repository

- `src/oflm/`: tokenization, reconstruction, simulation, and model code
- `scripts/`: data, training, and evaluation entry points
- `results/paper/`: compact outputs used in the paper
- `paper/`: LaTeX source, figures, and the latest compiled PDF

Raw market data, checkpoints, generated streams, and console logs are excluded
because they are large or reproducible intermediates.

## Setup

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/) are recommended.

```bash
uv sync
```

Each command exposes its options through `--help`. The main workflow is:

```bash
uv run python scripts/build_corpus.py --help
uv run python scripts/make_splits.py --help
uv run python scripts/train.py --help
uv run python scripts/eval_deep.py --help
uv run python scripts/pilot_inject.py --help
uv run python scripts/pnl_gap.py --help
```

The corpus uses the free first day of each month from Tardis.dev for Binance
USD-margined perpetual futures. The paper reports BTCUSDT, ETHUSDT, and SOLUSDT
from 2021 through mid-2026 with a strictly walk-forward July 2026 holdout.

## Paper

Regenerate the figures and compile the manuscript with:

```bash
uv run python paper/make_paper_figures.py
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

See [`results/paper/README.md`](results/paper/README.md) for output provenance
and [`paper/main.pdf`](paper/main.pdf) for the compiled manuscript.
