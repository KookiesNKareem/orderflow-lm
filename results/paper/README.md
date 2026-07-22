# Paper results

This directory contains the compact evaluation outputs used to build the paper.

- `distributional/`: free-rollout realism metrics by symbol
- `injection/`: the final 2,048-window paired-injection summaries, causal tests,
  impact-kernel estimates, and run metadata
- `kernel/`: truncation and segment-bootstrap robustness checks
- `execution/`: TWAP replay-gap and size-impact summaries

The final injection run used positional Gumbel common random numbers and the
step-22,000 checkpoint. Its checkpoint hash, source hashes, seed rule, and run
configuration are recorded in `injection/run_metadata.json`. Those hashes refer
to the source at execution time; later edits only shortened module documentation
and reorganized output paths.

Large raw token streams, generated rollout arrays, model checkpoints, and market
data are not stored in Git. The JSON files are sufficient to rebuild every paper
figure with:

```bash
uv run python paper/make_paper_figures.py
```
