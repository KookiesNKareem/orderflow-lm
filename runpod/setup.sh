#!/usr/bin/env bash
# One-time setup on any pod (CPU or GPU) with the network volume at /workspace.
# Re-running it updates the checkout. The repo, its .venv, the uv cache, data,
# and checkpoints all live on the volume, so they survive pod restarts.
#
#   bash setup.sh            # or: BRANCH=main bash setup.sh
set -euo pipefail

BRANCH=${BRANCH:-icml-week0}
REPO=/workspace/orderflow-lm
export UV_CACHE_DIR=/workspace/.uv-cache

if [ -d "$REPO/.git" ]; then
  git -C "$REPO" fetch -q origin "$BRANCH"
  git -C "$REPO" checkout -q "$BRANCH"
  git -C "$REPO" pull -q --ff-only origin "$BRANCH"
else
  git clone -q -b "$BRANCH" https://github.com/KookiesNKareem/orderflow-lm.git "$REPO"
fi

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

cd "$REPO"
uv sync
mkdir -p /workspace/checkpoints logs
echo "ready: $REPO @ $(git rev-parse --short HEAD), $(nproc) CPUs"
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
