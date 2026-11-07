#!/usr/bin/env bash
# Clean Data Encoder launcher (Linux / macOS)
#
# Runs through uv, which creates and syncs the virtualenv from pyproject.toml on
# first use. No pretrained backbone, so no HuggingFace / legacy-Keras / torch
# setup is needed - plain TensorFlow only.
#
#   ./run.sh run_ablation.py --dataset poi_library
#   ./run.sh train.py --run myrun query_instruction=noise_domain --eval-after
#   ./run.sh evaluate.py --run myrun --best
#   ./run.sh -m tensorboard.main --logdir runs
#
# Install uv once:  curl -LsSf https://astral.sh/uv/install.sh | sh

set -euo pipefail

export TF_CPP_MIN_LOG_LEVEL=2

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not on PATH. Install it with:"
    echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "then reopen this shell."
    exit 1
fi

if [ $# -eq 0 ]; then
    echo "Usage: ./run.sh <script.py> [args...]"
    echo
    echo "  ./run.sh run_ablation.py --dataset poi_library"
    echo "  ./run.sh train.py --run myrun --eval-after"
    echo "  ./run.sh evaluate.py --run myrun --best"
    echo "  ./run.sh -m tensorboard.main --logdir runs"
    exit 1
fi

exec uv run python "$@"
