#!/bin/bash
# Shared path defaults. Source this from server scripts.

# Set BENCHMARK_PERSIST to your durable storage root before sourcing, e.g.:
#   export BENCHMARK_PERSIST=/path/to/your/storage     # on a GPU cloud, use your network volume mount point
#   export BENCHMARK_PERSIST=/data/bench    # on your own machine
if [ -z "${BENCHMARK_PERSIST:-}" ]; then
    echo "WARNING: BENCHMARK_PERSIST not set; defaulting to current directory." >&2
    export BENCHMARK_PERSIST="$(pwd -P)"
fi

export BENCHMARK_HOME="${BENCHMARK_HOME:-$BENCHMARK_PERSIST/benchmark-ocr}"
export BENCHMARK_DATASETS="${BENCHMARK_DATASETS:-$BENCHMARK_PERSIST/datasets}"
export BENCHMARK_MODELS="${BENCHMARK_MODELS:-$BENCHMARK_PERSIST/models}"
export BENCHMARK_RESULTS="${BENCHMARK_RESULTS:-$BENCHMARK_PERSIST/results}"
export BENCHMARK_ENVS="${BENCHMARK_ENVS:-$BENCHMARK_PERSIST/envs}"

export HF_HOME="${HF_HOME:-$BENCHMARK_MODELS/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$BENCHMARK_MODELS/hf_home/hub}"
export PADDLEOCR_HOME="${PADDLEOCR_HOME:-$BENCHMARK_MODELS/paddleocr}"
