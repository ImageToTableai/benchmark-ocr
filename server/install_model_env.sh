#!/bin/bash
# Create or refresh one model-specific Python environment on the storage volume.
set -euo pipefail

cd "$(dirname "$0")/.."
source server/env.sh

MODEL="${1:?Usage: install_model_env.sh <model>}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ENV_ROOT="${BENCHMARK_ENV_ROOT:-$BENCHMARK_ENVS}"
ENV_DIR="$ENV_ROOT/$MODEL"
MODEL_REQ="deps/$MODEL.txt"
PACKAGE_CACHE="${BENCHMARK_PACKAGE_CACHE:-$BENCHMARK_MODELS/package-cache}"

detect_paddle_index_url() {
    if [ -n "${PADDLE_INDEX_URL:-}" ]; then
        echo "$PADDLE_INDEX_URL"
        return 0
    fi

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return 0
    fi

    local cuda_version
    cuda_version="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9][0-9]*\.[0-9]\).*/\1/p' | head -n 1)"

    case "$cuda_version" in
        13.0) echo "https://www.paddlepaddle.org.cn/packages/stable/cu130/" ;;
        12.9) echo "https://www.paddlepaddle.org.cn/packages/stable/cu129/" ;;
        12.6) echo "https://www.paddlepaddle.org.cn/packages/stable/cu126/" ;;
        11.8) echo "https://www.paddlepaddle.org.cn/packages/stable/cu118/" ;;
    esac
}

mkdir -p "$ENV_ROOT" "$PACKAGE_CACHE"

if [ ! -x "$ENV_DIR/bin/python" ]; then
    VENV_ARGS=()
    if [ "${BENCHMARK_SYSTEM_SITE_PACKAGES:-0}" = "1" ]; then
        VENV_ARGS+=(--system-site-packages)
    fi
    "$PYTHON_BIN" -m venv "${VENV_ARGS[@]}" "$ENV_DIR"
fi

ENV_PYTHON="$ENV_DIR/bin/python"

install_requirements() {
    # uv is fast for isolated envs, but pip correctly sees system site-packages.
    if [ "${BENCHMARK_USE_UV:-0}" = "1" ] \
        && [ "${BENCHMARK_SYSTEM_SITE_PACKAGES:-0}" != "1" ] \
        && command -v uv >/dev/null 2>&1; then
        UV_CACHE_DIR="$PACKAGE_CACHE" uv pip install --python "$ENV_PYTHON" "$@"
    else
        PIP_CACHE_DIR="$PACKAGE_CACHE" "$ENV_PYTHON" -m pip install "$@"
    fi
}

if [ "${BENCHMARK_INSTALL_ROOT_REQUIREMENTS:-1}" = "1" ]; then
    install_requirements -r requirements.txt
fi

if [ -f "$MODEL_REQ" ]; then
    if [ "$MODEL" = "paddleocr" ] || [ "$MODEL" = "paddleocr_vl" ]; then
        PADDLE_INDEX="$(detect_paddle_index_url)"
        if [ -n "$PADDLE_INDEX" ]; then
            echo "Using PaddlePaddle wheel index: $PADDLE_INDEX"
            PADDLE_REQS="$(mktemp)"
            OTHER_REQS="$(mktemp)"
            grep -E '^paddlepaddle-gpu([=<>!~ ]|$)' "$MODEL_REQ" > "$PADDLE_REQS" || true
            grep -Ev '^paddlepaddle-gpu([=<>!~ ]|$)' "$MODEL_REQ" > "$OTHER_REQS" || true
            if [ -s "$PADDLE_REQS" ]; then
                install_requirements -i "$PADDLE_INDEX" -r "$PADDLE_REQS"
            fi
            if [ -s "$OTHER_REQS" ]; then
                install_requirements -r "$OTHER_REQS"
            fi
            rm -f "$PADDLE_REQS" "$OTHER_REQS"
        else
            install_requirements -r "$MODEL_REQ"
        fi
    else
        install_requirements -r "$MODEL_REQ"
    fi
fi

if [ "$MODEL" = "tesseract" ] && ! command -v tesseract >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" = "0" ]; then
        apt-get update
        apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng
    else
        echo "WARNING: tesseract not found and not running as root. Install tesseract-ocr manually." >&2
    fi
fi

echo "$ENV_PYTHON"
