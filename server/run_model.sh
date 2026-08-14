#!/bin/bash
# Thin GPU entrypoint: install one model env, prepare samples, run, evaluate.
set -euo pipefail

cd "$(dirname "$0")/.."
source server/env.sh

MODEL="${1:-tesseract}"
DATASET="${2:-sroie}"
LIMIT="${3:-50}"
RUN_ID="${4:-${DATASET}-${MODEL}-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_TIER="${BENCHMARK_RUN_TIER:-smoke}"
EXPECTED_SPLIT="${BENCHMARK_EXPECTED_SPLIT:-}"
FIELD_POSTPROCESSOR="${BENCHMARK_FIELD_POSTPROCESSOR:-none}"

RUNNER_SCRIPT="scripts/run_${MODEL}.py"
if [ ! -f "$RUNNER_SCRIPT" ]; then
    echo "Missing runner: $RUNNER_SCRIPT" >&2
    exit 2
fi

case "$MODEL" in
    paddleocr|paddleocr_vl|tesseract)
        ENV_PYTHON="$(bash server/install_model_env.sh "$MODEL" | tail -n 1)"
        ;;
    *)
        ENV_PYTHON="$(BENCHMARK_SYSTEM_SITE_PACKAGES="${BENCHMARK_SYSTEM_SITE_PACKAGES:-1}" bash server/install_model_env.sh "$MODEL" | tail -n 1)"
        ;;
esac

case "$DATASET" in
    sroie)
        SAMPLES_FILE="$BENCHMARK_DATASETS/samples/sroie_test_${LIMIT}.jsonl"
        IMAGE_DIR="$BENCHMARK_DATASETS/processed/sroie_2019/images"
        mkdir -p "$(dirname "$SAMPLES_FILE")" "$IMAGE_DIR"
        if [ "${FORCE_PREPARE:-0}" = "1" ] || [ ! -f "$SAMPLES_FILE" ] || [ "$(wc -l < "$SAMPLES_FILE")" -lt "$LIMIT" ]; then
            "$ENV_PYTHON" scripts/prepare_sroie_samples.py \
                --split test \
                --limit "$LIMIT" \
                --samples-out "$SAMPLES_FILE" \
                --image-dir "$IMAGE_DIR" \
                --viewer-rows
        fi
        EXPECTED_SPLIT="${EXPECTED_SPLIT:-test}"
        if [ "${BENCHMARK_FIELD_POSTPROCESSOR+x}" != "x" ]; then
            FIELD_POSTPROCESSOR="sroie_receipt_regex"
        fi
        ;;
    cord|cord_v2)
        SAMPLES_FILE="$BENCHMARK_DATASETS/samples/cord_v2_test.jsonl"
        if [ ! -f "$SAMPLES_FILE" ]; then
            echo "Missing CORD test samples: $SAMPLES_FILE. Prepare and validate them before running." >&2
            exit 2
        fi
        CORD_SAMPLE_COUNT="$(wc -l < "$SAMPLES_FILE")"
        if [ "$LIMIT" != "$CORD_SAMPLE_COUNT" ]; then
            echo "CORD uses its reviewed fixed test file ($CORD_SAMPLE_COUNT samples); pass $CORD_SAMPLE_COUNT as the limit." >&2
            exit 2
        fi
        EXPECTED_SPLIT="${EXPECTED_SPLIT:-test}"
        ;;
    *)
        echo "Unsupported dataset: $DATASET" >&2
        exit 2
        ;;
esac

RESULT_DIR="$BENCHMARK_RESULTS/$RUN_ID"
PREDICTIONS_FILE="$RESULT_DIR/${MODEL}_predictions.jsonl"
METRICS_FILE="$RESULT_DIR/metrics.csv"
MANIFEST_FILE="$RESULT_DIR/manifest.json"
PERFORMANCE_FILE="$RESULT_DIR/performance.json"
WARMUP_SAMPLES="${BENCHMARK_WARMUP_SAMPLES:-}"
WARMUP_CONFIGURED=false
WARMUP_SAMPLES_SHA256=null
if [ -n "$WARMUP_SAMPLES" ] && [ ! -f "$WARMUP_SAMPLES" ]; then
    echo "Warmup sample file does not exist: $WARMUP_SAMPLES" >&2
    exit 2
fi
WARMUP_SAMPLE_COUNT=0
if [ -n "$WARMUP_SAMPLES" ]; then
    WARMUP_SAMPLE_COUNT="$(wc -l < "$WARMUP_SAMPLES")"
    "$ENV_PYTHON" scripts/validate_warmup_samples.py --scored-samples "$SAMPLES_FILE" --warmup-samples "$WARMUP_SAMPLES" >/dev/null
    WARMUP_SAMPLES_SHA256="\"$($ENV_PYTHON -c 'from pathlib import Path; from shared.run_contract import sha256_file; import sys; print(sha256_file(Path(sys.argv[1])))' "$WARMUP_SAMPLES")\""
    WARMUP_CONFIGURED=true
fi
MEASUREMENT_MODE="${BENCHMARK_MEASUREMENT_MODE:-cold_end_to_end}"
if [ -n "$WARMUP_SAMPLES" ] && [ -z "${BENCHMARK_MEASUREMENT_MODE:-}" ]; then
    MEASUREMENT_MODE="warm_then_scored"
fi
mkdir -p "$RESULT_DIR"

echo "=========================================="
echo "Run ID:        $RUN_ID"
echo "Model:         $MODEL"
echo "Dataset:       $DATASET"
echo "Sample limit:  $LIMIT"
echo "Run tier:      $RUN_TIER"
echo "Expected split:${EXPECTED_SPLIT:-<none>}"
echo "Project:       $PWD"
echo "Storage:       $BENCHMARK_PERSIST"
echo "Python:        $ENV_PYTHON"
echo "Samples:       $SAMPLES_FILE"
echo "Predictions:   $PREDICTIONS_FILE"
echo "Metrics:       $METRICS_FILE"
echo "Performance:   $PERFORMANCE_FILE"
echo "Warmup samples:${WARMUP_SAMPLES:-<none>} ($WARMUP_SAMPLE_COUNT)"
echo "GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'CPU only')"
echo "=========================================="

RUNNER_ARGS=(--run-id "$RUN_ID" --samples "$SAMPLES_FILE" --out "$PREDICTIONS_FILE")
if [ -n "$WARMUP_SAMPLES" ]; then
    RUNNER_ARGS+=(--warmup-samples "$WARMUP_SAMPLES")
fi
if [ "$MODEL" = "easyocr" ] && [ "$DATASET" != "sroie" ]; then
    RUNNER_ARGS+=(--timeout-mode process --timeout 60)
fi
RUN_STARTED_NS="$(date +%s%N)"
RUN_STARTED_AT_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$ENV_PYTHON" "$RUNNER_SCRIPT" "${RUNNER_ARGS[@]}"
RUN_FINISHED_NS="$(date +%s%N)"
RUN_FINISHED_AT_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
RUN_WALL_TIME_MS=$(( (RUN_FINISHED_NS - RUN_STARTED_NS) / 1000000 ))
printf '{\n  "schema_version": 1,\n  "measurement_mode": "%s",\n  "warmup_configured": %s,\n  "warmup_sample_count": %s,\n  "warmup_samples_sha256": %s,\n  "runner_started_at_utc": "%s",\n  "runner_finished_at_utc": "%s",\n  "run_wall_time_ms": %s\n}\n' \
    "$MEASUREMENT_MODE" "$WARMUP_CONFIGURED" "$WARMUP_SAMPLE_COUNT" "$WARMUP_SAMPLES_SHA256" "$RUN_STARTED_AT_UTC" "$RUN_FINISHED_AT_UTC" "$RUN_WALL_TIME_MS" > "$PERFORMANCE_FILE"

MANIFEST_ARGS=(
    --run-id "$RUN_ID"
    --run-tier "$RUN_TIER"
    --samples "$SAMPLES_FILE"
    --predictions "$PREDICTIONS_FILE"
    --runner "$RUNNER_SCRIPT"
    --model "$MODEL"
    --env-python "$ENV_PYTHON"
    --field-postprocessor "$FIELD_POSTPROCESSOR"
    --performance-file "$PERFORMANCE_FILE"
    --command "$ENV_PYTHON $RUNNER_SCRIPT ${RUNNER_ARGS[*]}"
    --out "$MANIFEST_FILE"
)
if [ -n "${BENCHMARK_GPU_LABEL:-}" ]; then
    MANIFEST_ARGS+=(--gpu-label "$BENCHMARK_GPU_LABEL")
fi
if [ -n "${BENCHMARK_GPU_PROVIDER:-}" ]; then
    MANIFEST_ARGS+=(--gpu-provider "$BENCHMARK_GPU_PROVIDER")
fi
if [ -n "${BENCHMARK_GPU_HOURLY_USD:-}" ]; then
    MANIFEST_ARGS+=(--gpu-hourly-usd "$BENCHMARK_GPU_HOURLY_USD")
fi
if [ -n "${BENCHMARK_PRICE_RECORDED_AT_UTC:-}" ]; then
    MANIFEST_ARGS+=(--price-recorded-at-utc "$BENCHMARK_PRICE_RECORDED_AT_UTC")
fi
if [ -n "$EXPECTED_SPLIT" ]; then
    MANIFEST_ARGS+=(--expected-split "$EXPECTED_SPLIT")
fi
"$ENV_PYTHON" scripts/create_run_manifest.py "${MANIFEST_ARGS[@]}"

METRICS_ARGS=(
    --run-id "$RUN_ID" \
    --samples "$SAMPLES_FILE" \
    --predictions "$PREDICTIONS_FILE" \
    --run-tier "$RUN_TIER" \
    --field-postprocessor "$FIELD_POSTPROCESSOR" \
    --performance-file "$PERFORMANCE_FILE" \
    --manifest "$MANIFEST_FILE" \
    --out "$METRICS_FILE"
)
if [ -n "${BENCHMARK_GPU_LABEL:-}" ]; then
    METRICS_ARGS+=(--gpu "$BENCHMARK_GPU_LABEL")
fi
if [ -n "${BENCHMARK_GPU_PROVIDER:-}" ]; then
    METRICS_ARGS+=(--gpu-provider "$BENCHMARK_GPU_PROVIDER")
fi
if [ -n "${BENCHMARK_GPU_HOURLY_USD:-}" ]; then
    METRICS_ARGS+=(--gpu-hourly-usd "$BENCHMARK_GPU_HOURLY_USD")
fi
if [ -n "${BENCHMARK_PRICE_RECORDED_AT_UTC:-}" ]; then
    METRICS_ARGS+=(--price-recorded-at-utc "$BENCHMARK_PRICE_RECORDED_AT_UTC")
fi
if [ -n "$EXPECTED_SPLIT" ]; then
    METRICS_ARGS+=(--expected-split "$EXPECTED_SPLIT")
fi
"$ENV_PYTHON" eval/build_metrics_csv.py "${METRICS_ARGS[@]}"

echo "=== Metrics ==="
cat "$METRICS_FILE"
echo "=== Done: $RESULT_DIR ==="
