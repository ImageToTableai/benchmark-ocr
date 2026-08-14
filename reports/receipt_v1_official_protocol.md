# Receipt v1 Official Protocol

Version: 0.1 frozen
Date: 2026-08-11

This protocol defines the first public-core receipt benchmark. It is the run
contract for official results; change it only by publishing a new protocol
version.

## Status

Protocol status: frozen.

This version governs the official runs. Smoke/provisional results produced
during validation are excluded from public tables; only runs carrying
`BENCHMARK_RUN_TIER=official` are publishable.

## Public-Core Scope

Receipt v1 covers public receipt OCR and receipt field extraction only.

Included:

- SROIE 2019 test split.
- CORD v2 reviewed test split.
- OCR text metrics for every model on both datasets.
- SROIE scalar field extraction through the named regex postprocessor where a
  model does not emit native fields.
- CORD structured scalar and menu metrics only when native structured fields
  are present.

Excluded from the public core:

- Private or unpublished document collections.
- DocILE, unless legal/use scope changes. Current known scope is
  non-commercial research only.
- FUNSD public-core results; keep it internal/research-only if used.
- Vietnamese OCR supplemental data; keep it outside Receipt v1 because it is
  text-only and partly synthetic.
- Labels-only datasets without images.

## Official Dataset Files

Official runs must use fixed JSONL files under `$BENCHMARK_DATASETS/samples`
(default `$BENCHMARK_PERSIST/datasets/samples`; see `server/env.sh`).
Every scored sample must have `source_split: test`.

| Dataset | Scored file | Warm-up file | Required split | Notes |
| --- | --- | --- | --- | --- |
| SROIE | `sroie_test_361.jsonl` | `sroie_warmup_5.jsonl` | `test` | Primary English receipt OCR and scalar field benchmark. |
| CORD v2 | `cord_v2_test.jsonl` | `cord_v2_warmup_5.jsonl` | `test` | Receipt structure and line-item candidate. Text metrics are reported, but language/layout caveats must be stated. |

Before running official results, record each scored file's SHA256 and audit path
in the methodology report. Warm-up samples must be validated as non-overlapping
with scored samples.

## Official Model Set

Receipt v1 first batch contains the current non-vLLM local group:

| Model ID | Runner | Track | Official v1 role |
| --- | --- | --- | --- |
| `tesseract` | `scripts/run_tesseract.py` | local | CPU/classic OCR baseline. |
| `paddleocr` | `scripts/run_paddleocr.py` | local | Strong classic OCR baseline. |
| `easyocr` | `scripts/run_easyocr.py` | local | Classic OCR baseline. |
| `doctr` | `scripts/run_doctr.py` | local | Neural OCR baseline. |
| `docling` | `scripts/run_docling.py` | local | Document parser baseline. |
| `paddleocr_vl` | `scripts/run_paddleocr_vl.py` | local | VLM OCR/document parser baseline without vLLM. |

Separate later batch:

- `surya2`
- `unlimited_ocr`

These require a vLLM/server image or external OpenAI-compatible backend and do
not block Receipt v1.

## Formal GPU

Smoke-tier results are not publishable. The formal run must record:

- `BENCHMARK_GPU_LABEL`
- `BENCHMARK_GPU_PROVIDER`
- `BENCHMARK_GPU_HOURLY_USD`
- `BENCHMARK_PRICE_RECORDED_AT_UTC`
- git commit
- run tier
- sample SHA256
- prediction SHA256
- performance SHA256

Preferred formal GPU order:

1. RTX 4090
2. RTX 3090
3. A100 when a high-end comparison is worth the extra cost

Do not merge smoke-tier numbers into official tables.

## Measurement Rules

Use the shared metric protocol in [../eval/metric_protocol.md](../eval/metric_protocol.md).

Required official settings:

- `BENCHMARK_RUN_TIER=official`
- `BENCHMARK_EXPECTED_SPLIT=test`
- `BENCHMARK_MEASUREMENT_MODE=warm_then_scored`
- fixed warm-up JSONL for each dataset
- one result directory per model and dataset
- no missing predictions in official runs

Latency reporting must include both:

- per-page p50/p95 latency from successful prediction records
- end-to-end wall-clock throughput from `performance.json`

Cost is reportable only when actual provider price metadata is present.

Field reporting must include both:

- summary rows in `metrics.csv`
- per-document field diagnostics in `field_details.csv`

CORD line items must report row-exact line-item precision/recall/F1 as the
primary line-item score. Column-level CORD menu metrics are diagnostic only and
must not be presented as a replacement for row-exact line-item matching.

## Official Commands

Set the run metadata once per formal GPU session:

```bash
export BENCHMARK_RUN_TIER=official
export BENCHMARK_EXPECTED_SPLIT=test
export BENCHMARK_MEASUREMENT_MODE=warm_then_scored
export BENCHMARK_GPU_LABEL="rtx_4090"
export BENCHMARK_GPU_PROVIDER="runpod"
export BENCHMARK_GPU_HOURLY_USD="<actual-price>"
export BENCHMARK_PRICE_RECORDED_AT_UTC="<YYYY-MM-DDTHH:MM:SSZ>"
```

Run SROIE:

```bash
export BENCHMARK_FIELD_POSTPROCESSOR=sroie_receipt_regex
for model in tesseract paddleocr easyocr doctr docling paddleocr_vl; do
  BENCHMARK_WARMUP_SAMPLES="$BENCHMARK_DATASETS/samples/sroie_warmup_5.jsonl" \
    bash server/run_model.sh "$model" sroie 361 "receipt-v1-sroie-${model}"
done
```

Run CORD v2 after confirming the exact fixed test count:

```bash
unset BENCHMARK_FIELD_POSTPROCESSOR
CORD_COUNT="$(wc -l < "$BENCHMARK_DATASETS/samples/cord_v2_test.jsonl")"
for model in tesseract paddleocr easyocr doctr docling paddleocr_vl; do
  BENCHMARK_WARMUP_SAMPLES="$BENCHMARK_DATASETS/samples/cord_v2_warmup_5.jsonl" \
    bash server/run_model.sh "$model" cord_v2 "$CORD_COUNT" "receipt-v1-cord-v2-${model}"
done
```

## Publication Gate

Do not publish a Receipt v1 report until every row below is true:

- Dataset audits report `technical_test_ready: true`.
- Every official result directory contains `manifest.json`, `performance.json`,
  `<model>_predictions.jsonl`, `metrics.csv`, and `field_details.csv`.
- Every official run has `run_tier=official` and `expected_split=test`.
- Smoke-tier result directories are excluded from public tables.
- `not_applicable` metrics are not converted to zero.
- CORD text/language caveats are stated in the limitations section.
- Every public number can be traced to a frozen metrics artifact.
