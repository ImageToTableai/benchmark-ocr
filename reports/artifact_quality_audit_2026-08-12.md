# Artifact Quality Audit, 2026-08-12

Scope: Receipt v1 benchmark run artifacts under `results/` and normalized fixed
samples under `datasets/samples/` (paths are relative to `$BENCHMARK_PERSIST`).

## Decision

The completed OCR predictions do not need an immediate GPU rerun for data
contract reasons. The reviewed sample files, prediction files, metrics files,
and performance files are internally consistent for the checked runs.

The Receipt v1 SROIE and CORD predictions and metrics should be kept; regenerate
or rerun only if schema v2 manifests are required for final public artifacts.

## Dataset Audit

| Sample file | Count | Split | Images | Text GT | Field GT | Status |
|---|---:|---|---:|---:|---:|---|
| `sroie_test_361.jsonl` | 361 | test | 0 missing | 361 | 361 flat | OK for official receipt OCR and SROIE fields |
| `cord_v2_test.jsonl` | 100 | test | 0 missing | 100 | 100 nested | OK for official CORD text; native CORD fields only when a model emits structured fields |

## Result Audit

Checked result families:

- `receipt-v1-sroie-*`
- `receipt-v1-cord-v2-*`

Findings:

- Prediction contracts passed for checked runs: sample IDs, run IDs, dataset IDs,
  duplicate IDs, and official coverage all matched.
- Existing manifest sample hashes and performance hashes matched their referenced
  files.
- Existing manifests are schema v1 because the schema v2 environment fingerprint
  was added after these runs.
- Speed information exists in `metrics.csv` and `performance.json`:
  `latency_p50_ms`, `latency_p95_ms`, `pages_per_minute`,
  `run_wall_time_ms`, `wall_clock_pages_per_minute`, and
  `cost_per_1000_pages`.

## Rerun Policy

No immediate GPU rerun is required for the checked prediction files.

Rerun only when one of these applies:

- The final public release requires schema v2 manifests generated in the same
  model environment, not backfilled from a generic Python environment.
- A model runner changes in a way that affects predictions, latency, warm-up,
  timeout, preprocessing, language settings, or model version.
- A sample file changes or is re-frozen with a different hash.
- The report scope changes from text OCR to native structured extraction.

For public benchmark artifacts, prefer a fresh run after the schema v2 manifest
change so the environment fingerprint is captured from the exact model Python
environment at run time.
