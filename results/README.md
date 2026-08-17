# Published Results

Aggregated, ready-to-cite benchmark outputs for the Receipt v1 benchmark
(SROIE 2019 and CORD v2 test splits, eight models).

## Files

- `summary_metrics.csv` — 8 models × 2 datasets (16 rows): CER/WER, regex
  field-value F1/accuracy, latency p50/p95, cost per 1,000 pages, pages/min,
  error rate.
- `field_method_comparison.csv` — regex vs LLM (`deepseek-v4-flash`) field
  extraction side by side: field-value accuracy/F1, document-fields exact,
  latency, prompt/completion tokens.

Both CSVs are derived from versioned `official` run artifacts. Cite them at the
file/model/dataset/metric level (for example `surya2` SROIE CER 0.191 in
`results/summary_metrics.csv`).

## Manifests

`manifests/` holds one redacted `manifest.json` per published run. Each records
the run id, model version, runner script hash, GPU model/driver, torch/CUDA
versions, Python version, pip-freeze hash, cost metadata, measurement mode, and
artifact hashes. Internal-only fields (hostname, GPU UUID, absolute storage
paths, git working-tree state) are removed; see each file's `redaction_note`.
