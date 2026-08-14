# Evaluation Scripts

Evaluation scripts read:

- `datasets/samples/*.jsonl`
- `results/<run_id>/*_predictions.jsonl`

They write:

- `results/<run_id>/metrics.csv`
- `results/<run_id>/field_details.csv`
- `results/<run_id>/performance.json` (when `server/run_model.sh` is used)
- optional `results/<run_id>/summary.md`

For an `official` run, create and pass `results/<run_id>/manifest.json` first.
The manifest validates the fixed sample file and records the runner/code/model
versions and runtime environment. Official evaluation requires every sample to
declare `source_split: test`.

First metrics to implement:

- CER / WER
- native field-value accuracy / document-fields exact / field-value F1
- CORD nested scalar fields plus order-independent menu-row precision/recall/F1
- CORD menu column diagnostics for item name, count, quantity, price, and item
  subtotal
- named postprocessor field metrics, reported separately from native model output
- per-field diagnostic rows in `field_details.csv`
- per-page latency p50 / p95, end-to-end wall time, and two throughput views
- actual-price cost per 1,000 pages and bootstrap 95% CIs where applicable
- error rate / success rate

`metric_status=not_applicable` means that the dataset schema or model output
cannot support that metric. It is not a score of zero.

The current `sroie_receipt_regex` postprocessor is a named SROIE receipt
baseline. It must not be used to score CORD. CORD has a dataset-specific
adapter in `eval/cord_metrics.py` for model-emitted native structured fields;
text-only models remain text/performance-only on CORD.

CORD amount normalization is intentionally conservative: clear thousands groups
such as `60.000` and `60,000` are normalized to `60000`, but decimal-like
values such as `12.30` remain distinct from `1230`.

Structured extraction datasets whose `gt_text` is a flattened field target are
handled by `eval/dataset_adapters.py`. `mychen76_invoices`, `fake_w2`, and
`invoices_donut` therefore report CER/WER as `not_applicable`; use their field
metrics instead. `invoices_donut` currently adapts scalar `header.*` and
`summary.*` fields only, with line-item rows reserved for a row-aware metric.

The normative calculation details are in [metric_protocol.md](metric_protocol.md).
For public cost metrics, supply `BENCHMARK_GPU_LABEL`,
`BENCHMARK_GPU_PROVIDER`, `BENCHMARK_GPU_HOURLY_USD`, and
`BENCHMARK_PRICE_RECORDED_AT_UTC` to `server/run_model.sh`; catalog GPU prices
are never substituted automatically.
