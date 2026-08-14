# Benchmark Release Checklist

Use this checklist before publishing any benchmark report or leaderboard table.

## Scope

- [ ] Report title names the exact scope, for example "Receipt OCR Benchmark",
      not a broader claim like "Best OCR Model" unless the data supports it.
- [ ] Document types, languages, and data sources are listed.
- [ ] Synthetic, noncommercial, research-only, internal-only, and blocked
      datasets are separated from public-core results.
- [ ] Private or unpublished document collections are not used, sampled,
      inspected, or referenced as benchmark data.

## Data

- [ ] Every scored sample comes from a fixed JSONL file.
- [ ] Every official scored sample has `source_split: test`.
- [ ] Dataset audit JSON exists and reports `technical_test_ready: true`.
- [ ] Sample count, sample SHA256, source URL, license, and source revision or
      retrieval date are recorded.
- [ ] Warm-up samples are fixed, unscored, and validated as non-overlapping with
      scored samples.

## Models

- [ ] Model version, runner script, dependency environment, backend type, and
      GPU/API runtime are recorded.
- [ ] Failed pages remain in the prediction JSONL and metrics; they are not
      silently excluded.
- [ ] Native structured fields and postprocessed fields are reported separately.
- [ ] VLM/API prompts and output JSON schema are frozen before the run.

## Metrics

- [ ] CER/WER normalization is described.
- [ ] Required fields and field normalization are described.
- [ ] Field-value accuracy, field-value F1, document-fields exact, and line-item
      metrics are included only where the dataset has matching ground truth.
- [ ] CORD column-level line-item metrics are labelled diagnostic, not a
      replacement for row-exact line-item F1.
- [ ] Amount/date/vendor normalization rules are documented and spot-checked
      against source annotations.
- [ ] Latency reports distinguish per-page latency from end-to-end wall time.
- [ ] Cost is `not_applicable` unless provider, GPU/API price, currency, and
      price timestamp are recorded.
- [ ] Confidence intervals are included where supported by the evaluator.

## Artifacts

- [ ] Result directory contains `manifest.json`, `performance.json`,
      `<model>_predictions.jsonl`, `metrics.csv`, and `field_details.csv`.
- [ ] Report tables are generated from frozen metrics artifacts, not manually
      rewritten numbers.
- [ ] Raw outputs or raw-output paths are preserved when a runner produces them.
- [ ] A limitations section states unsupported metrics and dataset caveats.

## Citation Block

Every public benchmark page should include a short citation/link block:

```text
If you use this benchmark, cite and link:
Benchmark OCR Team. "<Report title>." benchmark_ocr, <release date>.
Methodology, sample hashes, prediction JSONL, manifest, and metrics CSV are
available with the report artifacts.
```
