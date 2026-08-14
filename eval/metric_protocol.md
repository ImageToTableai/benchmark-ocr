# Benchmark Metric Protocol

Version 1.0, 2026-08-11. This protocol applies to every result labelled
`official`. Smoke and provisional results may use the same implementation, but
are not public comparison evidence.

## Data And Failures

- An official run uses one reviewed fixed JSONL file where every record has
  `source_split: test`.
- Every fixed sample must produce one prediction record. A non-`ok` record is a
  failure: it contributes to `error_rate` and `success_rate`, but is excluded
  from CER, WER, field, and per-page latency averages. Counts are always shown.
- Metrics are macro averages over successful documents unless the metric says
  otherwise. CORD menu metrics are corpus micro precision/recall/F1.

## Text Metrics

- Version 1.0 lowercases and collapses all whitespace to one ASCII space. It
  does not yet apply Unicode NFKC normalization.
- CER is Levenshtein distance divided by normalized ground-truth characters.
- WER is Levenshtein distance divided by whitespace-delimited ground-truth
  tokens. It is suitable for the current English and Indonesian receipt scope,
  not for CJK results. CJK requires a language-specific protocol revision.
- Some structured extraction datasets provide `gt_text` as flattened JSON-like
  field targets rather than page-level OCR transcription. For those datasets,
  CER/WER must be emitted as `metric_status=not_applicable`; they are evaluated
  with dataset field adapters instead.

## Field Metrics

- SROIE scalar fields compare normalized values after whitespace collapse.
  `field_value_accuracy` is the macro fraction of required fields correct;
  `document_fields_exact` requires all required fields; field F1 counts
  correctly named values against predicted and required values.
- CORD normalizes names with NFKC, case folding, and whitespace collapse.
  Amount fields remove comma and period separators only when they match clear
  thousands-grouping patterns such as `60.000`, `60,000`, or `1.234.567`.
  Decimal-like values such as `12.30` are not collapsed into `1230`. Quantity
  fields remove a trailing `x` and normalize a zero-only decimal suffix.
  Scalar totals and subtotals are scored separately from menu rows. Menu rows
  match only when every annotated value in that row matches, and are
  order-independent with duplicate-aware maximum matching.
- CORD also reports menu column diagnostics for `nm`, `num`, `cnt`, `price`,
  and `itemsubtotal`. These column metrics ignore row association and exist only
  to diagnose which part of a line item failed; the row-exact line-item F1
  remains the primary line-item metric.
- `native_*` metrics score structured fields emitted by a model. A named
  `postprocessed_*` metric scores only its declared postprocessor. They are
  never combined and `not_applicable` is not zero.
- `mychen76_invoices`, `fake_w2`, and `invoices_donut` are structured-target
  datasets in the current implementation. Their adapters compare scalar fields
  only. `invoices_donut` flattens `header.*` and `summary.*`; repeated `items`
  are held back until row-aware invoice line-item metrics are implemented.
- `field_details.csv` is emitted beside `metrics.csv` with per-document,
  per-field normalized values and match flags. Public field conclusions should
  be traceable to this file.

## Performance And Cost

- `latency_p50_ms` and `latency_p95_ms` summarize successful per-page runner
  calls. They exclude model construction that occurs before the sample loop.
- A steady-state comparison supplies `--warmup-samples` with a separate,
  unscored fixed JSONL. The runner fails if a warm-up call fails. The generated
  `performance.json` records whether warm-up was configured and the count.
- `run_wall_time_ms` measures the complete runner process from invocation to
  completion, including imports, model initialization, and any warm-up.
  `wall_clock_pages_per_minute` uses this value; `pages_per_minute` uses summed
  successful per-page latency.
- Quantiles use Hyndman-Fan type 7 linear interpolation. A published comparison
  reports both latency families and its measurement mode.
- `cost_per_1000_pages` is emitted only with an actual hourly USD price,
  provider label, price timestamp, and `run_wall_time_ms`. It uses the full
  runner wall time divided by successful pages, not a catalog GPU price.
- `manifest.json` schema v2 records the runner command, runner script hash,
  evaluator hash, git commit and dirty-status hashes, GPU/driver metadata,
  Python executable, key package versions, and `pip freeze` hash. Public
  reports should cite the manifest for reproducibility rather than copying
  environment facts by hand.

## Uncertainty

- Per-document CER, WER, error/success, and applicable document-level field
  metrics receive a two-sided percentile bootstrap 95% CI.
- The default is 2,000 deterministic resamples with seed `20260811`; the CSV
  records the method. A one-document smoke run has a degenerate interval and
  is not evidence for ranking models.
