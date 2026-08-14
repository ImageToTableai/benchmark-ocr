# Benchmark Methodology Template

Use this template for benchmark report drafts after result artifacts exist.

## Scope

- Report:
- Document type:
- Languages:
- Datasets:
- Models:
- Run tier:
- Hardware/API runtime:

## Datasets

| Dataset | Split | Samples | Ground truth | License | Sample SHA256 | Audit |
|---------|-------|---------|--------------|---------|---------------|-------|
| | | | | | | |

State whether each dataset is real, synthetic, mixed, internal-only, or
research-only. Do not mix these categories in one public-core ranking.

## Models

| Model | Version | Runner | Environment/backend | Notes |
|-------|---------|--------|---------------------|-------|
| | | | | |

## Metrics

Text metrics:

- CER:
- WER:

Field metrics:

- Field-value accuracy:
- Field-value F1:
- Document-fields exact:
- Required-fields exact:
- Line-item F1:
- Line-item column diagnostics:
- JSON/schema validity:
- Hallucinated fields rate:
- Normalization rules:

Performance metrics:

- Per-page latency p50/p95:
- End-to-end wall time:
- Pages per minute:
- Cost per 1,000 pages:

## Reproducibility

- Result directory:
- `manifest.json`:
- `performance.json`:
- Predictions JSONL:
- `metrics.csv`:
- `field_details.csv`:
- Git commit:
- Price timestamp:

## Findings

Only write findings that are directly supported by frozen metrics and inspected
prediction examples.

## Limitations

- Unsupported fields or metrics:
- Dataset limitations:
- Model/backend limitations:
- Cost/latency caveats:

## Citation

```text
Benchmark OCR Team. "<Report title>." benchmark_ocr, <release date>.
```
