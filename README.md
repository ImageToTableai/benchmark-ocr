# benchmark-ocr

Reproducible open-source OCR benchmark on public receipt datasets. Measures
five OCR/document-parsing models against fixed test splits of SROIE 2019 and
CORD v2, reporting CER/WER, field-extraction F1, latency, and cost.

Every result is produced under a frozen protocol and traced to a versioned run
artifact. No product, vendor, or commercial claim is made — this is a neutral
third-party benchmark.

## What's in scope

**Datasets** (fixed test splits only):

| Dataset | Split | Samples | Language |
|---------|-------|--------:|----------|
| SROIE 2019 (ICDAR 2019) | test | 361 | English receipts |
| CORD v2 (Clova AI) | test | 100 | Indonesian receipts |

**Models** (out-of-the-box, no fine-tuning):

| Model | Version | Type |
|-------|---------|------|
| Tesseract | 5.3.4 | Classic OCR engine |
| PaddleOCR | 3.7.0 | Deep-learning OCR |
| EasyOCR | 1.7.2 | Deep-learning OCR |
| docTR | v1.0.1 | Neural OCR |
| Docling | 2.119.0 | Document parser |

**Metrics**: CER, WER, field-value F1 (regex postprocessed), success rate,
p50/p95 latency, wall-clock pages/min, cost per 1,000 pages — all with bootstrap
95% confidence intervals.

## Directory layout

```text
datasets/registry.yaml      # dataset metadata: source, license, GT type
datasets/samples/*.jsonl    # fixed sample lists (image hashes + ground truth)
scripts/run_<model>.py      # one runner per model
eval/                       # CER/WER, field metrics, bootstrap CI, metrics CSV
shared/                     # result schema + run-contract validation
server/                     # run_model.sh, install_model_env.sh, env.sh
reports/                    # frozen protocol + artifact quality audit
deps/                       # per-model Python requirements
tests/                      # unit tests for the evaluation pipeline
```

## Quick start

### 1. Clone and set the storage root

```bash
git clone https://github.com/ImageToTableai/benchmark-ocr.git
cd benchmark-ocr
```

Set `BENCHMARK_PERSIST` to a writable directory where datasets, model caches,
environments, and results will live (a persistent volume on a GPU cloud, or a
local directory):

```bash
export BENCHMARK_PERSIST=/path/to/your/storage
source server/env.sh
```

### 2. Obtain the datasets

The sample lists in `datasets/samples/` reference images by relative path under
`$BENCHMARK_PERSIST/datasets/processed/`. Obtain the images from the public
sources recorded in `datasets/registry.yaml`:

- **SROIE 2019** — ICDAR 2019 Robust Reading Competition, Task 3
  (<https://rrc.cvc.uab.es/?ch=13>)
- **CORD v2** — Clova AI Research (<https://github.com/clovaai/cord>)

### 3. Install a model environment and run

```bash
bash server/install_model_env.sh tesseract
bash server/run_model.sh tesseract sroie 361 sroie-tesseract-v1
```

`run_model.sh` writes predictions, a `manifest.json`, and `performance.json`
under `$BENCHMARK_PERSIST/results/<run_id>/`. Compute metrics with:

```bash
python eval/build_metrics_csv.py \
  --run-id sroie-tesseract-v1 \
  --samples "$BENCHMARK_PERSIST/datasets/samples/sroie_test_361.jsonl" \
  --predictions "$BENCHMARK_PERSIST/results/sroie-tesseract-v1/tesseract_predictions.jsonl" \
  --run-tier official \
  --field-postprocessor sroie_receipt_regex \
  --out "$BENCHMARK_PERSIST/results/sroie-tesseract-v1/metrics.csv"
```

## Run tiers

Every run is one of three tiers:

- `smoke` — small run to verify install and output shape.
- `provisional` — exploratory; may use a non-final protocol.
- `official` — publishable. Requires `source_split=test` for every sample and a
  `manifest.json` with sample/code/model/runtime hashes. Only `official` runs
  are eligible for the leaderboard.

## Reproducibility

The protocol that governs every published number is
[`reports/receipt_v1_official_protocol.md`](reports/receipt_v1_official_protocol.md).
It defines the fixed sample files, the model set, measurement rules
(`warm_then_scored`, per-dataset warm-up files), and publication gates. The
artifact quality audit at
[`reports/artifact_quality_audit_2026-08-12.md`](reports/artifact_quality_audit_2026-08-12.md)
documents data-integrity checks over the released predictions.

## Citing

See [`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE).
