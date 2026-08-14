# Datasets

This directory is the home for lightweight benchmark inputs.

Keep tracked files small:

- `registry.yaml` records dataset metadata, source, license, and benchmark
  suitability.
- `samples/*.jsonl` records the fixed sample lists used by runners (sample IDs,
  image hashes, and ground-truth text/fields).

Do not commit raw images, downloaded archives, processed datasets, or large GT
dumps. Those belong in ignored directories such as `datasets/raw/`,
`datasets/processed/`, and `datasets/downloads/`.

## Sample list format

Each line of a sample JSONL is one JSON object. The `image_path` field is
relative to `$BENCHMARK_PERSIST/datasets/processed/`, and `source_repo` +
`source_key` uniquely identify the original image in the public dataset. The
remaining fields (`gt_text`, `gt_fields`, `source_split`, `doc_type`,
`language`) carry the ground truth and metadata needed by the evaluators.
