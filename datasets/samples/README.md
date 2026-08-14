# Sample Lists

Fixed JSONL sample lists live here — one line per scored sample, with the image
hash and ground truth needed by the evaluators.

Example row (note `image_path` is relative to
`$BENCHMARK_PERSIST/datasets/processed/`):

```json
{"sample_id":"sroie_test_X51005757222","dataset_id":"sroie_2019","source_repo":"jsdnrs/ICDAR2019-SROIE","source_split":"test","source_key":"X51005757222","image_path":"sroie_2019/images/sroie_test_X51005757222.png","gt_text":"...","gt_fields":{"total":"123.45"},"doc_type":"receipt","language":"en"}
```
