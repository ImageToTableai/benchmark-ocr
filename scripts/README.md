# Model Runners

One model gets one runner script.

Recommended shape:

```bash
source server/env.sh
ENV_PYTHON="$(bash server/install_model_env.sh paddleocr | tail -n 1)"
"$ENV_PYTHON" scripts/run_paddleocr.py \
  --run-id sroie-paddleocr-001 \
  --samples "$BENCHMARK_DATASETS/samples/sroie_test_361.jsonl" \
  --out "$BENCHMARK_RESULTS/sroie-paddleocr-001/paddleocr_predictions.jsonl"
```

Runner internals do not need to share an adapter. They only need to write the
standard prediction JSONL fields documented in `shared/result_schema.py`.

`create_run_manifest.py` is the companion script for a completed runner. It
validates the sample/prediction contract and writes the manifest required by an
official evaluation. Schema v2 also records runner command, runner/evaluator
hashes, git status hashes, GPU/driver metadata, Python executable, key package
versions, and a `pip freeze` hash. `server/run_model.sh` invokes it
automatically.
