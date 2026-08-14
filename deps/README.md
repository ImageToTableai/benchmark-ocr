# Model Dependencies

Each model runner gets its own optional dependency file:

```text
deps/tesseract.txt
deps/paddleocr.txt
deps/paddleocr_vl.txt
deps/unlimited_ocr.txt
deps/easyocr.txt
deps/doctr.txt
deps/docling.txt
deps/surya2.txt
```

`server/install_model_env.sh <model>` always installs the shared root
`requirements.txt` first, then installs `deps/<model>.txt` if it exists. The
environment is stored under `$BENCHMARK_ENVS/<model>` (defaults to
`$BENCHMARK_PERSIST/envs/<model>`; see `server/env.sh`).

Keep these files narrow. Do not put every OCR package into the shared
requirements file.

## Execution tracks

The benchmark is intentionally split into two execution tracks:

- `local_torch`: Tesseract, PaddleOCR, PaddleOCR-VL, EasyOCR, docTR, and
  Docling. These run in the current torch environment and may reuse the tested
  Torch/CUDA installation through their own model environments.
- `vllm_server`: Surya2 and Unlimited-OCR. These require a separate vLLM-capable
  image or an external OpenAI-compatible inference URL. Do not install vLLM into
  the current shared Torch environment just to unblock them.

Models that do not yet ship an official runner keep their dependency file
unclassified until the runner and backend requirements are verified. A model
that can optionally use vLLM is not automatically placed in the `vllm_server`
track.

For PaddleOCR runners, `server/install_model_env.sh` auto-detects the NVIDIA
CUDA version and installs `paddlepaddle-gpu` from the matching PaddlePaddle
wheel index before installing the remaining dependencies. Do not rely on PyPI's
default `paddlepaddle-gpu` package for GPU machines; it may resolve to a wheel for a
different CUDA line. Set `PADDLE_INDEX_URL` only when a machine image needs an
explicit override.

The official model set and its runner/dependency mapping are listed in
[`reports/receipt_v1_official_protocol.md`](../reports/receipt_v1_official_protocol.md).
