#!/usr/bin/env python3
"""Run Surya 2 through an OpenAI-compatible vLLM server for field extraction."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.result_schema import PredictionRecord


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def import_openai():
    try:
        from openai import OpenAI
    except Exception as exc:
        raise SystemExit("OpenAI client is required in the selected environment") from exc
    return OpenAI


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.as_posix())[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def schema_template(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: schema_template(item) for key, item in value.items()}
    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            return [schema_template(value[0])]
        return []
    return ""


def dataset_prompt(dataset_id: str, gt_fields: dict[str, Any]) -> str:
    template = schema_template(gt_fields)
    template_json = json.dumps(template, ensure_ascii=False, indent=2)
    if dataset_id == "sroie_2019":
        guidance = (
            "Extract only the receipt fields company, date, address, and total. "
            "Keep values exactly as they appear on the document when possible."
        )
    elif dataset_id == "cord_v2":
        guidance = (
            "Extract the receipt fields into the provided nested structure. "
            "Use menu rows when present, and keep amounts/counts as strings."
        )
    else:
        guidance = "Extract the document fields into the provided JSON structure."
    return (
        f"{guidance}\n"
        "Return only valid JSON. Do not include markdown fences or commentary.\n"
        "If a field is missing, use an empty string. If a list is missing, use an empty list.\n"
        "Use this exact JSON shape:\n"
        f"{template_json}"
    )


def extract_json_object(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty model response")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", stripped, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    start = min([index for index in (stripped.find("{"), stripped.find("[")) if index != -1], default=-1)
    if start == -1:
        raise ValueError("no JSON object found in model response")

    for end in range(len(stripped), start, -1):
        candidate = stripped[start:end]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("unable to parse model response as JSON")


def coerce_to_shape(schema: Any, value: Any) -> Any:
    if isinstance(schema, dict):
        source = value if isinstance(value, dict) else {}
        return {key: coerce_to_shape(subschema, source.get(key)) for key, subschema in schema.items()}
    if isinstance(schema, list):
        if not schema:
            return []
        item_schema = schema[0]
        if isinstance(value, list):
            return [coerce_to_shape(item_schema, item) for item in value]
        if value in (None, "", {}):
            return []
        if isinstance(value, dict):
            return [coerce_to_shape(item_schema, value)]
        return []
    if value is None:
        return ""
    return str(value).strip()


def serialize_response(response: Any) -> str:
    if hasattr(response, "model_dump_json"):
        return response.model_dump_json(indent=2)
    return json.dumps(response, ensure_ascii=False, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--warmup-samples", default=None, help="Unscored samples used only to warm runtime caches")
    parser.add_argument("--base-url", default=os.environ.get("SURYA_EXTRACT_BASE_URL", os.environ.get("SURYA_INFERENCE_URL", "http://localhost:8000/v1")))
    parser.add_argument("--model", default=os.environ.get("SURYA_EXTRACT_MODEL", "datalab-to/surya-ocr-2"))
    parser.add_argument("--api-key", default=os.environ.get("SURYA_EXTRACT_API_KEY", os.environ.get("OPENAI_API_KEY", os.environ.get("VLLM_API_KEY", "EMPTY"))))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("SURYA_EXTRACT_MAX_TOKENS", "1024")))
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    OpenAI = import_openai()
    client = OpenAI(api_key=args.api_key, base_url=args.base_url, timeout=args.timeout)
    samples = read_jsonl(Path(args.samples))
    out_path = Path(args.out)
    raw_dir = out_path.parent / "raw" / "surya2_extract"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    def infer_fields(sample: dict[str, Any]) -> tuple[dict[str, Any], str]:
        gt_fields = sample.get("gt_fields", {}) if isinstance(sample.get("gt_fields"), dict) else {}
        prompt = dataset_prompt(sample.get("dataset_id", ""), gt_fields)
        image_url = image_data_url(Path(sample["image_path"]))
        start = time.perf_counter()
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            max_tokens=args.max_tokens,
            temperature=0.0,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        content = response.choices[0].message.content or ""
        parsed = extract_json_object(content)
        fields = coerce_to_shape(schema_template(gt_fields), parsed)
        return {
            "fields": fields,
            "latency_ms": latency_ms,
            "content": content,
            "response": response,
        }, prompt

    if args.warmup_samples:
        for sample in read_jsonl(Path(args.warmup_samples)):
            infer_fields(sample)

    with out_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            try:
                result, prompt = infer_fields(sample)
                raw_path = raw_dir / f"{sample['sample_id']}.json"
                raw_path.write_text(
                    json.dumps(
                        {
                            "prompt": prompt,
                            "content": result["content"],
                            "response": json.loads(serialize_response(result["response"])),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                record = PredictionRecord(
                    run_id=args.run_id,
                    sample_id=sample["sample_id"],
                    dataset_id=sample["dataset_id"],
                    model="surya2_extract",
                    model_version=args.model,
                    fields=result["fields"],
                    latency_ms=result["latency_ms"],
                    status="ok",
                    raw_output_path=raw_path.as_posix(),
                )
            except Exception as exc:
                record = PredictionRecord(
                    run_id=args.run_id,
                    sample_id=sample["sample_id"],
                    dataset_id=sample["dataset_id"],
                    model="surya2_extract",
                    model_version=args.model,
                    status="error",
                    error=str(exc),
                )
            handle.write(json.dumps(record.to_json_dict(), ensure_ascii=False) + "\n")
            handle.flush()

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
