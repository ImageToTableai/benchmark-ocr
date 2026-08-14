#!/usr/bin/env python3
"""Write reproducibility metadata for one benchmark run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.run_contract import (
    read_jsonl,
    sha256_file,
    validate_prediction_contract,
    validate_sample_contract,
)


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def command_output_lines(command: list[str]) -> list[str]:
    output = command_output(command)
    return output.splitlines() if output else []


def current_git_metadata() -> dict[str, object]:
    git_commit = command_output(["git", "rev-parse", "HEAD"])
    git_status = command_output(["git", "status", "--porcelain"])
    diff_summary = command_output(["git", "diff", "--stat"])
    diff_name_status = command_output(["git", "diff", "--name-status"])
    return {
        "git_commit": git_commit,
        "git_dirty": bool(git_status) if git_commit is not None else None,
        "git_status_porcelain": git_status,
        "git_status_sha256": hashlib.sha256((git_status or "").encode("utf-8")).hexdigest() if git_status is not None else None,
        "git_diff_stat": diff_summary,
        "git_diff_name_status": diff_name_status,
        "git_diff_name_status_sha256": hashlib.sha256((diff_name_status or "").encode("utf-8")).hexdigest()
        if diff_name_status is not None
        else None,
    }


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def package_metadata() -> dict[str, object]:
    freeze_lines = command_output_lines([sys.executable, "-m", "pip", "freeze", "--all"])
    packages = {
        "python_executable": sys.executable,
        "pip_freeze_sha256": hashlib.sha256("\n".join(freeze_lines).encode("utf-8")).hexdigest() if freeze_lines else None,
        "pip_freeze_package_count": len(freeze_lines),
        "key_versions": {
            name: package_version(name)
            for name in (
                "torch",
                "torchvision",
                "torchaudio",
                "paddleocr",
                "paddlepaddle",
                "paddlepaddle-gpu",
                "paddlex",
                "easyocr",
                "python-doctr",
                "docling",
                "surya-ocr",
                "opencv-python",
                "pillow",
                "numpy",
            )
        },
    }
    return packages


def runtime_metadata() -> dict[str, object]:
    runtime: dict[str, object] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "machine": platform.machine(),
        "processor": platform.processor(),
        "gpu": command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "gpu_query": command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total,power.limit,pci.bus_id",
                "--format=csv,noheader",
            ]
        ),
        "nvidia_smi_version": command_output(["nvidia-smi", "--query", "--display=COMPUTE"]),
    }
    try:
        import torch

        runtime["torch_version"] = torch.__version__
        runtime["torch_cuda_version"] = torch.version.cuda
        runtime["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        runtime["torch_version"] = None
    return runtime


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-tier", choices=("smoke", "provisional", "official"), default="smoke")
    parser.add_argument("--expected-split", default=None)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--evaluator", default="eval/build_metrics_csv.py")
    parser.add_argument("--dataset-registry", default="datasets/registry.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--env-python", default=None)
    parser.add_argument("--field-postprocessor", default="none")
    parser.add_argument("--performance-file", default=None)
    parser.add_argument("--gpu-label", default=None)
    parser.add_argument("--gpu-provider", default=None)
    parser.add_argument("--gpu-hourly-usd", type=float, default=None)
    parser.add_argument("--price-recorded-at-utc", default=None)
    parser.add_argument("--command", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    samples_path = Path(args.samples)
    predictions_path = Path(args.predictions)
    runner_path = Path(args.runner)
    evaluator_path = Path(args.evaluator)
    registry_path = Path(args.dataset_registry)
    performance_path = Path(args.performance_file) if args.performance_file else None
    samples = read_jsonl(samples_path)
    predictions = read_jsonl(predictions_path)
    sample_info = validate_sample_contract(samples, args.run_tier, args.expected_split)
    prediction_info = validate_prediction_contract(predictions, samples, args.run_id, args.run_tier)
    if performance_path:
        try:
            performance = json.loads(performance_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Cannot read performance file {performance_path}: {exc}") from exc
        if not isinstance(performance, dict) or not isinstance(performance.get("run_wall_time_ms"), (int, float)):
            raise SystemExit("Performance file requires numeric run_wall_time_ms")
    else:
        performance = None
    if args.gpu_hourly_usd is not None:
        if args.gpu_hourly_usd < 0:
            raise SystemExit("--gpu-hourly-usd must be non-negative")
        if not args.gpu_provider or not args.price_recorded_at_utc:
            raise SystemExit("Cost metadata requires --gpu-provider and --price-recorded-at-utc")
    source = current_git_metadata()
    packages = package_metadata()

    manifest = {
        "schema_version": 2,
        "run_id": args.run_id,
        "run_tier": args.run_tier,
        "expected_split": args.expected_split,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "data": {
            **sample_info,
            "samples_path": str(samples_path),
            "samples_sha256": sha256_file(samples_path),
            "dataset_registry_path": str(registry_path),
            "dataset_registry_sha256": sha256_file(registry_path) if registry_path.is_file() else None,
        },
        "runner": {
            "path": str(runner_path),
            "sha256": sha256_file(runner_path),
            "command": args.command,
            "model": args.model,
            "env_python": args.env_python or sys.executable,
        },
        "predictions": {
            **prediction_info,
            "path": str(predictions_path),
            "sha256": sha256_file(predictions_path),
        },
        "evaluation": {
            "field_postprocessor": args.field_postprocessor,
            "evaluator_path": str(evaluator_path),
            "evaluator_sha256": sha256_file(evaluator_path) if evaluator_path.is_file() else None,
        },
        "performance": {
            "path": str(performance_path) if performance_path else None,
            "sha256": sha256_file(performance_path) if performance_path else None,
            "measurement_mode": performance.get("measurement_mode") if performance else None,
            "run_wall_time_ms": performance.get("run_wall_time_ms") if performance else None,
        },
        "cost": {
            "gpu_label": args.gpu_label,
            "gpu_provider": args.gpu_provider,
            "gpu_hourly_usd": args.gpu_hourly_usd,
            "price_recorded_at_utc": args.price_recorded_at_utc,
        },
        "source": source,
        "runtime": runtime_metadata(),
        "packages": packages,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
