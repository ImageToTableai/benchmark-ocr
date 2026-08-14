"""Shared JSONL result schema for benchmark runners.

Runners do not need to share a model adapter. They only need to emit one JSON
object per sample with these fields.
"""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PredictionRecord:
    """One line in results/<run_id>/<model>_predictions.jsonl."""

    run_id: str
    sample_id: str
    dataset_id: str
    model: str
    model_version: str
    text: str = ""
    fields: dict[str, Any] = field(default_factory=dict)
    tables: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float | None = None
    status: str = "ok"
    error: str | None = None
    raw_output_path: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)
