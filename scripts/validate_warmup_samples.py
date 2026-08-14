#!/usr/bin/env python3
"""Validate that unscored warm-up samples do not overlap a scored run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.run_contract import read_jsonl, validate_warmup_samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored-samples", required=True)
    parser.add_argument("--warmup-samples", required=True)
    args = parser.parse_args()
    result = validate_warmup_samples(read_jsonl(Path(args.scored_samples)), read_jsonl(Path(args.warmup_samples)))
    print(result["warmup_sample_count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
