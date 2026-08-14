"""Small text metrics for the local OCR benchmark."""

from __future__ import annotations

import re

try:
    from rapidfuzz.distance import Levenshtein
except ImportError:  # pragma: no cover - optional accelerator
    Levenshtein = None


def normalize_text(value: str) -> str:
    value = value.lower()
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    if Levenshtein is not None:
        return int(Levenshtein.distance(left, right))

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def cer(prediction: str, ground_truth: str) -> float:
    gt = normalize_text(ground_truth)
    pred = normalize_text(prediction)
    if not gt:
        return 0.0 if not pred else 1.0
    return edit_distance(pred, gt) / len(gt)


def wer(prediction: str, ground_truth: str) -> float:
    gt_words = normalize_text(ground_truth).split()
    pred_words = normalize_text(prediction).split()
    if not gt_words:
        return 0.0 if not pred_words else 1.0
    return edit_distance(pred_words, gt_words) / len(gt_words)
