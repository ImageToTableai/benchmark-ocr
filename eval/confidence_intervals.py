"""Deterministic bootstrap confidence intervals for per-document metrics."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


def quantile(values: Sequence[float], probability: float) -> float:
    """Return the Hyndman-Fan type 7 quantile used by the metric protocol."""
    if not values:
        raise ValueError("quantile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1")

    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    resamples: int = 2000,
    seed: int = 20260811,
) -> tuple[float, float] | None:
    """Return a two-sided percentile bootstrap 95% CI for a document mean."""
    if not values or resamples < 1:
        return None

    observations = [float(value) for value in values]
    if len(observations) == 1:
        return observations[0], observations[0]

    generator = random.Random(seed)
    count = len(observations)
    bootstrap_means = [
        sum(observations[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(resamples)
    ]
    return quantile(bootstrap_means, 0.025), quantile(bootstrap_means, 0.975)
