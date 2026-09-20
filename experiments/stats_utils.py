"""Small statistical helpers shared by paired experiment comparisons."""

from __future__ import annotations

import numpy as np


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    samples: int,
    generator: np.random.Generator,
) -> tuple[float, float, float]:
    """Return mean and percentile 95% CI from an IID array of analysis units.

    Callers are responsible for first reducing correlated observations to the
    intended independent unit, such as one mean difference per speaker.
    """

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError("values must not be empty")
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not np.isfinite(array).all():
        raise ValueError("values must be finite")
    draws = generator.choice(array, size=(samples, array.size), replace=True).mean(axis=1)
    return (
        float(array.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def paired_bootstrap_difference(
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    """Bootstrap the mean paired difference ``candidate - baseline``."""

    candidate_array = np.asarray(candidate, dtype=np.float64).reshape(-1)
    baseline_array = np.asarray(baseline, dtype=np.float64).reshape(-1)
    if candidate_array.shape != baseline_array.shape:
        raise ValueError("candidate and baseline must have identical shapes")
    estimate, low, high = bootstrap_mean_ci(
        candidate_array - baseline_array,
        samples=samples,
        generator=np.random.default_rng(seed),
    )
    return {
        "candidate_minus_baseline": estimate,
        "ci95_low": low,
        "ci95_high": high,
    }


def distribution_summary(values: np.ndarray) -> dict[str, float]:
    """Summarize finite values while ignoring infinities from near-zero radii."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError("values contain no finite observations")
    return {
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "q10": float(np.quantile(finite, 0.10)),
        "q90": float(np.quantile(finite, 0.90)),
    }
