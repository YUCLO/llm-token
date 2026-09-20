from __future__ import annotations

import numpy as np
import pytest

from experiments.stats_utils import (
    bootstrap_mean_ci,
    distribution_summary,
    paired_bootstrap_difference,
)


def test_paired_bootstrap_reports_candidate_minus_baseline():
    result = paired_bootstrap_difference(
        np.asarray([2.0, 4.0, 6.0]),
        np.asarray([1.0, 2.0, 3.0]),
        samples=2_000,
        seed=7,
    )
    assert result["candidate_minus_baseline"] == pytest.approx(2.0)
    assert result["ci95_low"] <= 2.0 <= result["ci95_high"]


def test_bootstrap_mean_requires_analysis_units():
    with pytest.raises(ValueError, match="must not be empty"):
        bootstrap_mean_ci(
            np.asarray([]),
            samples=10,
            generator=np.random.default_rng(1),
        )


def test_distribution_summary_ignores_nonfinite_gamma_outliers():
    result = distribution_summary(np.asarray([1.0, 2.0, np.inf]))
    assert result["mean"] == pytest.approx(1.5)
    assert result["median"] == pytest.approx(1.5)
