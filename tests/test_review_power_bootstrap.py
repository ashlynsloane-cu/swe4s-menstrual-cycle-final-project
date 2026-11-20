"""Larger synthetic data checks for power and resampling diversity."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "scripts" / "review"))
from statistical_diagnostics import (  # noqa: E402
    bootstrap_id_diagnostics,
    fast_permutation_test,
    simulate_power,
    synthetic_cycle_signatures,
    wilson_interval,
)


def test_25_and_50_participant_permutation_scenarios():
    """Compute/detect strong effects in larger cohorts."""
    for participants in (25, 50):
        ids, years, features = synthetic_cycle_signatures(
            participants=participants,
            cycles_per_person=4,
            stable_sd=4.0,
            seed=participants,
        )
        advantage, p = fast_permutation_test(
            ids, years, features, permutations=99, seed=14
        )
        assert advantage > 0
        assert p <= 0.05


def test_simulated_power_is_reproducible_and_bounded():
    args = dict(
        participants=25, cycles_per_person=4, stable_sd=1.0,
        simulations=4, permutations=49, seed=74,
    )
    result = simulate_power(**args)
    assert result == simulate_power(**args)
    assert 0 <= result["rejection_fraction"] <= 1
    assert 0 <= result["wilson_ci_95_low"]
    assert result["wilson_ci_95_low"] <= result["rejection_fraction"]
    assert result["rejection_fraction"] <= result["wilson_ci_95_high"] <= 1


def test_stratified_synthetic_null_is_finite_for_many_participants():
    ids, years, features = synthetic_cycle_signatures(
        participants=25, cycles_per_person=4, stable_sd=0, seed=36
    )
    statistic, p = fast_permutation_test(
        ids, years, features, permutations=49, stratified=True
    )
    assert np.isfinite(statistic)
    assert 0 < p <= 1


def test_profile_simulation_is_centered_by_signal():
    _, _, features = synthetic_cycle_signatures(
        participants=25, cycles_per_person=4, stable_sd=.5, seed=21
    )
    np.testing.assert_allclose(
        features.reshape(-1, 4, 4).mean(axis=-1), 0, atol=1e-12
    )


def test_bootstrap_diversity_matches_analytic_expectation():
    summary = bootstrap_id_diagnostics(range(25), draws=2000, seed=35)
    assert summary["participants"] == 25
    assert summary["unique_median"] < 25
    assert abs(
        summary["observed_mean_unique"]
        - summary["expected_unique_analytic"]
    ) < 1
    assert 1 < summary["effective_n_median"] < 25
    assert 0 <= summary[
        "probability_participant_sampled_at_least_10_times"
    ] <= 1


def test_invalid_bootstrap_and_wilson_inputs():
    with pytest.raises(ValueError):
        bootstrap_id_diagnostics([1])
    with pytest.raises(ValueError):
        wilson_interval(4, 3)


def test_wilson_zero_and_all_successes():
    assert wilson_interval(0, 50)[0] == pytest.approx(0)
    assert wilson_interval(50, 50)[1] == pytest.approx(1)


def test_fast_shuffle_matches_exact_aim1_source_when_available():
    """Check optimized vectorized code against the original source."""
    similarity = pytest.importorskip("mcphases.similarity")
    import pandas as pd

    ids, years, features = synthetic_cycle_signatures(
        participants=8, cycles_per_person=4, stable_sd=0.75, seed=23
    )
    signatures = pd.DataFrame(
        features, columns=[f"feature_{i}" for i in range(16)]
    )
    signatures.insert(0, "id", ids)
    signatures.insert(1, "study_interval", years)
    signatures.insert(2, "cycle_number", np.tile([1, 2, 3, 4], 8))
    pairs = similarity.calculate_pairwise_distances(signatures)
    reference = similarity.permutation_test(
        signatures, pairs, n_permutations=49, random_state=17
    )
    optimized = fast_permutation_test(
        ids, years, features, permutations=49, seed=17
    )
    assert optimized[0] == pytest.approx(reference[0], abs=1e-11)
    assert optimized[1] == reference[1]
