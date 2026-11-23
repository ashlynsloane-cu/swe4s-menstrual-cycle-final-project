"""Statistical behavior checks for the Aim 1 permutation null model."""

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from mcphases.permutation_validation import permute_participant_ids
from mcphases.similarity import (
    calculate_pairwise_distances,
    permutation_test,
)


def make_synthetic_signatures(seed, stable_person_signal):
    """Create cycles with a known within-person effect or the null."""
    rng = np.random.default_rng(seed)
    n_people, cycles_per_person = 8, 3
    ids = np.repeat(np.arange(n_people), cycles_per_person)

    effect = (
        np.repeat(np.linspace(-5, 5, n_people), cycles_per_person)
        if stable_person_signal
        else np.zeros(len(ids))
    )
    return pd.DataFrame(
        {
            "id": ids,
            "study_interval": np.full(len(ids), 2022),
            "cycle_number": np.tile(
                np.arange(1, cycles_per_person + 1), n_people
            ),
            "feature_a": effect + rng.normal(0, 1, len(ids)),
            "feature_b": -effect + rng.normal(0, 1, len(ids)),
        }
    )


def test_interval_stratified_permutation_preserves_id_multiplicities():
    ids = np.array([1, 1, 2, 2, 3, 3, 3, 4, 4])
    intervals = np.array([2022] * 4 + [2024] * 5)
    rng = np.random.default_rng(34)

    for _ in range(20):
        shuffled = permute_participant_ids(ids, rng, intervals)
        for interval in (2022, 2024):
            mask = intervals == interval
            assert Counter(shuffled[mask]) == Counter(ids[mask])


def test_stratification_requires_matching_interval_vector():
    with pytest.raises(ValueError, match="match"):
        permute_participant_ids(
            [1, 2, 3], np.random.default_rng(1), [2022]
        )


def test_permutation_reproducible_and_rejects_zero_iterations():
    signatures = make_synthetic_signatures(1, True)
    pairs = calculate_pairwise_distances(signatures)

    first = permutation_test(
        signatures, pairs, n_permutations=99, random_state=18
    )
    second = permutation_test(
        signatures, pairs, n_permutations=99, random_state=18
    )
    assert first[0] == second[0]
    assert first[1] == second[1]
    np.testing.assert_array_equal(first[2], second[2])

    with pytest.raises(ValueError, match="positive integer"):
        permutation_test(signatures, pairs, n_permutations=0)


def test_null_false_positive_rate_is_not_systematically_inflated():
    """Fixed-seed simulation checks inferential behavior, not runtime.

    This regression guard is intentionally loose: 3 or fewer discoveries
    in 12 independent null datasets at alpha=0.05. It is not a proof of
    exact type-I error control on real, dependent menstrual-cycle data.
    """
    significant = 0
    for seed in range(12):
        signatures = make_synthetic_signatures(seed, False)
        pairs = calculate_pairwise_distances(signatures)
        _, p_value, _ = permutation_test(
            signatures, pairs, n_permutations=99, random_state=23
        )
        significant += p_value < 0.05

    assert significant <= 3


def test_high_power_for_strong_stable_person_effect():
    """At least 7/8 fixed strong-effect datasets must be detectable."""
    significant = 0
    for seed in range(8):
        signatures = make_synthetic_signatures(seed, True)
        pairs = calculate_pairwise_distances(signatures)
        _, p_value, _ = permutation_test(
            signatures, pairs, n_permutations=99, random_state=23
        )
        significant += p_value < 0.05

    assert significant >= 7


def test_stratified_path_on_unequal_intervals():
    signatures = make_synthetic_signatures(10, True)
    signatures.loc[signatures.index[-8:], "study_interval"] = 2024
    pairs = calculate_pairwise_distances(signatures)

    advantage, p_value, null = permutation_test(
        signatures,
        pairs,
        n_permutations=31,
        random_state=42,
        stratify_by_study_interval=True,
    )
    assert np.isfinite(advantage)
    assert np.isfinite(null).all()
    assert 0 < p_value <= 1


def test_unstratified_null_exactly_matches_frozen_reference():
    """Regression-check the old permutation draws and one-sided p-value."""
    signatures = make_synthetic_signatures(27, True)
    pairs = calculate_pairwise_distances(signatures)
    observed, p_value, null = permutation_test(
        signatures, pairs, n_permutations=31, random_state=42
    )

    # Literal reference implementation of the pre-review shuffle.
    rng = np.random.default_rng(42)
    participant_ids = signatures["id"].to_numpy()
    advantages = []
    for _ in range(31):
        shuffled_ids = rng.permutation(participant_ids)
        permuted = pairs.copy()
        permuted["same_person"] = (
            shuffled_ids[permuted["index_a"].to_numpy()]
            == shuffled_ids[permuted["index_b"].to_numpy()]
        )
        within = permuted.loc[permuted["same_person"], "distance"].mean()
        between = permuted.loc[~permuted["same_person"], "distance"].mean()
        advantages.append(between - within)

    np.testing.assert_array_equal(null, np.asarray(advantages))
    assert p_value == (1 + np.sum(null >= observed)) / 32


def test_single_participant_is_rejected_for_inference():
    signatures = make_synthetic_signatures(1, True)
    signatures["id"] = 1
    pairs = calculate_pairwise_distances(signatures)
    with pytest.raises(ValueError, match="within-person and between-person"):
        permutation_test(signatures, pairs, n_permutations=10)


def test_constant_feature_is_rejected():
    signatures = make_synthetic_signatures(1, True)
    signatures["feature_a"] = 1.0
    with pytest.raises(ValueError, match="positive finite variance"):
        calculate_pairwise_distances(signatures)
