"""Cycle-signature similarity analysis for Aim 1."""

from itertools import combinations

import numpy as np
import pandas as pd


def select_repeated_cycle_cohort(signatures):
    """Keep participants with at least two eligible cycles."""
    counts = signatures.groupby("id").size()

    repeated_ids = counts[counts >= 2].index

    repeated_signatures = signatures[
        signatures["id"].isin(repeated_ids)
    ]

    return repeated_signatures


def build_cycle_pairs(signatures):
    """Build every unique pair of cycle signatures."""
    pairs = []

    for (index_a, cycle_a), (index_b, cycle_b) in combinations(
        signatures.iterrows(),
        2,
    ):
        pairs.append(
            {
                "id_a": cycle_a["id"],
                "cycle_number_a": cycle_a["cycle_number"],
                "id_b": cycle_b["id"],
                "cycle_number_b": cycle_b["cycle_number"],
                "same_person": cycle_a["id"] == cycle_b["id"],
            }
        )

    return pd.DataFrame(pairs)


def get_feature_columns(signatures):
    """Return physiological feature columns used for similarity."""
    identifier_columns = {
        "id",
        "study_interval",
        "cycle_number",
    }

    feature_columns = [
        column
        for column in signatures.columns
        if column not in identifier_columns
    ]

    return feature_columns


def calculate_feature_variances(signatures, feature_columns):
    """Calculate cohort-level variance for each physiological feature."""
    return signatures[feature_columns].var()


def standardized_euclidean_distance(
    cycle_a,
    cycle_b,
    feature_columns,
    feature_variances,
):
    """Calculate standardized Euclidean distance between two cycles."""
    differences = cycle_a[feature_columns] - cycle_b[feature_columns]

    scaled_squared_differences = (
        differences ** 2
    ) / feature_variances

    distance = scaled_squared_differences.sum() ** 0.5

    return distance


def calculate_pairwise_distances(
    signatures,
    feature_columns=None,
):
    """Calculate standardized Euclidean distance for every unique cycle."""
    complete = signatures.dropna().reset_index(drop=True)

    if feature_columns is None:
        feature_columns = get_feature_columns(complete)

    feature_variances = calculate_feature_variances(
        complete,
        feature_columns,
    )

    pairs = []

    for (index_a, cycle_a), (index_b, cycle_b) in combinations(
        complete.iterrows(),
        2,
    ):
        distance = standardized_euclidean_distance(
            cycle_a,
            cycle_b,
            feature_columns,
            feature_variances,
        )

        pairs.append(
            {
                "index_a": index_a,
                "index_b": index_b,
                "id_a": cycle_a["id"],
                "cycle_number_a": cycle_a["cycle_number"],
                "id_b": cycle_b["id"],
                "cycle_number_b": cycle_b["cycle_number"],
                "same_person": cycle_a["id"] == cycle_b["id"],
                "distance": distance,
            }
        )

    return pd.DataFrame(pairs)


def calculate_similarity_advantage(pairwise_distances):
    """Calculate the within-person similarity advantage."""
    same_person = pairwise_distances[
        pairwise_distances["same_person"]
    ]

    different_person = pairwise_distances[
        ~pairwise_distances["same_person"]
    ]

    same_mean = same_person["distance"].mean()
    different_mean = different_person["distance"].mean()

    similarity_advantage = different_mean - same_mean

    return similarity_advantage


def permutation_test(
    signatures,
    pairwise_distances,
    n_permutations=10000,
    random_state=42,
):
    """Perform a participant-label permutation test."""
    rng = np.random.default_rng(random_state)

    complete = signatures.dropna().reset_index(drop=True)

    observed_advantage = calculate_similarity_advantage(
        pairwise_distances
    )

    participant_ids = complete["id"].to_numpy()

    permuted_advantages = []

    for _ in range(n_permutations):
        shuffled_ids = rng.permutation(participant_ids)

        permuted_pairs = pairwise_distances.copy()

        permuted_pairs["same_person"] = (
            shuffled_ids[
                permuted_pairs["index_a"].to_numpy()
            ]
            == shuffled_ids[
                permuted_pairs["index_b"].to_numpy()
            ]
        )

        permuted_advantage = calculate_similarity_advantage(
            permuted_pairs
        )

        permuted_advantages.append(permuted_advantage)

    permuted_advantages = np.array(permuted_advantages)

    p_value = (
        np.sum(
            permuted_advantages >= observed_advantage
        )
        + 1
    ) / (n_permutations + 1)

    return observed_advantage, p_value, permuted_advantages


def get_signal_feature_columns(signatures, signal):
    """Return feature columns belonging to one physiological signal."""
    return [
        column
        for column in signatures.columns
        if column.startswith(signal)
    ]


def signal_specific_reproducibility(
    signatures,
    n_permutations=10000,
    random_state=42,
):
    """Compare reproducibility across individual physiological signals."""
    signals = (
        "resting_heart_rate",
        "hrv_rmssd",
        "nightly_temperature",
        "respiratory_rate",
    )

    results = []

    for signal in signals:
        feature_columns = get_signal_feature_columns(
            signatures,
            signal,
        )

        pairwise_distances = calculate_pairwise_distances(
            signatures,
            feature_columns=feature_columns,
        )

        observed_advantage, p_value, _ = permutation_test(
            signatures,
            pairwise_distances,
            n_permutations=n_permutations,
            random_state=random_state,
        )

        results.append(
            {
                "signal": signal,
                "similarity_advantage": observed_advantage,
                "p_value": p_value,
            }
        )

    all_pairwise_distances = calculate_pairwise_distances(
        signatures
    )

    observed_advantage, p_value, _ = permutation_test(
        signatures,
        all_pairwise_distances,
        n_permutations=n_permutations,
        random_state=random_state,
    )

    results.append(
        {
            "signal": "all_signals",
            "similarity_advantage": observed_advantage,
            "p_value": p_value,
        }
    )

    return pd.DataFrame(results)
