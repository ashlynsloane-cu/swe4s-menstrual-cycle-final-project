"""Regularized Mahalanobis sensitivity analysis for Aim 1."""

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from mcphases.similarity import (
    calculate_similarity_advantage,
    get_feature_columns,
    get_signal_feature_columns,
    permutation_test,
)


SIGNALS = (
    "resting_heart_rate",
    "hrv_rmssd",
    "nightly_temperature",
    "respiratory_rate",
)


def fit_regularized_precision(signatures, feature_columns):
    """Fit a Ledoit-Wolf precision matrix after feature standardization."""
    features = signatures[feature_columns].astype(float)

    feature_means = features.mean()
    feature_scales = features.std(ddof=0)

    if (
        not np.isfinite(feature_scales).all()
        or (feature_scales <= 0).any()
    ):
        raise ValueError(
            "Features must have positive finite scale for "
            "regularized Mahalanobis distance."
        )

    standardized = (
        features - feature_means
    ) / feature_scales

    estimator = LedoitWolf().fit(
        standardized.to_numpy()
    )

    return (
        feature_means,
        feature_scales,
        estimator.precision_,
    )


def regularized_mahalanobis_distance(
    cycle_a,
    cycle_b,
    feature_columns,
    feature_means,
    feature_scales,
    precision,
):
    """Calculate regularized Mahalanobis distance between two cycles."""
    standardized_a = (
        cycle_a[feature_columns].astype(float)
        - feature_means
    ) / feature_scales

    standardized_b = (
        cycle_b[feature_columns].astype(float)
        - feature_means
    ) / feature_scales

    difference = (
        standardized_a.to_numpy()
        - standardized_b.to_numpy()
    )

    squared_distance = (
        difference
        @ precision
        @ difference
    )

    return float(
        np.sqrt(max(squared_distance, 0.0))
    )


def calculate_mahalanobis_pairwise_distances(
    signatures,
    feature_columns=None,
):
    """Calculate regularized Mahalanobis distance for each cycle pair."""
    complete = signatures.dropna().reset_index(drop=True)

    if feature_columns is None:
        feature_columns = get_feature_columns(
            complete
        )

    (
        feature_means,
        feature_scales,
        precision,
    ) = fit_regularized_precision(
        complete,
        feature_columns,
    )

    pairs = []

    for (index_a, cycle_a), (
        index_b,
        cycle_b,
    ) in combinations(
        complete.iterrows(),
        2,
    ):
        distance = regularized_mahalanobis_distance(
            cycle_a,
            cycle_b,
            feature_columns,
            feature_means,
            feature_scales,
            precision,
        )

        pairs.append(
            {
                "index_a": index_a,
                "index_b": index_b,
                "id_a": cycle_a["id"],
                "cycle_number_a": cycle_a[
                    "cycle_number"
                ],
                "id_b": cycle_b["id"],
                "cycle_number_b": cycle_b[
                    "cycle_number"
                ],
                "same_person": (
                    cycle_a["id"]
                    == cycle_b["id"]
                ),
                "distance": distance,
                "distance_metric": (
                    "regularized_mahalanobis"
                ),
            }
        )

    return pd.DataFrame(pairs)


def mahalanobis_signal_specific_reproducibility(
    signatures,
    n_permutations=10000,
    random_state=42,
    stratify_by_study_interval=False,
):
    """Repeat Aim 1 inference for each signal using Mahalanobis distance."""
    results = []

    for signal in SIGNALS:
        feature_columns = get_signal_feature_columns(
            signatures,
            signal,
        )

        pairwise_distances = (
            calculate_mahalanobis_pairwise_distances(
                signatures,
                feature_columns=feature_columns,
            )
        )

        (
            observed_advantage,
            p_value,
            _,
        ) = permutation_test(
            signatures,
            pairwise_distances,
            n_permutations=n_permutations,
            random_state=random_state,
            stratify_by_study_interval=(
                stratify_by_study_interval
            ),
        )

        results.append(
            {
                "signal": signal,
                "similarity_advantage": (
                    observed_advantage
                ),
                "p_value": p_value,
                "distance_metric": (
                    "regularized_mahalanobis"
                ),
            }
        )

    all_pairwise = (
        calculate_mahalanobis_pairwise_distances(
            signatures
        )
    )

    (
        observed_advantage,
        p_value,
        _,
    ) = permutation_test(
        signatures,
        all_pairwise,
        n_permutations=n_permutations,
        random_state=random_state,
        stratify_by_study_interval=(
            stratify_by_study_interval
        ),
    )

    results.append(
        {
            "signal": "all_signals",
            "similarity_advantage": (
                observed_advantage
            ),
            "p_value": p_value,
            "distance_metric": (
                "regularized_mahalanobis"
            ),
        }
    )

    return pd.DataFrame(results)
