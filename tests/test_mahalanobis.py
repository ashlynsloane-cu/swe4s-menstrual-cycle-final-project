"""Tests for regularized Mahalanobis Aim 1 sensitivity analysis."""

import pandas as pd
import pytest

from mcphases.mahalanobis import (
    calculate_mahalanobis_pairwise_distances,
    fit_regularized_precision,
    mahalanobis_signal_specific_reproducibility,
    regularized_mahalanobis_distance,
)


def test_fit_regularized_precision():
    signatures = pd.DataFrame(
        {
            "feature_a": [1.0, 1.2, 3.0, 3.2, 5.0, 5.2],
            "feature_b": [2.0, 2.3, 6.0, 6.4, 10.0, 10.5],
        }
    )

    means, scales, precision = fit_regularized_precision(
        signatures,
        ["feature_a", "feature_b"],
    )

    assert len(means) == 2
    assert len(scales) == 2
    assert precision.shape == (2, 2)
    assert (scales > 0).all()


def test_regularized_mahalanobis_zero_for_same_cycle():
    signatures = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "feature_b": [2.0, 3.0, 5.0, 8.0],
        }
    )

    feature_columns = [
        "feature_a",
        "feature_b",
    ]

    means, scales, precision = fit_regularized_precision(
        signatures,
        feature_columns,
    )

    cycle = signatures.iloc[0]

    distance = regularized_mahalanobis_distance(
        cycle,
        cycle,
        feature_columns,
        means,
        scales,
        precision,
    )

    assert distance == pytest.approx(0.0)


def test_mahalanobis_pairwise_distances():
    signatures = pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3],
            "study_interval": [2022] * 6,
            "cycle_number": [1, 2, 1, 2, 1, 2],
            "feature_a": [1.0, 1.1, 3.0, 3.1, 5.0, 5.2],
            "feature_b": [2.0, 2.2, 6.0, 6.1, 10.0, 10.3],
        }
    )

    result = calculate_mahalanobis_pairwise_distances(
        signatures
    )

    assert len(result) == 15
    assert (result["distance"] >= 0).all()
    assert result["distance"].notna().all()
    assert result["same_person"].sum() == 3

    assert set(result["distance_metric"]) == {
        "regularized_mahalanobis"
    }


def test_signal_specific_mahalanobis():
    signatures = pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3],
            "study_interval": [2022] * 6,
            "cycle_number": [1, 2, 1, 2, 1, 2],
            "resting_heart_rate_menstrual_centered": [
                -1.0,
                -1.1,
                0.2,
                0.1,
                1.1,
                1.0,
            ],
            "resting_heart_rate_luteal_centered": [
                1.0,
                1.1,
                -0.2,
                -0.1,
                -1.1,
                -1.0,
            ],
            "hrv_rmssd_menstrual_centered": [
                -2.0,
                -2.1,
                0.4,
                0.3,
                2.2,
                2.1,
            ],
            "hrv_rmssd_luteal_centered": [
                2.0,
                2.1,
                -0.4,
                -0.3,
                -2.2,
                -2.1,
            ],
            "nightly_temperature_menstrual_centered": [
                -0.2,
                -0.3,
                0.0,
                0.1,
                0.3,
                0.2,
            ],
            "nightly_temperature_luteal_centered": [
                0.2,
                0.3,
                0.0,
                -0.1,
                -0.3,
                -0.2,
            ],
            "respiratory_rate_menstrual_centered": [
                -0.5,
                -0.6,
                0.0,
                0.1,
                0.6,
                0.5,
            ],
            "respiratory_rate_luteal_centered": [
                0.5,
                0.6,
                0.0,
                -0.1,
                -0.6,
                -0.5,
            ],
        }
    )

    result = mahalanobis_signal_specific_reproducibility(
        signatures,
        n_permutations=10,
        random_state=42,
    )

    assert len(result) == 5

    assert set(result["signal"]) == {
        "resting_heart_rate",
        "hrv_rmssd",
        "nightly_temperature",
        "respiratory_rate",
        "all_signals",
    }

    assert result["p_value"].between(
        0.0,
        1.0,
    ).all()

    assert set(result["distance_metric"]) == {
        "regularized_mahalanobis"
    }
