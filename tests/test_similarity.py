import pandas as pd
import pytest


from mcphases.similarity import (
    build_cycle_pairs,
    calculate_similarity_advantage,
    get_feature_columns,
    permutation_test,
    select_repeated_cycle_cohort,
    calculate_feature_variances,
    standardized_euclidean_distance,
    calculate_pairwise_distances,
    calculate_similarity_advantage,
    get_signal_feature_columns,
    signal_specific_reproducibility,
)


def test_select_repeated_cycle_cohort():
    test = pd.DataFrame(
        {
            "id": [1, 1, 2, 3, 3, 3],
            "cycle_number": [1, 2, 1, 1, 2, 3],
        }
    )

    result = select_repeated_cycle_cohort(test)

    assert result["id"].nunique() == 2
    assert set(result["id"]) == {1, 3}


def test_build_cycle_pairs():
    test = pd.DataFrame(
        {
            "id": [1, 1, 2],
            "cycle_number": [1, 2, 1],
        }
    )

    result = build_cycle_pairs(test)

    assert len(result) == 3
    assert result["same_person"].sum() == 1


def test_get_feature_columns():
    test = pd.DataFrame(
        {
            "id": [1, 1, 2],
            "study_interval": [2022, 2022, 2022],
            "cycle_number": [1, 2, 1],
            "feature_a": [0.1, 0.2, 0.3],
            "feature_b": [1.0, 1.1, 1.2],
        }
    )
    result = get_feature_columns(test)

    assert result == ["feature_a", "feature_b"]


def test_calculate_feature_variances():
    test = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0],
            "feature_b": [10.0, 20.0, 30.0],
        }
    )

    feature_columns = ["feature_a", "feature_b"]

    result = calculate_feature_variances(
        test,
        feature_columns,
    )

    assert result["feature_a"] == 1.0
    assert result["feature_b"] == 100.0


def test_standardized_euclidean_distance():
    cycle_a = pd.Series(
        {
            "feature_a": 1.0,
            "feature_b": 2.0,
        }
    )

    cycle_b = pd.Series(
        {
            "feature_a": 2.0,
            "feature_b": 4.0,
        }
    )

    feature_columns = ["feature_a", "feature_b"]

    feature_variances = pd.Series(
        {
            "feature_a": 1.0,
            "feature_b": 1.0,
        }
    )
    result = standardized_euclidean_distance(
        cycle_a,
        cycle_b,
        feature_columns,
        feature_variances,
    )

    assert result == pytest.approx(5 ** 0.5)


def test_calculate_pairwaise_distances():
    test = pd.DataFrame(
        {
            "id": [1, 1, 2],
            "study_interval": [2022, 2022, 2022],
            "cycle_number": [1, 2, 1],
            "feature_a": [1.0, 2.0, 3.0],
            "feature_b": [2.0, 4.0, 6.0],
        }
    )

    result = calculate_pairwise_distances(test)

    assert len(result) == 3
    assert result["same_person"].sum() == 1
    assert "distance" in result.columns
    assert result["distance"].notna().all()


def test_calculate_similarity_advantage():
    test = pd.DataFrame(
        {
            "id_a": [1, 1, 2],
            "cycle_number_a": [1, 2, 1],
            "id_b": [1, 2, 2],
            "cycle_number_b": [2, 1, 1],
            "same_person": [True, False, True],
            "distance": [0.5, 1.0, 0.3],
        }
    )

    result = calculate_similarity_advantage(test)

    assert result == pytest.approx(0.6)


def test_permutation_test():
    signatures = pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "study_interval": [2022, 2022, 2022, 2022],
            "cycle_number": [1, 2, 1, 2],
            "feature_a": [1.0, 1.2, 3.0, 3.2],
            "feature_b": [2.0, 2.2, 6.0, 6.2],
        }
    )

    pairwise_distances = calculate_pairwise_distances(signatures)
    observed, p_value, null_distribution = permutation_test(
        signatures,
        pairwise_distances,
        n_permutations=10,
        random_state=42,
    )
    assert isinstance(observed, float)
    assert 0.0 <= p_value <= 1.0
    assert len(null_distribution) == 10


def test_get_signal_feature_columns():
    test = pd.DataFrame(
        {
            "id": [1],
            "resting_heart_rate_menstrual_centered": [1.0],
            "resting_heart_rate_luteal_centered": [2.0],
            "hrv_rmssd_menstrual_centered": [3.0],
        }
    )

    result = get_signal_feature_columns(
        test,
        "resting_heart_rate",
    )

    assert result == [
        "resting_heart_rate_menstrual_centered",
        "resting_heart_rate_luteal_centered",
    ]


def test_signal_specific_reproducibility():
    signatures = pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "study_interval": [2022, 2022, 2022, 2022],
            "cycle_number": [1, 2, 1, 2],
            "resting_heart_rate_menstrual_centered": [
                -1.0,
                -1.1,
                1.0,
                1.1,
            ],
            "resting_heart_rate_luteal_centered": [
                1.0,
                1.1,
                -1.0,
                -1.1,
            ],
            "hrv_rmssd_menstrual_centered": [
                -2.0,
                -2.1,
                2.0,
                2.1,
            ],
            "hrv_rmssd_luteal_centered": [
                2.0,
                2.1,
                -2.0,
                -2.1,
            ],
            "nightly_temperature_menstrual_centered": [
                -0.2,
                -0.3,
                0.2,
                0.3,
            ],
            "nightly_temperature_luteal_centered": [
                0.2,
                0.3,
                -0.2,
                -0.3,
            ],
            "respiratory_rate_menstrual_centered": [
                -0.5,
                -0.6,
                0.5,
                0.6,
            ],
            "respiratory_rate_luteal_centered": [
                0.5,
                0.6,
                -0.5,
                -0.6,
            ],
        }
    )

    result = signal_specific_reproducibility(
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
    assert result["p_value"].between(0.0, 1.0).all()
