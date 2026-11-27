"""Tests for the Aim 1 Mahalanobis runner."""

from pathlib import Path

import numpy as np
import pandas as pd

import mcphases.run_aim1_mahalanobis as runner


def test_run_aim1_mahalanobis_writes_outputs(
    tmp_path,
    monkeypatch,
):
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

    monkeypatch.setattr(
        runner,
        "load_aim1_data",
        lambda: pd.DataFrame(),
    )

    monkeypatch.setattr(
        runner,
        "build_all_cycle_signatures",
        lambda _: signatures,
    )

    output_dir = tmp_path / "aim1_mahalanobis"

    runner.run_aim1_mahalanobis(
        output_dir=output_dir,
        n_permutations=10,
    )

    expected_files = [
        "cycle_signatures.csv",
        "pairwise_distances.csv",
        "signal_specific_reproducibility.csv",
        "permutation_null_distribution.csv",
    ]

    for filename in expected_files:
        assert (
            output_dir / filename
        ).exists()

    pairwise = pd.read_csv(
        output_dir / "pairwise_distances.csv"
    )

    assert not pairwise.empty
    assert np.isfinite(
        pairwise["distance"]
    ).all()

    assert set(
        pairwise["distance_metric"]
    ) == {
        "regularized_mahalanobis"
    }

    signal_results = pd.read_csv(
        output_dir
        / "signal_specific_reproducibility.csv"
    )

    assert len(signal_results) == 5

    null_distribution = pd.read_csv(
        output_dir
        / "permutation_null_distribution.csv"
    )

    assert len(null_distribution) == 10
