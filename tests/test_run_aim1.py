import pandas as pd

from mcphases.run_aim1 import load_aim1_data, run_aim1


def test_load_aim1_data(tmp_path):
    """Retain only daily rows from Aim 1 eligible cycles."""
    daily = pd.DataFrame(
        {
            "id": [1, 1, 1, 1],
            "study_interval": [2022, 2022, 2022, 2022],
            "cycle_number": [1, 1, 2, 2],
            "phase": [
                "Menstrual",
                "Luteal",
                "Menstrual",
                "Luteal",
            ],
        }
    )

    cycle_qc = pd.DataFrame(
        {
            "id": [1, 1],
            "study_interval": [2022, 2022],
            "cycle_number": [1, 2],
            "aim1_primary_eligible": [True, False],
        }
    )

    daily_path = tmp_path / "daily.csv"
    cycle_qc_path = tmp_path / "cycle_qc.csv"

    daily.to_csv(daily_path, index=False)
    cycle_qc.to_csv(cycle_qc_path, index=False)

    eligible_data = load_aim1_data(
        daily_path=daily_path,
        cycle_qc_path=cycle_qc_path,
    )

    assert len(eligible_data) == 2
    assert set(eligible_data["cycle_number"]) == {1}


def test_run_aim1(tmp_path):
    """Run the full Aim 1 workflow and save expected outputs."""
    daily = pd.DataFrame(
        {
            "id": [
                1, 1, 1, 1,
                1, 1, 1, 1,
                2, 2, 2, 2,
                2, 2, 2, 2,
            ],
            "study_interval": [2022] * 16,
            "cycle_number": [
                1, 1, 1, 1,
                2, 2, 2, 2,
                1, 1, 1, 1,
                2, 2, 2, 2,
            ],
            "phase": [
                "Menstrual",
                "Follicular",
                "Fertility",
                "Luteal",
            ] * 4,
            "resting_heart_rate": [
                60, 62, 64, 66,
                61, 62, 65, 67,
                70, 73, 74, 76,
                69, 72, 75, 78,
            ],
            "hrv_rmssd": [
                40, 42, 44, 46,
                41, 43, 44, 48,
                35, 38, 39, 42,
                34, 36, 40, 43,
            ],
            "nightly_temperature": [
                36.0, 36.1, 36.2, 36.3,
                36.0, 36.2, 36.25, 36.4,
                35.9, 36.0, 36.3, 36.5,
                35.8, 36.1, 36.2, 36.6,
            ],
            "respiratory_rate": [
                14.0, 14.1, 14.2, 14.3,
                14.0, 14.2, 14.3, 14.5,
                15.0, 15.1, 15.4, 15.6,
                14.9, 15.2, 15.3, 15.7,
            ],
        }
    )

    cycle_qc = pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "study_interval": [2022, 2022, 2022, 2022],
            "cycle_number": [1, 2, 1, 2],
            "aim1_primary_eligible": [True, True, True, True],
        }
    )

    daily_path = tmp_path / "daily.csv"
    cycle_qc_path = tmp_path / "cycle_qc.csv"
    output_path = tmp_path / "cycle_signatures.csv"
    pairwise_output_path = tmp_path / "pairwise_distances.csv"
    signal_output_path = (
        tmp_path / "signal_specific_reproducibility.csv"
    )
    permutation_output_path = (
        tmp_path / "permutation_null_distribution.csv"
    )

    daily.to_csv(daily_path, index=False)
    cycle_qc.to_csv(cycle_qc_path, index=False)

    run_aim1(
        daily_path=daily_path,
        cycle_qc_path=cycle_qc_path,
        output_path=output_path,
        pairwise_output_path=pairwise_output_path,
        signal_output_path=signal_output_path,
        permutation_output_path=permutation_output_path,
        n_permutations=10,
    )

    assert output_path.exists()
    assert pairwise_output_path.exists()
    assert signal_output_path.exists()
    assert permutation_output_path.exists()

    signatures = pd.read_csv(output_path)
    pairwise = pd.read_csv(pairwise_output_path)
    signal_results = pd.read_csv(signal_output_path)
    permutation_results = pd.read_csv(permutation_output_path)

    assert len(signatures) == 4
    assert signatures.shape[1] == 19

    assert len(pairwise) == 6
    assert pairwise["same_person"].sum() == 2

    assert len(signal_results) == 5
    assert len(permutation_results) == 10
