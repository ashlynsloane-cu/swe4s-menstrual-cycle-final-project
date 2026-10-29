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
    """Run Aim 1 and save one cycle signature."""
    daily = pd.DataFrame(
        {
            "id": [1, 1, 1, 1],
            "study_interval": [2022, 2022, 2022, 2022],
            "cycle_number": [1, 1, 1, 1],
            "phase": [
                "Menstrual",
                "Follicular",
                "Fertility",
                "Luteal",
            ],
            "resting_heart_rate": [60, 62, 64, 66],
            "hrv_rmssd": [40, 42, 44, 46],
            "nightly_temperature": [36.0, 36.1, 36.2, 36.3],
            "respiratory_rate": [14.0, 14.1, 14.2, 14.3],
        }
    )

    cycle_qc = pd.DataFrame(
        {
            "id": [1],
            "study_interval": [2022],
            "cycle_number": [1],
            "aim1_primary_eligible": [True],
        }
    )

    daily_path = tmp_path / "daily.csv"
    cycle_qc_path = tmp_path / "cycle_qc.csv"
    output_path = tmp_path / "cycle_signatures.csv"

    daily.to_csv(daily_path, index=False)
    cycle_qc.to_csv(cycle_qc_path, index=False)

    run_aim1(
        daily_path=daily_path,
        cycle_qc_path=cycle_qc_path,
        output_path=output_path,
    )

    assert output_path.exists()
    signatures = pd.read_csv(output_path)
    assert len(signatures) == 1
    assert signatures.shape[1] == 19
