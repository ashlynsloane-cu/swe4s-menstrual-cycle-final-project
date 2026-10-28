import pandas as pd

from mcphases.features import (
    CORE_SIGNALS,
    calculate_phase_means,
    build_cycle_signature,
    calculate_centered_phase_means,
    calculate_cycle_means,
    calculate_phase_means,
    build_all_cycle_signatures
)


def make_test_cycle():
    """Create a simple menstrual cycle for testing."""
    return pd.DataFrame(
        {
            "phase": [
                "Menstrual",
                "Menstrual",
                "Follicular",
                "Follicular",
                "Fertility",
                "Fertility",
                "Luteal",
                "Luteal",
            ],
            "resting_heart_rate": [60, 62, 64, 66, 68, 70, 72, 74],
            "hrv_rmssd": [40, 42, 38, 40, 36, 38, 34, 36],
            "nightly_temperature": [
                36.0,
                36.2,
                36.1,
                36.3,
                36.4,
                36.6,
                36.7,
                36.9,
            ],
            "respiratory_rate": [
                14.0,
                14.2,
                14.1,
                14.3,
                14.4,
                14.6,
                14.7,
                14.9,
            ],
        }
    )


def test_calculate_phase_means():
    cycle_data = make_test_cycle()

    phase_means = calculate_phase_means(cycle_data)

    assert phase_means.loc["Menstrual", "resting_heart_rate"] == 61
    assert phase_means.loc["Follicular", "resting_heart_rate"] == 65


def test_calculate_cycle_means():
    cycle_data = make_test_cycle()

    cycle_means = calculate_cycle_means(cycle_data)

    assert cycle_means["resting_heart_rate"] == 67
    assert cycle_means["hrv_rmssd"] == 38


def test_calculate_centered_phase_means():
    cycle_data = make_test_cycle()

    centered = calculate_centered_phase_means(cycle_data)

    assert centered.loc["Menstrual", "resting_heart_rate"] == -6
    assert centered.loc["Luteal", "resting_heart_rate"] == 6


def test_build_cycle_signature():
    cycle_data = make_test_cycle()

    signature = build_cycle_signature(cycle_data)

    assert len(signature) == 16
    assert signature["resting_heart_rate_menstrual_centered"] == -6
    assert signature["resting_heart_rate_luteal_centered"] == 6


def test_build_all_cycle_signatures():
    cycle1 = make_test_cycle()
    cycle1["id"] = 1
    cycle1["study_interval"] = 2022
    cycle1["cycle_number"] = 1

    cycle2 = make_test_cycle()
    cycle2["id"] = 1
    cycle2["study_interval"] = 2022
    cycle2["cycle_number"] = 2

    eligible_data = pd.concat(
        [cycle1, cycle2],
        ignore_index=True,
    )

    signatures = build_all_cycle_signatures(eligible_data)

    assert len(signatures) == 2
    assert set(signatures["cycle_number"]) == {1, 2}
