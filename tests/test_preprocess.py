import pandas as pd

from mcphases.preprocess import (
    assign_cycles,
    merge_signals,
    normalize_phase,
)


def test_normalize_phase():
    phases = pd.Series([
        "menstrual",
        " Follicular ",
        "fertile",
        "LUTEAL",
    ])
    observed = normalize_phase(phases).tolist()
    expected = [
        "Menstrual",
        "Follicular",
        "Fertility",
        "Luteal",
    ]
    assert observed == expected


def test_assign_cycles_detects_repeated_cycles():
    dataframe = pd.DataFrame(
        {
            "id": ["P01"] * 8,
            "study_interval": [1] * 8,
            "day_in_study": list(range(1, 9)),
            "phase": [
                "Menstrual",
                "Follicular",
                "Fertility",
                "Luteal",
                "Menstrual",
                "Follicular",
                "Fertility",
                "Luteal",
            ],
        }
    )

    observed = assign_cycles(dataframe)

    assert observed["cycle_number"].tolist() == [
        1, 1, 1, 1, 2, 2, 2, 2
    ]
    assert observed["complete_cycle"].all()
    assert observed["valid_phase_order"].all()


def test_merge_signals_keeps_master_days():
    daily = pd.DataFrame(
        {
            "id": ["P01", "P01"],
            "day_in_study": [1, 2],
        }
    )
    rhr = pd.DataFrame(
        {
            "id": ["P01"],
            "day_in_study": [2],
            "resting_heart_rate": [61.0],
        }
    )

    observed = merge_signals(daily, [rhr])

    assert len(observed) == 2
    assert pd.isna(observed.loc[0, "resting_heart_rate"])
    assert observed.loc[1, "resting_heart_rate"] == 61.0
