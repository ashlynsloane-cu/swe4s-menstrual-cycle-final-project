import pandas as pd

from mcphases.preprocess import (
    SignalSpec,
    assign_cycles,
    collapse_daily_rows,
    load_signal,
    merge_signals,
    normalize_phase,
)


def test_normalize_phase():
    phases = pd.Series(
        [
            "menstrual",
            " Follicular ",
            "fertile",
            "LUTEAL",
        ]
    )

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
            "study_interval": [2022] * 8,
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
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
    ]

    assert observed["complete_cycle"].all()
    assert observed["valid_phase_order"].all()


def test_collapse_daily_rows_keeps_intervals_separate():
    dataframe = pd.DataFrame(
        {
            "id": ["P01", "P01"],
            "study_interval": [2022, 2024],
            "day_in_study": [1, 1],
            "phase": ["Menstrual", "Menstrual"],
            "lh": [2.0, 8.0],
        }
    )

    observed = collapse_daily_rows(dataframe)

    assert len(observed) == 2
    assert set(observed["study_interval"]) == {2022, 2024}


def test_merge_signals_keeps_master_days():
    daily = pd.DataFrame(
        {
            "id": ["P01", "P01"],
            "study_interval": [2022, 2022],
            "day_in_study": [1, 2],
        }
    )

    rhr = pd.DataFrame(
        {
            "id": ["P01"],
            "study_interval": [2022],
            "day_in_study": [2],
            "resting_heart_rate": [61.0],
        }
    )

    observed = merge_signals(
        daily,
        [rhr],
    )

    assert len(observed) == 2
    assert pd.isna(
        observed.loc[
            0,
            "resting_heart_rate",
        ]
    )
    assert (
        observed.loc[
            1,
            "resting_heart_rate",
        ]
        == 61.0
    )


def test_merge_signals_does_not_cross_intervals():
    daily = pd.DataFrame(
        {
            "id": ["P01", "P01"],
            "study_interval": [2022, 2024],
            "day_in_study": [1, 1],
        }
    )

    rhr = pd.DataFrame(
        {
            "id": ["P01"],
            "study_interval": [2024],
            "day_in_study": [1],
            "resting_heart_rate": [70.0],
        }
    )

    observed = merge_signals(
        daily,
        [rhr],
    )

    value_2022 = observed.loc[
        observed["study_interval"] == 2022,
        "resting_heart_rate",
    ].iloc[0]

    value_2024 = observed.loc[
        observed["study_interval"] == 2024,
        "resting_heart_rate",
    ].iloc[0]

    assert pd.isna(value_2022)
    assert value_2024 == 70.0


def test_load_signal_removes_nonpositive_values(
    tmp_path,
):
    dataframe = pd.DataFrame(
        {
            "id": [
                "P01",
                "P01",
                "P01",
                "P01",
            ],
            "study_interval": [
                2022,
                2022,
                2022,
                2022,
            ],
            "day_in_study": [
                1,
                1,
                1,
                1,
            ],
            "rmssd": [
                0.0,
                20.0,
                40.0,
                60.0,
            ],
        }
    )

    path = tmp_path / "hrv.csv"

    dataframe.to_csv(
        path,
        index=False,
    )

    spec = SignalSpec(
        name="hrv_rmssd",
        filename="hrv.csv",
        day_column="day_in_study",
        value_column="rmssd",
        aggregation="median",
        positive_only=True,
    )

    observed = load_signal(
        tmp_path,
        spec,
        required=True,
    )

    assert len(observed) == 1
    assert (
        observed.loc[
            0,
            "hrv_rmssd",
        ]
        == 40.0
    )


def test_respiratory_rate_removes_zero(
    tmp_path,
):
    dataframe = pd.DataFrame(
        {
            "id": [
                "P01",
                "P01",
                "P01",
            ],
            "study_interval": [
                2022,
                2022,
                2022,
            ],
            "day_in_study": [
                1,
                1,
                2,
            ],
            "full_sleep_breathing_rate": [
                0.0,
                15.5,
                16.0,
            ],
        }
    )

    path = tmp_path / "respiratory.csv"

    dataframe.to_csv(
        path,
        index=False,
    )

    spec = SignalSpec(
        name="respiratory_rate",
        filename="respiratory.csv",
        day_column="day_in_study",
        value_column=(
            "full_sleep_breathing_rate"
        ),
        aggregation="median",
        positive_only=True,
    )

    observed = load_signal(
        tmp_path,
        spec,
        required=True,
    )

    assert len(observed) == 2

    assert (
        observed.loc[
            0,
            "respiratory_rate",
        ]
        == 15.5
    )

    assert (
        observed.loc[
            1,
            "respiratory_rate",
        ]
        == 16.0
    )
