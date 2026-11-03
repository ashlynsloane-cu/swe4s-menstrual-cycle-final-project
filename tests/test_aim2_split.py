import pandas as pd

from mcphases.aim2_split import (
    add_aim2_complete_case,
    build_primary_manifest,
    prepare_valid_cycles,
)


def make_daily():
    rows = []

    phases = [
        "Menstrual",
        "Follicular",
        "Fertility",
        "Luteal",
    ]

    for participant_id in [1, 2, 3]:
        for cycle_number in [1, 2, 3]:
            for offset, phase in enumerate(phases):
                rows.append(
                    {
                        "id": participant_id,
                        "study_interval": 2022,
                        "day_in_study": (
                            cycle_number * 10 + offset
                        ),
                        "cycle_number": cycle_number,
                        "phase": phase,
                        "resting_heart_rate": 60.0,
                        "hrv_rmssd": 50.0,
                        "nightly_temperature": 34.0,
                        "respiratory_rate": 15.0,
                    }
                )

    return pd.DataFrame(rows)


def make_cycle_qc():
    rows = []

    for participant_id in [1, 2, 3]:
        for cycle_number in [1, 2, 3]:
            rows.append(
                {
                    "id": participant_id,
                    "study_interval": 2022,
                    "cycle_number": cycle_number,
                    "cycle_start_day": cycle_number * 10,
                    "complete_cycle": True,
                    "valid_phase_order": True,
                }
            )

    return pd.DataFrame(rows)


def test_complete_case_requires_all_predictors():
    daily = make_daily()

    daily.loc[
        0,
        "hrv_rmssd",
    ] = pd.NA

    observed = add_aim2_complete_case(daily)

    assert not observed.loc[
        0,
        "aim2_complete_case",
    ]

    assert observed.loc[
        1,
        "aim2_complete_case",
    ]


def test_valid_cycles_are_chronologically_numbered():
    observed = prepare_valid_cycles(
        make_cycle_qc()
    )

    person = observed[
        observed["id"] == 1
    ]

    assert (
        person["target_cycle_index"].tolist()
        == [1, 2, 3]
    )


def test_population_training_excludes_target():
    manifest, _ = build_primary_manifest(
        make_daily(),
        make_cycle_qc(),
    )

    fold = manifest[
        manifest["target_id"] == 1
    ]

    population = fold[
        fold["split_role"] == "population_train"
    ]

    assert not (
        population["source_id"] == 1
    ).any()


def test_target_cycles_have_correct_roles():
    manifest, _ = build_primary_manifest(
        make_daily(),
        make_cycle_qc(),
    )

    fold = manifest[
        manifest["target_id"] == 1
    ]

    cycle_1 = fold[
        fold["split_role"]
        == "personal_calibration_cycle1"
    ]

    cycle_2 = fold[
        fold["split_role"]
        == "personal_calibration_cycle2"
    ]

    future = fold[
        fold["split_role"] == "future_test"
    ]

    assert set(
        cycle_1["target_cycle_index"]
    ) == {1}

    assert set(
        cycle_2["target_cycle_index"]
    ) == {2}

    assert set(
        future["target_cycle_index"]
    ) == {3}


def test_same_future_data_supports_all_model_comparisons():
    manifest, _ = build_primary_manifest(
        make_daily(),
        make_cycle_qc(),
    )

    fold = manifest[
        manifest["target_id"] == 1
    ]

    future = fold[
        fold["split_role"] == "future_test"
    ]

    assert len(future) == 4

    assert set(future["phase"]) == {
        "Menstrual",
        "Follicular",
        "Fertility",
        "Luteal",
    }


def test_no_duplicate_day_assignments_within_fold():
    manifest, _ = build_primary_manifest(
        make_daily(),
        make_cycle_qc(),
    )

    keys = [
        "target_id",
        "source_id",
        "study_interval",
        "day_in_study",
    ]

    assert not manifest.duplicated(keys).any()


def test_missing_first_cycle_data_blocks_matched_cohort():
    daily = make_daily()

    mask = (
        (daily["id"] == 1)
        & (daily["cycle_number"] == 1)
    )

    daily.loc[
        mask,
        "hrv_rmssd",
    ] = pd.NA

    _, summary = build_primary_manifest(
        daily,
        make_cycle_qc(),
    )

    participant = summary[
        summary["target_id"] == 1
    ].iloc[0]

    assert participant["cal1_days"] == 0
    assert not participant[
        "primary_matched_eligible"
    ]
