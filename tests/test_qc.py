import pandas as pd

from mcphases.qc import (
    add_cycle_analysis_flags,
    add_participant_analysis_flags,
)


def make_cycle_summary():
    return pd.DataFrame(
        {
            "id": [
                "P01",
                "P01",
                "P01",
                "P02",
                "P02",
            ],
            "study_interval": [
                2022,
                2022,
                2024,
                2022,
                2022,
            ],
            "cycle_number": [
                1,
                2,
                1,
                1,
                2,
            ],
            "cycle_start_day": [
                1,
                30,
                900,
                1,
                30,
            ],
            "complete_cycle": [
                True,
                True,
                True,
                True,
                True,
            ],
            "valid_phase_order": [
                True,
                True,
                True,
                True,
                True,
            ],
            "resting_heart_rate_coverage": [
                0.70,
                0.90,
                0.90,
                0.90,
                0.90,
            ],
            "hrv_rmssd_coverage": [
                0.70,
                0.90,
                0.90,
                0.90,
                0.90,
            ],
            "nightly_temperature_coverage": [
                0.70,
                0.90,
                0.90,
                0.90,
                0.90,
            ],
            "respiratory_rate_coverage": [
                0.70,
                0.90,
                0.90,
                0.90,
                0.90,
            ],
        }
    )


def test_cycle_flags_use_inclusive_70_percent():
    observed = add_cycle_analysis_flags(
        make_cycle_summary()
    )

    first = observed[
        (observed["id"] == "P01")
        & (observed["study_interval"] == 2022)
        & (observed["cycle_number"] == 1)
    ].iloc[0]

    assert first["passes_core_70pct"]
    assert not first["passes_core_80pct"]
    assert first["aim1_primary_eligible"]
    assert not first["aim1_sensitivity_80pct"]


def test_last_cycle_is_right_censored():
    observed = add_cycle_analysis_flags(
        make_cycle_summary()
    )

    last_cycle = observed[
        (observed["id"] == "P01")
        & (observed["study_interval"] == 2022)
        & (observed["cycle_number"] == 2)
    ].iloc[0]

    assert not last_cycle["closed_by_next_menses"]
    assert not last_cycle["aim1_primary_eligible"]


def test_next_cycle_closes_previous_cycle():
    observed = add_cycle_analysis_flags(
        make_cycle_summary()
    )

    first_cycle = observed[
        (observed["id"] == "P02")
        & (observed["cycle_number"] == 1)
    ].iloc[0]

    assert first_cycle["closed_by_next_menses"]
    assert first_cycle["aim1_primary_eligible"]


def test_invalid_phase_order_is_not_eligible():
    data = make_cycle_summary()

    data.loc[
        (data["id"] == "P02")
        & (data["cycle_number"] == 1),
        "valid_phase_order",
    ] = False

    observed = add_cycle_analysis_flags(data)

    cycle = observed[
        (observed["id"] == "P02")
        & (observed["cycle_number"] == 1)
    ].iloc[0]

    assert cycle["closed_by_next_menses"]
    assert cycle["passes_core_70pct"]
    assert not cycle["aim1_primary_eligible"]


def test_participant_flags_use_eligible_cycles():
    cycles = pd.DataFrame(
        {
            "id": [
                "P01",
                "P01",
                "P01",
                "P02",
                "P02",
            ],
            "study_interval": [
                2022,
                2022,
                2024,
                2022,
                2024,
            ],
            "aim1_primary_eligible": [
                True,
                True,
                True,
                True,
                False,
            ],
        }
    )

    participants = pd.DataFrame(
        {
            "id": [
                "P01",
                "P02",
                "P03",
            ]
        }
    )

    observed = add_participant_analysis_flags(
        participants,
        cycles,
    )

    p01 = observed[
        observed["id"] == "P01"
    ].iloc[0]

    p02 = observed[
        observed["id"] == "P02"
    ].iloc[0]

    p03 = observed[
        observed["id"] == "P03"
    ].iloc[0]

    assert p01["n_aim1_primary_cycles"] == 3
    assert p01["aim1_repeated_cycle_candidate"]
    assert p01["long_term_candidate"]

    assert p02["n_aim1_primary_cycles"] == 1
    assert not p02["aim1_repeated_cycle_candidate"]
    assert not p02["long_term_candidate"]

    assert p03["n_aim1_primary_cycles"] == 0
    assert not p03["aim1_repeated_cycle_candidate"]
    assert not p03["long_term_candidate"]
