"""Audit one-cycle and two-cycle calibration designs for Aim 2."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PHASES = (
    "Menstrual",
    "Follicular",
    "Fertility",
    "Luteal",
)

PREDICTORS = (
    "resting_heart_rate",
    "hrv_rmssd",
    "nightly_temperature",
    "respiratory_rate",
)

CYCLE_KEYS = (
    "id",
    "study_interval",
    "cycle_number",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Audit chronological personal-calibration windows "
            "for Aim 2."
        )
    )

    parser.add_argument(
        "--daily",
        type=Path,
        default=Path(
            "results/preprocessed/"
            "mcphases_daily_master.csv"
        ),
    )

    parser.add_argument(
        "--cycle-qc",
        type=Path,
        default=Path(
            "results/preprocessed/"
            "mcphases_cycle_qc.csv"
        ),
    )

    return parser.parse_args()


def prepare_data(
    daily: pd.DataFrame,
    cycle_qc: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare complete-case days and valid biological cycles."""

    daily = daily.copy()
    cycle_qc = cycle_qc.copy()

    daily["aim2_complete_case"] = (
        daily["phase"].isin(PHASES)
        & daily[list(PREDICTORS)].notna().all(axis=1)
    )

    valid_cycles = cycle_qc[
        cycle_qc["complete_cycle"]
        & cycle_qc["valid_phase_order"]
    ].copy()

    valid_cycles = valid_cycles.sort_values(
        [
            "id",
            "study_interval",
            "cycle_start_day",
            "cycle_number",
        ]
    ).reset_index(drop=True)

    valid_cycles["chronological_cycle"] = (
        valid_cycles.groupby("id").cumcount() + 1
    )

    return daily, valid_cycles


def summarize_window(
    daily: pd.DataFrame,
    valid_cycles: pd.DataFrame,
    n_calibration_cycles: int,
) -> pd.DataFrame:
    """Summarize calibration and future-test data by participant."""

    records = []

    for participant_id, participant_cycles in (
        valid_cycles.groupby("id")
    ):
        participant_cycles = participant_cycles.sort_values(
            [
                "study_interval",
                "cycle_start_day",
                "cycle_number",
            ]
        )

        if len(participant_cycles) <= n_calibration_cycles:
            continue

        calibration_cycles = participant_cycles.iloc[
            :n_calibration_cycles
        ]

        test_cycles = participant_cycles.iloc[
            n_calibration_cycles:
        ]

        calibration_keys = calibration_cycles[
            list(CYCLE_KEYS)
        ]

        test_keys = test_cycles[
            list(CYCLE_KEYS)
        ]

        participant_days = daily[
            daily["id"] == participant_id
        ].copy()

        calibration = participant_days.merge(
            calibration_keys,
            on=list(CYCLE_KEYS),
            how="inner",
        )

        test = participant_days.merge(
            test_keys,
            on=list(CYCLE_KEYS),
            how="inner",
        )

        calibration = calibration[
            calibration["aim2_complete_case"]
        ]

        test = test[
            test["aim2_complete_case"]
        ]

        calibration_counts = (
            calibration["phase"]
            .value_counts()
            .reindex(PHASES, fill_value=0)
        )

        test_counts = (
            test["phase"]
            .value_counts()
            .reindex(PHASES, fill_value=0)
        )

        record = {
            "id": participant_id,
            "n_calibration_cycles": n_calibration_cycles,
            "n_total_valid_cycles": len(participant_cycles),
            "n_test_cycles": len(test_cycles),
            "calibration_total_days": len(calibration),
            "calibration_n_phases": int(
                (calibration_counts > 0).sum()
            ),
            "test_total_days": len(test),
            "test_n_phases": int(
                (test_counts > 0).sum()
            ),
        }

        for phase in PHASES:
            label = phase.lower()

            record[
                f"calibration_{label}_days"
            ] = int(calibration_counts[phase])

            record[
                f"test_{label}_days"
            ] = int(test_counts[phase])

        records.append(record)

    return pd.DataFrame(records)


def print_window_summary(
    audit: pd.DataFrame,
    n_cycles: int,
) -> None:
    """Print summary for one calibration design."""

    print("\n" + "=" * 78)
    print(
        f"{n_cycles}-CYCLE PERSONAL CALIBRATION"
    )
    print("=" * 78)

    print(
        "\nParticipants with enough biological cycles:",
        len(audit),
    )

    if audit.empty:
        return

    print("\nCalibration complete-case days:")
    print(
        audit["calibration_total_days"].describe(
            percentiles=[
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
            ]
        )
    )

    print("\nFuture test complete-case days:")
    print(
        audit["test_total_days"].describe(
            percentiles=[
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
            ]
        )
    )

    print("\nCalibration phase representation:")
    print(
        audit["calibration_n_phases"]
        .value_counts()
        .sort_index()
    )

    print("\nFuture-test phase representation:")
    print(
        audit["test_n_phases"]
        .value_counts()
        .sort_index()
    )

    has_calibration = (
        audit["calibration_total_days"] > 0
    )

    has_test = (
        audit["test_total_days"] > 0
    )

    hybrid_possible = (
        has_calibration
        & has_test
    )

    personal_two_class = (
        (audit["calibration_n_phases"] >= 2)
        & has_test
    )

    personal_four_class = (
        (audit["calibration_n_phases"] == 4)
        & has_test
    )

    print("\nModel-feasibility counts:")

    print(
        "Population-only testable:",
        int(has_test.sum()),
    )

    print(
        "Hybrid calibration possible:",
        int(hybrid_possible.sum()),
    )

    print(
        "Personal-only has >=2 training phases:",
        int(personal_two_class.sum()),
    )

    print(
        "Personal-only has all 4 training phases:",
        int(personal_four_class.sum()),
    )

    print(
        "\nHybrid participants with at least "
        "N complete-case calibration days:"
    )

    for threshold in [1, 5, 10, 20, 30, 40]:
        eligible = (
            (audit["calibration_total_days"] >= threshold)
            & has_test
        )

        print(
            f">={threshold:2d} days: "
            f"{int(eligible.sum())}"
        )

    print("\nPer-participant details:")

    columns = [
        "id",
        "n_total_valid_cycles",
        "n_test_cycles",
        "calibration_total_days",
        "calibration_n_phases",
        "calibration_menstrual_days",
        "calibration_follicular_days",
        "calibration_fertility_days",
        "calibration_luteal_days",
        "test_total_days",
        "test_n_phases",
    ]

    print(
        audit[columns]
        .sort_values(
            [
                "calibration_total_days",
                "id",
            ]
        )
        .to_string(index=False)
    )


def summarize_longitudinal(
    daily: pd.DataFrame,
    valid_cycles: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize 2022 calibration and 2024 testing."""

    records = []

    for participant_id, participant_cycles in (
        valid_cycles.groupby("id")
    ):
        cycles_2022 = participant_cycles[
            participant_cycles["study_interval"] == 2022
        ]

        cycles_2024 = participant_cycles[
            participant_cycles["study_interval"] == 2024
        ]

        if cycles_2022.empty or cycles_2024.empty:
            continue

        participant_days = daily[
            daily["id"] == participant_id
        ].copy()

        calibration = participant_days[
            participant_days["study_interval"] == 2022
        ]

        test = participant_days[
            participant_days["study_interval"] == 2024
        ]

        valid_2022_keys = cycles_2022[
            list(CYCLE_KEYS)
        ]

        valid_2024_keys = cycles_2024[
            list(CYCLE_KEYS)
        ]

        calibration = calibration.merge(
            valid_2022_keys,
            on=list(CYCLE_KEYS),
            how="inner",
        )

        test = test.merge(
            valid_2024_keys,
            on=list(CYCLE_KEYS),
            how="inner",
        )

        calibration = calibration[
            calibration["aim2_complete_case"]
        ]

        test = test[
            test["aim2_complete_case"]
        ]

        calibration_counts = (
            calibration["phase"]
            .value_counts()
            .reindex(PHASES, fill_value=0)
        )

        test_counts = (
            test["phase"]
            .value_counts()
            .reindex(PHASES, fill_value=0)
        )

        records.append(
            {
                "id": participant_id,
                "n_2022_cycles": len(cycles_2022),
                "n_2024_cycles": len(cycles_2024),
                "calibration_days_2022": len(calibration),
                "calibration_phases_2022": int(
                    (calibration_counts > 0).sum()
                ),
                "test_days_2024": len(test),
                "test_phases_2024": int(
                    (test_counts > 0).sum()
                ),
            }
        )

    return pd.DataFrame(records)


def print_longitudinal_summary(
    audit: pd.DataFrame,
) -> None:
    """Print 2022-to-2024 audit."""

    print("\n" + "=" * 78)
    print("LONGITUDINAL: 2022 CALIBRATION -> 2024 TEST")
    print("=" * 78)

    print(
        "\nParticipants with valid cycles "
        "in both intervals:",
        len(audit),
    )

    if audit.empty:
        return

    usable = audit[
        (audit["calibration_days_2022"] > 0)
        & (audit["test_days_2024"] > 0)
    ]

    print(
        "Participants with usable 2022 calibration "
        "and 2024 test data:",
        len(usable),
    )

    print(
        "Participants with all 4 phases observed "
        "in 2022 calibration:",
        int(
            (
                usable["calibration_phases_2022"] == 4
            ).sum()
        ),
    )

    print(
        "Participants with all 4 phases observed "
        "in 2024 test data:",
        int(
            (
                usable["test_phases_2024"] == 4
            ).sum()
        ),
    )

    print("\nPer-participant longitudinal details:")

    print(
        audit.sort_values("id").to_string(index=False)
    )


def main() -> None:
    """Run calibration-window audits."""

    args = parse_args()

    daily = pd.read_csv(args.daily)
    cycle_qc = pd.read_csv(args.cycle_qc)

    daily, valid_cycles = prepare_data(
        daily,
        cycle_qc,
    )

    print("=" * 78)
    print("AIM 2 ONE-CYCLE VS TWO-CYCLE CALIBRATION AUDIT")
    print("=" * 78)

    print(
        "\nComplete valid biological cycles:",
        len(valid_cycles),
    )

    print(
        "Participants:",
        valid_cycles["id"].nunique(),
    )

    for n_cycles in (1, 2):
        audit = summarize_window(
            daily,
            valid_cycles,
            n_calibration_cycles=n_cycles,
        )

        print_window_summary(
            audit,
            n_cycles,
        )

    longitudinal = summarize_longitudinal(
        daily,
        valid_cycles,
    )

    print_longitudinal_summary(
        longitudinal,
    )


if __name__ == "__main__":
    main()
