"""Audit matched one-cycle and two-cycle calibration for Aim 2."""

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

KEYS = [
    "id",
    "study_interval",
    "cycle_number",
]


def count_phase_days(data):
    """Count complete-case observations in each phase."""

    counts = (
        data["phase"]
        .value_counts()
        .reindex(PHASES, fill_value=0)
    )

    return {
        phase: int(counts[phase])
        for phase in PHASES
    }


def select_cycle_days(
    daily,
    cycles,
):
    """Select complete-case days belonging to specified cycles."""

    if cycles.empty:
        return daily.iloc[0:0].copy()

    selected = daily.merge(
        cycles[KEYS],
        on=KEYS,
        how="inner",
    )

    return selected[
        selected["aim2_complete_case"]
    ].copy()


def main():
    """Run the matched calibration audit."""

    daily = pd.read_csv(
        Path(
            "results/preprocessed/"
            "mcphases_daily_master.csv"
        )
    )

    qc = pd.read_csv(
        Path(
            "results/preprocessed/"
            "mcphases_cycle_qc.csv"
        )
    )

    daily["aim2_complete_case"] = (
        daily["phase"].isin(PHASES)
        & daily[list(PREDICTORS)]
        .notna()
        .all(axis=1)
    )

    valid = qc[
        qc["complete_cycle"]
        & qc["valid_phase_order"]
    ].copy()

    valid = valid.sort_values(
        [
            "id",
            "study_interval",
            "cycle_start_day",
            "cycle_number",
        ]
    )

    records = []

    for participant_id, cycles in valid.groupby("id"):
        cycles = cycles.sort_values(
            [
                "study_interval",
                "cycle_start_day",
                "cycle_number",
            ]
        ).reset_index(drop=True)

        if len(cycles) < 3:
            continue

        person = daily[
            daily["id"] == participant_id
        ].copy()

        cycle_1 = cycles.iloc[[0]]
        cycles_1_2 = cycles.iloc[:2]
        future_cycles = cycles.iloc[2:]

        cal_1 = select_cycle_days(
            person,
            cycle_1,
        )

        cal_2 = select_cycle_days(
            person,
            cycles_1_2,
        )

        test = select_cycle_days(
            person,
            future_cycles,
        )

        phase_1 = count_phase_days(cal_1)
        phase_2 = count_phase_days(cal_2)
        phase_test = count_phase_days(test)

        record = {
            "id": participant_id,
            "n_valid_cycles": len(cycles),
            "n_test_cycles": len(future_cycles),
            "cal1_days": len(cal_1),
            "cal2_days": len(cal_2),
            "test_days": len(test),
            "cal1_n_phases": sum(
                value > 0
                for value in phase_1.values()
            ),
            "cal2_n_phases": sum(
                value > 0
                for value in phase_2.values()
            ),
            "test_n_phases": sum(
                value > 0
                for value in phase_test.values()
            ),
        }

        for phase in PHASES:
            label = phase.lower()

            record[f"cal1_{label}"] = phase_1[phase]
            record[f"cal2_{label}"] = phase_2[phase]
            record[f"test_{label}"] = phase_test[phase]

        records.append(record)

    audit = pd.DataFrame(records)

    audit["population_testable"] = (
        audit["test_days"] > 0
    )

    audit["hybrid_1_testable"] = (
        (audit["cal1_days"] > 0)
        & audit["population_testable"]
    )

    audit["hybrid_2_testable"] = (
        (audit["cal2_days"] > 0)
        & audit["population_testable"]
    )

    audit["matched_hybrid_testable"] = (
        audit["hybrid_1_testable"]
        & audit["hybrid_2_testable"]
    )

    audit["personal_1_four_class"] = (
        (audit["cal1_n_phases"] == 4)
        & audit["population_testable"]
    )

    audit["personal_2_four_class"] = (
        (audit["cal2_n_phases"] == 4)
        & audit["population_testable"]
    )

    print("=" * 76)
    print("AIM 2 MATCHED CALIBRATION AUDIT")
    print("=" * 76)

    print(
        "\nParticipants with >=3 complete valid cycles:",
        len(audit),
    )

    print(
        "Participants with future complete-case test data:",
        int(audit["population_testable"].sum()),
    )

    print(
        "1-cycle hybrid feasible:",
        int(audit["hybrid_1_testable"].sum()),
    )

    print(
        "2-cycle hybrid feasible:",
        int(audit["hybrid_2_testable"].sum()),
    )

    print(
        "BOTH hybrid models feasible on same test set:",
        int(audit["matched_hybrid_testable"].sum()),
    )

    print(
        "\nPersonal-only 1-cycle four-class feasible:",
        int(audit["personal_1_four_class"].sum()),
    )

    print(
        "Personal-only 2-cycle four-class feasible:",
        int(audit["personal_2_four_class"].sum()),
    )

    matched = audit[
        audit["matched_hybrid_testable"]
    ].copy()

    print("\nMATCHED PRIMARY COHORT")
    print("----------------------")

    print("Participants:", len(matched))

    print(
        "Future test cycles:",
        int(matched["n_test_cycles"].sum()),
    )

    print(
        "Future test days:",
        int(matched["test_days"].sum()),
    )

    print("\nCalibration days:")

    print(
        matched[
            [
                "cal1_days",
                "cal2_days",
            ]
        ].describe(
            percentiles=[
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
            ]
        )
    )

    print("\nFuture test phase counts:")

    test_columns = [
        f"test_{phase.lower()}"
        for phase in PHASES
    ]

    print(
        matched[test_columns].sum()
    )

    print(
        "\nNumber of phases represented "
        "in calibration:"
    )

    print("\nAfter one cycle:")
    print(
        matched["cal1_n_phases"]
        .value_counts()
        .sort_index()
    )

    print("\nAfter two cycles:")
    print(
        matched["cal2_n_phases"]
        .value_counts()
        .sort_index()
    )

    print("\nPer-participant matched cohort:")

    columns = [
        "id",
        "n_valid_cycles",
        "n_test_cycles",
        "cal1_days",
        "cal1_n_phases",
        "cal2_days",
        "cal2_n_phases",
        "test_days",
        "test_n_phases",
    ]

    print(
        matched[columns]
        .sort_values("id")
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
