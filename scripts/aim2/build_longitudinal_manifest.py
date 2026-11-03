"""Build the 2022-to-2024 longitudinal Aim 2 manifest."""

from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "results/preprocessed/"
    "mcphases_daily_complete_cycles.csv"
)

OUTPUT_DIR = Path(
    "results/aim2/longitudinal"
)

PHASES = [
    "Menstrual",
    "Follicular",
    "Fertility",
    "Luteal",
]

SIGNALS = [
    "resting_heart_rate",
    "hrv_rmssd",
    "nightly_temperature",
    "respiratory_rate",
]

DEVELOPMENT_ID = 22


def prepare_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Restrict to complete-case wearable observations."""

    mask = (
        data["phase"].isin(PHASES)
        & data[SIGNALS].notna().all(axis=1)
    )

    return data.loc[mask].copy()


def cycle_summary(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize each participant-interval-cycle."""

    grouped = (
        data.groupby(
            [
                "id",
                "study_interval",
                "cycle_number",
            ],
            as_index=False,
        )
        .agg(
            n_days=(
                "day_in_study",
                "nunique",
            ),
            n_phases=(
                "phase",
                "nunique",
            ),
        )
    )

    grouped["all_four_phases"] = (
        grouped["n_phases"] == 4
    )

    grouped = grouped.sort_values(
        [
            "id",
            "study_interval",
            "cycle_number",
        ]
    )

    grouped["interval_cycle_index"] = (
        grouped.groupby(
            [
                "id",
                "study_interval",
            ]
        )
        .cumcount()
        + 1
    )

    return grouped


def main() -> None:
    """Build longitudinal eligibility manifest."""

    data = pd.read_csv(
        INPUT_PATH
    )

    data = prepare_data(
        data
    )

    cycles = cycle_summary(
        data
    )

    rows = []

    participant_ids = sorted(
        set(
            cycles.loc[
                cycles["study_interval"] == 2022,
                "id",
            ]
        )
        & set(
            cycles.loc[
                cycles["study_interval"] == 2024,
                "id",
            ]
        )
    )

    for participant_id in participant_ids:
        person = cycles[
            cycles["id"] == participant_id
        ].copy()

        cycles_2022 = person[
            person["study_interval"] == 2022
        ].copy()

        cycles_2024 = person[
            person["study_interval"] == 2024
        ].copy()

        valid_2022 = (
            cycles_2022[
                cycles_2022["all_four_phases"]
            ]
            .sort_values("interval_cycle_index")
            .copy()
        )

        valid_2024 = (
            cycles_2024[
                cycles_2024["all_four_phases"]
            ]
            .sort_values("interval_cycle_index")
            .copy()
        )

        # Calibration is exactly the first two valid
        # 2022 cycles, matching the primary Aim 2 design.
        calibration = valid_2022.head(2)

        # All valid 2024 cycles are held-out longitudinal test.
        test = valid_2024.copy()

        eligible = (
            len(calibration) == 2
            and len(test) >= 1
        )

        calibration_cycle_numbers = (
            ",".join(
                calibration[
                    "cycle_number"
                ]
                .astype(int)
                .astype(str)
            )
        )

        test_cycle_numbers = (
            ",".join(
                test[
                    "cycle_number"
                ]
                .astype(int)
                .astype(str)
            )
        )

        cal_days = int(
            calibration["n_days"].sum()
        )

        test_days = int(
            test["n_days"].sum()
        )

        rows.append(
            {
                "target_id": int(
                    participant_id
                ),
                "eligible": eligible,
                "development_id": (
                    int(participant_id)
                    == DEVELOPMENT_ID
                ),
                "n_2022_valid_cycles": int(
                    len(valid_2022)
                ),
                "n_2024_valid_cycles": int(
                    len(valid_2024)
                ),
                "calibration_cycle_numbers":
                    calibration_cycle_numbers,
                "test_cycle_numbers":
                    test_cycle_numbers,
                "calibration_days":
                    cal_days,
                "test_days":
                    test_days,
            }
        )

    manifest = pd.DataFrame(
        rows
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest.to_csv(
        OUTPUT_DIR
        / "longitudinal_manifest_all.csv",
        index=False,
    )

    primary = manifest[
        manifest["eligible"]
        & ~manifest["development_id"]
    ].copy()

    primary.to_csv(
        OUTPUT_DIR
        / "longitudinal_manifest_primary.csv",
        index=False,
    )

    sensitivity = manifest[
        manifest["eligible"]
    ].copy()

    sensitivity.to_csv(
        OUTPUT_DIR
        / "longitudinal_manifest_with_id22.csv",
        index=False,
    )

    print("=" * 76)
    print("AIM 2 LONGITUDINAL COHORT")
    print("=" * 76)

    print(
        "\nParticipants with data in both intervals:",
        len(manifest),
    )

    print(
        "Eligible with >=2 valid 2022 cycles "
        "and >=1 valid 2024 cycle:",
        int(
            manifest["eligible"].sum()
        ),
    )

    print(
        "Primary longitudinal cohort "
        "(excluding development ID 22):",
        len(primary),
    )

    print(
        "\nPrimary calibration days:",
        int(
            primary[
                "calibration_days"
            ].sum()
        ),
    )

    print(
        "Primary 2024 test days:",
        int(
            primary[
                "test_days"
            ].sum()
        ),
    )

    print(
        "\nPRIMARY MANIFEST"
    )

    print(
        primary.to_string(
            index=False
        )
    )

    print(
        "\nOutputs written to:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
