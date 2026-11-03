"""Build the final within-interval Aim 2 primary split manifest."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PHASES = [
    "Menstrual",
    "Follicular",
    "Fertility",
    "Luteal",
]

PREDICTORS = [
    "resting_heart_rate",
    "hrv_rmssd",
    "nightly_temperature",
    "respiratory_rate",
]

DAILY_PATH = Path(
    "results/preprocessed/mcphases_daily_master.csv"
)

CYCLE_PATH = Path(
    "results/preprocessed/mcphases_cycle_qc.csv"
)

OUTPUT_DIR = Path(
    "results/aim2/splits"
)


def main() -> None:
    """Build deterministic within-interval calibration/test splits."""

    daily = pd.read_csv(DAILY_PATH)
    cycles = pd.read_csv(CYCLE_PATH)

    usable = daily[
        daily["phase"].isin(PHASES)
        & daily[PREDICTORS].notna().all(axis=1)
    ].copy()

    valid_cycles = cycles[
        cycles["complete_cycle"]
        & cycles["valid_phase_order"]
    ].copy()

    valid_cycles = valid_cycles.sort_values(
        [
            "id",
            "study_interval",
            "cycle_start_day",
            "cycle_number",
        ]
    )

    valid_cycles["interval_cycle_index"] = (
        valid_cycles.groupby(
            [
                "id",
                "study_interval",
            ]
        ).cumcount()
        + 1
    )

    manifest_parts = []
    summary_rows = []

    for (
        target_id,
        target_interval,
    ), group in valid_cycles.groupby(
        [
            "id",
            "study_interval",
        ]
    ):
        group = group.sort_values(
            "interval_cycle_index"
        )

        if len(group) < 3:
            continue

        calibration_cycles = group[
            group["interval_cycle_index"].isin(
                [1, 2]
            )
        ]

        test_cycles = group[
            group["interval_cycle_index"] >= 3
        ]

        def select_days(
            selected_cycles: pd.DataFrame,
        ) -> pd.DataFrame:
            keys = selected_cycles[
                [
                    "id",
                    "study_interval",
                    "cycle_number",
                    "interval_cycle_index",
                ]
            ]

            return usable.merge(
                keys,
                on=[
                    "id",
                    "study_interval",
                    "cycle_number",
                ],
                how="inner",
                validate="many_to_one",
            )

        calibration = select_days(
            calibration_cycles
        )

        test = select_days(
            test_cycles
        )

        if calibration.empty or test.empty:
            continue

        # Population-only must know nothing personal
        # about the target participant, including data
        # from the target's other study interval.
        population = usable[
            usable["id"] != target_id
        ].copy()

        population_rows = population[
            [
                "id",
                "study_interval",
                "day_in_study",
                "cycle_number",
                "phase",
            ]
        ].copy()

        population_rows["interval_cycle_index"] = pd.NA
        population_rows["split_role"] = (
            "population_train"
        )

        calibration_rows = calibration[
            [
                "id",
                "study_interval",
                "day_in_study",
                "cycle_number",
                "interval_cycle_index",
                "phase",
            ]
        ].copy()

        calibration_rows["split_role"] = (
            "personal_calibration"
        )

        test_rows = test[
            [
                "id",
                "study_interval",
                "day_in_study",
                "cycle_number",
                "interval_cycle_index",
                "phase",
            ]
        ].copy()

        test_rows["split_role"] = "future_test"

        fold = pd.concat(
            [
                population_rows,
                calibration_rows,
                test_rows,
            ],
            ignore_index=True,
        )

        fold = fold.rename(
            columns={
                "id": "source_id",
            }
        )

        fold.insert(
            0,
            "target_id",
            target_id,
        )

        fold.insert(
            1,
            "target_interval",
            target_interval,
        )

        # Leakage checks.
        population_check = fold[
            fold["split_role"]
            == "population_train"
        ]

        if (
            population_check["source_id"]
            == target_id
        ).any():
            raise RuntimeError(
                f"Target {target_id} leaked into "
                "population training."
            )

        target_rows = fold[
            fold["split_role"].isin(
                [
                    "personal_calibration",
                    "future_test",
                ]
            )
        ]

        if not (
            target_rows["source_id"]
            == target_id
        ).all():
            raise RuntimeError(
                "Non-target observations found in "
                "personal/test data."
            )

        if not (
            target_rows["study_interval"]
            == target_interval
        ).all():
            raise RuntimeError(
                "Cross-interval data found in primary "
                "calibration/test split."
            )

        if not (
            calibration_rows[
                "interval_cycle_index"
            ].isin([1, 2])
        ).all():
            raise RuntimeError(
                "Invalid calibration cycle."
            )

        if not (
            test_rows[
                "interval_cycle_index"
            ] >= 3
        ).all():
            raise RuntimeError(
                "Invalid future test cycle."
            )

        manifest_parts.append(fold)

        summary_rows.append(
            {
                "target_id": target_id,
                "target_interval": target_interval,
                "n_valid_cycles": len(group),
                "n_calibration_days": len(
                    calibration
                ),
                "n_test_cycles": len(
                    test_cycles
                ),
                "n_test_days": len(test),
                "n_calibration_phases": (
                    calibration["phase"].nunique()
                ),
                "n_test_phases": (
                    test["phase"].nunique()
                ),
            }
        )

    manifest = pd.concat(
        manifest_parts,
        ignore_index=True,
    )

    summary = pd.DataFrame(
        summary_rows
    ).sort_values(
        [
            "target_id",
            "target_interval",
        ]
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest.to_csv(
        OUTPUT_DIR
        / "aim2_within_interval_manifest.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "aim2_within_interval_summary.csv",
        index=False,
    )

    print("=" * 68)
    print("FINAL AIM 2 WITHIN-INTERVAL PRIMARY SPLIT")
    print("=" * 68)

    print(
        "Participant-intervals:",
        len(summary),
    )

    print(
        "Unique participants:",
        summary["target_id"].nunique(),
    )

    print(
        "Future test cycles:",
        int(summary["n_test_cycles"].sum()),
    )

    print(
        "Future test days:",
        int(summary["n_test_days"].sum()),
    )

    print("\nBy study interval:")

    print(
        summary.groupby(
            "target_interval"
        ).agg(
            participant_intervals=(
                "target_id",
                "size",
            ),
            unique_participants=(
                "target_id",
                "nunique",
            ),
            test_cycles=(
                "n_test_cycles",
                "sum",
            ),
            test_days=(
                "n_test_days",
                "sum",
            ),
        ).to_string()
    )


if __name__ == "__main__":
    main()
