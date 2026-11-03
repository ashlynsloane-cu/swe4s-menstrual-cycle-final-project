"""Build leakage-safe 2022-to-2024 Aim 2 longitudinal folds."""

from pathlib import Path

import numpy as np
import pandas as pd


MASTER_PATH = Path(
    "results/preprocessed/"
    "mcphases_daily_master.csv"
)

MANIFEST_PATH = Path(
    "results/aim2/longitudinal/"
    "longitudinal_manifest_primary.csv"
)

OUTPUT_ROOT = Path(
    "results/aim2/longitudinal_folds"
)

AUDIT_PATH = OUTPUT_ROOT / "longitudinal_fold_audit.csv"

PHASES = [
    "Menstrual",
    "Follicular",
    "Fertility",
    "Luteal",
]

SIGNALS = {
    "resting_heart_rate": "rhr",
    "hrv_rmssd": "hrv",
    "nightly_temperature": "temp",
    "respiratory_rate": "resp",
}


def parse_cycle_numbers(
    value,
) -> list[int]:
    """Convert comma-separated manifest cycle numbers to integers."""

    if pd.isna(value):
        return []

    return [
        int(float(item.strip()))
        for item in str(value).split(",")
        if item.strip()
    ]


def prepare_master(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Restrict to complete-case observations used by Aim 2."""

    mask = (
        data["phase"].isin(PHASES)
        & data[list(SIGNALS)].notna().all(axis=1)
    )

    output = data.loc[mask].copy()

    output["source_id"] = (
        output["id"].astype(int)
    )

    output["study_interval"] = (
        output["study_interval"].astype(int)
    )

    return output


def add_population_baselines(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Center population rows within source participant and interval."""

    output = data.copy()

    grouping = [
        "source_id",
        "study_interval",
    ]

    for signal, short_name in SIGNALS.items():
        baseline_column = (
            f"{short_name}_baseline"
        )

        within_column = (
            f"{short_name}_within"
        )

        output[baseline_column] = (
            output.groupby(grouping)[signal]
            .transform("mean")
        )

        output[within_column] = (
            output[signal]
            - output[baseline_column]
        )

    return output


def target_baselines(
    calibration: pd.DataFrame,
) -> dict[str, float]:
    """Calculate target baselines using 2022 calibration only."""

    return {
        short_name: float(
            calibration[signal].mean()
        )
        for signal, short_name in SIGNALS.items()
    }


def add_target_baselines(
    data: pd.DataFrame,
    baselines: dict[str, float],
) -> pd.DataFrame:
    """Apply frozen 2022 target baselines to calibration or test rows."""

    output = data.copy()

    for signal, short_name in SIGNALS.items():
        baseline_column = (
            f"{short_name}_baseline"
        )

        within_column = (
            f"{short_name}_within"
        )

        output[baseline_column] = (
            baselines[short_name]
        )

        output[within_column] = (
            output[signal]
            - output[baseline_column]
        )

    return output


def add_common_columns(
    data: pd.DataFrame,
    target_id: int,
    split_role: str,
) -> pd.DataFrame:
    """Add fold metadata matching the primary Aim 2 files."""

    output = data.copy()

    output["target_id"] = target_id

    # The prediction target is always 2024 in this analysis.
    output["target_interval"] = 2024

    output["split_role"] = split_role

    if "phase_manifest" not in output.columns:
        output["phase_manifest"] = (
            output["phase"]
        )

    if "cycle_number_manifest" not in output.columns:
        output["cycle_number_manifest"] = (
            output["cycle_number"]
        )

    return output


def add_target_cycle_index(
    data: pd.DataFrame,
    cycle_numbers: list[int],
) -> pd.DataFrame:
    """Add ordered cycle indices for target calibration/test cycles."""

    output = data.copy()

    lookup = {
        cycle_number: index + 1
        for index, cycle_number in enumerate(
            cycle_numbers
        )
    }

    output["interval_cycle_index"] = (
        output["cycle_number"].map(
            lookup
        )
    )

    return output


def add_population_cycle_index(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Add cycle index where a population row has a known cycle."""

    output = data.copy()

    output["interval_cycle_index"] = np.nan

    known = output[
        output["cycle_number"].notna()
    ].copy()

    if known.empty:
        return output

    unique_cycles = (
        known[
            [
                "source_id",
                "study_interval",
                "cycle_number",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "source_id",
                "study_interval",
                "cycle_number",
            ]
        )
    )

    unique_cycles[
        "interval_cycle_index"
    ] = (
        unique_cycles.groupby(
            [
                "source_id",
                "study_interval",
            ]
        )
        .cumcount()
        + 1
    )

    lookup = {
        (
            int(row.source_id),
            int(row.study_interval),
            row.cycle_number,
        ): row.interval_cycle_index
        for row in unique_cycles.itertuples()
    }

    mask = output["cycle_number"].notna()

    output.loc[
        mask,
        "interval_cycle_index",
    ] = [
        lookup.get(
            (
                int(source_id),
                int(interval),
                cycle_number,
            ),
            np.nan,
        )
        for source_id, interval, cycle_number in zip(
            output.loc[mask, "source_id"],
            output.loc[mask, "study_interval"],
            output.loc[mask, "cycle_number"],
        )
    ]

    return output


def reorder_columns(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Put fold metadata first while preserving all master columns."""

    preferred = [
        "target_id",
        "target_interval",
        "source_id",
        "study_interval",
        "day_in_study",
        "cycle_number_manifest",
        "phase_manifest",
        "interval_cycle_index",
        "split_role",
    ]

    remaining = [
        column
        for column in data.columns
        if column not in preferred
    ]

    return data[
        preferred + remaining
    ]


def main() -> None:
    """Build all primary longitudinal folds."""

    master = pd.read_csv(
        MASTER_PATH
    )

    master = prepare_master(
        master
    )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_rows = []

    for _, row in manifest.iterrows():
        target_id = int(
            row["target_id"]
        )

        calibration_cycles = parse_cycle_numbers(
            row["calibration_cycle_numbers"]
        )

        test_cycles = parse_cycle_numbers(
            row["test_cycle_numbers"]
        )

        # ------------------------------------------------------
        # Population training
        #
        # Completely remove target participant from both years.
        # ------------------------------------------------------

        population = master[
            master["source_id"] != target_id
        ].copy()

        assert (
            target_id
            not in set(
                population["source_id"]
            )
        )

        population = (
            add_population_baselines(
                population
            )
        )

        population = add_common_columns(
            population,
            target_id,
            "population_train",
        )

        population = (
            add_population_cycle_index(
                population
            )
        )

        # ------------------------------------------------------
        # Target 2022 personal calibration
        # ------------------------------------------------------

        calibration = master[
            (master["source_id"] == target_id)
            & (
                master["study_interval"]
                == 2022
            )
            & (
                master["cycle_number"]
                .isin(calibration_cycles)
            )
        ].copy()

        calibration = add_common_columns(
            calibration,
            target_id,
            "personal_calibration",
        )

        calibration = add_target_cycle_index(
            calibration,
            calibration_cycles,
        )

        if calibration.empty:
            raise RuntimeError(
                f"ID {target_id}: "
                "no calibration rows."
            )

        observed_calibration_cycles = sorted(
            calibration[
                "cycle_number"
            ]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        if (
            observed_calibration_cycles
            != sorted(calibration_cycles)
        ):
            raise RuntimeError(
                f"ID {target_id}: calibration "
                f"cycle mismatch. Expected "
                f"{calibration_cycles}, observed "
                f"{observed_calibration_cycles}."
            )

        baselines = target_baselines(
            calibration
        )

        calibration = add_target_baselines(
            calibration,
            baselines,
        )

        # ------------------------------------------------------
        # Target 2024 future test
        #
        # Crucially, 2024 values are NOT used to estimate
        # the target baseline.
        # ------------------------------------------------------

        future_test = master[
            (master["source_id"] == target_id)
            & (
                master["study_interval"]
                == 2024
            )
            & (
                master["cycle_number"]
                .isin(test_cycles)
            )
        ].copy()

        future_test = add_common_columns(
            future_test,
            target_id,
            "future_test",
        )

        future_test = add_target_cycle_index(
            future_test,
            test_cycles,
        )

        if future_test.empty:
            raise RuntimeError(
                f"ID {target_id}: "
                "no 2024 test rows."
            )

        observed_test_cycles = sorted(
            future_test[
                "cycle_number"
            ]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        if (
            observed_test_cycles
            != sorted(test_cycles)
        ):
            raise RuntimeError(
                f"ID {target_id}: test "
                f"cycle mismatch. Expected "
                f"{test_cycles}, observed "
                f"{observed_test_cycles}."
            )

        future_test = add_target_baselines(
            future_test,
            baselines,
        )

        # ------------------------------------------------------
        # Safety checks
        # ------------------------------------------------------

        assert (
            calibration[
                "study_interval"
            ].eq(2022).all()
        )

        assert (
            future_test[
                "study_interval"
            ].eq(2024).all()
        )

        for short_name in SIGNALS.values():
            baseline_column = (
                f"{short_name}_baseline"
            )

            cal_values = calibration[
                baseline_column
            ].unique()

            test_values = future_test[
                baseline_column
            ].unique()

            if len(cal_values) != 1:
                raise RuntimeError(
                    f"ID {target_id}: "
                    f"{baseline_column} is "
                    "not frozen in calibration."
                )

            if len(test_values) != 1:
                raise RuntimeError(
                    f"ID {target_id}: "
                    f"{baseline_column} is "
                    "not frozen in test."
                )

            if not np.isclose(
                cal_values[0],
                test_values[0],
            ):
                raise RuntimeError(
                    f"ID {target_id}: "
                    f"{baseline_column} changed "
                    "between 2022 and 2024."
                )

        # ------------------------------------------------------
        # Save
        # ------------------------------------------------------

        population = reorder_columns(
            population
        )

        calibration = reorder_columns(
            calibration
        )

        future_test = reorder_columns(
            future_test
        )

        fold_dir = (
            OUTPUT_ROOT
            / f"id{target_id}"
        )

        fold_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        population.to_csv(
            fold_dir
            / "population_train.csv",
            index=False,
        )

        calibration.to_csv(
            fold_dir
            / "personal_calibration.csv",
            index=False,
        )

        future_test.to_csv(
            fold_dir
            / "future_test.csv",
            index=False,
        )

        audit_rows.append(
            {
                "target_id": target_id,
                "population_days": len(
                    population
                ),
                "population_participants": (
                    population[
                        "source_id"
                    ].nunique()
                ),
                "calibration_cycles": (
                    len(calibration_cycles)
                ),
                "calibration_days": len(
                    calibration
                ),
                "test_cycles": len(
                    test_cycles
                ),
                "test_days": len(
                    future_test
                ),
                "calibration_phases": (
                    calibration[
                        "phase"
                    ].nunique()
                ),
                "test_phases": (
                    future_test[
                        "phase"
                    ].nunique()
                ),
                "rhr_2022_baseline": (
                    baselines["rhr"]
                ),
                "hrv_2022_baseline": (
                    baselines["hrv"]
                ),
                "temp_2022_baseline": (
                    baselines["temp"]
                ),
                "resp_2022_baseline": (
                    baselines["resp"]
                ),
                "rhr_test_mean_shift": (
                    future_test[
                        "resting_heart_rate"
                    ].mean()
                    - baselines["rhr"]
                ),
                "hrv_test_mean_shift": (
                    future_test[
                        "hrv_rmssd"
                    ].mean()
                    - baselines["hrv"]
                ),
                "temp_test_mean_shift": (
                    future_test[
                        "nightly_temperature"
                    ].mean()
                    - baselines["temp"]
                ),
                "resp_test_mean_shift": (
                    future_test[
                        "respiratory_rate"
                    ].mean()
                    - baselines["resp"]
                ),
            }
        )

    audit = pd.DataFrame(
        audit_rows
    )

    audit.to_csv(
        AUDIT_PATH,
        index=False,
    )

    print("=" * 78)
    print("AIM 2 LONGITUDINAL FOLDS BUILT")
    print("=" * 78)

    print(
        "\nFolds:",
        len(audit),
    )

    print(
        "Unique targets:",
        audit["target_id"].nunique(),
    )

    print(
        "Calibration days:",
        audit[
            "calibration_days"
        ].sum(),
    )

    print(
        "2024 test days:",
        audit[
            "test_days"
        ].sum(),
    )

    print(
        "\nFOLD AUDIT"
    )

    print(
        audit.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.3f}"
            ),
        )
    )

    print(
        "\nOutputs written to:",
        OUTPUT_ROOT,
    )


if __name__ == "__main__":
    main()
