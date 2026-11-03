"""Build centered physiological features for final Aim 2 primary folds."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DAILY_PATH = Path(
    "results/preprocessed/mcphases_daily_master.csv"
)

MANIFEST_PATH = Path(
    "results/aim2/splits/"
    "aim2_within_interval_manifest.csv"
)

SUMMARY_PATH = Path(
    "results/aim2/splits/"
    "aim2_within_interval_summary.csv"
)

OUTPUT_DIR = Path(
    "results/aim2/centered_primary"
)

PREDICTORS = {
    "resting_heart_rate": "rhr",
    "hrv_rmssd": "hrv",
    "nightly_temperature": "temp",
    "respiratory_rate": "resp",
}


def attach_daily(
    manifest: pd.DataFrame,
    daily: pd.DataFrame,
) -> pd.DataFrame:
    """Attach physiological measurements to manifest rows."""

    return manifest.merge(
        daily,
        left_on=[
            "source_id",
            "study_interval",
            "day_in_study",
        ],
        right_on=[
            "id",
            "study_interval",
            "day_in_study",
        ],
        how="left",
        validate="many_to_one",
        suffixes=("_manifest", ""),
    )


def calculate_person_baselines(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate one baseline per source participant."""

    baselines = (
        data.groupby(
            [
                "source_id",
                "study_interval",
            ],
            as_index=False,
        )[list(PREDICTORS)]
        .mean()
    )

    return baselines.rename(
        columns={
            predictor: f"{short}_baseline"
            for predictor, short in PREDICTORS.items()
        }
    )


def add_centered_features(
    data: pd.DataFrame,
    baselines: pd.DataFrame,
) -> pd.DataFrame:
    """Add baseline and within-person deviation features."""

    output = data.merge(
        baselines,
        on=[
            "source_id",
            "study_interval",
        ],
        how="left",
        validate="many_to_one",
    )

    for predictor, short in PREDICTORS.items():
        baseline = f"{short}_baseline"

        if output[baseline].isna().any():
            raise RuntimeError(
                f"Missing {baseline} values."
            )

        output[f"{short}_within"] = (
            output[predictor]
            - output[baseline]
        )

    return output


def verify_target(
    calibration: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Verify target centering uses calibration data only."""

    for predictor, short in PREDICTORS.items():
        baseline = f"{short}_baseline"
        within = f"{short}_within"

        calibration_mean = calibration[
            within
        ].mean()

        if abs(calibration_mean) > 1e-10:
            raise RuntimeError(
                f"{short} calibration deviations "
                "do not average to zero."
            )

        target_baseline = calibration[
            baseline
        ].iloc[0]

        test_baselines = test[
            baseline
        ].unique()

        if len(test_baselines) != 1:
            raise RuntimeError(
                f"Multiple {short} test baselines."
            )

        if abs(
            test_baselines[0]
            - target_baseline
        ) > 1e-12:
            raise RuntimeError(
                f"{short} test baseline changed "
                "after calibration."
            )


def main() -> None:
    """Build all centered primary folds."""

    daily = pd.read_csv(DAILY_PATH)
    manifest = pd.read_csv(MANIFEST_PATH)
    summary = pd.read_csv(SUMMARY_PATH)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_rows = []

    for _, row in summary.iterrows():
        target_id = row["target_id"]
        target_interval = row[
            "target_interval"
        ]

        fold = manifest[
            (manifest["target_id"] == target_id)
            & (
                manifest["target_interval"]
                == target_interval
            )
        ].copy()

        population_manifest = fold[
            fold["split_role"]
            == "population_train"
        ]

        calibration_manifest = fold[
            fold["split_role"]
            == "personal_calibration"
        ]

        test_manifest = fold[
            fold["split_role"]
            == "future_test"
        ]

        population = attach_daily(
            population_manifest,
            daily,
        )

        calibration = attach_daily(
            calibration_manifest,
            daily,
        )

        test = attach_daily(
            test_manifest,
            daily,
        )

        # Population baselines are calculated only
        # from population-training observations.
        population_baselines = (
            calculate_person_baselines(
                population
            )
        )

        # Target baseline is calculated only from
        # cycles 1-2 in the target interval.
        target_baseline = (
            calculate_person_baselines(
                calibration
            )
        )

        if len(target_baseline) != 1:
            raise RuntimeError(
                "Target calibration produced more "
                "than one personal baseline."
            )

        population_centered = (
            add_centered_features(
                population,
                population_baselines,
            )
        )

        calibration_centered = (
            add_centered_features(
                calibration,
                target_baseline,
            )
        )

        test_centered = (
            add_centered_features(
                test,
                target_baseline,
            )
        )

        verify_target(
            calibration_centered,
            test_centered,
        )

        fold_name = (
            f"id{int(target_id)}_"
            f"{int(target_interval)}"
        )

        fold_dir = (
            OUTPUT_DIR
            / fold_name
        )

        fold_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        population_centered.to_csv(
            fold_dir
            / "population_train.csv",
            index=False,
        )

        calibration_centered.to_csv(
            fold_dir
            / "personal_calibration.csv",
            index=False,
        )

        test_centered.to_csv(
            fold_dir
            / "future_test.csv",
            index=False,
        )

        target_record = {
            "target_id": target_id,
            "target_interval": target_interval,
            "population_days": len(
                population_centered
            ),
            "calibration_days": len(
                calibration_centered
            ),
            "test_days": len(
                test_centered
            ),
        }

        for _, short in PREDICTORS.items():
            target_record[
                f"{short}_baseline"
            ] = calibration_centered[
                f"{short}_baseline"
            ].iloc[0]

            target_record[
                f"{short}_future_mean_deviation"
            ] = test_centered[
                f"{short}_within"
            ].mean()

        audit_rows.append(
            target_record
        )

    audit = pd.DataFrame(
        audit_rows
    )

    audit.to_csv(
        OUTPUT_DIR
        / "centered_fold_audit.csv",
        index=False,
    )

    print("=" * 68)
    print("AIM 2 CENTERED PRIMARY FOLDS COMPLETE")
    print("=" * 68)

    print(
        "Participant-interval folds:",
        len(audit),
    )

    print(
        "Unique participants:",
        audit["target_id"].nunique(),
    )

    print(
        "Total calibration days:",
        int(
            audit[
                "calibration_days"
            ].sum()
        ),
    )

    print(
        "Total future test days:",
        int(
            audit[
                "test_days"
            ].sum()
        ),
    )

    print("\nTarget future deviation summary:")

    columns = [
        "rhr_future_mean_deviation",
        "hrv_future_mean_deviation",
        "temp_future_mean_deviation",
        "resp_future_mean_deviation",
    ]

    print(
        audit[columns]
        .describe()
        .loc[
            [
                "mean",
                "std",
                "min",
                "50%",
                "max",
            ]
        ]
        .to_string()
    )


if __name__ == "__main__":
    main()
