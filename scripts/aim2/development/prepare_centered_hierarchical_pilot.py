"""Prepare within-person centered physiology for the Aim 2 pilot."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_DIR = Path("results/aim2/pilot")
OUTPUT_DIR = Path("results/aim2/pilot/centered")

TARGET_ID = 22

PREDICTORS = {
    "resting_heart_rate": "rhr",
    "hrv_rmssd": "hrv",
    "nightly_temperature": "temp",
    "respiratory_rate": "resp",
}


def calculate_population_baselines(
    population: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate each training participant's physiological baseline."""

    baseline_columns = list(PREDICTORS)

    baselines = (
        population.groupby(
            "source_id",
            as_index=False,
        )[baseline_columns]
        .mean()
    )

    baselines = baselines.rename(
        columns={
            predictor: f"{short}_baseline"
            for predictor, short in PREDICTORS.items()
        }
    )

    return baselines


def calculate_target_baseline(
    calibration: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate target baseline using calibration cycles only."""

    if calibration["source_id"].nunique() != 1:
        raise ValueError(
            "Calibration data must contain exactly one target."
        )

    if calibration["source_id"].iloc[0] != TARGET_ID:
        raise ValueError(
            f"Expected target {TARGET_ID} calibration data."
        )

    record = {
        "source_id": TARGET_ID,
    }

    for predictor, short in PREDICTORS.items():
        record[
            f"{short}_baseline"
        ] = calibration[predictor].mean()

    return pd.DataFrame([record])


def add_centered_features(
    data: pd.DataFrame,
    baselines: pd.DataFrame,
) -> pd.DataFrame:
    """Attach baseline and within-person deviation features."""

    output = data.merge(
        baselines,
        on="source_id",
        how="left",
        validate="many_to_one",
    )

    baseline_names = [
        f"{short}_baseline"
        for short in PREDICTORS.values()
    ]

    if output[baseline_names].isna().any().any():
        raise ValueError(
            "Missing physiological baseline after merge."
        )

    for predictor, short in PREDICTORS.items():
        output[
            f"{short}_within"
        ] = (
            output[predictor]
            - output[f"{short}_baseline"]
        )

    return output


def verify_target_centering(
    calibration: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Verify target baselines and chronology behavior."""

    print("\nTARGET 22 BASELINES FROM CYCLES 1-2")
    print("=" * 68)

    for predictor, short in PREDICTORS.items():
        baseline = calibration[
            f"{short}_baseline"
        ].iloc[0]

        calibration_mean = calibration[
            predictor
        ].mean()

        calibration_deviation_mean = calibration[
            f"{short}_within"
        ].mean()

        test_deviation_mean = test[
            f"{short}_within"
        ].mean()

        test_baselines = test[
            f"{short}_baseline"
        ].unique()

        if len(test_baselines) != 1:
            raise ValueError(
                f"Test data contain multiple {short} baselines."
            )

        if not pd.isna(test_baselines[0]):
            if abs(test_baselines[0] - baseline) > 1e-12:
                raise ValueError(
                    f"{short} test baseline differs from "
                    "calibration baseline."
                )

        print(
            f"{short:5s} "
            f"baseline={baseline:8.3f}  "
            f"cal_mean={calibration_mean:8.3f}  "
            f"cal_dev_mean={calibration_deviation_mean:+8.3f}  "
            f"test_dev_mean={test_deviation_mean:+8.3f}"
        )


def print_population_summary(
    population: pd.DataFrame,
    baselines: pd.DataFrame,
) -> None:
    """Print population baseline and centered-feature summaries."""

    print("\nPOPULATION TRAINING BASELINES")
    print("=" * 68)

    print(
        f"Training participants: "
        f"{baselines['source_id'].nunique()}"
    )

    for _, short in PREDICTORS.items():
        series = baselines[
            f"{short}_baseline"
        ]

        print(
            f"{short:5s} "
            f"mean={series.mean():8.3f}  "
            f"sd={series.std():8.3f}  "
            f"min={series.min():8.3f}  "
            f"max={series.max():8.3f}"
        )

    print("\nPOPULATION WITHIN-PERSON DEVIATIONS")
    print("=" * 68)

    for _, short in PREDICTORS.items():
        series = population[
            f"{short}_within"
        ]

        print(
            f"{short:5s} "
            f"mean={series.mean():+8.3f}  "
            f"sd={series.std():8.3f}  "
            f"min={series.min():+8.3f}  "
            f"max={series.max():+8.3f}"
        )


def main() -> None:
    """Build centered pilot datasets."""

    population = pd.read_csv(
        INPUT_DIR
        / "target22_population_train.csv"
    )

    calibration = pd.read_csv(
        INPUT_DIR
        / "target22_personal_calibration.csv"
    )

    test = pd.read_csv(
        INPUT_DIR
        / "target22_future_test.csv"
    )

    population_baselines = (
        calculate_population_baselines(
            population
        )
    )

    target_baseline = (
        calculate_target_baseline(
            calibration
        )
    )

    population_centered = add_centered_features(
        population,
        population_baselines,
    )

    calibration_centered = add_centered_features(
        calibration,
        target_baseline,
    )

    test_centered = add_centered_features(
        test,
        target_baseline,
    )

    hybrid_centered = pd.concat(
        [
            population_centered,
            calibration_centered,
        ],
        ignore_index=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    population_centered.to_csv(
        OUTPUT_DIR
        / "target22_population_centered.csv",
        index=False,
    )

    calibration_centered.to_csv(
        OUTPUT_DIR
        / "target22_calibration_centered.csv",
        index=False,
    )

    test_centered.to_csv(
        OUTPUT_DIR
        / "target22_future_test_centered.csv",
        index=False,
    )

    hybrid_centered.to_csv(
        OUTPUT_DIR
        / "target22_hybrid_centered.csv",
        index=False,
    )

    population_baselines.to_csv(
        OUTPUT_DIR
        / "population_participant_baselines.csv",
        index=False,
    )

    target_baseline.to_csv(
        OUTPUT_DIR
        / "target22_calibration_baseline.csv",
        index=False,
    )

    print("=" * 68)
    print("AIM 2 PERSON-CENTERED PILOT PREPARATION")
    print("=" * 68)

    print(
        "Population days:",
        len(population_centered),
    )

    print(
        "Calibration days:",
        len(calibration_centered),
    )

    print(
        "Future test days:",
        len(test_centered),
    )

    print_population_summary(
        population_centered,
        population_baselines,
    )

    verify_target_centering(
        calibration_centered,
        test_centered,
    )

    print("\nCentered datasets written to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
