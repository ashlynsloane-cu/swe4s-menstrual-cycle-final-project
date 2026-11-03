"""Prepare one Aim 2 fold for hierarchical model development."""

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


TARGET_ID = 22

PREDICTORS = [
    "resting_heart_rate",
    "hrv_rmssd",
    "nightly_temperature",
    "respiratory_rate",
]

Z_COLUMNS = [
    "rhr_z",
    "hrv_z",
    "temp_z",
    "resp_z",
]


def main():
    """Prepare population, hybrid, and test tables for one target."""

    daily = pd.read_csv(
        Path(
            "results/preprocessed/"
            "mcphases_daily_master.csv"
        )
    )

    manifest = pd.read_csv(
        Path(
            "results/aim2/splits/"
            "aim2_primary_split_manifest.csv"
        )
    )

    fold = manifest[
        manifest["target_id"] == TARGET_ID
    ].copy()

    population_manifest = fold[
        fold["split_role"] == "population_train"
    ]

    calibration_manifest = fold[
        fold["split_role"].isin(
            [
                "personal_calibration_cycle1",
                "personal_calibration_cycle2",
            ]
        )
    ]

    test_manifest = fold[
        fold["split_role"] == "future_test"
    ]

    def attach_data(part):
        merged = part.merge(
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
            validate="one_to_one",
            suffixes=("_manifest", ""),
        )

        return merged

    population = attach_data(population_manifest)
    calibration = attach_data(calibration_manifest)
    test = attach_data(test_manifest)

    # Scaling is learned ONLY from population training data.
    scaler = StandardScaler()
    scaler.fit(population[PREDICTORS])

    population[Z_COLUMNS] = scaler.transform(
        population[PREDICTORS]
    )

    calibration[Z_COLUMNS] = scaler.transform(
        calibration[PREDICTORS]
    )

    test[Z_COLUMNS] = scaler.transform(
        test[PREDICTORS]
    )

    # Hybrid training = population + target cycles 1-2.
    hybrid = pd.concat(
        [
            population,
            calibration,
        ],
        ignore_index=True,
    )

    # Keep IDs categorical for hierarchical grouping.
    population["model_id"] = (
        population["source_id"].astype(str)
    )

    calibration["model_id"] = (
        calibration["source_id"].astype(str)
    )

    hybrid["model_id"] = (
        hybrid["source_id"].astype(str)
    )

    test["model_id"] = (
        test["source_id"].astype(str)
    )

    outdir = Path(
        "results/aim2/pilot"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    population.to_csv(
        outdir / "target22_population_train.csv",
        index=False,
    )

    calibration.to_csv(
        outdir / "target22_personal_calibration.csv",
        index=False,
    )

    hybrid.to_csv(
        outdir / "target22_hybrid_train.csv",
        index=False,
    )

    test.to_csv(
        outdir / "target22_future_test.csv",
        index=False,
    )

    print("=" * 68)
    print("TARGET 22 HIERARCHICAL PILOT")
    print("=" * 68)

    print("Population training days:", len(population))
    print("Personal calibration days:", len(calibration))
    print("Hybrid training days:", len(hybrid))
    print("Future test days:", len(test))

    print("\nCalibration phases:")
    print(
        calibration["phase"].value_counts()
    )

    print("\nFuture test phases:")
    print(
        test["phase"].value_counts()
    )

    print("\nScaling parameters learned from population only:")

    for predictor, mean, scale in zip(
        PREDICTORS,
        scaler.mean_,
        scaler.scale_,
    ):
        print(
            f"{predictor:24s} "
            f"mean={mean:8.3f} "
            f"sd={scale:8.3f}"
        )


if __name__ == "__main__":
    main()
