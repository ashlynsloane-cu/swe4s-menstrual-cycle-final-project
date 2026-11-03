"""Diagnose why empirical Aim 2 personalization fails."""

from pathlib import Path

import numpy as np
import pandas as pd


PRIMARY_FOLDS = Path(
    "results/aim2/centered_primary"
)

LONGITUDINAL_FOLDS = Path(
    "results/aim2/longitudinal_folds"
)

PRIMARY_RESULTS = Path(
    "results/aim2/primary_models/"
    "all_fold_results.csv"
)

LONGITUDINAL_RESULTS = Path(
    "results/aim2/longitudinal_models/"
    "all_fold_results.csv"
)

TRANSFER_PATH = Path(
    "results/aim2/profile_transfer/"
    "fold_profile_transfer.csv"
)

REPEATABILITY_PATH = Path(
    "results/aim2/simulation_components/"
    "phase_profile_repeatability.csv"
)

OUTPUT_DIR = Path(
    "results/aim2/personalization_failure"
)

SIGNALS = {
    "rhr": "resting_heart_rate",
    "hrv": "hrv_rmssd",
    "temp": "nightly_temperature",
    "resp": "respiratory_rate",
}


def model_differences(
    path: Path,
    analysis: str,
) -> pd.DataFrame:
    """Calculate personalized-minus-population performance."""

    results = pd.read_csv(
        path
    )

    index_columns = [
        "target_id",
        "target_interval",
    ]

    metrics = [
        "accuracy",
        "log_loss",
        "brier_score",
    ]

    population = results[
        results["model"]
        == "population_only"
    ][
        index_columns
        + metrics
    ].copy()

    personalized = results[
        results["model"]
        == "hybrid_personalized"
    ][
        index_columns
        + metrics
    ].copy()

    merged = population.merge(
        personalized,
        on=index_columns,
        suffixes=(
            "_population",
            "_personalized",
        ),
        validate="one_to_one",
    )

    for metric in metrics:
        merged[
            f"delta_{metric}"
        ] = (
            merged[
                f"{metric}_personalized"
            ]
            - merged[
                f"{metric}_population"
            ]
        )

    merged["analysis"] = analysis

    merged["fold"] = (
        "id"
        + merged[
            "target_id"
        ].astype(int).astype(str)
    )

    if analysis == "within_interval":
        merged["fold"] = (
            merged["fold"]
            + "_"
            + merged[
                "target_interval"
            ].astype(int).astype(str)
        )

    return merged


def fold_drift(
    fold_root: Path,
    analysis: str,
    raw_sd: dict,
) -> pd.DataFrame:
    """Calculate calibration-to-test physiological baseline drift."""

    rows = []

    for fold_dir in sorted(
        fold_root.glob("id*")
    ):
        calibration_path = (
            fold_dir
            / "personal_calibration.csv"
        )

        test_path = (
            fold_dir
            / "future_test.csv"
        )

        if not (
            calibration_path.exists()
            and test_path.exists()
        ):
            continue

        calibration = pd.read_csv(
            calibration_path
        )

        test = pd.read_csv(
            test_path
        )

        target_id = int(
            calibration[
                "target_id"
            ].iloc[0]
        )

        target_interval = int(
            test[
                "study_interval"
            ].iloc[0]
        )

        row = {
            "analysis": analysis,
            "fold": fold_dir.name,
            "target_id": target_id,
            "target_interval": (
                target_interval
            ),
        }

        standardized_absolute = []

        for short_name, column in (
            SIGNALS.items()
        ):
            calibration_mean = float(
                calibration[
                    column
                ].mean()
            )

            test_mean = float(
                test[
                    column
                ].mean()
            )

            shift = (
                test_mean
                - calibration_mean
            )

            standardized_shift = (
                shift
                / raw_sd[
                    short_name
                ]
            )

            row[
                f"{short_name}_baseline_shift"
            ] = shift

            row[
                f"{short_name}_baseline_shift_z"
            ] = standardized_shift

            row[
                f"{short_name}_abs_baseline_shift_z"
            ] = abs(
                standardized_shift
            )

            standardized_absolute.append(
                abs(
                    standardized_shift
                )
            )

        row[
            "mean_abs_baseline_shift_z"
        ] = float(
            np.mean(
                standardized_absolute
            )
        )

        row[
            "max_abs_baseline_shift_z"
        ] = float(
            np.max(
                standardized_absolute
            )
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def transfer_wide(
    transfer: pd.DataFrame,
) -> pd.DataFrame:
    """Convert signal-specific transfer metrics to fold-wide columns."""

    selected = transfer[
        [
            "analysis",
            "fold",
            "target_id",
            "target_interval",
            "signal",
            "calibration_cycle_corr",
            "cal_to_test_corr",
        ]
    ].copy()

    wide = selected.pivot_table(
        index=[
            "analysis",
            "fold",
            "target_id",
            "target_interval",
        ],
        columns="signal",
        values=[
            "calibration_cycle_corr",
            "cal_to_test_corr",
        ],
    )

    wide.columns = [
        f"{signal}_{metric}"
        for metric, signal
        in wide.columns
    ]

    return (
        wide.reset_index()
    )


def spearman_table(
    data: pd.DataFrame,
    analysis: str,
) -> pd.DataFrame:
    """Calculate descriptive Spearman associations."""

    subset = data[
        data["analysis"]
        == analysis
    ].copy()

    predictors = [
        "rhr_cal_to_test_corr",
        "hrv_cal_to_test_corr",
        "temp_cal_to_test_corr",
        "resp_cal_to_test_corr",
        "rhr_abs_baseline_shift_z",
        "hrv_abs_baseline_shift_z",
        "temp_abs_baseline_shift_z",
        "resp_abs_baseline_shift_z",
        "mean_abs_baseline_shift_z",
        "max_abs_baseline_shift_z",
    ]

    outcomes = [
        "delta_accuracy",
        "delta_log_loss",
        "delta_brier_score",
    ]

    rows = []

    for predictor in predictors:
        if predictor not in subset:
            continue

        for outcome in outcomes:
            pair = subset[
                [
                    predictor,
                    outcome,
                ]
            ].dropna()

            if len(pair) < 4:
                continue

            correlation = pair[
                predictor
            ].corr(
                pair[outcome],
                method="spearman",
            )

            rows.append(
                {
                    "analysis": analysis,
                    "predictor": predictor,
                    "outcome": outcome,
                    "n": len(pair),
                    "spearman_rho": (
                        correlation
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def main():
    """Run personalization-failure diagnostics."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    repeatability = pd.read_csv(
        REPEATABILITY_PATH
    )

    raw_sd = {
        row["signal"]: float(
            row["raw_sd"]
        )
        for _, row
        in repeatability.iterrows()
    }

    primary_perf = model_differences(
        PRIMARY_RESULTS,
        "within_interval",
    )

    longitudinal_perf = (
        model_differences(
            LONGITUDINAL_RESULTS,
            "longitudinal_2022_to_2024",
        )
    )

    performance = pd.concat(
        [
            primary_perf,
            longitudinal_perf,
        ],
        ignore_index=True,
    )

    primary_drift = fold_drift(
        PRIMARY_FOLDS,
        "within_interval",
        raw_sd,
    )

    longitudinal_drift = fold_drift(
        LONGITUDINAL_FOLDS,
        "longitudinal_2022_to_2024",
        raw_sd,
    )

    drift = pd.concat(
        [
            primary_drift,
            longitudinal_drift,
        ],
        ignore_index=True,
    )

    transfer = pd.read_csv(
        TRANSFER_PATH
    )

    transfer = transfer_wide(
        transfer
    )

    combined = performance.merge(
        drift,
        on=[
            "analysis",
            "fold",
            "target_id",
            "target_interval",
        ],
        how="left",
        validate="one_to_one",
    )

    combined = combined.merge(
        transfer,
        on=[
            "analysis",
            "fold",
            "target_id",
            "target_interval",
        ],
        how="left",
        validate="one_to_one",
    )

    correlations = pd.concat(
        [
            spearman_table(
                combined,
                "within_interval",
            ),
            spearman_table(
                combined,
                "longitudinal_2022_to_2024",
            ),
        ],
        ignore_index=True,
    )

    combined.to_csv(
        OUTPUT_DIR
        / "fold_failure_diagnostics.csv",
        index=False,
    )

    correlations.to_csv(
        OUTPUT_DIR
        / "descriptive_correlations.csv",
        index=False,
    )

    drift_summary = (
        combined.groupby(
            "analysis"
        )[
            [
                "rhr_abs_baseline_shift_z",
                "hrv_abs_baseline_shift_z",
                "temp_abs_baseline_shift_z",
                "resp_abs_baseline_shift_z",
                "mean_abs_baseline_shift_z",
            ]
        ]
        .median()
        .reset_index()
    )

    drift_summary.to_csv(
        OUTPUT_DIR
        / "baseline_drift_summary.csv",
        index=False,
    )

    print("=" * 88)
    print("PERSONALIZATION FAILURE DIAGNOSTICS")
    print("=" * 88)

    print("\nMEDIAN ABSOLUTE BASELINE DRIFT")
    print("(units = signal-wide SD)")
    print(
        drift_summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.3f}"
            ),
        )
    )

    print(
        "\nLONGITUDINAL TARGET-LEVEL DIAGNOSTICS"
    )

    longitudinal = combined[
        combined["analysis"]
        == "longitudinal_2022_to_2024"
    ][
        [
            "target_id",
            "delta_accuracy",
            "delta_log_loss",
            "delta_brier_score",
            "rhr_cal_to_test_corr",
            "hrv_cal_to_test_corr",
            "rhr_baseline_shift_z",
            "hrv_baseline_shift_z",
            "temp_baseline_shift_z",
            "resp_baseline_shift_z",
            "mean_abs_baseline_shift_z",
        ]
    ]

    print(
        longitudinal.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.3f}"
            ),
        )
    )

    print(
        "\nDESCRIPTIVE SPEARMAN CORRELATIONS"
    )
    print(
        "NOTE: exploratory only; small numbers "
        "of independent folds."
    )

    interesting = correlations[
        correlations["outcome"]
        == "delta_log_loss"
    ].copy()

    interesting[
        "abs_rho"
    ] = interesting[
        "spearman_rho"
    ].abs()

    interesting = interesting.sort_values(
        [
            "analysis",
            "abs_rho",
        ],
        ascending=[
            True,
            False,
        ],
    )

    print(
        interesting.drop(
            columns="abs_rho"
        ).to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.3f}"
            ),
        )
    )

    print(
        "\nOutputs:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
