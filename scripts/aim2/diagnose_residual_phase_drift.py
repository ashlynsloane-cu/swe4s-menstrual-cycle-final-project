"""Test whether phase-pattern drift explains residual longitudinal failure."""

from pathlib import Path

import numpy as np
import pandas as pd


ORACLE_METRICS = Path(
    "results/aim2/oracle_recenter/fold_metrics.csv"
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
    "results/aim2/residual_phase_drift"
)

SIGNALS = [
    "rhr",
    "hrv",
    "temp",
    "resp",
]


def performance_table() -> pd.DataFrame:
    """Calculate remaining oracle performance penalty."""

    metrics = pd.read_csv(
        ORACLE_METRICS
    )

    pivot = metrics.pivot(
        index="target_id",
        columns="model",
        values=[
            "accuracy",
            "log_loss",
            "brier_score",
        ],
    )

    pivot.columns = [
        f"{metric}_{model}"
        for metric, model in pivot.columns
    ]

    pivot = pivot.reset_index()

    pivot[
        "residual_log_loss_penalty"
    ] = (
        pivot[
            "log_loss_oracle_2024_recentered"
        ]
        - pivot[
            "log_loss_population_only"
        ]
    )

    pivot[
        "residual_brier_penalty"
    ] = (
        pivot[
            "brier_score_oracle_2024_recentered"
        ]
        - pivot[
            "brier_score_population_only"
        ]
    )

    pivot[
        "residual_accuracy_difference"
    ] = (
        pivot[
            "accuracy_oracle_2024_recentered"
        ]
        - pivot[
            "accuracy_population_only"
        ]
    )

    pivot[
        "baseline_log_loss_rescue"
    ] = (
        pivot[
            "log_loss_hybrid_personalized"
        ]
        - pivot[
            "log_loss_oracle_2024_recentered"
        ]
    )

    return pivot


def transfer_table() -> pd.DataFrame:
    """Create participant-wide phase-profile transfer table."""

    transfer = pd.read_csv(
        TRANSFER_PATH
    )

    transfer = transfer[
        transfer["analysis"]
        == "longitudinal_2022_to_2024"
    ].copy()

    keep = [
        "calibration_cycle_corr",
        "cal_to_test_corr",
        "cal_to_test_rmse",
        "sign_agreement",
    ]

    wide = transfer.pivot(
        index="target_id",
        columns="signal",
        values=keep,
    )

    wide.columns = [
        f"{signal}_{metric}"
        for metric, signal in wide.columns
    ]

    return wide.reset_index()


def stable_transfer_score(
    combined: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Weight transfer by estimated stable person-phase variance."""

    repeatability = pd.read_csv(
        REPEATABILITY_PATH
    )

    stable_variance = {
        row["signal"]: float(
            row[
                "stable_person_phase_variance"
            ]
        )
        for _, row in repeatability.iterrows()
    }

    positive_signals = [
        signal
        for signal in SIGNALS
        if stable_variance.get(
            signal,
            0.0,
        ) > 0
    ]

    output = combined.copy()

    scores = []

    calibration_scores = []

    for _, row in output.iterrows():
        transfer_values = []
        calibration_values = []
        weights = []

        for signal in positive_signals:
            weight = stable_variance[
                signal
            ]

            transfer_value = row.get(
                f"{signal}_cal_to_test_corr"
            )

            calibration_value = row.get(
                f"{signal}_calibration_cycle_corr"
            )

            if pd.notna(
                transfer_value
            ):
                transfer_values.append(
                    float(
                        transfer_value
                    )
                )

                weights.append(
                    weight
                )

            if pd.notna(
                calibration_value
            ):
                calibration_values.append(
                    (
                        float(
                            calibration_value
                        ),
                        weight,
                    )
                )

        if weights:
            score = np.average(
                transfer_values,
                weights=weights,
            )
        else:
            score = np.nan

        if calibration_values:
            values = [
                value
                for value, _
                in calibration_values
            ]

            cal_weights = [
                weight
                for _, weight
                in calibration_values
            ]

            calibration_score = np.average(
                values,
                weights=cal_weights,
            )
        else:
            calibration_score = np.nan

        scores.append(
            score
        )

        calibration_scores.append(
            calibration_score
        )

    output[
        "stable_signal_transfer_score"
    ] = scores

    output[
        "stable_signal_calibration_agreement"
    ] = calibration_scores

    return (
        output,
        stable_variance,
    )


def association(
    data: pd.DataFrame,
    predictor: str,
    outcome: str,
) -> dict:
    """Calculate Spearman association and leave-one-out range."""

    pair = data[
        [
            "target_id",
            predictor,
            outcome,
        ]
    ].dropna()

    rho = pair[
        predictor
    ].corr(
        pair[outcome],
        method="spearman",
    )

    leave_one_out = []

    if len(pair) >= 5:
        for target_id in pair[
            "target_id"
        ]:
            subset = pair[
                pair["target_id"]
                != target_id
            ]

            value = subset[
                predictor
            ].corr(
                subset[outcome],
                method="spearman",
            )

            leave_one_out.append(
                value
            )

    return {
        "predictor": predictor,
        "outcome": outcome,
        "n": len(pair),
        "spearman_rho": rho,
        "loo_rho_min": (
            np.nanmin(
                leave_one_out
            )
            if leave_one_out
            else np.nan
        ),
        "loo_rho_max": (
            np.nanmax(
                leave_one_out
            )
            if leave_one_out
            else np.nan
        ),
    }


def main():
    """Run residual phase-drift diagnostic."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    performance = performance_table()

    transfer = transfer_table()

    combined = performance.merge(
        transfer,
        on="target_id",
        how="inner",
        validate="one_to_one",
    )

    (
        combined,
        stable_variance,
    ) = stable_transfer_score(
        combined
    )

    predictors = [
        "stable_signal_transfer_score",
        "rhr_cal_to_test_corr",
        "hrv_cal_to_test_corr",
        "temp_cal_to_test_corr",
        "resp_cal_to_test_corr",
        "stable_signal_calibration_agreement",
    ]

    outcomes = [
        "residual_log_loss_penalty",
        "residual_brier_penalty",
        "residual_accuracy_difference",
    ]

    rows = []

    for predictor in predictors:
        for outcome in outcomes:
            rows.append(
                association(
                    combined,
                    predictor,
                    outcome,
                )
            )

    associations = pd.DataFrame(
        rows
    )

    combined.to_csv(
        OUTPUT_DIR
        / "participant_residual_penalty.csv",
        index=False,
    )

    associations.to_csv(
        OUTPUT_DIR
        / "descriptive_associations.csv",
        index=False,
    )

    print("=" * 88)
    print("RESIDUAL PHASE-PATTERN DRIFT DIAGNOSTIC")
    print("=" * 88)

    print(
        "\nSTABLE PERSON-PHASE VARIANCE "
        "USED FOR TRANSFER SCORE"
    )

    for signal in SIGNALS:
        print(
            f"{signal:>5}: "
            f"{stable_variance.get(signal, np.nan):.4f}"
        )

    print(
        "\nPARTICIPANT-LEVEL RESULTS"
    )

    display_columns = [
        "target_id",
        "residual_log_loss_penalty",
        "baseline_log_loss_rescue",
        "stable_signal_transfer_score",
        "rhr_cal_to_test_corr",
        "hrv_cal_to_test_corr",
    ]

    print(
        combined[
            display_columns
        ]
        .sort_values(
            "residual_log_loss_penalty",
            ascending=False,
        )
        .to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.3f}"
            ),
        )
    )

    print(
        "\nPHASE TRANSFER VS "
        "RESIDUAL LOG-LOSS PENALTY"
    )

    log_loss = associations[
        associations["outcome"]
        == "residual_log_loss_penalty"
    ].copy()

    print(
        log_loss.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.3f}"
            ),
        )
    )

    print(
        "\nInterpretation direction:"
    )

    print(
        "If better phase-profile transfer explains "
        "the remaining failure, transfer correlations "
        "should have NEGATIVE rho with the residual "
        "log-loss penalty."
    )

    print(
        "\nLeave-one-out ranges show whether the "
        "association depends heavily on one participant."
    )

    print(
        "\nOutputs:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
