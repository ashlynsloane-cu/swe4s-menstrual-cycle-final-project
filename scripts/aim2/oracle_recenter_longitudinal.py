"""Test whether 2022-to-2024 baseline drift causes personalization failure."""

from pathlib import Path
import importlib.util

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

RUNNER_PATH = Path(
    "scripts/aim2/run_longitudinal_hierarchical_models.py"
)

FOLD_ROOT = Path(
    "results/aim2/longitudinal_folds"
)

MODEL_ROOT = Path(
    "results/aim2/longitudinal_models"
)

OUTPUT_ROOT = Path(
    "results/aim2/oracle_recenter"
)

SIGNALS = {
    "rhr": "resting_heart_rate",
    "hrv": "hrv_rmssd",
    "temp": "nightly_temperature",
    "resp": "respiratory_rate",
}

PHASES = [
    "Menstrual",
    "Follicular",
    "Fertility",
    "Luteal",
]

MODELS = [
    "population_only",
    "hybrid_personalized",
    "oracle_2024_recentered",
]


def load_runner():
    """Import the frozen longitudinal runner."""

    spec = importlib.util.spec_from_file_location(
        "longitudinal_runner",
        RUNNER_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not import longitudinal runner."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def make_hybrid_model(
    runner,
    hybrid: pd.DataFrame,
):
    """Reconstruct the already-fitted Bambi model specification."""

    model = bmb.Model(
        runner.FORMULA,
        hybrid,
        family="categorical",
        categorical=["model_id"],
        priors=runner.make_priors(),
        auto_scale=False,
        center_predictors=False,
        dropna=False,
    )

    model.build()

    return model


def use_calibration_group(
    test: pd.DataFrame,
    hybrid: pd.DataFrame,
    target_id: int,
) -> pd.DataFrame:
    """Map 2024 test data to the target's 2022 random effects."""

    output = test.copy()

    calibration_group = (
        f"{target_id}_2022"
    )

    categories = (
        hybrid["model_id"]
        .cat.categories
    )

    if calibration_group not in categories:
        raise RuntimeError(
            f"Missing model group {calibration_group}."
        )

    output["model_id"] = calibration_group

    output["model_id"] = pd.Categorical(
        output["model_id"],
        categories=categories,
    )

    return output


def make_oracle_raw_test(
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Recenter target test data on its observed 2024 means."""

    output = test.copy()

    baselines = {}

    for short_name, raw_column in SIGNALS.items():
        baseline_column = (
            f"{short_name}_baseline"
        )

        within_column = (
            f"{short_name}_within"
        )

        baseline = float(
            output[raw_column].mean()
        )

        baselines[short_name] = baseline

        output[baseline_column] = baseline

        output[within_column] = (
            output[raw_column]
            - baseline
        )

    return output, baselines


def saved_probability_matrix(
    predictions: pd.DataFrame,
    labels: list[str],
) -> np.ndarray:
    """Extract saved probabilities in Bambi label order."""

    columns = [
        f"prob_{label.lower()}"
        for label in labels
    ]

    return predictions[
        columns
    ].to_numpy(
        dtype=float
    )


def evaluate(
    truth: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
) -> dict:
    """Calculate multiclass performance metrics."""

    label_to_index = {
        label: index
        for index, label in enumerate(labels)
    }

    truth_index = np.array(
        [
            label_to_index[value]
            for value in truth
        ],
        dtype=int,
    )

    predicted_index = probabilities.argmax(
        axis=1
    )

    predicted = np.array(
        [
            labels[index]
            for index in predicted_index
        ]
    )

    rows = np.arange(
        len(truth)
    )

    true_probability = probabilities[
        rows,
        truth_index,
    ]

    one_hot = np.zeros_like(
        probabilities
    )

    one_hot[
        rows,
        truth_index,
    ] = 1.0

    return {
        "n_days": len(truth),
        "accuracy": accuracy_score(
            truth,
            predicted,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            truth,
            predicted,
        ),
        "macro_f1": f1_score(
            truth,
            predicted,
            labels=PHASES,
            average="macro",
            zero_division=0,
        ),
        "log_loss": float(
            -np.mean(
                np.log(
                    np.clip(
                        true_probability,
                        1e-15,
                        1.0,
                    )
                )
            )
        ),
        "brier_score": float(
            np.mean(
                np.sum(
                    (
                        probabilities
                        - one_hot
                    )
                    ** 2,
                    axis=1,
                )
            )
        ),
        "mean_true_probability": float(
            np.mean(
                true_probability
            )
        ),
    }


def make_prediction_frame(
    test: pd.DataFrame,
    probabilities: np.ndarray,
    labels: list[str],
    model_name: str,
) -> pd.DataFrame:
    """Build a standardized prediction table."""

    output = test[
        [
            "target_id",
            "target_interval",
            "source_id",
            "study_interval",
            "day_in_study",
            "cycle_number",
            "interval_cycle_index",
            "phase",
        ]
    ].copy()

    output["model"] = model_name

    output["predicted_phase"] = [
        labels[index]
        for index in probabilities.argmax(
            axis=1
        )
    ]

    for index, label in enumerate(labels):
        output[
            f"prob_{label.lower()}"
        ] = probabilities[
            :,
            index,
        ]

    return output


def analyze_fold(
    runner,
    target_id: int,
) -> tuple[list[dict], pd.DataFrame, dict]:
    """Run original-reproduction and oracle predictions for one target."""

    fold_name = (
        f"id{target_id}"
    )

    fold_dir = (
        FOLD_ROOT
        / fold_name
    )

    model_dir = (
        MODEL_ROOT
        / fold_name
    )

    population_raw = pd.read_csv(
        fold_dir
        / "population_train.csv"
    )

    calibration_raw = pd.read_csv(
        fold_dir
        / "personal_calibration.csv"
    )

    test_raw = pd.read_csv(
        fold_dir
        / "future_test.csv"
    )

    saved = pd.read_csv(
        model_dir
        / "predictions.csv"
    )

    (
        population,
        calibration,
        test,
    ) = runner.standardize_fold(
        population_raw.copy(),
        calibration_raw.copy(),
        test_raw.copy(),
    )

    population = runner.add_model_group(
        population
    )

    calibration = runner.add_model_group(
        calibration
    )

    test = runner.add_model_group(
        test
    )

    hybrid = pd.concat(
        [
            population,
            calibration,
        ],
        ignore_index=True,
    )

    hybrid["model_id"] = (
        hybrid["model_id"]
        .astype(str)
        .astype("category")
    )

    model = make_hybrid_model(
        runner,
        hybrid,
    )

    idata = az.from_netcdf(
        model_dir
        / "hybrid_posterior.nc"
    )

    original_test = use_calibration_group(
        test,
        hybrid,
        target_id,
    )

    (
        reproduced_probability,
        labels,
    ) = runner.extract_probabilities(
        model,
        idata,
        original_test,
        include_group_specific=True,
    )

    saved_hybrid = saved[
        saved["model"]
        == "hybrid_personalized"
    ].reset_index(
        drop=True
    )

    saved_population = saved[
        saved["model"]
        == "population_only"
    ].reset_index(
        drop=True
    )

    if len(saved_hybrid) != len(test):
        raise RuntimeError(
            f"{fold_name}: saved/test row mismatch."
        )

    saved_hybrid_probability = (
        saved_probability_matrix(
            saved_hybrid,
            labels,
        )
    )

    maximum_difference = float(
        np.max(
            np.abs(
                reproduced_probability
                - saved_hybrid_probability
            )
        )
    )

    if maximum_difference > 1e-6:
        raise RuntimeError(
            f"{fold_name}: failed reproduction. "
            f"Max probability difference = "
            f"{maximum_difference:.8f}"
        )

    oracle_raw, oracle_baselines = (
        make_oracle_raw_test(
            test_raw
        )
    )

    (
        _,
        _,
        oracle_test,
    ) = runner.standardize_fold(
        population_raw.copy(),
        calibration_raw.copy(),
        oracle_raw,
    )

    oracle_test = runner.add_model_group(
        oracle_test
    )

    oracle_test = use_calibration_group(
        oracle_test,
        hybrid,
        target_id,
    )

    (
        oracle_probability,
        oracle_labels,
    ) = runner.extract_probabilities(
        model,
        idata,
        oracle_test,
        include_group_specific=True,
    )

    if oracle_labels != labels:
        raise RuntimeError(
            f"{fold_name}: category order changed."
        )

    population_probability = (
        saved_probability_matrix(
            saved_population,
            labels,
        )
    )

    truth = test[
        "phase"
    ].astype(str).to_numpy()

    probabilities = {
        "population_only": (
            population_probability
        ),
        "hybrid_personalized": (
            saved_hybrid_probability
        ),
        "oracle_2024_recentered": (
            oracle_probability
        ),
    }

    metric_rows = []

    prediction_frames = []

    for model_name, probability in (
        probabilities.items()
    ):
        metrics = evaluate(
            truth,
            probability,
            labels,
        )

        metric_rows.append(
            {
                "target_id": target_id,
                "target_interval": 2024,
                "model": model_name,
                **metrics,
            }
        )

        prediction_frames.append(
            make_prediction_frame(
                test,
                probability,
                labels,
                model_name,
            )
        )

    baseline_row = {
        "target_id": target_id,
        "max_reproduction_difference": (
            maximum_difference
        ),
    }

    for short_name in SIGNALS:
        original_baseline = float(
            test_raw[
                f"{short_name}_baseline"
            ].iloc[0]
        )

        oracle_baseline = (
            oracle_baselines[
                short_name
            ]
        )

        baseline_row[
            f"{short_name}_2022_baseline"
        ] = original_baseline

        baseline_row[
            f"{short_name}_2024_oracle_baseline"
        ] = oracle_baseline

        baseline_row[
            f"{short_name}_baseline_change"
        ] = (
            oracle_baseline
            - original_baseline
        )

    return (
        metric_rows,
        pd.concat(
            prediction_frames,
            ignore_index=True,
        ),
        baseline_row,
    )


def pooled_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate pooled metrics for all targets."""

    rows = []

    labels = PHASES

    for model_name in MODELS:
        subset = predictions[
            predictions["model"]
            == model_name
        ]

        probabilities = subset[
            [
                f"prob_{label.lower()}"
                for label in labels
            ]
        ].to_numpy()

        metrics = evaluate(
            subset["phase"]
            .astype(str)
            .to_numpy(),
            probabilities,
            labels,
        )

        rows.append(
            {
                "model": model_name,
                **metrics,
            }
        )

    return pd.DataFrame(
        rows
    )


def cluster_bootstrap(
    predictions: pd.DataFrame,
    n_boot: int = 5000,
    seed: int = 6118,
) -> pd.DataFrame:
    """Bootstrap model differences by participant."""

    rng = np.random.default_rng(
        seed
    )

    participants = np.array(
        sorted(
            predictions[
                "target_id"
            ].unique()
        )
    )

    metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "log_loss",
        "brier_score",
        "mean_true_probability",
    ]

    rows = []

    for iteration in range(
        n_boot
    ):
        sampled = rng.choice(
            participants,
            size=len(participants),
            replace=True,
        )

        pieces = []

        for draw_index, target_id in enumerate(
            sampled
        ):
            part = predictions[
                predictions["target_id"]
                == target_id
            ].copy()

            part["_bootstrap_id"] = (
                draw_index
            )

            pieces.append(
                part
            )

        sample = pd.concat(
            pieces,
            ignore_index=True,
        )

        model_metrics = {}

        for model_name in MODELS:
            subset = sample[
                sample["model"]
                == model_name
            ]

            probabilities = subset[
                [
                    f"prob_{label.lower()}"
                    for label in PHASES
                ]
            ].to_numpy()

            model_metrics[
                model_name
            ] = evaluate(
                subset["phase"]
                .astype(str)
                .to_numpy(),
                probabilities,
                PHASES,
            )

        row = {
            "iteration": iteration,
        }

        for metric in metrics:
            row[
                f"oracle_minus_hybrid_{metric}"
            ] = (
                model_metrics[
                    "oracle_2024_recentered"
                ][metric]
                - model_metrics[
                    "hybrid_personalized"
                ][metric]
            )

            row[
                f"oracle_minus_population_{metric}"
            ] = (
                model_metrics[
                    "oracle_2024_recentered"
                ][metric]
                - model_metrics[
                    "population_only"
                ][metric]
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def summarize_bootstrap(
    bootstrap: pd.DataFrame,
    observed: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize participant-bootstrap differences."""

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "log_loss",
        "brier_score",
        "mean_true_probability",
    ]

    indexed = observed.set_index(
        "model"
    )

    rows = []

    comparisons = {
        "oracle_minus_hybrid": (
            "oracle_2024_recentered",
            "hybrid_personalized",
        ),
        "oracle_minus_population": (
            "oracle_2024_recentered",
            "population_only",
        ),
    }

    for comparison, (
        first,
        second,
    ) in comparisons.items():
        for metric in metric_names:
            column = (
                f"{comparison}_{metric}"
            )

            values = bootstrap[
                column
            ]

            observed_difference = (
                indexed.loc[
                    first,
                    metric,
                ]
                - indexed.loc[
                    second,
                    metric,
                ]
            )

            rows.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "observed_difference": (
                        observed_difference
                    ),
                    "ci_2.5": values.quantile(
                        0.025
                    ),
                    "ci_97.5": values.quantile(
                        0.975
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def main():
    """Run oracle baseline recentering diagnostic."""

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    runner = load_runner()

    targets = [
        26,
        30,
        33,
        38,
        41,
        43,
        47,
        48,
        50,
    ]

    metric_rows = []
    prediction_frames = []
    baseline_rows = []

    for target_id in targets:
        print(
            f"Processing ID {target_id}..."
        )

        (
            fold_metrics,
            fold_predictions,
            baseline_row,
        ) = analyze_fold(
            runner,
            target_id,
        )

        metric_rows.extend(
            fold_metrics
        )

        prediction_frames.append(
            fold_predictions
        )

        baseline_rows.append(
            baseline_row
        )

    fold_metrics = pd.DataFrame(
        metric_rows
    )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    baselines = pd.DataFrame(
        baseline_rows
    )

    overall = pooled_metrics(
        predictions
    )

    bootstrap = cluster_bootstrap(
        predictions
    )

    bootstrap_summary = summarize_bootstrap(
        bootstrap,
        overall,
    )

    fold_metrics.to_csv(
        OUTPUT_ROOT
        / "fold_metrics.csv",
        index=False,
    )

    predictions.to_csv(
        OUTPUT_ROOT
        / "all_predictions.csv",
        index=False,
    )

    baselines.to_csv(
        OUTPUT_ROOT
        / "baseline_changes.csv",
        index=False,
    )

    overall.to_csv(
        OUTPUT_ROOT
        / "overall_metrics.csv",
        index=False,
    )

    bootstrap.to_csv(
        OUTPUT_ROOT
        / "participant_bootstrap.csv",
        index=False,
    )

    bootstrap_summary.to_csv(
        OUTPUT_ROOT
        / "bootstrap_summary.csv",
        index=False,
    )

    print("\n" + "=" * 86)
    print("ORACLE 2024 RECENTERING DIAGNOSTIC")
    print("=" * 86)

    print(
        "\nMaximum reproduction error:",
        baselines[
            "max_reproduction_difference"
        ].max(),
    )

    print("\nOVERALL PERFORMANCE")
    print(
        overall.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    print(
        "\nORACLE DIFFERENCES"
    )

    print(
        bootstrap_summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    print(
        "\nFOLD LOG LOSS"
    )

    log_loss = fold_metrics.pivot(
        index="target_id",
        columns="model",
        values="log_loss",
    ).reset_index()

    print(
        log_loss.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    print(
        "\nOutputs:",
        OUTPUT_ROOT,
    )


if __name__ == "__main__":
    main()
