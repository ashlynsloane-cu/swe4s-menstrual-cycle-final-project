"""Diagnose weak population-level phase prediction in Aim 2."""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler


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

SUMMARY_PATH = Path(
    "results/aim2/splits/"
    "aim2_within_interval_summary.csv"
)

FOLD_ROOT = Path(
    "results/aim2/centered_primary"
)

HIERARCHICAL_PATH = Path(
    "results/aim2/primary_models/"
    "all_predictions.csv"
)

OUTPUT_DIR = Path(
    "results/aim2/population_diagnostics"
)


def reorder_probabilities(
    model: LogisticRegression,
    probabilities: np.ndarray,
) -> np.ndarray:
    """Reorder model probabilities into PHASES order."""

    model_classes = list(
        model.classes_
    )

    indices = [
        model_classes.index(phase)
        for phase in PHASES
    ]

    return probabilities[
        :,
        indices,
    ]


def predicted_labels(
    probabilities: np.ndarray,
) -> np.ndarray:
    """Convert probabilities to phase calls."""

    return np.asarray(PHASES)[
        probabilities.argmax(axis=1)
    ]


def true_probabilities(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    """Return probability assigned to each true phase."""

    lookup = {
        phase: index
        for index, phase in enumerate(PHASES)
    }

    indices = np.array(
        [
            lookup[phase]
            for phase in y_true
        ]
    )

    return probabilities[
        np.arange(len(y_true)),
        indices,
    ]


def log_loss_safe(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """Calculate order-safe multiclass log loss."""

    true_prob = true_probabilities(
        y_true,
        probabilities,
    )

    return float(
        -np.mean(
            np.log(
                np.clip(
                    true_prob,
                    1e-15,
                    1.0,
                )
            )
        )
    )


def brier_score(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """Calculate multiclass Brier score."""

    lookup = {
        phase: index
        for index, phase in enumerate(PHASES)
    }

    observed = np.zeros_like(
        probabilities,
        dtype=float,
    )

    for row, phase in enumerate(y_true):
        observed[
            row,
            lookup[phase],
        ] = 1.0

    return float(
        np.mean(
            np.sum(
                (
                    probabilities
                    - observed
                ) ** 2,
                axis=1,
            )
        )
    )


def make_prediction_table(
    test: pd.DataFrame,
    probabilities: np.ndarray,
    model_name: str,
    target_id: int,
    target_interval: int,
) -> pd.DataFrame:
    """Create one prediction table."""

    output = pd.DataFrame(
        {
            "target_id": target_id,
            "target_interval": target_interval,
            "day_in_study": test[
                "day_in_study"
            ].to_numpy(),
            "phase": test[
                "phase"
            ].to_numpy(),
            "model": model_name,
            "predicted_phase": (
                predicted_labels(
                    probabilities
                )
            ),
        }
    )

    for index, phase in enumerate(PHASES):
        output[
            f"prob_{phase.lower()}"
        ] = probabilities[
            :,
            index,
        ]

    return output


def evaluate(
    data: pd.DataFrame,
) -> dict[str, float]:
    """Calculate pooled diagnostic metrics."""

    y_true = data[
        "phase"
    ].to_numpy()

    y_pred = data[
        "predicted_phase"
    ].to_numpy()

    probabilities = data[
        [
            f"prob_{phase.lower()}"
            for phase in PHASES
        ]
    ].to_numpy()

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "balanced_accuracy": (
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=PHASES,
            average="macro",
            zero_division=0,
        ),
        "log_loss": log_loss_safe(
            y_true,
            probabilities,
        ),
        "brier_score": brier_score(
            y_true,
            probabilities,
        ),
        "mean_true_probability": float(
            true_probabilities(
                y_true,
                probabilities,
            ).mean()
        ),
    }


def class_summary(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize phase-specific behavior."""

    y_true = data[
        "phase"
    ].to_numpy()

    y_pred = data[
        "predicted_phase"
    ].to_numpy()

    recalls = recall_score(
        y_true,
        y_pred,
        labels=PHASES,
        average=None,
        zero_division=0,
    )

    prediction_counts = (
        pd.Series(y_pred)
        .value_counts()
    )

    rows = []

    for index, phase in enumerate(PHASES):
        rows.append(
            {
                "phase": phase,
                "n_true": int(
                    np.sum(
                        y_true == phase
                    )
                ),
                "n_predicted": int(
                    prediction_counts.get(
                        phase,
                        0,
                    )
                ),
                "recall": recalls[index],
            }
        )

    return pd.DataFrame(rows)


def fit_fold_models(
    population: pd.DataFrame,
    test: pd.DataFrame,
    target_id: int,
    target_interval: int,
) -> list[pd.DataFrame]:
    """Fit prior-only and logistic population diagnostics."""

    scaler = StandardScaler()

    x_train = scaler.fit_transform(
        population[PREDICTORS]
    )

    x_test = scaler.transform(
        test[PREDICTORS]
    )

    y_train = population[
        "phase"
    ].to_numpy()

    # ----------------------------------------------------------
    # 1. Training-prior-only baseline
    # ----------------------------------------------------------

    training_counts = (
        population["phase"]
        .value_counts(normalize=True)
    )

    prior = np.array(
        [
            training_counts.get(
                phase,
                0.0,
            )
            for phase in PHASES
        ]
    )

    prior_probabilities = np.tile(
        prior,
        (
            len(test),
            1,
        ),
    )

    # ----------------------------------------------------------
    # 2. Ordinary multinomial logistic regression
    # ----------------------------------------------------------

    logistic = LogisticRegression(
        solver="lbfgs",
        max_iter=5000,
        random_state=20261014,
    )

    logistic.fit(
        x_train,
        y_train,
    )

    logistic_probabilities = (
        reorder_probabilities(
            logistic,
            logistic.predict_proba(
                x_test
            ),
        )
    )

    # ----------------------------------------------------------
    # 3. Class-balanced multinomial logistic regression
    # ----------------------------------------------------------

    balanced = LogisticRegression(
        solver="lbfgs",
        max_iter=5000,
        class_weight="balanced",
        random_state=20261014,
    )

    balanced.fit(
        x_train,
        y_train,
    )

    balanced_probabilities = (
        reorder_probabilities(
            balanced,
            balanced.predict_proba(
                x_test
            ),
        )
    )

    return [
        make_prediction_table(
            test,
            prior_probabilities,
            "training_prior_only",
            target_id,
            target_interval,
        ),
        make_prediction_table(
            test,
            logistic_probabilities,
            "plain_logistic",
            target_id,
            target_interval,
        ),
        make_prediction_table(
            test,
            balanced_probabilities,
            "balanced_logistic",
            target_id,
            target_interval,
        ),
    ]


def hierarchical_predictions() -> pd.DataFrame:
    """Load existing frozen population-model predictions."""

    predictions = pd.read_csv(
        HIERARCHICAL_PATH
    )

    predictions = predictions[
        predictions["model"]
        == "population_only"
    ].copy()

    keep = [
        "target_id",
        "target_interval",
        "day_in_study",
        "phase",
        "predicted_phase",
    ]

    probability_columns = [
        f"prob_{phase.lower()}"
        for phase in PHASES
    ]

    output = predictions[
        keep
        + probability_columns
    ].copy()

    output["model"] = (
        "hierarchical_population"
    )

    return output


def main() -> None:
    """Run all population-model diagnostics."""

    summary = pd.read_csv(
        SUMMARY_PATH
    )

    prediction_tables = []

    for _, row in summary.iterrows():
        target_id = int(
            row["target_id"]
        )

        target_interval = int(
            row["target_interval"]
        )

        folder = (
            FOLD_ROOT
            / (
                f"id{target_id}_"
                f"{target_interval}"
            )
        )

        population = pd.read_csv(
            folder
            / "population_train.csv"
        )

        test = pd.read_csv(
            folder
            / "future_test.csv"
        )

        prediction_tables.extend(
            fit_fold_models(
                population,
                test,
                target_id,
                target_interval,
            )
        )

    prediction_tables.append(
        hierarchical_predictions()
    )

    predictions = pd.concat(
        prediction_tables,
        ignore_index=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        OUTPUT_DIR
        / "population_model_predictions.csv",
        index=False,
    )

    print("=" * 76)
    print("POPULATION MODEL DIAGNOSTIC")
    print("=" * 76)

    metric_rows = []

    model_order = [
        "training_prior_only",
        "plain_logistic",
        "balanced_logistic",
        "hierarchical_population",
    ]

    for model_name in model_order:
        data = predictions[
            predictions["model"]
            == model_name
        ].copy()

        metrics = evaluate(
            data
        )

        metric_rows.append(
            {
                "model": model_name,
                **metrics,
            }
        )

    metrics_table = pd.DataFrame(
        metric_rows
    )

    metrics_table.to_csv(
        OUTPUT_DIR
        / "population_model_metrics.csv",
        index=False,
    )

    print("\nOVERALL METRICS")
    print(
        metrics_table.to_string(
            index=False
        )
    )

    for model_name in model_order:
        data = predictions[
            predictions["model"]
            == model_name
        ]

        print(
            "\n"
            + "=" * 76
        )

        print(
            model_name.upper()
        )

        print(
            "=" * 76
        )

        classes = class_summary(
            data
        )

        print(
            classes.to_string(
                index=False
            )
        )

        matrix = confusion_matrix(
            data["phase"],
            data["predicted_phase"],
            labels=PHASES,
        )

        matrix_table = pd.DataFrame(
            matrix,
            index=[
                f"true_{phase}"
                for phase in PHASES
            ],
            columns=[
                f"pred_{phase}"
                for phase in PHASES
            ],
        )

        print("\nConfusion matrix:")
        print(
            matrix_table.to_string()
        )

        matrix_table.to_csv(
            OUTPUT_DIR
            / (
                "confusion_"
                f"{model_name}.csv"
            )
        )


if __name__ == "__main__":
    main()
