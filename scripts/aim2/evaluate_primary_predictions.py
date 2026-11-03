"""Evaluate frozen Aim 2 population and personalized predictions."""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


INPUT_PATH = Path(
    "results/aim2/primary_models/all_predictions.csv"
)

OUTPUT_DIR = Path(
    "results/aim2/evaluation"
)

PHASES = [
    "Menstrual",
    "Follicular",
    "Fertility",
    "Luteal",
]

MODELS = [
    "population_only",
    "hybrid_personalized",
]

KEY_COLUMNS = [
    "target_id",
    "target_interval",
    "source_id",
    "study_interval",
    "day_in_study",
    "cycle_number",
    "interval_cycle_index",
    "phase",
]

N_BOOTSTRAPS = 5000
SEED = 20261014


def probability_columns() -> list[str]:
    """Return phase-probability columns in PHASES order."""

    return [
        f"prob_{phase.lower()}"
        for phase in PHASES
    ]


def brier_score(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """Calculate multiclass Brier score."""

    phase_lookup = {
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
            phase_lookup[phase],
        ] = 1.0

    return float(
        np.mean(
            np.sum(
                (probabilities - observed) ** 2,
                axis=1,
            )
        )
    )


def true_phase_probability(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    """Return probability assigned to the observed phase."""

    phase_lookup = {
        phase: index
        for index, phase in enumerate(PHASES)
    }

    indices = np.array(
        [
            phase_lookup[phase]
            for phase in y_true
        ]
    )

    return probabilities[
        np.arange(len(y_true)),
        indices,
    ]


def calculate_metrics(
    data: pd.DataFrame,
) -> dict[str, float]:
    """Calculate pooled prediction metrics."""

    y_true = data["phase"].to_numpy()
    y_pred = data["predicted_phase"].to_numpy()

    probabilities = data[
        probability_columns()
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
        "log_loss": float(
            -np.mean(
                np.log(
                    np.clip(
                        true_phase_probability(
                            y_true,
                            probabilities,
                        ),
                        1e-15,
                        1.0,
                    )
                )
            )
        ),
        "brier_score": brier_score(
            y_true,
            probabilities,
        ),
        "mean_true_phase_probability": float(
            true_phase_probability(
                y_true,
                probabilities,
            ).mean()
        ),
    }


def class_diagnostics(
    data: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    """Calculate class-specific prediction diagnostics."""

    y_true = data["phase"].to_numpy()
    y_pred = data["predicted_phase"].to_numpy()

    probabilities = data[
        probability_columns()
    ].to_numpy()

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=PHASES,
            zero_division=0,
        )
    )

    predicted_counts = (
        pd.Series(y_pred)
        .value_counts()
    )

    rows = []

    for index, phase in enumerate(PHASES):
        phase_mask = y_true == phase

        phase_probability = (
            probabilities[
                phase_mask,
                index,
            ]
        )

        rows.append(
            {
                "model": model_name,
                "phase": phase,
                "n_true": int(
                    support[index]
                ),
                "n_predicted": int(
                    predicted_counts.get(
                        phase,
                        0,
                    )
                ),
                "precision": precision[index],
                "recall": recall[index],
                "f1": f1[index],
                "mean_probability_when_true": (
                    float(
                        phase_probability.mean()
                    )
                    if len(phase_probability)
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def confusion_table(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Return confusion matrix as a labeled table."""

    matrix = confusion_matrix(
        data["phase"],
        data["predicted_phase"],
        labels=PHASES,
    )

    return pd.DataFrame(
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


def prepare_paired(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Create one matched row per held-out day."""

    population = predictions[
        predictions["model"]
        == "population_only"
    ].copy()

    hybrid = predictions[
        predictions["model"]
        == "hybrid_personalized"
    ].copy()

    probability_cols = probability_columns()

    population = population[
        KEY_COLUMNS
        + ["predicted_phase"]
        + probability_cols
    ]

    hybrid = hybrid[
        KEY_COLUMNS
        + ["predicted_phase"]
        + probability_cols
    ]

    paired = population.merge(
        hybrid,
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
        suffixes=("_population", "_hybrid"),
    )

    if len(paired) != len(population):
        raise RuntimeError(
            "Population and hybrid predictions "
            "are not perfectly matched."
        )

    return paired


def paired_metric_values(
    data: pd.DataFrame,
) -> dict[str, float]:
    """Calculate Hybrid minus Population metric differences."""

    pop = pd.DataFrame(
        {
            "phase": data["phase"],
            "predicted_phase": (
                data[
                    "predicted_phase_population"
                ]
            ),
        }
    )

    hybrid = pd.DataFrame(
        {
            "phase": data["phase"],
            "predicted_phase": (
                data[
                    "predicted_phase_hybrid"
                ]
            ),
        }
    )

    for column in probability_columns():
        pop[column] = data[
            f"{column}_population"
        ]

        hybrid[column] = data[
            f"{column}_hybrid"
        ]

    pop_metrics = calculate_metrics(pop)
    hybrid_metrics = calculate_metrics(hybrid)

    return {
        metric: (
            hybrid_metrics[metric]
            - pop_metrics[metric]
        )
        for metric in pop_metrics
    }


def cluster_bootstrap(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    """Bootstrap unique participants as clusters."""

    rng = np.random.default_rng(SEED)

    participant_ids = np.array(
        sorted(
            paired["target_id"]
            .unique()
        )
    )

    participant_indices = {
        participant_id: np.flatnonzero(
            paired["target_id"].to_numpy()
            == participant_id
        )
        for participant_id in participant_ids
    }

    rows = []

    for bootstrap_index in range(
        N_BOOTSTRAPS
    ):
        sampled_ids = rng.choice(
            participant_ids,
            size=len(participant_ids),
            replace=True,
        )

        sampled_indices = np.concatenate(
            [
                participant_indices[
                    participant_id
                ]
                for participant_id
                in sampled_ids
            ]
        )

        sample = paired.iloc[
            sampled_indices
        ]

        differences = paired_metric_values(
            sample
        )

        differences["bootstrap"] = (
            bootstrap_index
        )

        rows.append(differences)

    return pd.DataFrame(rows)


def bootstrap_summary(
    observed: dict[str, float],
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize participant-cluster bootstrap intervals."""

    rows = []

    for metric, value in observed.items():
        values = bootstrap[
            metric
        ].to_numpy()

        rows.append(
            {
                "metric": metric,
                "hybrid_minus_population": value,
                "ci_2.5": np.quantile(
                    values,
                    0.025,
                ),
                "ci_97.5": np.quantile(
                    values,
                    0.975,
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    """Run complete frozen-primary evaluation."""

    predictions = pd.read_csv(
        INPUT_PATH
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("AIM 2 PRIMARY PREDICTION DIAGNOSTICS")
    print("=" * 72)

    print(
        "Unique participants:",
        predictions["target_id"].nunique(),
    )

    print(
        "Participant-intervals:",
        predictions[
            [
                "target_id",
                "target_interval",
            ]
        ]
        .drop_duplicates()
        .shape[0],
    )

    print(
        "Held-out days:",
        len(
            predictions[
                predictions["model"]
                == "population_only"
            ]
        ),
    )

    metric_rows = []
    class_tables = []

    for model_name in MODELS:
        data = predictions[
            predictions["model"]
            == model_name
        ].copy()

        metrics = calculate_metrics(
            data
        )

        metric_rows.append(
            {
                "model": model_name,
                **metrics,
            }
        )

        class_table = class_diagnostics(
            data,
            model_name,
        )

        class_tables.append(
            class_table
        )

        confusion = confusion_table(
            data
        )

        confusion.to_csv(
            OUTPUT_DIR
            / f"confusion_{model_name}.csv"
        )

        print(
            f"\n{model_name.upper()}"
        )
        print("-" * 72)

        for metric, value in metrics.items():
            print(
                f"{metric:32s} "
                f"{value:.4f}"
            )

        print("\nClass diagnostics:")

        print(
            class_table.to_string(
                index=False
            )
        )

        print("\nConfusion matrix:")

        print(
            confusion.to_string()
        )

    metrics_table = pd.DataFrame(
        metric_rows
    )

    classes_table = pd.concat(
        class_tables,
        ignore_index=True,
    )

    metrics_table.to_csv(
        OUTPUT_DIR
        / "overall_metrics.csv",
        index=False,
    )

    classes_table.to_csv(
        OUTPUT_DIR
        / "class_diagnostics.csv",
        index=False,
    )

    paired = prepare_paired(
        predictions
    )

    observed = paired_metric_values(
        paired
    )

    print("\n" + "=" * 72)
    print("HYBRID MINUS POPULATION")
    print("=" * 72)

    for metric, value in observed.items():
        print(
            f"{metric:32s} "
            f"{value:+.4f}"
        )

    print(
        "\nRunning participant-cluster "
        f"bootstrap ({N_BOOTSTRAPS} resamples)..."
    )

    bootstrap = cluster_bootstrap(
        paired
    )

    summary = bootstrap_summary(
        observed,
        bootstrap,
    )

    bootstrap.to_csv(
        OUTPUT_DIR
        / "cluster_bootstrap.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "cluster_bootstrap_summary.csv",
        index=False,
    )

    print("\nBootstrap summary:")

    print(
        summary.to_string(
            index=False,
        )
    )

    print(
        "\nNOTE:"
    )

    print(
        "For accuracy, balanced accuracy, "
        "macro-F1, and mean true-phase "
        "probability, positive differences "
        "favor personalization."
    )

    print(
        "For log loss and Brier score, "
        "negative differences favor "
        "personalization."
    )


if __name__ == "__main__":
    main()
