"""Evaluate 2022-to-2024 Aim 2 longitudinal predictions."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

INPUT_PATH = Path(
    "results/aim2/longitudinal_models/"
    "all_predictions.csv"
)

OUTPUT_DIR = Path(
    "results/aim2/longitudinal_evaluation"
)

PHASES = [
    "Menstrual",
    "Follicular",
    "Fertility",
    "Luteal",
]

PROB_COLUMNS = {
    "Menstrual": "prob_menstrual",
    "Follicular": "prob_follicular",
    "Fertility": "prob_fertility",
    "Luteal": "prob_luteal",
}

MODELS = [
    "population_only",
    "hybrid_personalized",
]


def evaluate(
    data: pd.DataFrame,
) -> dict[str, float]:
    """Calculate multiclass prediction metrics."""

    y_true = data["phase"].to_numpy()
    y_pred = data["predicted_phase"].to_numpy()

    probabilities = data[
        [
            PROB_COLUMNS[phase]
            for phase in PHASES
        ]
    ].to_numpy()

    phase_to_index = {
        phase: index
        for index, phase in enumerate(PHASES)
    }

    true_indices = np.array(
        [
            phase_to_index[phase]
            for phase in y_true
        ]
    )

    row_indices = np.arange(
        len(data)
    )

    true_probabilities = probabilities[
        row_indices,
        true_indices,
    ]

    clipped = np.clip(
        true_probabilities,
        1e-15,
        1.0,
    )

    one_hot = np.zeros_like(
        probabilities
    )

    one_hot[
        row_indices,
        true_indices,
    ] = 1.0

    return {
        "n_days": len(data),
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
                np.log(clipped)
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
                true_probabilities
            )
        ),
    }


def class_summary(
    data: pd.DataFrame,
    model_name: str,
) -> list[dict]:
    """Summarize predictions separately for each phase."""

    rows = []

    for phase in PHASES:
        true_mask = (
            data["phase"] == phase
        )

        predicted_mask = (
            data["predicted_phase"] == phase
        )

        n_true = int(
            true_mask.sum()
        )

        correct = int(
            (
                true_mask
                & predicted_mask
            ).sum()
        )

        recall = (
            correct / n_true
            if n_true
            else np.nan
        )

        rows.append(
            {
                "model": model_name,
                "phase": phase,
                "n_true": n_true,
                "n_predicted": int(
                    predicted_mask.sum()
                ),
                "recall": recall,
                "mean_true_probability": (
                    data.loc[
                        true_mask,
                        PROB_COLUMNS[phase],
                    ].mean()
                ),
            }
        )

    return rows


def participant_bootstrap(
    data: pd.DataFrame,
    n_boot: int = 5000,
    seed: int = 6118,
) -> pd.DataFrame:
    """Paired participant-cluster bootstrap."""

    participants = np.array(
        sorted(
            data["target_id"]
            .unique()
        )
    )

    rng = np.random.default_rng(
        seed
    )

    rows = []

    for iteration in range(n_boot):
        sampled_ids = rng.choice(
            participants,
            size=len(participants),
            replace=True,
        )

        sampled_parts = []

        for bootstrap_index, target_id in enumerate(
            sampled_ids
        ):
            part = data[
                data["target_id"]
                == target_id
            ].copy()

            # Give repeated sampled participants unique
            # bootstrap-cluster labels.
            part[
                "_bootstrap_cluster"
            ] = bootstrap_index

            sampled_parts.append(
                part
            )

        sample = pd.concat(
            sampled_parts,
            ignore_index=True,
        )

        metrics = {}

        for model_name in MODELS:
            model_data = sample[
                sample["model"]
                == model_name
            ]

            metrics[model_name] = evaluate(
                model_data
            )

        row = {
            "iteration": iteration,
        }

        for metric in [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "log_loss",
            "brier_score",
            "mean_true_probability",
        ]:
            row[
                f"{metric}_difference"
            ] = (
                metrics[
                    "hybrid_personalized"
                ][metric]
                - metrics[
                    "population_only"
                ][metric]
            )

        rows.append(row)

    return pd.DataFrame(
        rows
    )


def main() -> None:
    """Run longitudinal evaluation."""

    data = pd.read_csv(
        INPUT_PATH
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if len(data) != 1402:
        raise RuntimeError(
            "Expected 1402 prediction rows "
            f"but found {len(data)}."
        )

    if data["target_id"].nunique() != 9:
        raise RuntimeError(
            "Expected 9 longitudinal targets."
        )

    overall_rows = []
    class_rows = []

    for model_name in MODELS:
        model_data = data[
            data["model"] == model_name
        ].copy()

        metrics = evaluate(
            model_data
        )

        metrics["model"] = model_name

        overall_rows.append(
            metrics
        )

        class_rows.extend(
            class_summary(
                model_data,
                model_name,
            )
        )

    overall = pd.DataFrame(
        overall_rows
    )

    overall = overall[
        [
            "model",
            "n_days",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "log_loss",
            "brier_score",
            "mean_true_probability",
        ]
    ]

    classes = pd.DataFrame(
        class_rows
    )

    bootstrap = participant_bootstrap(
        data
    )

    summary_rows = []

    for column in bootstrap.columns:
        if not column.endswith(
            "_difference"
        ):
            continue

        values = bootstrap[
            column
        ].to_numpy()

        metric = column.replace(
            "_difference",
            "",
        )

        observed = (
            overall.loc[
                overall["model"]
                == "hybrid_personalized",
                metric,
            ].iloc[0]
            - overall.loc[
                overall["model"]
                == "population_only",
                metric,
            ].iloc[0]
        )

        summary_rows.append(
            {
                "metric": metric,
                "hybrid_minus_population": (
                    observed
                ),
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

    bootstrap_summary = pd.DataFrame(
        summary_rows
    )

    overall.to_csv(
        OUTPUT_DIR
        / "overall_metrics.csv",
        index=False,
    )

    classes.to_csv(
        OUTPUT_DIR
        / "class_metrics.csv",
        index=False,
    )

    bootstrap.to_csv(
        OUTPUT_DIR
        / "participant_bootstrap.csv",
        index=False,
    )

    bootstrap_summary.to_csv(
        OUTPUT_DIR
        / "bootstrap_summary.csv",
        index=False,
    )

    print("=" * 78)
    print("LONGITUDINAL AIM 2 EVALUATION")
    print("=" * 78)

    print("\nOVERALL")
    print(
        overall.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    print("\nCLASS PERFORMANCE")
    print(
        classes.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    print(
        "\nHYBRID MINUS POPULATION"
        "\nParticipant-cluster bootstrap, "
        "5000 iterations"
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
        "\nOutputs written to:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
