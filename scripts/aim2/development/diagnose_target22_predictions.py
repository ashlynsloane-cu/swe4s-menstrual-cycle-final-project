"""Diagnose why personalization changes Target 22 predictions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)


PHASES = [
    "Menstrual",
    "Follicular",
    "Fertility",
    "Luteal",
]

INPUT_DIR = Path(
    "results/aim2/pilot/hierarchical_models"
)

MANIFEST_PATH = Path(
    "results/aim2/splits/"
    "aim2_primary_split_manifest.csv"
)

TARGET_ID = 22


def load_predictions(
    filename: str,
) -> pd.DataFrame:
    """Load one prediction table."""

    path = INPUT_DIR / filename

    data = pd.read_csv(path)

    if "target_cycle_index" not in data.columns:
        manifest = pd.read_csv(
            MANIFEST_PATH
        )

        cycle_lookup = manifest[
            manifest["target_id"] == TARGET_ID
        ][
            [
                "source_id",
                "study_interval",
                "day_in_study",
                "target_cycle_index",
            ]
        ].drop_duplicates()

        data = data.merge(
            cycle_lookup,
            on=[
                "source_id",
                "study_interval",
                "day_in_study",
            ],
            how="left",
            validate="one_to_one",
        )

        if data[
            "target_cycle_index"
        ].isna().any():
            raise ValueError(
                "Could not recover target_cycle_index "
                "for all prediction rows."
            )

    required = {
        "phase",
        "predicted_phase",
        "cycle_number",
        "target_cycle_index",
    }

    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"{filename} is missing columns: "
            + ", ".join(sorted(missing))
        )

    return data


def print_predicted_counts(
    name: str,
    data: pd.DataFrame,
) -> None:
    """Print predicted phase distribution."""

    print("\nPREDICTED PHASE COUNTS")
    print(name)
    print("-" * 60)

    counts = (
        data["predicted_phase"]
        .value_counts()
        .reindex(PHASES, fill_value=0)
    )

    print(counts.to_string())


def print_confusion(
    name: str,
    data: pd.DataFrame,
) -> None:
    """Print confusion matrix."""

    matrix = confusion_matrix(
        data["phase"],
        data["predicted_phase"],
        labels=PHASES,
    )

    output = pd.DataFrame(
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

    print("\nCONFUSION MATRIX")
    print(name)
    print("-" * 60)
    print(output.to_string())


def phase_recall(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate recall separately for each true phase."""

    records = []

    for phase in PHASES:
        subset = data[
            data["phase"] == phase
        ]

        if subset.empty:
            recall = np.nan
        else:
            recall = (
                subset["predicted_phase"]
                == phase
            ).mean()

        records.append(
            {
                "phase": phase,
                "n_days": len(subset),
                "recall": recall,
            }
        )

    return pd.DataFrame(records)


def true_phase_probability(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate probability assigned to the true phase."""

    output = data.copy()

    probabilities = []

    for _, row in output.iterrows():
        column = (
            "prob_"
            + str(row["phase"]).lower()
        )

        probabilities.append(
            row[column]
        )

    output["true_phase_probability"] = (
        probabilities
    )

    return output


def print_phase_diagnostics(
    name: str,
    data: pd.DataFrame,
) -> None:
    """Print phase-specific diagnostic summaries."""

    with_probability = (
        true_phase_probability(data)
    )

    recall = phase_recall(data)

    probability_summary = (
        with_probability.groupby(
            "phase",
            observed=False,
        )["true_phase_probability"]
        .mean()
        .reindex(PHASES)
        .rename("mean_true_phase_probability")
        .reset_index()
    )

    summary = recall.merge(
        probability_summary,
        on="phase",
        how="left",
    )

    print("\nPHASE-SPECIFIC PERFORMANCE")
    print(name)
    print("-" * 60)

    print(
        summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.3f}"
            ),
        )
    )


def cycle_metrics(
    data: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    """Calculate future performance for each cycle."""

    records = []

    for cycle, subset in data.groupby(
        "target_cycle_index"
    ):
        records.append(
            {
                "model": model_name,
                "target_cycle_index": cycle,
                "n_days": len(subset),
                "accuracy": accuracy_score(
                    subset["phase"],
                    subset["predicted_phase"],
                ),
                "macro_f1": f1_score(
                    subset["phase"],
                    subset["predicted_phase"],
                    labels=PHASES,
                    average="macro",
                    zero_division=0,
                ),
            }
        )

    return pd.DataFrame(records)


def compare_changed_predictions(
    common: pd.DataFrame,
    personalized: pd.DataFrame,
) -> None:
    """Summarize days whose prediction changed."""

    keys = [
        "source_id",
        "study_interval",
        "day_in_study",
        "cycle_number",
        "phase",
    ]

    merged = common[
        keys + ["predicted_phase"]
    ].merge(
        personalized[
            keys + ["predicted_phase"]
        ],
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=(
            "_common",
            "_personalized",
        ),
    )

    merged["prediction_changed"] = (
        merged["predicted_phase_common"]
        != merged[
            "predicted_phase_personalized"
        ]
    )

    changed = merged[
        merged["prediction_changed"]
    ]

    print("\nPREDICTIONS CHANGED BY PERSONALIZATION")
    print("-" * 60)

    print(
        f"Changed days: "
        f"{len(changed)} / {len(merged)} "
        f"({len(changed) / len(merged):.1%})"
    )

    if changed.empty:
        return

    transitions = (
        changed.groupby(
            [
                "predicted_phase_common",
                "predicted_phase_personalized",
            ]
        )
        .size()
        .reset_index(name="n_days")
        .sort_values(
            "n_days",
            ascending=False,
        )
    )

    print("\nMost common prediction changes:")

    print(
        transitions.to_string(
            index=False,
        )
    )


def main() -> None:
    """Run Target 22 prediction diagnostics."""

    population = load_predictions(
        "population_only_future_predictions.csv"
    )

    common = load_predictions(
        "hybrid_common_only_future_predictions.csv"
    )

    personalized = load_predictions(
        "hybrid_personalized_future_predictions.csv"
    )

    calibration = load_predictions(
        "hybrid_personalized_calibration_predictions.csv"
    )

    models = [
        (
            "Population-only: future",
            population,
        ),
        (
            "Hybrid common-only: future",
            common,
        ),
        (
            "Hybrid personalized: future",
            personalized,
        ),
        (
            "Hybrid personalized: calibration",
            calibration,
        ),
    ]

    for name, data in models:
        print_predicted_counts(
            name,
            data,
        )

        print_confusion(
            name,
            data,
        )

        print_phase_diagnostics(
            name,
            data,
        )

    compare_changed_predictions(
        common,
        personalized,
    )

    cycle_results = pd.concat(
        [
            cycle_metrics(
                population,
                "population_only",
            ),
            cycle_metrics(
                common,
                "hybrid_common_only",
            ),
            cycle_metrics(
                personalized,
                "hybrid_personalized",
            ),
        ],
        ignore_index=True,
    )

    print("\nPER-FUTURE-CYCLE PERFORMANCE")
    print("-" * 60)

    print(
        cycle_results.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.3f}"
            ),
        )
    )

    cycle_results.to_csv(
        INPUT_DIR
        / "target22_per_cycle_diagnostics.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
