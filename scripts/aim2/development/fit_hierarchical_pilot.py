"""Fit and diagnose hierarchical Aim 2 models for Target 22."""

from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
)


PHASES = [
    "Menstrual",
    "Follicular",
    "Fertility",
    "Luteal",
]

FORMULA = (
    "phase ~ "
    "rhr_z + hrv_z + temp_z + resp_z"
    " + (1 | model_id)"
    " + (0 + rhr_z | model_id)"
    " + (0 + hrv_z | model_id)"
    " + (0 + temp_z | model_id)"
    " + (0 + resp_z | model_id)"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Fit and diagnose hierarchical models "
            "for the Target 22 Aim 2 pilot."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/aim2/pilot"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/aim2/pilot/hierarchical_models"
        ),
    )

    parser.add_argument(
        "--draws",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--tune",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--chains",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20261014,
    )

    return parser.parse_args()


def prepare_model_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Set categorical types required by Bambi."""

    output = data.copy()

    output["phase"] = pd.Categorical(
        output["phase"],
        categories=PHASES,
    )

    output["model_id"] = (
        output["model_id"]
        .astype(str)
        .astype("category")
    )

    return output


def fit_model(
    data: pd.DataFrame,
    draws: int,
    tune: int,
    chains: int,
    seed: int,
):
    """Fit one hierarchical categorical model."""

    model = bmb.Model(
        FORMULA,
        data,
        family="categorical",
        categorical=["model_id"],
        dropna=False,
    )

    print("\nMODEL SPECIFICATION")
    print("-------------------")
    print(model)

    idata = model.fit(
        draws=draws,
        tune=tune,
        chains=chains,
        cores=chains,
        target_accept=0.95,
        random_seed=seed,
    )

    return model, idata


def extract_probabilities(
    model,
    idata,
    data: pd.DataFrame,
    include_group_specific: bool,
) -> tuple[np.ndarray, list[str]]:
    """Extract posterior-mean class probabilities."""

    predicted = model.predict(
        idata,
        data=data,
        kind="response_params",
        inplace=False,
        include_group_specific=include_group_specific,
    )

    probabilities = predicted.posterior["p"]

    mean_probabilities = probabilities.mean(
        dim=("chain", "draw")
    )

    observation_dimensions = [
        dimension
        for dimension in mean_probabilities.dims
        if "obs" in dimension
    ]

    if len(observation_dimensions) != 1:
        raise RuntimeError(
            "Could not uniquely identify observation "
            "dimension in predicted probabilities."
        )

    observation_dimension = observation_dimensions[0]

    category_dimensions = [
        dimension
        for dimension in mean_probabilities.dims
        if dimension != observation_dimension
    ]

    if len(category_dimensions) != 1:
        raise RuntimeError(
            "Could not uniquely identify category "
            "dimension in predicted probabilities."
        )

    category_dimension = category_dimensions[0]

    mean_probabilities = (
        mean_probabilities.transpose(
            observation_dimension,
            category_dimension,
        )
    )

    labels = [
        str(value)
        for value in mean_probabilities.coords[
            category_dimension
        ].values
    ]

    return (
        mean_probabilities.to_numpy(),
        labels,
    )


def multiclass_brier(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
) -> float:
    """Calculate multiclass Brier score."""

    label_to_index = {
        label: index
        for index, label in enumerate(labels)
    }

    observed = np.zeros_like(
        probabilities,
        dtype=float,
    )

    for row, label in enumerate(y_true):
        observed[
            row,
            label_to_index[label],
        ] = 1.0

    squared_error = (
        probabilities
        - observed
    ) ** 2

    return float(
        np.mean(
            np.sum(
                squared_error,
                axis=1,
            )
        )
    )


def calculate_metrics(
    data: pd.DataFrame,
    probabilities: np.ndarray,
    labels: list[str],
) -> dict[str, float]:
    """Calculate predictive metrics."""

    y_true = (
        data["phase"]
        .astype(str)
        .to_numpy()
    )

    predicted_indices = probabilities.argmax(
        axis=1
    )

    y_pred = np.asarray(labels)[
        predicted_indices
    ]

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
        "log_loss": log_loss(
            y_true,
            probabilities,
            labels=labels,
        ),
        "brier_score": multiclass_brier(
            y_true,
            probabilities,
            labels,
        ),
    }


def diagnostic_summary(
    idata,
) -> dict[str, float]:
    """Return key MCMC diagnostics."""

    summary = az.summary(
        idata,
        kind="diagnostics",
    )

    divergences = int(
        idata.sample_stats[
            "diverging"
        ].sum()
    )

    return {
        "divergences": divergences,
        "max_rhat": float(
            summary["r_hat"].max()
        ),
        "min_ess_bulk": float(
            summary["ess_bulk"].min()
        ),
        "min_ess_tail": float(
            summary["ess_tail"].min()
        ),
    }


def save_predictions(
    data: pd.DataFrame,
    probabilities: np.ndarray,
    labels: list[str],
    path: Path,
) -> None:
    """Save posterior-mean probabilities."""

    output = data[
        [
            "source_id",
            "study_interval",
            "day_in_study",
            "cycle_number",
            "phase",
        ]
    ].copy()

    predicted_indices = probabilities.argmax(
        axis=1
    )

    output["predicted_phase"] = np.asarray(
        labels
    )[predicted_indices]

    for index, label in enumerate(labels):
        output[
            f"prob_{label.lower()}"
        ] = probabilities[:, index]

    output.to_csv(
        path,
        index=False,
    )


def evaluate_predictions(
    label: str,
    model,
    idata,
    data: pd.DataFrame,
    include_group_specific: bool,
    output_path: Path,
    evaluation_set: str,
) -> dict[str, object]:
    """Predict, score, and save one evaluation."""

    probabilities, labels = extract_probabilities(
        model,
        idata,
        data,
        include_group_specific=(
            include_group_specific
        ),
    )

    metrics = calculate_metrics(
        data,
        probabilities,
        labels,
    )

    save_predictions(
        data,
        probabilities,
        labels,
        output_path,
    )

    return {
        "evaluation": label,
        "evaluation_set": evaluation_set,
        "n_days": len(data),
        "group_specific": include_group_specific,
        **metrics,
    }


def print_diagnostics(
    name: str,
    diagnostics: dict[str, float],
) -> None:
    """Print MCMC diagnostics."""

    print("\n" + name)
    print("-" * len(name))

    for key, value in diagnostics.items():
        print(f"{key:20s}: {value}")


def print_metrics(
    title: str,
    result: dict[str, object],
) -> None:
    """Print prediction metrics."""

    print("\n" + title)
    print("-" * len(title))

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "log_loss",
        "brier_score",
    ]

    for key in metric_names:
        print(
            f"{key:20s}: "
            f"{float(result[key]):.4f}"
        )


def main() -> None:
    """Run the Target 22 hierarchical diagnostic pilot."""

    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    population = prepare_model_data(
        pd.read_csv(
            args.input_dir
            / "target22_population_train.csv"
        )
    )

    calibration = prepare_model_data(
        pd.read_csv(
            args.input_dir
            / "target22_personal_calibration.csv"
        )
    )

    hybrid = prepare_model_data(
        pd.read_csv(
            args.input_dir
            / "target22_hybrid_train.csv"
        )
    )

    test = prepare_model_data(
        pd.read_csv(
            args.input_dir
            / "target22_future_test.csv"
        )
    )

    print("=" * 72)
    print("AIM 2 HIERARCHICAL DIAGNOSTIC PILOT: TARGET 22")
    print("=" * 72)

    print(
        "Population training days:",
        len(population),
    )

    print(
        "Personal calibration days:",
        len(calibration),
    )

    print(
        "Hybrid training days:",
        len(hybrid),
    )

    print(
        "Future test days:",
        len(test),
    )

    # ------------------------------------------------------------
    # MODEL 1: population-only
    # ------------------------------------------------------------

    print("\n" + "=" * 72)
    print("FIT 1: POPULATION-ONLY MODEL")
    print("=" * 72)

    population_model, population_idata = fit_model(
        population,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        seed=args.seed,
    )

    population_diagnostics = diagnostic_summary(
        population_idata
    )

    print_diagnostics(
        "POPULATION MODEL MCMC DIAGNOSTICS",
        population_diagnostics,
    )

    population_future = evaluate_predictions(
        label="population_only_future",
        model=population_model,
        idata=population_idata,
        data=test,
        include_group_specific=False,
        output_path=(
            args.output_dir
            / "population_only_future_predictions.csv"
        ),
        evaluation_set="future_test",
    )

    print_metrics(
        "POPULATION-ONLY: FUTURE TEST",
        population_future,
    )

    population_idata.to_netcdf(
        args.output_dir
        / "population_only_posterior.nc"
    )

    # ------------------------------------------------------------
    # MODEL 2: hybrid model
    # ------------------------------------------------------------

    print("\n" + "=" * 72)
    print("FIT 2: HYBRID TWO-CYCLE MODEL")
    print("=" * 72)

    hybrid_model, hybrid_idata = fit_model(
        hybrid,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        seed=args.seed,
    )

    hybrid_diagnostics = diagnostic_summary(
        hybrid_idata
    )

    print_diagnostics(
        "HYBRID MODEL MCMC DIAGNOSTICS",
        hybrid_diagnostics,
    )

    # Diagnostic A:
    # Hybrid fit, but remove Person 22-specific effects when
    # predicting future data.
    hybrid_common_future = evaluate_predictions(
        label="hybrid_common_only_future",
        model=hybrid_model,
        idata=hybrid_idata,
        data=test,
        include_group_specific=False,
        output_path=(
            args.output_dir
            / "hybrid_common_only_future_predictions.csv"
        ),
        evaluation_set="future_test",
    )

    print_metrics(
        "HYBRID FIT, COMMON EFFECTS ONLY: FUTURE TEST",
        hybrid_common_future,
    )

    # Diagnostic B:
    # Full hybrid prediction: common population effects plus
    # Person 22-specific effects learned from cycles 1-2.
    hybrid_personalized_future = evaluate_predictions(
        label="hybrid_personalized_future",
        model=hybrid_model,
        idata=hybrid_idata,
        data=test,
        include_group_specific=True,
        output_path=(
            args.output_dir
            / "hybrid_personalized_future_predictions.csv"
        ),
        evaluation_set="future_test",
    )

    print_metrics(
        "HYBRID PERSONALIZED: FUTURE TEST",
        hybrid_personalized_future,
    )

    # Diagnostic C:
    # In-sample calibration fit. This is NOT held-out performance.
    # It asks only whether the learned Person 22 effects describe
    # cycles 1-2 that generated those effects.
    hybrid_personalized_calibration = evaluate_predictions(
        label="hybrid_personalized_calibration",
        model=hybrid_model,
        idata=hybrid_idata,
        data=calibration,
        include_group_specific=True,
        output_path=(
            args.output_dir
            / "hybrid_personalized_calibration_predictions.csv"
        ),
        evaluation_set="calibration_in_sample",
    )

    print_metrics(
        "HYBRID PERSONALIZED: CALIBRATION CYCLES 1-2",
        hybrid_personalized_calibration,
    )

    hybrid_idata.to_netcdf(
        args.output_dir
        / "hybrid_two_cycle_posterior.nc"
    )

    # ------------------------------------------------------------
    # Combined comparison
    # ------------------------------------------------------------

    results = pd.DataFrame(
        [
            population_future,
            hybrid_common_future,
            hybrid_personalized_future,
            hybrid_personalized_calibration,
        ]
    )

    results.to_csv(
        args.output_dir
        / "target22_diagnostic_comparison.csv",
        index=False,
    )

    print("\n" + "=" * 72)
    print("TARGET 22 DIAGNOSTIC COMPARISON")
    print("=" * 72)

    display_columns = [
        "evaluation",
        "evaluation_set",
        "n_days",
        "group_specific",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "log_loss",
        "brier_score",
    ]

    print(
        results[
            display_columns
        ].to_string(
            index=False,
        )
    )

    future_rows = results[
        results["evaluation_set"]
        == "future_test"
    ].set_index(
        "evaluation"
    )

    population_macro_f1 = future_rows.loc[
        "population_only_future",
        "macro_f1",
    ]

    common_macro_f1 = future_rows.loc[
        "hybrid_common_only_future",
        "macro_f1",
    ]

    personalized_macro_f1 = future_rows.loc[
        "hybrid_personalized_future",
        "macro_f1",
    ]

    print("\n" + "=" * 72)
    print("KEY DIAGNOSTIC DIFFERENCES")
    print("=" * 72)

    print(
        "Hybrid common-only minus population-only "
        "macro-F1: "
        f"{common_macro_f1 - population_macro_f1:+.4f}"
    )

    print(
        "Hybrid personalized minus hybrid common-only "
        "macro-F1: "
        f"{personalized_macro_f1 - common_macro_f1:+.4f}"
    )

    print(
        "Hybrid personalized minus population-only "
        "macro-F1: "
        f"{personalized_macro_f1 - population_macro_f1:+.4f}"
    )

    print(
        "\nNOTE: Calibration-cycle performance is in-sample "
        "and is diagnostic only."
    )


if __name__ == "__main__":
    main()
