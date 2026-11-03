"""Run the frozen Aim 2 population and personalized models."""

from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    log_loss,
)
from sklearn.preprocessing import StandardScaler


PHASES = [
    "Menstrual",
    "Follicular",
    "Fertility",
    "Luteal",
]

RAW_COLUMNS = [
    "resting_heart_rate",
    "hrv_rmssd",
    "nightly_temperature",
    "respiratory_rate",
]

RAW_Z_COLUMNS = [
    "rhr_z",
    "hrv_z",
    "temp_z",
    "resp_z",
]

WITHIN_COLUMNS = [
    "rhr_within",
    "hrv_within",
    "temp_within",
    "resp_within",
]

WITHIN_Z_COLUMNS = [
    "rhr_within_z",
    "hrv_within_z",
    "temp_within_z",
    "resp_within_z",
]

FORMULA = (
    "phase ~ "
    "rhr_z + hrv_z + temp_z + resp_z"
    " + (1 | model_id)"
    " + (0 + rhr_within_z | model_id)"
    " + (0 + hrv_within_z | model_id)"
    " + (0 + temp_within_z | model_id)"
    " + (0 + resp_within_z | model_id)"
)

INPUT_ROOT = Path(
    "results/aim2/longitudinal_folds"
)

SUMMARY_PATH = Path(
    "results/aim2/longitudinal/"
    "longitudinal_manifest_primary.csv"
)

DEFAULT_OUTPUT = Path(
    "results/aim2/longitudinal_models"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run frozen hierarchical Aim 2 models "
            "for 2022-to-2024 longitudinal folds."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--draws",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--tune",
        type=int,
        default=1000,
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

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip folds with a completed result file.",
    )

    return parser.parse_args()


def group_prior(
    scale: float,
) -> bmb.Prior:
    """Construct a regularizing group-specific prior."""

    return bmb.Prior(
        "Normal",
        mu=0,
        sigma=bmb.Prior(
            "HalfNormal",
            sigma=scale,
        ),
    )


def make_priors() -> dict:
    """Return the frozen Aim 2 prior specification."""

    return {
        "Intercept": bmb.Prior(
            "Normal",
            mu=0,
            sigma=1.5,
        ),
        "rhr_z": bmb.Prior(
            "Normal",
            mu=0,
            sigma=1,
        ),
        "hrv_z": bmb.Prior(
            "Normal",
            mu=0,
            sigma=1,
        ),
        "temp_z": bmb.Prior(
            "Normal",
            mu=0,
            sigma=1,
        ),
        "resp_z": bmb.Prior(
            "Normal",
            mu=0,
            sigma=1,
        ),
        "1|model_id": group_prior(1.0),
        "rhr_within_z|model_id": (
            group_prior(0.75)
        ),
        "hrv_within_z|model_id": (
            group_prior(0.75)
        ),
        "temp_within_z|model_id": (
            group_prior(0.75)
        ),
        "resp_within_z|model_id": (
            group_prior(0.75)
        ),
    }


def add_model_group(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Create participant-interval model groups."""

    output = data.copy()

    output["model_id"] = (
        output["source_id"]
        .astype(int)
        .astype(str)
        + "_"
        + output["study_interval"]
        .astype(int)
        .astype(str)
    )

    output["model_id"] = (
        output["model_id"]
        .astype("category")
    )

    return output


def standardize_fold(
    population: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Fit scalers on population training only."""

    population = population.copy()
    calibration = calibration.copy()
    test = test.copy()

    raw_scaler = StandardScaler()

    raw_scaler.fit(
        population[RAW_COLUMNS]
    )

    for data in [
        population,
        calibration,
        test,
    ]:
        data[RAW_Z_COLUMNS] = (
            raw_scaler.transform(
                data[RAW_COLUMNS]
            )
        )

    within_scaler = StandardScaler()

    within_scaler.fit(
        population[WITHIN_COLUMNS]
    )

    for data in [
        population,
        calibration,
        test,
    ]:
        data[WITHIN_Z_COLUMNS] = (
            within_scaler.transform(
                data[WITHIN_COLUMNS]
            )
        )

    return (
        population,
        calibration,
        test,
    )


def fit_model(
    data: pd.DataFrame,
    args: argparse.Namespace,
):
    """Fit the frozen hierarchical categorical model."""

    model = bmb.Model(
        FORMULA,
        data,
        family="categorical",
        categorical=["model_id"],
        priors=make_priors(),
        auto_scale=False,
        center_predictors=False,
        dropna=False,
    )

    idata = model.fit(
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        cores=args.chains,
        target_accept=0.95,
        random_seed=args.seed,
    )

    return model, idata


def diagnostics(
    idata,
) -> dict[str, float]:
    """Extract convergence diagnostics."""

    summary = az.summary(
        idata,
        kind="diagnostics",
    )

    return {
        "divergences": int(
            idata.sample_stats[
                "diverging"
            ].sum()
        ),
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


def extract_probabilities(
    model,
    idata,
    test: pd.DataFrame,
    include_group_specific: bool,
) -> tuple[np.ndarray, list[str]]:
    """Extract posterior-mean class probabilities."""

    predicted = model.predict(
        idata,
        data=test,
        kind="response_params",
        inplace=False,
        include_group_specific=(
            include_group_specific
        ),
    )

    probability_data = (
        predicted.posterior["p"]
        .mean(
            dim=("chain", "draw")
        )
    )

    observation_dims = [
        dim
        for dim in probability_data.dims
        if "obs" in dim
    ]

    if len(observation_dims) != 1:
        raise RuntimeError(
            "Could not identify prediction "
            "observation dimension."
        )

    observation_dim = observation_dims[0]

    category_dims = [
        dim
        for dim in probability_data.dims
        if dim != observation_dim
    ]

    if len(category_dims) != 1:
        raise RuntimeError(
            "Could not identify prediction "
            "category dimension."
        )

    category_dim = category_dims[0]

    probability_data = (
        probability_data.transpose(
            observation_dim,
            category_dim,
        )
    )

    labels = [
        str(value)
        for value in probability_data.coords[
            category_dim
        ].values
    ]

    return (
        probability_data.to_numpy(),
        labels,
    )


def multiclass_brier(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
) -> float:
    """Calculate multiclass Brier score."""

    lookup = {
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
            lookup[label],
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


def evaluate(
    data: pd.DataFrame,
    probabilities: np.ndarray,
    labels: list[str],
) -> dict[str, float]:
    """Calculate metrics valid for every fold."""

    y_true = (
        data["phase"]
        .astype(str)
        .to_numpy()
    )

    predicted = np.asarray(labels)[
        probabilities.argmax(
            axis=1
        )
    ]

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                predicted,
            )
        ),
        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=labels,
            )
        ),
        "brier_score": (
            multiclass_brier(
                y_true,
                probabilities,
                labels,
            )
        ),
    }


def prediction_table(
    test: pd.DataFrame,
    probabilities: np.ndarray,
    labels: list[str],
    model_name: str,
    target_id: int,
    target_interval: int,
) -> pd.DataFrame:
    """Create a daily prediction table."""

    columns = [
        "source_id",
        "study_interval",
        "day_in_study",
        "cycle_number",
        "interval_cycle_index",
        "phase",
    ]

    output = test[
        [
            column
            for column in columns
            if column in test.columns
        ]
    ].copy()

    output.insert(
        0,
        "target_id",
        target_id,
    )

    output.insert(
        1,
        "target_interval",
        target_interval,
    )

    output["model"] = model_name

    output["predicted_phase"] = (
        np.asarray(labels)[
            probabilities.argmax(
                axis=1
            )
        ]
    )

    for index, label in enumerate(labels):
        output[
            f"prob_{label.lower()}"
        ] = probabilities[:, index]

    return output


def run_fold(
    target_id: int,
    target_interval: int,
    args: argparse.Namespace,
) -> None:
    """Fit and evaluate one participant-interval fold."""

    fold_name = f"id{target_id}"

    input_dir = (
        INPUT_ROOT
        / fold_name
    )

    output_dir = (
        args.output_dir
        / fold_name
    )

    result_path = (
        output_dir
        / "fold_result.csv"
    )

    if (
        args.resume
        and result_path.exists()
    ):
        print(
            f"\nSKIP {fold_name}: "
            "completed result exists."
        )
        return

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    population = pd.read_csv(
        input_dir
        / "population_train.csv"
    )

    calibration = pd.read_csv(
        input_dir
        / "personal_calibration.csv"
    )

    test = pd.read_csv(
        input_dir
        / "future_test.csv"
    )

    population, calibration, test = (
        standardize_fold(
            population,
            calibration,
            test,
        )
    )

    population = add_model_group(
        population
    )

    calibration = add_model_group(
        calibration
    )

    test = add_model_group(
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

    print("\n" + "=" * 72)
    print(
        f"FOLD {fold_name}"
    )
    print("=" * 72)

    print(
        "Population days:",
        len(population),
    )

    print(
        "Calibration days:",
        len(calibration),
    )

    print(
        "Future test days:",
        len(test),
    )

    print("\nFitting population-only model...")

    population_model, population_idata = (
        fit_model(
            population,
            args,
        )
    )

    population_diag = diagnostics(
        population_idata
    )

    population_prob, labels = (
        extract_probabilities(
            population_model,
            population_idata,
            test,
            include_group_specific=False,
        )
    )

    population_metrics = evaluate(
        test,
        population_prob,
        labels,
    )

    print("\nFitting personalized hybrid model...")

    hybrid_model, hybrid_idata = (
        fit_model(
            hybrid,
            args,
        )
    )

    hybrid_diag = diagnostics(
        hybrid_idata
    )

    # Longitudinal transfer:
    # use the target's person-specific effects learned from
    # their 2022 calibration cycles when predicting 2024.
    #
    # The observations themselves remain 2024 data; only
    # the random-effect group is carried forward from 2022.
    hybrid_test = test.copy()

    calibration_group = (
        f"{target_id}_2022"
    )

    known_groups = (
        hybrid["model_id"]
        .astype(str)
        .unique()
    )

    if calibration_group not in known_groups:
        raise RuntimeError(
            f"ID {target_id}: calibration model group "
            f"{calibration_group} was not found in "
            "hybrid training data."
        )

    hybrid_test["model_id"] = (
        calibration_group
    )

    hybrid_test["model_id"] = pd.Categorical(
        hybrid_test["model_id"],
        categories=(
            hybrid["model_id"]
            .cat.categories
        ),
    )

    hybrid_prob, hybrid_labels = (
        extract_probabilities(
            hybrid_model,
            hybrid_idata,
            hybrid_test,
            include_group_specific=True,
        )
    )

    if hybrid_labels != labels:
        raise RuntimeError(
            "Population and hybrid category "
            "orders differ."
        )

    hybrid_metrics = evaluate(
        test,
        hybrid_prob,
        labels,
    )

    population_predictions = (
        prediction_table(
            test,
            population_prob,
            labels,
            "population_only",
            target_id,
            target_interval,
        )
    )

    hybrid_predictions = (
        prediction_table(
            test,
            hybrid_prob,
            labels,
            "hybrid_personalized",
            target_id,
            target_interval,
        )
    )

    predictions = pd.concat(
        [
            population_predictions,
            hybrid_predictions,
        ],
        ignore_index=True,
    )

    predictions.to_csv(
        output_dir
        / "predictions.csv",
        index=False,
    )

    result_rows = []

    for (
        model_name,
        metrics,
        diag,
    ) in [
        (
            "population_only",
            population_metrics,
            population_diag,
        ),
        (
            "hybrid_personalized",
            hybrid_metrics,
            hybrid_diag,
        ),
    ]:
        result_rows.append(
            {
                "target_id": target_id,
                "target_interval": (
                    target_interval
                ),
                "model": model_name,
                "n_test_days": len(test),
                **metrics,
                **diag,
            }
        )

    results = pd.DataFrame(
        result_rows
    )

    results.to_csv(
        result_path,
        index=False,
    )

    population_idata.to_netcdf(
        output_dir
        / "population_posterior.nc"
    )

    hybrid_idata.to_netcdf(
        output_dir
        / "hybrid_posterior.nc"
    )

    print("\nFOLD RESULT")
    print(results.to_string(index=False))


def combine_results(
    args: argparse.Namespace,
) -> None:
    """Combine all completed fold outputs."""

    result_files = sorted(
        args.output_dir.glob(
            "id*/fold_result.csv"
        )
    )

    prediction_files = sorted(
        args.output_dir.glob(
            "id*/predictions.csv"
        )
    )

    if result_files:
        results = pd.concat(
            [
                pd.read_csv(path)
                for path in result_files
            ],
            ignore_index=True,
        )

        results.to_csv(
            args.output_dir
            / "all_fold_results.csv",
            index=False,
        )

    if prediction_files:
        predictions = pd.concat(
            [
                pd.read_csv(path)
                for path in prediction_files
            ],
            ignore_index=True,
        )

        predictions.to_csv(
            args.output_dir
            / "all_predictions.csv",
            index=False,
        )


def main() -> None:
    """Run all final longitudinal Aim 2 folds."""

    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = pd.read_csv(
        SUMMARY_PATH
    )

    print("=" * 72)
    print("FINAL AIM 2 LONGITUDINAL HIERARCHICAL ANALYSIS")
    print("=" * 72)

    print(
        "Folds:",
        len(summary),
    )

    print(
        "Draws:",
        args.draws,
    )

    print(
        "Tune:",
        args.tune,
    )

    print(
        "Chains:",
        args.chains,
    )

    for index, row in summary.iterrows():
        target_id = int(
            row["target_id"]
        )

        # Longitudinal test interval is always 2024.
        target_interval = 2024

        print(
            f"\nFold {index + 1}/{len(summary)}"
        )

        run_fold(
            target_id,
            target_interval,
            args,
        )

        combine_results(args)

    combine_results(args)

    print("\n" + "=" * 72)
    print("LONGITUDINAL MODEL RUN COMPLETE")
    print("=" * 72)

    print(
        "Results:",
        args.output_dir
        / "all_fold_results.csv",
    )

    print(
        "Predictions:",
        args.output_dir
        / "all_predictions.csv",
    )


if __name__ == "__main__":
    main()
