"""Convergence-only dry run for the final Aim 2 model."""

from __future__ import annotations

from pathlib import Path

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


TARGET_ID = 26
TARGET_INTERVAL = 2022

INPUT_DIR = Path(
    "results/aim2/centered_primary/"
    f"id{TARGET_ID}_{TARGET_INTERVAL}"
)

OUTPUT_DIR = Path(
    "results/aim2/model_dry_run/"
    f"id{TARGET_ID}_{TARGET_INTERVAL}"
)

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

COMMON_TERMS = [
    "Intercept",
    "rhr_z",
    "hrv_z",
    "temp_z",
    "resp_z",
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


def group_prior(scale: float) -> bmb.Prior:
    """Return a regularizing hierarchical Normal prior."""

    return bmb.Prior(
        "Normal",
        mu=0,
        sigma=bmb.Prior(
            "HalfNormal",
            sigma=scale,
        ),
    )


def make_priors() -> dict:
    """Define explicit weakly regularizing priors."""

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
        "rhr_within_z|model_id": group_prior(0.75),
        "hrv_within_z|model_id": group_prior(0.75),
        "temp_within_z|model_id": group_prior(0.75),
        "resp_within_z|model_id": group_prior(0.75),
    }


def add_model_group(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Create participant-interval hierarchical groups."""

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


def standardize(
    population: pd.DataFrame,
    calibration: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize using population-training data only."""

    population = population.copy()
    calibration = calibration.copy()

    raw_scaler = StandardScaler()

    raw_scaler.fit(
        population[RAW_COLUMNS]
    )

    population[RAW_Z_COLUMNS] = (
        raw_scaler.transform(
            population[RAW_COLUMNS]
        )
    )

    calibration[RAW_Z_COLUMNS] = (
        raw_scaler.transform(
            calibration[RAW_COLUMNS]
        )
    )

    within_scaler = StandardScaler()

    within_scaler.fit(
        population[WITHIN_COLUMNS]
    )

    population[WITHIN_Z_COLUMNS] = (
        within_scaler.transform(
            population[WITHIN_COLUMNS]
        )
    )

    calibration[WITHIN_Z_COLUMNS] = (
        within_scaler.transform(
            calibration[WITHIN_COLUMNS]
        )
    )

    return population, calibration


def fit_model(
    data: pd.DataFrame,
    name: str,
):
    """Fit one categorical hierarchical model."""

    print("\n" + "=" * 72)
    print(name)
    print("=" * 72)

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

    print(model)

    idata = model.fit(
        draws=500,
        tune=500,
        chains=4,
        cores=4,
        target_accept=0.95,
        random_seed=20261014,
    )

    return model, idata


def diagnostics(
    idata,
    name: str,
) -> None:
    """Print convergence diagnostics."""

    summary = az.summary(
        idata,
        kind="diagnostics",
    )

    divergences = int(
        idata.sample_stats[
            "diverging"
        ].sum()
    )

    print("\n" + name)
    print("-" * 72)

    print(
        "divergences:",
        divergences,
    )

    print(
        "max R-hat:",
        float(
            summary["r_hat"].max()
        ),
    )

    print(
        "min ESS bulk:",
        float(
            summary["ess_bulk"].min()
        ),
    )

    print(
        "min ESS tail:",
        float(
            summary["ess_tail"].min()
        ),
    )


def print_group_variances(
    idata,
    name: str,
) -> None:
    """Print learned participant-effect variation."""

    print("\n" + name)
    print("-" * 72)

    for variable in idata.posterior.data_vars:
        if (
            "model_id_sigma" not in variable
        ):
            continue

        summary = az.summary(
            idata,
            var_names=[variable],
        )

        columns = [
            column
            for column in [
                "mean",
                "sd",
                "r_hat",
                "ess_bulk",
            ]
            if column in summary.columns
        ]

        print(f"\n{variable}")

        print(
            summary[
                columns
            ].to_string()
        )


def compare_common_effects(
    population_idata,
    hybrid_idata,
) -> None:
    """Check whether calibration changes population effects."""

    print("\nCOMMON-EFFECT POSTERIOR SHIFTS")
    print("-" * 72)

    for term in COMMON_TERMS:
        population_mean = (
            population_idata.posterior[
                term
            ]
            .mean(
                dim=("chain", "draw")
            )
        )

        hybrid_mean = (
            hybrid_idata.posterior[
                term
            ]
            .mean(
                dim=("chain", "draw")
            )
        )

        difference = (
            hybrid_mean
            - population_mean
        )

        max_shift = float(
            np.abs(
                difference.values
            ).max()
        )

        print(
            f"{term:12s} "
            f"max_abs_shift={max_shift:.4f}"
        )


def main() -> None:
    """Run convergence-only model development."""

    population = pd.read_csv(
        INPUT_DIR
        / "population_train.csv"
    )

    calibration = pd.read_csv(
        INPUT_DIR
        / "personal_calibration.csv"
    )

    # IMPORTANT:
    # future_test.csv is deliberately never loaded.

    population, calibration = standardize(
        population,
        calibration,
    )

    population = add_model_group(
        population
    )

    calibration = add_model_group(
        calibration
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

    print("=" * 72)
    print("AIM 2 CENTERED MODEL DRY RUN")
    print("=" * 72)

    print(
        "Target:",
        TARGET_ID,
    )

    print(
        "Study interval:",
        TARGET_INTERVAL,
    )

    print(
        "Population days:",
        len(population),
    )

    print(
        "Calibration days:",
        len(calibration),
    )

    print(
        "Future test data loaded: NO",
    )

    population_model, population_idata = (
        fit_model(
            population,
            "POPULATION MODEL",
        )
    )

    _, hybrid_idata = fit_model(
        hybrid,
        "HYBRID MODEL",
    )

    diagnostics(
        population_idata,
        "POPULATION DIAGNOSTICS",
    )

    diagnostics(
        hybrid_idata,
        "HYBRID DIAGNOSTICS",
    )

    print_group_variances(
        population_idata,
        "POPULATION GROUP-SPECIFIC VARIATION",
    )

    compare_common_effects(
        population_idata,
        hybrid_idata,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    population_idata.to_netcdf(
        OUTPUT_DIR
        / "population_posterior.nc"
    )

    hybrid_idata.to_netcdf(
        OUTPUT_DIR
        / "hybrid_posterior.nc"
    )

    print(
        "\nDry run complete."
    )

    print(
        "No future test outcomes were loaded "
        "or evaluated."
    )


if __name__ == "__main__":
    main()
