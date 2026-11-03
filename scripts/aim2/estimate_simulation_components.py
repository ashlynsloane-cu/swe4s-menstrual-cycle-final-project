"""Estimate mcPHASES variance components for Aim 2 simulation."""

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "results/preprocessed/"
    "mcphases_daily_complete_cycles.csv"
)

OUTPUT_DIR = Path(
    "results/aim2/simulation_components"
)

PHASES = [
    "Menstrual",
    "Follicular",
    "Fertility",
    "Luteal",
]

SIGNALS = {
    "resting_heart_rate": "rhr",
    "hrv_rmssd": "hrv",
    "nightly_temperature": "temp",
    "respiratory_rate": "resp",
}


def prepare_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Restrict to complete-case valid phase observations."""

    mask = (
        data["phase"].isin(PHASES)
        & data[list(SIGNALS)].notna().all(axis=1)
    )

    output = data.loc[mask].copy()

    output["cycle_id"] = (
        output["id"].astype(int).astype(str)
        + "_"
        + output["study_interval"]
        .astype(int)
        .astype(str)
        + "_"
        + output["cycle_number"]
        .astype(int)
        .astype(str)
    )

    return output


def standardize_signal(
    data: pd.DataFrame,
    signal: str,
) -> tuple[pd.DataFrame, float, float]:
    """Standardize one signal across included observations."""

    output = data.copy()

    mean = output[signal].mean()
    sd = output[signal].std(ddof=1)

    output["value_z"] = (
        output[signal] - mean
    ) / sd

    return output, mean, sd


def decompose_signal(
    data: pd.DataFrame,
    signal: str,
    short_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decompose one signal into hierarchical components."""

    x, raw_mean, raw_sd = standardize_signal(
        data,
        signal,
    )

    # ----------------------------------------------------------
    # 1. Population phase pattern
    # ----------------------------------------------------------

    phase_means = (
        x.groupby("phase")["value_z"]
        .mean()
        .reindex(PHASES)
    )

    x["population_phase_mean"] = (
        x["phase"].map(
            phase_means
        )
    )

    x["after_population_phase"] = (
        x["value_z"]
        - x["population_phase_mean"]
    )

    # ----------------------------------------------------------
    # 2. Stable participant-by-phase signature
    #
    # Average residual physiology for each person and phase,
    # pooling across that person's observed cycles.
    # ----------------------------------------------------------

    person_phase = (
        x.groupby(
            [
                "id",
                "phase",
            ],
            as_index=False,
        )["after_population_phase"]
        .mean()
        .rename(
            columns={
                "after_population_phase":
                "person_phase_effect"
            }
        )
    )

    x = x.merge(
        person_phase,
        on=[
            "id",
            "phase",
        ],
        how="left",
        validate="many_to_one",
    )

    x["after_person_phase"] = (
        x["after_population_phase"]
        - x["person_phase_effect"]
    )

    # ----------------------------------------------------------
    # 3. Cycle-by-phase deviation
    #
    # For a particular person's particular cycle, how much
    # does that phase differ from their stable phase pattern?
    # ----------------------------------------------------------

    cycle_phase = (
        x.groupby(
            [
                "cycle_id",
                "phase",
            ],
            as_index=False,
        )["after_person_phase"]
        .mean()
        .rename(
            columns={
                "after_person_phase":
                "cycle_phase_effect"
            }
        )
    )

    x = x.merge(
        cycle_phase,
        on=[
            "cycle_id",
            "phase",
        ],
        how="left",
        validate="many_to_one",
    )

    # ----------------------------------------------------------
    # 4. Day-level residual
    # ----------------------------------------------------------

    x["day_residual"] = (
        x["after_person_phase"]
        - x["cycle_phase_effect"]
    )

    # ----------------------------------------------------------
    # Variance summaries
    # ----------------------------------------------------------

    population_phase_variance = float(
        phase_means.var(ddof=1)
    )

    person_phase_variance = float(
        person_phase[
            "person_phase_effect"
        ].var(ddof=1)
    )

    cycle_phase_variance = float(
        cycle_phase[
            "cycle_phase_effect"
        ].var(ddof=1)
    )

    day_variance = float(
        x["day_residual"].var(ddof=1)
    )

    total_component_variance = (
        population_phase_variance
        + person_phase_variance
        + cycle_phase_variance
        + day_variance
    )

    summary = pd.DataFrame(
        [
            {
                "signal": short_name,
                "raw_mean": raw_mean,
                "raw_sd": raw_sd,
                "n_days": len(x),
                "n_participants": (
                    x["id"].nunique()
                ),
                "n_cycles": (
                    x["cycle_id"].nunique()
                ),
                "population_phase_variance":
                    population_phase_variance,
                "person_phase_variance":
                    person_phase_variance,
                "cycle_phase_variance":
                    cycle_phase_variance,
                "day_variance":
                    day_variance,
                "component_total":
                    total_component_variance,
                "person_fraction": (
                    person_phase_variance
                    / total_component_variance
                ),
                "cycle_fraction": (
                    cycle_phase_variance
                    / total_component_variance
                ),
                "day_fraction": (
                    day_variance
                    / total_component_variance
                ),
            }
        ]
    )

    # Save phase means too because the simulation will
    # use these as the population phase pattern.
    phase_table = (
        phase_means.rename(
            "population_phase_mean_z"
        )
        .reset_index()
    )

    phase_table.insert(
        0,
        "signal",
        short_name,
    )

    return (
        summary,
        phase_table,
        person_phase,
        cycle_phase,
        x,
    )


def main() -> None:
    """Estimate simulation components for all wearable signals."""

    data = pd.read_csv(
        INPUT_PATH
    )

    data = prepare_data(
        data
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries = []
    phase_tables = []

    for signal, short_name in SIGNALS.items():
        (
            summary,
            phase_table,
            person_phase,
            cycle_phase,
            decomposed,
        ) = decompose_signal(
            data,
            signal,
            short_name,
        )

        summaries.append(
            summary
        )

        phase_tables.append(
            phase_table
        )

        person_phase.to_csv(
            OUTPUT_DIR
            / (
                f"{short_name}_"
                "person_phase_effects.csv"
            ),
            index=False,
        )

        cycle_phase.to_csv(
            OUTPUT_DIR
            / (
                f"{short_name}_"
                "cycle_phase_effects.csv"
            ),
            index=False,
        )

        decomposed[
            [
                "id",
                "study_interval",
                "cycle_number",
                "cycle_id",
                "day_in_study",
                "phase",
                "value_z",
                "population_phase_mean",
                "person_phase_effect",
                "cycle_phase_effect",
                "day_residual",
            ]
        ].to_csv(
            OUTPUT_DIR
            / (
                f"{short_name}_"
                "decomposition.csv"
            ),
            index=False,
        )

    summary = pd.concat(
        summaries,
        ignore_index=True,
    )

    phase_patterns = pd.concat(
        phase_tables,
        ignore_index=True,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "variance_components.csv",
        index=False,
    )

    phase_patterns.to_csv(
        OUTPUT_DIR
        / "population_phase_patterns.csv",
        index=False,
    )

    print("=" * 78)
    print("AIM 2 SIMULATION COMPONENT ESTIMATION")
    print("=" * 78)

    print(
        "\nIncluded complete-case days:",
        len(data),
    )

    print(
        "Participants:",
        data["id"].nunique(),
    )

    print(
        "Cycles:",
        data["cycle_id"].nunique(),
    )

    display_columns = [
        "signal",
        "population_phase_variance",
        "person_phase_variance",
        "cycle_phase_variance",
        "day_variance",
        "person_fraction",
        "cycle_fraction",
        "day_fraction",
    ]

    print("\nVARIANCE COMPONENTS")
    print(
        summary[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print("\nPOPULATION PHASE PATTERNS")
    print(
        phase_patterns.pivot(
            index="phase",
            columns="signal",
            values="population_phase_mean_z",
        )
        .reindex(PHASES)
        .round(3)
        .to_string()
    )

    print(
        "\nOutputs written to:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
