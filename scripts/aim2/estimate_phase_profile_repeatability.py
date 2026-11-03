"""Estimate cycle-to-cycle repeatability of personal phase profiles."""

from __future__ import annotations

from itertools import combinations
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
    """Restrict to complete-case phase observations."""

    mask = (
        data["phase"].isin(PHASES)
        & data[list(SIGNALS)].notna().all(axis=1)
    )

    output = data.loc[mask].copy()

    output["person_interval"] = (
        output["id"].astype(int).astype(str)
        + "_"
        + output["study_interval"]
        .astype(int)
        .astype(str)
    )

    output["cycle_id"] = (
        output["person_interval"]
        + "_"
        + output["cycle_number"]
        .astype(int)
        .astype(str)
    )

    return output


def analyze_signal(
    data: pd.DataFrame,
    signal: str,
    short_name: str,
) -> tuple[dict, pd.DataFrame]:
    """Estimate repeatability of phase-specific profiles."""

    x = data.copy()

    signal_mean = x[signal].mean()
    signal_sd = x[signal].std(ddof=1)

    x["value_z"] = (
        x[signal] - signal_mean
    ) / signal_sd

    # Population-average phase pattern.
    phase_means = (
        x.groupby("phase")["value_z"]
        .mean()
        .reindex(PHASES)
    )

    x["population_phase_mean"] = (
        x["phase"].map(phase_means)
    )

    x["after_population_phase"] = (
        x["value_z"]
        - x["population_phase_mean"]
    )

    # Person-level baseline variation, separate from
    # phase-specific personalization.
    person_baselines = (
        x.groupby("person_interval")[
            "after_population_phase"
        ]
        .mean()
    )

    person_baseline_variance = float(
        person_baselines.var(ddof=1)
    )

    # One physiological mean for each cycle x phase.
    cells = (
        x.groupby(
            [
                "person_interval",
                "cycle_id",
                "phase",
            ],
            as_index=False,
        )
        .agg(
            cell_mean_z=(
                "value_z",
                "mean",
            ),
            n_days=(
                "value_z",
                "size",
            ),
        )
    )

    # Keep cycles with observations for all four phases.
    profiles = cells.pivot(
        index=[
            "person_interval",
            "cycle_id",
        ],
        columns="phase",
        values="cell_mean_z",
    )

    profiles = profiles.reindex(
        columns=PHASES
    )

    profiles = profiles.dropna(
        how="any"
    )

    # Remove the population-average phase pattern.
    for phase in PHASES:
        profiles[phase] = (
            profiles[phase]
            - phase_means[phase]
        )

    # Remove each cycle's overall level.
    #
    # This is crucial: what remains is the SHAPE of that
    # person's phase pattern, not whether they simply have
    # high or low HR/HRV/temp/resp overall.
    profiles = profiles.sub(
        profiles.mean(axis=1),
        axis=0,
    )

    paired_x = []
    paired_y = []
    pair_records = []

    profile_reset = (
        profiles.reset_index()
    )

    for person_interval, group in (
        profile_reset.groupby(
            "person_interval"
        )
    ):
        if len(group) < 2:
            continue

        records = list(
            group.to_dict(
                orient="records"
            )
        )

        for first, second in combinations(
            records,
            2,
        ):
            for phase in PHASES:
                value_a = first[phase]
                value_b = second[phase]

                paired_x.append(
                    value_a
                )

                paired_y.append(
                    value_b
                )

                pair_records.append(
                    {
                        "signal": short_name,
                        "person_interval":
                            person_interval,
                        "cycle_a":
                            first["cycle_id"],
                        "cycle_b":
                            second["cycle_id"],
                        "phase": phase,
                        "profile_a":
                            value_a,
                        "profile_b":
                            value_b,
                    }
                )

    paired_x = np.asarray(
        paired_x,
        dtype=float,
    )

    paired_y = np.asarray(
        paired_y,
        dtype=float,
    )

    if len(paired_x) < 2:
        raise RuntimeError(
            f"Not enough repeated cycles for "
            f"{short_name}."
        )

    repeatability = float(
        np.corrcoef(
            paired_x,
            paired_y,
        )[0, 1]
    )

    covariance = float(
        np.cov(
            paired_x,
            paired_y,
            ddof=1,
        )[0, 1]
    )

    total_profile_variance = float(
        (
            paired_x.var(ddof=1)
            + paired_y.var(ddof=1)
        )
        / 2
    )

    stable_variance = float(
        np.clip(
            covariance,
            0.0,
            total_profile_variance,
        )
    )

    transient_variance = float(
        total_profile_variance
        - stable_variance
    )

    # Day-to-day noise around each cycle-phase mean.
    cell_lookup = cells.rename(
        columns={
            "cell_mean_z":
                "observed_cycle_phase_mean"
        }
    )[
        [
            "person_interval",
            "cycle_id",
            "phase",
            "observed_cycle_phase_mean",
        ]
    ]

    day_data = x.merge(
        cell_lookup,
        on=[
            "person_interval",
            "cycle_id",
            "phase",
        ],
        how="left",
        validate="many_to_one",
    )

    day_data["day_residual"] = (
        day_data["value_z"]
        - day_data[
            "observed_cycle_phase_mean"
        ]
    )

    day_variance = float(
        day_data[
            "day_residual"
        ].var(ddof=1)
    )

    repeated_people = (
        profile_reset.groupby(
            "person_interval"
        )["cycle_id"]
        .nunique()
    )

    repeated_people = int(
        (repeated_people >= 2).sum()
    )

    summary = {
        "signal": short_name,
        "raw_mean": signal_mean,
        "raw_sd": signal_sd,
        "n_complete_profile_cycles":
            len(profiles),
        "n_repeated_person_intervals":
            repeated_people,
        "n_cycle_pairs": len(pair_records) // len(PHASES),
        "person_baseline_variance":
            person_baseline_variance,
        "profile_repeatability":
            repeatability,
        "stable_person_phase_variance":
            stable_variance,
        "transient_cycle_phase_variance":
            transient_variance,
        "day_variance":
            day_variance,
    }

    return (
        summary,
        pd.DataFrame(pair_records),
    )


def main() -> None:
    """Estimate phase-profile repeatability."""

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
    pair_tables = []

    for signal, short_name in SIGNALS.items():
        summary, pairs = analyze_signal(
            data,
            signal,
            short_name,
        )

        summaries.append(
            summary
        )

        pair_tables.append(
            pairs
        )

    summary_table = pd.DataFrame(
        summaries
    )

    pair_table = pd.concat(
        pair_tables,
        ignore_index=True,
    )

    summary_table.to_csv(
        OUTPUT_DIR
        / "phase_profile_repeatability.csv",
        index=False,
    )

    pair_table.to_csv(
        OUTPUT_DIR
        / "cycle_profile_pairs.csv",
        index=False,
    )

    print("=" * 78)
    print("PHASE-PROFILE REPEATABILITY FOR AIM 2 SIMULATION")
    print("=" * 78)

    columns = [
        "signal",
        "n_complete_profile_cycles",
        "n_repeated_person_intervals",
        "n_cycle_pairs",
        "person_baseline_variance",
        "profile_repeatability",
        "stable_person_phase_variance",
        "transient_cycle_phase_variance",
        "day_variance",
    ]

    print(
        summary_table[
            columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print(
        "\nInterpretation:"
    )

    print(
        "profile_repeatability > 0 means a "
        "person's phase-specific deviation tends "
        "to recur in later cycles."
    )

    print(
        "stable_person_phase_variance is the "
        "cross-cycle covariance of that phase "
        "profile."
    )


if __name__ == "__main__":
    main()
