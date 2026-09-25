"""Quality-control rules for the mcPHASES analysis dataset."""

from __future__ import annotations

import logging

import pandas as pd


LOGGER = logging.getLogger(__name__)

CORE_SIGNALS = (
    "resting_heart_rate",
    "hrv_rmssd",
    "nightly_temperature",
    "respiratory_rate",
)


def add_cycle_analysis_flags(
    cycle_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Add cycle-level Aim 1 quality-control flags."""

    data = cycle_summary.copy()

    if data.empty:
        for column in (
            "closed_by_next_menses",
            "passes_core_70pct",
            "passes_core_80pct",
            "aim1_primary_eligible",
            "aim1_sensitivity_80pct",
        ):
            data[column] = pd.Series(dtype=bool)

        return data

    required = {
        "id",
        "study_interval",
        "cycle_number",
        "cycle_start_day",
        "complete_cycle",
        "valid_phase_order",
    }

    missing = sorted(
        required - set(data.columns)
    )

    if missing:
        raise ValueError(
            "Cycle summary is missing required column(s): "
            + ", ".join(missing)
        )

    data = data.sort_values(
        [
            "id",
            "study_interval",
            "cycle_number",
        ]
    ).reset_index(drop=True)

    # Because our cycles start at menstruation, the start of
    # the next cycle represents the next observed menstrual onset.
    data["next_cycle_start_day"] = (
        data.groupby(
            [
                "id",
                "study_interval",
            ]
        )["cycle_start_day"]
        .shift(-1)
    )

    data["closed_by_next_menses"] = (
        data["next_cycle_start_day"].notna()
    )

    coverage_columns = [
        f"{signal}_coverage"
        for signal in CORE_SIGNALS
    ]

    missing_coverage = [
        column
        for column in coverage_columns
        if column not in data.columns
    ]

    if missing_coverage:
        LOGGER.warning(
            "Cannot evaluate core coverage; missing: %s",
            ", ".join(missing_coverage),
        )

        data["passes_core_70pct"] = False
        data["passes_core_80pct"] = False

    else:
        data["passes_core_70pct"] = (
            data[coverage_columns]
            .ge(0.70)
            .all(axis=1)
        )

        data["passes_core_80pct"] = (
            data[coverage_columns]
            .ge(0.80)
            .all(axis=1)
        )

    biological_cycle = (
        data["complete_cycle"].fillna(False)
        & data["valid_phase_order"].fillna(False)
        & data["closed_by_next_menses"].fillna(False)
    )

    data["aim1_primary_eligible"] = (
        biological_cycle
        & data["passes_core_70pct"]
    )

    data["aim1_sensitivity_80pct"] = (
        biological_cycle
        & data["passes_core_80pct"]
    )

    return data


def add_participant_analysis_flags(
    participant_summary: pd.DataFrame,
    cycle_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Add participant-level Aim 1 eligibility flags."""

    summary = participant_summary.copy()

    if "id" not in summary.columns:
        raise ValueError(
            "Participant summary is missing required column: id"
        )

    required = {
        "id",
        "study_interval",
        "aim1_primary_eligible",
    }

    missing = sorted(
        required - set(cycle_summary.columns)
    )

    if missing:
        raise ValueError(
            "Cycle summary is missing required column(s): "
            + ", ".join(missing)
        )

    eligible = cycle_summary[
        cycle_summary["aim1_primary_eligible"]
    ].copy()

    cycle_counts = (
        eligible.groupby("id")
        .size()
        .rename("n_aim1_primary_cycles")
        .reset_index()
    )

    interval_counts = (
        eligible[
            [
                "id",
                "study_interval",
            ]
        ]
        .drop_duplicates()
        .groupby("id")
        .size()
        .rename(
            "n_intervals_with_aim1_cycle"
        )
        .reset_index()
    )

    summary = summary.merge(
        cycle_counts,
        on="id",
        how="left",
    )

    summary = summary.merge(
        interval_counts,
        on="id",
        how="left",
    )

    summary["n_aim1_primary_cycles"] = (
        summary["n_aim1_primary_cycles"]
        .fillna(0)
        .astype(int)
    )

    summary["n_intervals_with_aim1_cycle"] = (
        summary["n_intervals_with_aim1_cycle"]
        .fillna(0)
        .astype(int)
    )

    # Primary Aim 1 requires repeated eligible cycles.
    summary["aim1_repeated_cycle_candidate"] = (
        summary["n_aim1_primary_cycles"] >= 2
    )

    # Long-term analysis requires usable cycles in both intervals.
    summary["long_term_candidate"] = (
        summary["n_intervals_with_aim1_cycle"] >= 2
    )

    return summary
