"""Cycle-level physiological feature extraction for Aim 1."""

import pandas as pd

CORE_SIGNALS = (
    "resting_heart_rate",
    "hrv_rmssd",
    "nightly_temperature",
    "respiratory_rate",
)

PHASES = (
    "Menstrual",
    "Follicular",
    "Fertility",
    "Luteal",
)


def calculate_phase_means(cycle_data):
    """Calculate mean metrics for each menstrual phase within a cycle."""
    phase_means = cycle_data.groupby("phase")[list(CORE_SIGNALS)].mean()

    return phase_means


def calculate_cycle_means(cycle_data):
    """Calculate mean metrics across the entire menstrual cycle."""
    cycle_means = cycle_data[list(CORE_SIGNALS)].mean()

    return cycle_means


def calculate_centered_phase_means(cycle_data):
    """Center each phase mean relative to the cycle-wide mean."""
    phase_means = calculate_phase_means(cycle_data)
    cycle_means = calculate_cycle_means(cycle_data)

    centered = phase_means - cycle_means

    return centered


def build_cycle_signature(cycle_data):
    """Build a 16-feature within-cycle physiological signature."""
    centered = calculate_centered_phase_means(cycle_data)

    features = {}

    for signal in CORE_SIGNALS:
        for phase in PHASES:
            feature_name = f"{signal}_{phase.lower()}_centered"
            features[feature_name] = centered.loc[phase, signal]

    return pd.Series(features)


def build_all_cycle_signatures(eligible_data):
    """Build one physiological signature for each eligible cycle."""
    signatures = []

    grouped = eligible_data.groupby(
        ["id", "study_interval", "cycle_number"]
    )

    for (participant_id, study_interval, cycle_number), cycle_data in grouped:
        signature = build_cycle_signature(cycle_data)

        signature["id"] = participant_id
        signature["study_interval"] = study_interval
        signature["cycle_number"] = cycle_number

        signatures.append(signature)

    return pd.DataFrame(signatures)
