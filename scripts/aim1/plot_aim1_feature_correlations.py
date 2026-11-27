"""Visualize correlations among Aim 1 cycle-signature features."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mcphases.similarity import (
    get_feature_columns,
    select_repeated_cycle_cohort,
)


SIGNAL_LABELS = {
    "resting_heart_rate": "RHR",
    "hrv_rmssd": "HRV",
    "nightly_temperature": "Temp",
    "respiratory_rate": "RR",
}

PHASES = (
    "menstrual",
    "follicular",
    "fertility",
    "luteal",
)


def calculate_correlations(signatures):
    """Calculate Pearson correlations in the analysis cohort."""
    complete = (
        select_repeated_cycle_cohort(signatures)
        .dropna()
        .reset_index(drop=True)
    )

    features = get_feature_columns(complete)

    if len(complete) < 3 or not features:
        raise ValueError("Insufficient complete signature data.")

    ordered = []

    for signal in SIGNAL_LABELS:
        for phase in PHASES:
            ordered.extend(
                column
                for column in features
                if column.startswith(signal)
                and column.endswith(phase)
                and column not in ordered
            )

        ordered.extend(
            column
            for column in features
            if column.startswith(signal)
            and column not in ordered
        )

    ordered.extend(
        column for column in features
        if column not in ordered
    )

    correlations = complete[ordered].corr(
        method="pearson"
    )

    if not np.isfinite(correlations.to_numpy()).all():
        raise ValueError("Undefined feature correlations.")

    return correlations


def format_label(feature):
    """Shorten feature names for plotting."""
    for signal, label in SIGNAL_LABELS.items():
        if feature.startswith(signal):
            suffix = feature[len(signal):]
            return label + suffix.replace("_", " ")

    return feature.replace("_", " ")


def make_figure(correlations):
    """Create a diverging feature-correlation heatmap."""
    fig, ax = plt.subplots(figsize=(10, 9))

    image = ax.imshow(
        correlations.to_numpy(),
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
    )

    labels = [
        format_label(feature)
        for feature in correlations.columns
    ]

    positions = np.arange(len(labels))

    ax.set_xticks(positions)
    ax.set_yticks(positions)

    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
        fontsize=9,
    )

    ax.set_yticklabels(
        labels,
        fontsize=9,
    )

    for row in range(len(labels)):
        for col in range(len(labels)):
            value = correlations.iloc[row, col]

            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color=(
                    "white"
                    if abs(value) > 0.65
                    else "#222222"
                ),
            )

    groups = [
        next(
            (
                signal
                for signal in SIGNAL_LABELS
                if feature.startswith(signal)
            ),
            feature,
        )
        for feature in correlations.columns
    ]

    for index in range(1, len(groups)):
        if groups[index] != groups[index - 1]:
            ax.axhline(
                index - 0.5,
                color="white",
                linewidth=2.5,
            )
            ax.axvline(
                index - 0.5,
                color="white",
                linewidth=2.5,
            )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    colorbar.set_label(
        "Pearson correlation (r)"
    )

    ax.set_title(
        "Correlation structure of physiological "
        "cycle signatures",
        fontsize=14,
        fontweight="bold",
        pad=18,
    )

    ax.tick_params(length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()

    return fig


def main():
    """Generate the feature correlation matrix and figure."""
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--signatures",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures/aim1"),
    )

    args = parser.parse_args()

    plt.switch_backend("Agg")

    signatures = pd.read_csv(args.signatures)
    correlations = calculate_correlations(signatures)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    correlations.to_csv(
        args.output_dir
        / "aim1_feature_correlations.csv"
    )

    fig = make_figure(correlations)

    for extension in ("png", "pdf", "svg"):
        fig.savefig(
            args.output_dir
            / f"aim1_feature_correlations.{extension}",
            dpi=300,
            bbox_inches="tight",
        )

    plt.close(fig)


if __name__ == "__main__":
    main()
