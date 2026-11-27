"""Compare Aim 1 similarity advantages across distance metrics."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


LABEL_MAP = {
    "resting_heart_rate": "Resting heart rate",
    "nightly_temperature": "Nightly temperature",
    "hrv_rmssd": "HRV (RMSSD)",
    "respiratory_rate": "Respiratory rate",
    "all_signals": "All signals",
}

PLOT_ORDER = [
    "resting_heart_rate",
    "nightly_temperature",
    "hrv_rmssd",
    "respiratory_rate",
    "all_signals",
]

EUCLIDEAN_COLOR = "#0072B2"
MAHALANOBIS_COLOR = "#D55E00"
CONNECTOR_COLOR = "#A0A0A0"


def load_results(path):
    """Load signal-specific reproducibility results."""
    results = pd.read_csv(path)

    required = {
        "signal",
        "similarity_advantage",
        "p_value",
    }

    missing = required - set(results.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    return results


def prepare_comparison(euclidean, mahalanobis):
    """Join both distance-metric results by physiological signal."""
    comparison = euclidean[
        [
            "signal",
            "similarity_advantage",
            "p_value",
        ]
    ].rename(
        columns={
            "similarity_advantage": "euclidean_advantage",
            "p_value": "euclidean_p",
        }
    )

    mahalanobis_subset = mahalanobis[
        [
            "signal",
            "similarity_advantage",
            "p_value",
        ]
    ].rename(
        columns={
            "similarity_advantage": "mahalanobis_advantage",
            "p_value": "mahalanobis_p",
        }
    )

    comparison = comparison.merge(
        mahalanobis_subset,
        on="signal",
        how="inner",
        validate="one_to_one",
    )

    comparison["plot_order"] = comparison[
        "signal"
    ].map(
        {
            signal: index
            for index, signal in enumerate(
                PLOT_ORDER
            )
        }
    )

    comparison = comparison.sort_values(
        "plot_order"
    ).reset_index(drop=True)

    return comparison


def make_figure(comparison):
    """Build a dumbbell plot comparing both distance metrics."""
    fig, ax = plt.subplots(
        figsize=(9.5, 5.8)
    )

    for index, row in comparison.iterrows():
        y = index

        ax.plot(
            [
                row["euclidean_advantage"],
                row["mahalanobis_advantage"],
            ],
            [y, y],
            color=CONNECTOR_COLOR,
            linewidth=2,
            zorder=1,
        )

        ax.scatter(
            row["euclidean_advantage"],
            y,
            s=100,
            color=EUCLIDEAN_COLOR,
            label=(
                "Standardized Euclidean"
                if index == 0
                else None
            ),
            zorder=2,
        )

        ax.scatter(
            row["mahalanobis_advantage"],
            y,
            s=100,
            color=MAHALANOBIS_COLOR,
            label=(
                "Regularized Mahalanobis"
                if index == 0
                else None
            ),
            zorder=2,
        )

    ax.axvline(
        0,
        color="#555555",
        linewidth=1,
        linestyle="--",
    )

    ax.set_yticks(
        range(len(comparison))
    )

    tick_labels = [
        LABEL_MAP[signal]
        for signal in comparison["signal"]
    ]

    ax.set_yticklabels(
        tick_labels,
        fontsize=11,
    )

    for tick, signal in zip(
        ax.get_yticklabels(),
        comparison["signal"],
    ):
        if signal == "all_signals":
            tick.set_fontweight("bold")

    ax.invert_yaxis()

    ax.set_xlabel(
        "Similarity advantage\n"
        "(different-person mean distance "
        "− same-person mean distance)",
        fontsize=11,
    )

    ax.set_title(
        "Signal-specific patterns are consistent "
        "across distance metrics",
        loc="left",
        fontsize=14,
        fontweight="bold",
    )

    ax.grid(
        axis="x",
        color="#D5D5D5",
        alpha=0.5,
        linewidth=0.8,
    )

    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.tick_params(
        axis="y",
        length=0,
    )

    ax.legend(
        frameon=False,
        loc="upper left",
    )

    fig.tight_layout()

    return fig


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--euclidean",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--mahalanobis",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/figures/aim1"
        ),
    )

    return parser.parse_args()


def main():
    """Generate the Aim 1 metric-comparison figure."""
    args = parse_args()

    plt.switch_backend("Agg")

    euclidean = load_results(
        args.euclidean
    )

    mahalanobis = load_results(
        args.mahalanobis
    )

    comparison = prepare_comparison(
        euclidean,
        mahalanobis,
    )

    fig = make_figure(
        comparison
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        args.output_dir
        / "aim1_metric_comparison.csv",
        index=False,
    )

    stem = (
        args.output_dir
        / "aim1_metric_comparison"
    )

    fig.savefig(
        stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    fig.savefig(
        stem.with_suffix(".svg"),
        bbox_inches="tight",
    )

    plt.close(fig)


if __name__ == "__main__":
    main()
