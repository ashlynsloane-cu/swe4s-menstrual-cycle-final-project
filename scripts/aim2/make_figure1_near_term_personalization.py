"""Create Aim 2 Figure 1: near-term two-cycle personalization.

Panel A summarizes the leakage-free within-interval design.
Panels B-D are generated directly from the finalized prediction table.
"""

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


PREDICTION_PATH = Path(
    "results/aim2/primary_models/all_predictions.csv"
)

ASSET_DIR = Path(
    "assets/aim2"
)

OUTPUT_DIR = Path(
    "results/figures/aim2"
)

MODELS = [
    "population_only",
    "hybrid_personalized",
]

PHASES = [
    "Menstrual",
    "Follicular",
    "Fertility",
    "Luteal",
]

PHASE_COLORS = {
    "Menstrual": "#D94A4A",
    "Follicular": "#E8A9B6",
    "Fertility": "#8C70B7",
    "Luteal": "#3E6FB6",
}

PROB_COLUMNS = {
    "Menstrual": "prob_menstrual",
    "Follicular": "prob_follicular",
    "Fertility": "prob_fertility",
    "Luteal": "prob_luteal",
}


def validate_inputs():
    """Check required prediction and icon files."""

    required = [
        PREDICTION_PATH,
        ASSET_DIR
        / "icon_population_phase.png",
        ASSET_DIR
        / "icon_person_signature.png",
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        message = "\n".join(
            f"  - {path}"
            for path in missing
        )

        raise FileNotFoundError(
            "Figure 1 is missing required inputs:\n"
            f"{message}"
        )


def panel_label(
    axis,
    label,
):
    """Place a panel letter."""

    axis.text(
        -0.12,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top",
    )


def add_icon(
    axis,
    image_path,
    xy,
    zoom,
):
    """Place a fixed transparent icon."""

    image = mpimg.imread(
        image_path
    )

    image_box = OffsetImage(
        image,
        zoom=zoom,
    )

    artist = AnnotationBbox(
        image_box,
        xy,
        xycoords="axes fraction",
        frameon=False,
        box_alignment=(
            0.5,
            0.5,
        ),
        pad=0,
    )

    axis.add_artist(
        artist
    )


def draw_design_panel(
    axis,
):
    """Draw the within-interval two-cycle evaluation design."""

    axis.axis(
        "off"
    )

    axis.set_xlim(
        0,
        1,
    )

    axis.set_ylim(
        0,
        1,
    )

    cycle_icon = (
        ASSET_DIR
        / "icon_population_phase.png"
    )

    person_icon = (
        ASSET_DIR
        / "icon_person_signature.png"
    )

    axis.set_title(
        "Near-term personalization design",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )

    axis.text(
        0.15,
        0.88,
        "Cycles 1–2",
        ha="center",
        fontsize=10,
        fontweight="bold",
    )

    axis.text(
        0.50,
        0.88,
        "Estimate personal\nsignature",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="#0072B2",
    )

    axis.text(
        0.85,
        0.88,
        "Future\ncycle(s)",
        ha="center",
        fontsize=10,
        fontweight="bold",
    )

    add_icon(
        axis,
        cycle_icon,
        (
            0.15,
            0.69,
        ),
        zoom=0.21,
    )

    add_icon(
        axis,
        person_icon,
        (
            0.50,
            0.69,
        ),
        zoom=0.19,
    )

    add_icon(
        axis,
        cycle_icon,
        (
            0.85,
            0.69,
        ),
        zoom=0.21,
    )

    axis.annotate(
        "",
        xy=(
            0.38,
            0.69,
        ),
        xytext=(
            0.26,
            0.69,
        ),
        xycoords="axes fraction",
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.4,
        },
    )

    axis.annotate(
        "",
        xy=(
            0.73,
            0.69,
        ),
        xytext=(
            0.62,
            0.69,
        ),
        xycoords="axes fraction",
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.4,
        },
    )

    population_box = FancyBboxPatch(
        (
            0.12,
            0.30,
        ),
        0.34,
        0.19,
        transform=axis.transAxes,
        boxstyle="round,pad=0.01",
        facecolor="#EEEEEE",
        edgecolor="#666666",
        linewidth=1,
    )

    personalized_box = FancyBboxPatch(
        (
            0.54,
            0.30,
        ),
        0.34,
        0.19,
        transform=axis.transAxes,
        boxstyle="round,pad=0.01",
        facecolor="#E5F1F8",
        edgecolor="#0072B2",
        linewidth=1.5,
    )

    axis.add_patch(
        population_box
    )

    axis.add_patch(
        personalized_box
    )

    axis.text(
        0.29,
        0.425,
        "Population model",
        transform=axis.transAxes,
        ha="center",
        fontsize=9.5,
        fontweight="bold",
    )

    axis.text(
        0.29,
        0.35,
        "No target-participant history",
        transform=axis.transAxes,
        ha="center",
        fontsize=8.5,
    )

    axis.text(
        0.71,
        0.425,
        "Personalized model",
        transform=axis.transAxes,
        ha="center",
        fontsize=9.5,
        fontweight="bold",
        color="#0072B2",
    )

    axis.text(
        0.71,
        0.35,
        "Adds target cycles 1–2",
        transform=axis.transAxes,
        ha="center",
        fontsize=8.5,
    )

    axis.text(
        0.50,
        0.15,
        "Both models predict the same held-out future days.",
        transform=axis.transAxes,
        ha="center",
        fontsize=9,
        fontweight="bold",
    )

    axis.text(
        0.50,
        0.06,
        "Target participant excluded from population training.",
        transform=axis.transAxes,
        ha="center",
        fontsize=8.2,
        style="italic",
    )


def evaluate_classification(
    data,
):
    """Calculate classification metrics."""

    truth = data[
        "phase"
    ].to_numpy()

    predicted = data[
        "predicted_phase"
    ].to_numpy()

    return {
        "accuracy": accuracy_score(
            truth,
            predicted,
        ),
        "balanced_accuracy": (
            balanced_accuracy_score(
                truth,
                predicted,
            )
        ),
        "macro_f1": f1_score(
            truth,
            predicted,
            labels=PHASES,
            average="macro",
            zero_division=0,
        ),
    }


def evaluate_probabilistic(
    data,
):
    """Calculate log loss and multiclass Brier score."""

    probabilities = data[
        [
            PROB_COLUMNS[
                phase
            ]
            for phase in PHASES
        ]
    ].to_numpy()

    phase_to_index = {
        phase: index
        for index, phase in enumerate(
            PHASES
        )
    }

    truth_index = np.array(
        [
            phase_to_index[
                phase
            ]
            for phase in data[
                "phase"
            ]
        ],
        dtype=int,
    )

    rows = np.arange(
        len(data)
    )

    true_probability = probabilities[
        rows,
        truth_index,
    ]

    one_hot = np.zeros_like(
        probabilities
    )

    one_hot[
        rows,
        truth_index,
    ] = 1.0

    return {
        "log_loss": float(
            -np.mean(
                np.log(
                    np.clip(
                        true_probability,
                        1e-15,
                        1.0,
                    )
                )
            )
        ),
        "brier_score": float(
            np.mean(
                np.sum(
                    (
                        probabilities
                        - one_hot
                    )
                    ** 2,
                    axis=1,
                )
            )
        ),
    }


def classification_bootstrap(
    predictions,
    n_boot=5000,
    seed=6118,
):
    """Paired participant-cluster bootstrap for classification deltas."""

    rng = np.random.default_rng(
        seed
    )

    participants = np.array(
        sorted(
            predictions[
                "target_id"
            ].unique()
        )
    )

    metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
    ]

    rows = []

    for iteration in range(
        n_boot
    ):
        sampled = rng.choice(
            participants,
            size=len(participants),
            replace=True,
        )

        pieces = []

        for draw_index, target_id in enumerate(
            sampled
        ):
            part = predictions[
                predictions[
                    "target_id"
                ]
                == target_id
            ].copy()

            part[
                "_bootstrap_cluster"
            ] = draw_index

            pieces.append(
                part
            )

        sample = pd.concat(
            pieces,
            ignore_index=True,
        )

        model_metrics = {}

        for model_name in MODELS:
            subset = sample[
                sample[
                    "model"
                ]
                == model_name
            ]

            model_metrics[
                model_name
            ] = evaluate_classification(
                subset
            )

        row = {
            "iteration": iteration,
        }

        for metric in metrics:
            row[
                metric
            ] = (
                model_metrics[
                    "hybrid_personalized"
                ][metric]
                - model_metrics[
                    "population_only"
                ][metric]
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def classification_summary(
    predictions,
    bootstrap,
):
    """Summarize classification differences and bootstrap intervals."""

    metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
    ]

    model_metrics = {}

    for model_name in MODELS:
        model_metrics[
            model_name
        ] = evaluate_classification(
            predictions[
                predictions[
                    "model"
                ]
                == model_name
            ]
        )

    rows = []

    for metric in metrics:
        observed = (
            model_metrics[
                "hybrid_personalized"
            ][metric]
            - model_metrics[
                "population_only"
            ][metric]
        )

        rows.append(
            {
                "metric": metric,
                "difference": observed,
                "ci_2.5": bootstrap[
                    metric
                ].quantile(
                    0.025
                ),
                "ci_97.5": bootstrap[
                    metric
                ].quantile(
                    0.975
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def draw_classification_panel(
    axis,
    summary,
):
    """Plot classification differences with participant-bootstrap CIs."""

    labels = {
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced\naccuracy",
        "macro_f1": "Macro-F1",
    }

    order = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
    ]

    data = (
        summary.set_index(
            "metric"
        )
        .loc[
            order
        ]
        .reset_index()
    )

    y = np.arange(
        len(data)
    )

    values = data[
        "difference"
    ].to_numpy()

    low = data[
        "ci_2.5"
    ].to_numpy()

    high = data[
        "ci_97.5"
    ].to_numpy()

    xerr = np.vstack(
        [
            values - low,
            high - values,
        ]
    )

    axis.errorbar(
        values,
        y,
        xerr=xerr,
        fmt="o",
        markersize=7,
        linewidth=1.5,
        capsize=4,
        color="#0072B2",
    )

    axis.axvline(
        0,
        color="#666666",
        linestyle="--",
        linewidth=1,
    )

    axis.set_yticks(
        y
    )

    axis.set_yticklabels(
        [
            labels[
                metric
            ]
            for metric in order
        ]
    )

    axis.invert_yaxis()

    axis.set_xlabel(
        "Personalized − population"
    )

    axis.set_title(
        "No reliable improvement in discrimination",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )

    axis.text(
        0.98,
        0.985,
        "Right of 0 = personalization better",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color="#0072B2",
        fontweight="bold",
    )

    axis.spines[
        "top"
    ].set_visible(
        False
    )

    axis.spines[
        "right"
    ].set_visible(
        False
    )


def draw_probabilistic_panel(
    axis,
    predictions,
):
    """Compare population and personalized probabilistic scores."""

    rows = []

    for model_name in MODELS:
        metrics = evaluate_probabilistic(
            predictions[
                predictions[
                    "model"
                ]
                == model_name
            ]
        )

        rows.append(
            {
                "model": model_name,
                **metrics,
            }
        )

    scores = (
        pd.DataFrame(
            rows
        )
        .set_index(
            "model"
        )
    )

    categories = [
        "Log loss",
        "Brier score",
    ]

    population = [
        scores.loc[
            "population_only",
            "log_loss",
        ],
        scores.loc[
            "population_only",
            "brier_score",
        ],
    ]

    personalized = [
        scores.loc[
            "hybrid_personalized",
            "log_loss",
        ],
        scores.loc[
            "hybrid_personalized",
            "brier_score",
        ],
    ]

    x = np.arange(
        len(categories)
    )

    width = 0.34

    axis.bar(
        x - width / 2,
        population,
        width,
        label="Population",
        color="#999999",
    )

    axis.bar(
        x + width / 2,
        personalized,
        width,
        label="Personalized",
        color="#D55E00",
    )

    for position, value in zip(
        x - width / 2,
        population,
    ):
        axis.text(
            position,
            value + 0.025,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    for position, value in zip(
        x + width / 2,
        personalized,
    ):
        axis.text(
            position,
            value + 0.025,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    axis.set_xticks(
        x
    )

    axis.set_xticklabels(
        categories
    )

    axis.set_ylabel(
        "Prediction error\n(lower is better)"
    )

    axis.set_title(
        "Probabilistic performance worsens",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )

    axis.legend(
        frameon=False,
        fontsize=9,
    )

    axis.spines[
        "top"
    ].set_visible(
        False
    )

    axis.spines[
        "right"
    ].set_visible(
        False
    )


def prediction_distribution(
    predictions,
):
    """Calculate true and predicted phase distributions."""

    population = predictions[
        predictions[
            "model"
        ]
        == "population_only"
    ].copy()

    personalized = predictions[
        predictions[
            "model"
        ]
        == "hybrid_personalized"
    ].copy()

    true_counts = (
        population[
            "phase"
        ]
        .value_counts()
        .reindex(
            PHASES,
            fill_value=0,
        )
    )

    population_counts = (
        population[
            "predicted_phase"
        ]
        .value_counts()
        .reindex(
            PHASES,
            fill_value=0,
        )
    )

    personalized_counts = (
        personalized[
            "predicted_phase"
        ]
        .value_counts()
        .reindex(
            PHASES,
            fill_value=0,
        )
    )

    distributions = pd.DataFrame(
        {
            "True phases": (
                100
                * true_counts
                / true_counts.sum()
            ),
            "Population predictions": (
                100
                * population_counts
                / population_counts.sum()
            ),
            "Personalized predictions": (
                100
                * personalized_counts
                / personalized_counts.sum()
            ),
        }
    ).T

    return distributions


def draw_distribution_panel(
    axis,
    predictions,
):
    """Plot true versus predicted phase composition."""

    data = prediction_distribution(
        predictions
    )

    y = np.arange(
        len(data)
    )

    left = np.zeros(
        len(data)
    )

    for phase in PHASES:
        values = data[
            phase
        ].to_numpy()

        axis.barh(
            y,
            values,
            left=left,
            label=phase,
            color=PHASE_COLORS[
                phase
            ],
        )

        left += values

    axis.set_yticks(
        y
    )

    axis.set_yticklabels(
        data.index
    )

    axis.invert_yaxis()

    axis.set_xlim(
        0,
        100,
    )

    axis.set_xlabel(
        "Share of days (%)"
    )

    axis.set_title(
        "Personalization changes class balance, not overall accuracy",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )

    axis.legend(
        frameon=False,
        fontsize=8,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(
            0.5,
            -0.34,
        ),
    )

    axis.spines[
        "top"
    ].set_visible(
        False
    )

    axis.spines[
        "right"
    ].set_visible(
        False
    )


def main():
    """Create the four-panel near-term personalization figure."""

    validate_inputs()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions = pd.read_csv(
        PREDICTION_PATH
    )

    if len(
        predictions
    ) != 886:
        raise RuntimeError(
            "Expected 886 prediction rows "
            "(443 days × 2 models), found "
            f"{len(predictions)}."
        )

    if (
        predictions[
            "target_id"
        ].nunique()
        != 16
    ):
        raise RuntimeError(
            "Expected 16 unique target participants."
        )

    bootstrap = classification_bootstrap(
        predictions,
        n_boot=5000,
        seed=6118,
    )

    classification = classification_summary(
        predictions,
        bootstrap,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            12,
            8.5,
        ),
        constrained_layout=True,
    )

    draw_design_panel(
        axes[
            0,
            0,
        ]
    )

    draw_classification_panel(
        axes[
            0,
            1,
        ],
        classification,
    )

    draw_probabilistic_panel(
        axes[
            1,
            0,
        ],
        predictions,
    )

    draw_distribution_panel(
        axes[
            1,
            1,
        ],
        predictions,
    )

    for axis, label in zip(
        axes.flat,
        [
            "A",
            "B",
            "C",
            "D",
        ],
    ):
        panel_label(
            axis,
            label,
        )

    fig.suptitle(
        "Does two-cycle personalization improve future phase prediction?",
        fontsize=16,
        fontweight="bold",
    )

    output_paths = [
        (
            OUTPUT_DIR
            / "figure1_near_term_personalization.png"
        ),
        (
            OUTPUT_DIR
            / "figure1_near_term_personalization.pdf"
        ),
        (
            OUTPUT_DIR
            / "figure1_near_term_personalization.svg"
        ),
    ]

    fig.savefig(
        output_paths[0],
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        output_paths[1],
        bbox_inches="tight",
    )

    fig.savefig(
        output_paths[2],
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print("Saved:")

    for path in output_paths:
        print(path)

    print("\nClassification differences:")
    print(
        classification.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )


if __name__ == "__main__":
    main()
