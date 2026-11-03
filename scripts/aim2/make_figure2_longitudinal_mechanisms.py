"""Create Aim 2 Figure 2: longitudinal personalization mechanisms."""

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyBboxPatch


ORACLE_DIR = Path(
    "results/aim2/oracle_recenter"
)

FAILURE_DIR = Path(
    "results/aim2/personalization_failure"
)

RESIDUAL_DIR = Path(
    "results/aim2/residual_phase_drift"
)

OUTPUT_DIR = Path(
    "results/figures/aim2"
)

ASSET_DIR = Path(
    "assets/aim2"
)


MODEL_ORDER = [
    "population_only",
    "hybrid_personalized",
    "oracle_2024_recentered",
]

MODEL_LABELS = [
    "Population",
    "2022\npersonalized",
    "2024 baseline\nupdated",
]

MODEL_COLORS = [
    "#4D4D4D",
    "#D55E00",
    "#0072B2",
]

SIGNALS = [
    "rhr",
    "hrv",
    "temp",
    "resp",
]

SIGNAL_LABELS = [
    "RHR",
    "HRV",
    "Temperature",
    "Respiration",
]


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


def add_design_icon(
    axis,
    path,
    xy,
    zoom,
):
    """Place a fixed schematic asset in Panel A."""

    image = mpimg.imread(
        path
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
    """Draw a clean icon-based longitudinal design."""

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

    for icon_path in [
        cycle_icon,
        person_icon,
    ]:
        if not icon_path.exists():
            raise FileNotFoundError(
                f"Missing Figure 2 asset: {icon_path}"
            )

    axis.set_title(
        "Longitudinal personalization design",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )

    # --------------------------------------------------------------
    # Top row: calibration -> signature -> held-out test
    # --------------------------------------------------------------

    x_positions = [
        0.18,
        0.50,
        0.82,
    ]

    labels = [
        "2022 calibration",
        "Personal signature",
        "2024 test",
    ]

    label_colors = [
        "#222222",
        "#0072B2",
        "#222222",
    ]

    for x, label, color in zip(
        x_positions,
        labels,
        label_colors,
    ):
        axis.text(
            x,
            0.88,
            label,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=color,
        )

    add_design_icon(
        axis,
        cycle_icon,
        (
            0.18,
            0.69,
        ),
        zoom=0.21,
    )

    add_design_icon(
        axis,
        person_icon,
        (
            0.50,
            0.69,
        ),
        zoom=0.19,
    )

    add_design_icon(
        axis,
        cycle_icon,
        (
            0.82,
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
            0.29,
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
            0.70,
            0.69,
        ),
        xytext=(
            0.61,
            0.69,
        ),
        xycoords="axes fraction",
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.4,
        },
    )

    # --------------------------------------------------------------
    # Simple branch from 2024 test into two baseline conditions
    # --------------------------------------------------------------

    axis.plot(
        [
            0.82,
            0.82,
        ],
        [
            0.58,
            0.53,
        ],
        transform=axis.transAxes,
        color="#555555",
        linewidth=1.1,
    )

    axis.plot(
        [
            0.32,
            0.82,
        ],
        [
            0.53,
            0.53,
        ],
        transform=axis.transAxes,
        color="#555555",
        linewidth=1.1,
    )

    for x in [
        0.34,
        0.72,
    ]:
        axis.annotate(
            "",
            xy=(
                x,
                0.45,
            ),
            xytext=(
                x,
                0.53,
            ),
            xycoords="axes fraction",
            arrowprops={
                "arrowstyle": "->",
                "linewidth": 1.1,
                "color": "#555555",
            },
        )

    # --------------------------------------------------------------
    # Minimal baseline condition labels
    # --------------------------------------------------------------

    axis.text(
        0.34,
        0.39,
        "Original prediction",
        transform=axis.transAxes,
        ha="center",
        fontsize=9.5,
        fontweight="bold",
    )

    axis.plot(
        [
            0.20,
            0.48,
        ],
        [
            0.32,
            0.32,
        ],
        transform=axis.transAxes,
        color="#555555",
        linewidth=2.2,
    )

    axis.text(
        0.34,
        0.255,
        "Use 2022 baseline",
        transform=axis.transAxes,
        ha="center",
        fontsize=8.8,
    )

    axis.text(
        0.72,
        0.39,
        "Baseline-updated diagnostic",
        transform=axis.transAxes,
        ha="center",
        fontsize=9.5,
        fontweight="bold",
        color="#0072B2",
    )

    axis.plot(
        [
            0.59,
            0.85,
        ],
        [
            0.295,
            0.295,
        ],
        transform=axis.transAxes,
        color="#AAAAAA",
        linestyle="--",
        linewidth=1.2,
    )

    axis.plot(
        [
            0.59,
            0.85,
        ],
        [
            0.34,
            0.34,
        ],
        transform=axis.transAxes,
        color="#0072B2",
        linewidth=2.2,
    )

    axis.annotate(
        "",
        xy=(
            0.57,
            0.34,
        ),
        xytext=(
            0.57,
            0.295,
        ),
        xycoords="axes fraction",
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.1,
            "color": "#0072B2",
        },
    )

    axis.text(
        0.72,
        0.255,
        "Use observed 2024 baseline",
        transform=axis.transAxes,
        ha="center",
        fontsize=8.8,
    )

    # --------------------------------------------------------------
    # One-line interpretation
    # --------------------------------------------------------------

    axis.text(
        0.50,
        0.12,
        "Same 2022 personal signature; only the baseline changes.",
        transform=axis.transAxes,
        ha="center",
        fontsize=9,
        fontweight="bold",
    )

    axis.text(
        0.50,
        0.035,
        "2024 baseline uses future information and is diagnostic only.",
        transform=axis.transAxes,
        ha="center",
        fontsize=8,
        style="italic",
    )


def draw_log_loss_panel(
    axis,
):
    """Plot participant-level longitudinal log loss."""

    metrics = pd.read_csv(
        ORACLE_DIR
        / "fold_metrics.csv"
    )

    overall = pd.read_csv(
        ORACLE_DIR
        / "overall_metrics.csv"
    )

    pivot = metrics.pivot(
        index="target_id",
        columns="model",
        values="log_loss",
    )

    x = np.arange(
        len(MODEL_ORDER)
    )

    for _, row in pivot.iterrows():
        values = [
            row[model]
            for model in MODEL_ORDER
        ]

        axis.plot(
            x,
            values,
            marker="o",
            linewidth=0.9,
            alpha=0.40,
            color="#777777",
        )

    pooled = (
        overall.set_index(
            "model"
        )["log_loss"]
    )

    pooled_values = [
        pooled.loc[model]
        for model in MODEL_ORDER
    ]

    axis.plot(
        x,
        pooled_values,
        marker="o",
        linewidth=2.2,
        markersize=7,
        color="black",
        zorder=5,
    )

    for position, value, color in zip(
        x,
        pooled_values,
        MODEL_COLORS,
    ):
        axis.scatter(
            position,
            value,
            s=70,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            zorder=6,
        )

        axis.text(
            position,
            value + 0.16,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    excess_before = (
        pooled.loc[
            "hybrid_personalized"
        ]
        - pooled.loc[
            "population_only"
        ]
    )

    excess_after = (
        pooled.loc[
            "oracle_2024_recentered"
        ]
        - pooled.loc[
            "population_only"
        ]
    )

    rescue_fraction = (
        1.0
        - excess_after
        / excess_before
    )

    axis.text(
        0.98,
        0.96,
        (
            "Updating the baseline removes\n"
            f"{100 * rescue_fraction:.0f}% of excess "
            "log-loss penalty"
        ),
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )

    axis.set_xticks(
        x
    )

    axis.set_xticklabels(
        MODEL_LABELS
    )

    axis.set_ylabel(
        "Log loss\n(lower is better)"
    )

    axis.set_title(
        "Updating the baseline rescues probabilistic performance",
        loc="left",
        fontsize=12,
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


def draw_baseline_panel(
    axis,
):
    """Plot same-period versus longitudinal baseline drift."""

    drift = pd.read_csv(
        FAILURE_DIR
        / "baseline_drift_summary.csv"
    )

    drift = drift.set_index(
        "analysis"
    )

    within = drift.loc[
        "within_interval"
    ]

    longitudinal = drift.loc[
        "longitudinal_2022_to_2024"
    ]

    within_values = [
        within[
            f"{signal}_abs_baseline_shift_z"
        ]
        for signal in SIGNALS
    ]

    longitudinal_values = [
        longitudinal[
            f"{signal}_abs_baseline_shift_z"
        ]
        for signal in SIGNALS
    ]

    x = np.arange(
        len(SIGNALS)
    )

    width = 0.36

    axis.bar(
        x - width / 2,
        within_values,
        width,
        label="Later cycles, same period",
        color="#999999",
    )

    longitudinal_bars = axis.bar(
        x + width / 2,
        longitudinal_values,
        width,
        label="2022 → 2024",
        color="#D55E00",
    )

    for bar, value in zip(
        longitudinal_bars,
        longitudinal_values,
    ):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
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
        SIGNAL_LABELS
    )

    axis.set_ylabel(
        "Median absolute baseline shift\n(signal-wide SD)"
    )

    axis.set_title(
        "Baselines drift across years",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )

    axis.legend(
        frameon=False,
        fontsize=8.5,
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


def draw_transfer_panel(
    axis,
):
    """Plot phase-profile transfer against residual failure."""

    data = pd.read_csv(
        RESIDUAL_DIR
        / "participant_residual_penalty.csv"
    )

    x = data[
        "stable_signal_transfer_score"
    ].to_numpy()

    y = data[
        "residual_log_loss_penalty"
    ].to_numpy()

    rho = pd.Series(
        x
    ).corr(
        pd.Series(y),
        method="spearman",
    )

    axis.scatter(
        x,
        y,
        s=55,
        color="#0072B2",
        edgecolor="black",
        linewidth=0.7,
        zorder=3,
    )

    for _, row in data.iterrows():
        axis.annotate(
            str(
                int(
                    row["target_id"]
                )
            ),
            (
                row[
                    "stable_signal_transfer_score"
                ],
                row[
                    "residual_log_loss_penalty"
                ],
            ),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    if len(data) >= 2:
        slope, intercept = np.polyfit(
            x,
            y,
            1,
        )

        x_line = np.linspace(
            x.min(),
            x.max(),
            100,
        )

        axis.plot(
            x_line,
            slope * x_line + intercept,
            linewidth=1.5,
            linestyle="--",
            color="#555555",
        )

    axis.axhline(
        0,
        linewidth=1,
        linestyle=":",
        color="#777777",
    )

    axis.axvline(
        0,
        linewidth=1,
        linestyle=":",
        color="#777777",
    )

    axis.text(
        0.97,
        0.95,
        (
            f"Spearman ρ = {rho:.3f}\n"
            "Leave-one-out: −0.810 to −0.405\n"
            "n = 9; descriptive"
        ),
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )

    axis.set_xlabel(
        "2022 → 2024 personal phase-pattern transfer\n"
        "(weighted RHR + HRV correlation)"
    )

    axis.set_ylabel(
        "Remaining log-loss penalty\n"
        "(updated baseline − population)"
    )

    axis.set_title(
        "Poor phase-pattern transfer tracks residual error",
        loc="left",
        fontsize=12,
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


def main():
    """Create the four-panel figure."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8.5),
        constrained_layout=True,
    )

    draw_design_panel(
        axes[0, 0]
    )

    draw_log_loss_panel(
        axes[0, 1]
    )

    draw_baseline_panel(
        axes[1, 0]
    )

    draw_transfer_panel(
        axes[1, 1]
    )

    for axis, label in zip(
        axes.flat,
        ["A", "B", "C", "D"],
    ):
        panel_label(
            axis,
            label,
        )

    fig.suptitle(
        "Why does personalization fail to transfer across years?",
        fontsize=16,
        fontweight="bold",
    )

    png_path = (
        OUTPUT_DIR
        / "figure2_longitudinal_mechanisms.png"
    )

    pdf_path = (
        OUTPUT_DIR
        / "figure2_longitudinal_mechanisms.pdf"
    )

    svg_path = (
        OUTPUT_DIR
        / "figure2_longitudinal_mechanisms.svg"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    fig.savefig(
        svg_path,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print("Saved:")
    print(png_path)
    print(pdf_path)
    print(svg_path)


if __name__ == "__main__":
    main()
