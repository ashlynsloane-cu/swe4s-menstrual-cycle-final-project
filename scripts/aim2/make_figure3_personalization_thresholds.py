"""Create Aim 2 Figure 3: personalization calibration requirements.

Panel A uses fixed, version-controlled schematic assets.
Panels B and C are generated directly from analysis outputs.

Expected assets:
    assets/aim2/icon_population_phase.png
    assets/aim2/icon_person_signature.png
    assets/aim2/icon_cycle_variation.png
    assets/aim2/icon_day_noise.png
"""

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyBboxPatch, Rectangle


CYCLE_DEPTH_DIR = Path(
    "results/aim2/cycle_depth_simulation"
)

PRECISION_DIR = Path(
    "results/aim2/nominal_precision_check"
)

THRESHOLD_DIR = Path(
    "results/aim2/variance_cycle_thresholds"
)

ASSET_DIR = Path(
    "assets/aim2"
)

OUTPUT_DIR = Path(
    "results/figures/aim2"
)

ASSETS = {
    "population": (
        ASSET_DIR
        / "icon_population_phase.png"
    ),
    "personal": (
        ASSET_DIR
        / "icon_person_signature.png"
    ),
    "cycle": (
        ASSET_DIR
        / "icon_cycle_variation.png"
    ),
    "noise": (
        ASSET_DIR
        / "icon_day_noise.png"
    ),
}

STABLE_SCALES = [
    0.5,
    1.0,
    2.0,
]

CYCLE_SCALES = [
    0.5,
    1.0,
    2.0,
]


def validate_inputs():
    """Check that required result files and schematic assets exist."""

    required_results = [
        (
            CYCLE_DEPTH_DIR
            / "simulation_summary.csv"
        ),
        (
            PRECISION_DIR
            / "log_loss_summary.csv"
        ),
        (
            THRESHOLD_DIR
            / "cycle_thresholds.csv"
        ),
    ]

    missing = [
        path
        for path in (
            required_results
            + list(
                ASSETS.values()
            )
        )
        if not path.exists()
    ]

    if missing:
        formatted = "\n".join(
            f"  - {path}"
            for path in missing
        )

        raise FileNotFoundError(
            "Figure 3 is missing required inputs:\n"
            f"{formatted}"
        )


def panel_label(
    axis,
    label,
):
    """Place a panel letter."""

    axis.text(
        -0.11,
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
    """Place a transparent schematic asset at fixed axes coordinates."""

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


def draw_concept_panel(
    axis,
):
    """Draw schematic concept using fixed image assets."""

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

    axis.set_title(
        "What determines whether personalization can be learned?",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )

    centers = [
        0.13,
        0.38,
        0.63,
        0.87,
    ]

    labels = [
        "Population\nphase pattern",
        "Stable personal\nsignature",
        "Cycle-to-cycle\nvariation",
        "Day-level\nnoise",
    ]

    label_colors = [
        "#222222",
        "#0072B2",
        "#D55E00",
        "#444444",
    ]

    for center, label, color in zip(
        centers,
        labels,
        label_colors,
    ):
        axis.text(
            center,
            0.84,
            label,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=color,
        )

    # Fixed image assets. Their sizes can be tuned with zoom
    # without editing the image files themselves.
    add_icon(
        axis,
        ASSETS["population"],
        (
            0.13,
            0.56,
        ),
        zoom=0.28,
    )

    add_icon(
        axis,
        ASSETS["personal"],
        (
            0.38,
            0.56,
        ),
        zoom=0.26,
    )

    add_icon(
        axis,
        ASSETS["cycle"],
        (
            0.63,
            0.56,
        ),
        zoom=0.27,
    )

    add_icon(
        axis,
        ASSETS["noise"],
        (
            0.87,
            0.56,
        ),
        zoom=0.27,
    )

    for x in [
        0.255,
        0.505,
        0.755,
    ]:
        axis.text(
            x,
            0.56,
            "+",
            fontsize=22,
            ha="center",
            va="center",
            fontweight="bold",
        )

    axis.annotate(
        "",
        xy=(
            0.50,
            0.27,
        ),
        xytext=(
            0.50,
            0.37,
        ),
        xycoords="axes fraction",
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.5,
        },
    )

    callout = FancyBboxPatch(
        (
            0.27,
            0.12,
        ),
        0.46,
        0.10,
        transform=axis.transAxes,
        boxstyle="round,pad=0.01",
        facecolor="#E5F1F8",
        edgecolor="none",
    )

    axis.add_patch(
        callout
    )

    axis.text(
        0.50,
        0.17,
        (
            "More calibration cycles estimate "
            "the stable personal signature"
        ),
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )

    axis.text(
        0.50,
        0.045,
        (
            "More history helps when repeatable personal signal "
            "is large relative to cycle variability."
        ),
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=8.8,
    )


def load_observed_curve():
    """Load observed simulation and high-precision 1-2 cycle estimates."""

    summary = pd.read_csv(
        CYCLE_DEPTH_DIR
        / "simulation_summary.csv"
    )

    data = summary[
        summary["metric"]
        == "log_loss"
    ][
        [
            "depth",
            "mean_difference",
            "ci_2.5",
            "ci_97.5",
        ]
    ].copy()

    precision = pd.read_csv(
        PRECISION_DIR
        / "log_loss_summary.csv"
    )

    # Supersede the lower-precision 1- and 2-cycle
    # estimates with the 5,000-replicate precision run.
    for _, row in precision.iterrows():
        depth = int(
            row["depth"]
        )

        mask = (
            data["depth"]
            == depth
        )

        if mask.sum() != 1:
            raise RuntimeError(
                "Could not uniquely identify "
                f"cycle depth {depth}."
            )

        for column in [
            "mean_difference",
            "ci_2.5",
            "ci_97.5",
        ]:
            data.loc[
                mask,
                column,
            ] = row[
                column
            ]

    return data.sort_values(
        "depth"
    )


def draw_observed_panel(
    axis,
):
    """Plot the observed stationary mcPHASES simulation."""

    data = load_observed_curve()

    x = data[
        "depth"
    ].to_numpy()

    y = data[
        "mean_difference"
    ].to_numpy()

    low = data[
        "ci_2.5"
    ].to_numpy()

    high = data[
        "ci_97.5"
    ].to_numpy()

    axis.fill_between(
        x,
        low,
        high,
        alpha=0.75,
        linewidth=0,
        color="#C7DDEB",
    )

    axis.plot(
        x,
        y,
        marker="o",
        linewidth=2,
        color="#0072B2",
    )

    axis.axhline(
        0,
        linestyle="--",
        linewidth=1.3,
        color="#444444",
    )

    row_one = data[
        data["depth"]
        == 1
    ].iloc[0]

    row_two = data[
        data["depth"]
        == 2
    ].iloc[0]

    axis.scatter(
        2,
        row_two[
            "mean_difference"
        ],
        s=100,
        color="#D55E00",
        edgecolor="black",
        linewidth=0.8,
        zorder=5,
    )

    axis.annotate(
        "1 cycle: interval crosses 0",
        xy=(
            1,
            row_one[
                "mean_difference"
            ],
        ),
        xytext=(
            1.23,
            0.006,
        ),
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1,
        },
        fontsize=8.5,
        ha="left",
    )

    axis.annotate(
        "Reliable threshold = 2 cycles",
        xy=(
            2,
            row_two[
                "mean_difference"
            ],
        ),
        xytext=(
            3.0,
            -0.020,
        ),
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1,
        },
        fontsize=9,
        ha="left",
    )

    axis.text(
        0.02,
        0.04,
        "Below 0 = personalization predicts better",
        transform=axis.transAxes,
        fontsize=8.5,
        fontweight="bold",
        color="#0072B2",
    )

    axis.text(
        0.98,
        0.05,
        "Metric: log loss",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )

    axis.set_xticks(
        x
    )

    axis.set_xlabel(
        "Prior calibration cycles"
    )

    axis.set_ylabel(
        "Change in prediction error\n"
        "(personalized − population)"
    )

    axis.set_title(
        "Observed stationary mcPHASES scenario",
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


def load_threshold_matrix():
    """Load threshold matrix with precision-corrected observed cell."""

    thresholds = pd.read_csv(
        THRESHOLD_DIR
        / "cycle_thresholds.csv"
    )

    precision = pd.read_csv(
        PRECISION_DIR
        / "log_loss_summary.csv"
    )

    reliable = precision[
        precision[
            "ci_97.5"
        ]
        < 0
    ]

    if reliable.empty:
        raise RuntimeError(
            "Precision check found no reliable "
            "crossover by two cycles."
        )

    observed_threshold = int(
        reliable.iloc[0][
            "depth"
        ]
    )

    observed_mask = (
        np.isclose(
            thresholds[
                "stable_signal_scale"
            ],
            1.0,
        )
        & np.isclose(
            thresholds[
                "cycle_variance_scale"
            ],
            1.0,
        )
    )

    if observed_mask.sum() != 1:
        raise RuntimeError(
            "Could not uniquely identify "
            "the observed mcPHASES cell."
        )

    thresholds.loc[
        observed_mask,
        "reliable_log_loss_crossover",
    ] = observed_threshold

    matrix = thresholds.pivot(
        index="stable_signal_scale",
        columns="cycle_variance_scale",
        values="reliable_log_loss_crossover",
    )

    return matrix.reindex(
        index=STABLE_SCALES,
        columns=CYCLE_SCALES,
    )


def draw_heatmap_panel(
    axis,
):
    """Plot reliable calibration-cycle thresholds."""

    matrix = load_threshold_matrix()

    values = matrix.to_numpy(
        dtype=float
    )

    display_values = np.where(
        np.isnan(values),
        18,
        values,
    )

    image = axis.imshow(
        display_values,
        cmap="YlOrRd",
        vmin=1,
        vmax=18,
        aspect="auto",
    )

    axis.set_xticks(
        range(3)
    )

    axis.set_xticklabels(
        [
            "Lower\n(0.5×)",
            "Observed\n(1×)",
            "Higher\n(2×)",
        ]
    )

    axis.set_yticks(
        range(3)
    )

    axis.set_yticklabels(
        [
            "Weaker\n(0.5×)",
            "Observed\n(1×)",
            "Stronger\n(2×)",
        ]
    )

    axis.set_xlabel(
        "Cycle-to-cycle variation",
        labelpad=10,
    )

    axis.set_ylabel(
        "Stable personal signal"
    )

    axis.set_title(
        "Cycles required depend on biological stability",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )

    for row_index, stable_scale in enumerate(
        STABLE_SCALES
    ):
        for column_index, cycle_scale in enumerate(
            CYCLE_SCALES
        ):
            value = matrix.loc[
                stable_scale,
                cycle_scale,
            ]

            label = (
                ">16"
                if pd.isna(
                    value
                )
                else str(
                    int(
                        value
                    )
                )
            )

            axis.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=15,
                fontweight="bold",
            )

    axis.add_patch(
        Rectangle(
            (
                0.5,
                0.5,
            ),
            1,
            1,
            fill=False,
            edgecolor="#0072B2",
            linewidth=3,
        )
    )

    colorbar = plt.colorbar(
        image,
        ax=axis,
        fraction=0.045,
        pad=0.04,
    )

    colorbar.set_label(
        "Cycles required"
    )

    axis.text(
        0.50,
        -0.27,
        "Blue outline = observed mcPHASES",
        transform=axis.transAxes,
        ha="center",
        fontsize=8.5,
        fontweight="bold",
        color="#0072B2",
    )

    axis.text(
        0.50,
        -0.37,
        (
            "Cell value = calibration cycles required "
            "for reliable improvement"
        ),
        transform=axis.transAxes,
        ha="center",
        fontsize=8,
        style="italic",
    )


def main():
    """Create the final three-panel Figure 3."""

    validate_inputs()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig = plt.figure(
        figsize=(
            14,
            7.4,
        )
    )

    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[
            0.95,
            1.05,
        ],
        width_ratios=[
            1,
            1.05,
        ],
        hspace=0.40,
        wspace=0.30,
    )

    axis_a = fig.add_subplot(
        grid[0, :]
    )

    axis_b = fig.add_subplot(
        grid[1, 0]
    )

    axis_c = fig.add_subplot(
        grid[1, 1]
    )

    draw_concept_panel(
        axis_a
    )

    draw_observed_panel(
        axis_b
    )

    draw_heatmap_panel(
        axis_c
    )

    for axis, label in zip(
        [
            axis_a,
            axis_b,
            axis_c,
        ],
        [
            "A",
            "B",
            "C",
        ],
    ):
        panel_label(
            axis,
            label,
        )

    fig.suptitle(
        "How much history is needed for personalization?",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    output_paths = [
        (
            OUTPUT_DIR
            / "figure3_personalization_thresholds.png"
        ),
        (
            OUTPUT_DIR
            / "figure3_personalization_thresholds.pdf"
        ),
        (
            OUTPUT_DIR
            / "figure3_personalization_thresholds.svg"
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


if __name__ == "__main__":
    main()
