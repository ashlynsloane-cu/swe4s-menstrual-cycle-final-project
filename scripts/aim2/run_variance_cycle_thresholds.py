"""Map personalization cycle requirements across variance scenarios."""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_cycle_depth_simulation as sim


DEFAULT_OUTPUT = Path(
    "results/aim2/variance_cycle_thresholds"
)

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

DEPTHS = [
    1,
    2,
    3,
    4,
    6,
    8,
    10,
    12,
    16,
]


def parse_args():
    """Parse command-line options."""

    parser = argparse.ArgumentParser(
        description=(
            "Estimate calibration-cycle requirements "
            "across stable-signal and cycle-noise levels."
        )
    )

    parser.add_argument(
        "--replicates",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--participants",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=6118,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    return parser.parse_args()


def summarize_scenario(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize log-loss differences by cycle depth."""

    rows = []

    for depth in DEPTHS:
        subset = results[
            results["depth"] == depth
        ]

        values = subset[
            "delta_log_loss"
        ].to_numpy()

        rows.append(
            {
                "depth": depth,
                "mean_difference": (
                    values.mean()
                ),
                "median_difference": (
                    np.median(values)
                ),
                "ci_2.5": np.quantile(
                    values,
                    0.025,
                ),
                "ci_97.5": np.quantile(
                    values,
                    0.975,
                ),
                "probability_personalized_better": (
                    np.mean(values < 0)
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def find_thresholds(
    summary: pd.DataFrame,
) -> tuple:
    """Find expected and reliable log-loss crossover."""

    expected = summary[
        summary["mean_difference"] < 0
    ]

    reliable = summary[
        summary["ci_97.5"] < 0
    ]

    expected_depth = (
        int(expected.iloc[0]["depth"])
        if not expected.empty
        else np.nan
    )

    reliable_depth = (
        int(reliable.iloc[0]["depth"])
        if not reliable.empty
        else np.nan
    )

    return (
        expected_depth,
        reliable_depth,
    )


def make_heatmap(
    thresholds: pd.DataFrame,
    output_dir: Path,
):
    """Plot reliable cycle threshold across scenarios."""

    matrix = thresholds.pivot(
        index="stable_signal_scale",
        columns="cycle_variance_scale",
        values="reliable_log_loss_crossover",
    )

    matrix = matrix.reindex(
        index=STABLE_SCALES,
        columns=CYCLE_SCALES,
    )

    display = matrix.to_numpy(
        dtype=float
    )

    # Values beyond tested range are shown one unit
    # above the maximum purely for plotting.
    plot_values = np.where(
        np.isnan(display),
        max(DEPTHS) + 1,
        display,
    )

    fig, ax = plt.subplots(
        figsize=(7.0, 5.2)
    )

    image = ax.imshow(
        plot_values,
        aspect="auto",
    )

    ax.set_xticks(
        range(len(CYCLE_SCALES))
    )

    ax.set_xticklabels(
        [
            f"{value:g}×"
            for value in CYCLE_SCALES
        ]
    )

    ax.set_yticks(
        range(len(STABLE_SCALES))
    )

    ax.set_yticklabels(
        [
            f"{value:g}×"
            for value in STABLE_SCALES
        ]
    )

    ax.set_xlabel(
        "Cycle-to-cycle variance"
    )

    ax.set_ylabel(
        "Stable person-specific signal"
    )

    ax.set_title(
        "Calibration cycles required for "
        "reliable personalization"
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

            if pd.isna(value):
                label = ">16"
            else:
                label = str(
                    int(value)
                )

            ax.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
            )

    fig.colorbar(
        image,
        ax=ax,
        label="Prior calibration cycles",
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "reliable_cycle_threshold_heatmap.png",
        dpi=300,
    )

    plt.close(fig)


def main():
    """Run variance-by-cycle-depth sensitivity analysis."""

    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        population_means,
        stable_var,
        transient_var,
        day_var,
        repeatability,
    ) = sim.load_parameters()

    (
        templates,
        template_ids,
        priors,
    ) = sim.load_cycle_templates()

    scenarios = [
        (
            stable_scale,
            cycle_scale,
        )
        for stable_scale in STABLE_SCALES
        for cycle_scale in CYCLE_SCALES
    ]

    seeds = np.random.SeedSequence(
        args.seed
    ).spawn(
        len(scenarios)
    )

    all_results = []
    all_summaries = []
    threshold_rows = []

    for scenario_index, (
        stable_scale,
        cycle_scale,
    ) in enumerate(scenarios):

        print("\n" + "=" * 82)
        print(
            "STABLE SIGNAL:",
            f"{stable_scale:g}x",
            "| CYCLE VARIANCE:",
            f"{cycle_scale:g}x",
        )
        print("=" * 82)

        scenario_stable = (
            stable_var
            * stable_scale
        )

        scenario_transient = (
            transient_var
            * cycle_scale
        )

        rng = np.random.default_rng(
            seeds[scenario_index]
        )

        rows = []

        total = (
            args.replicates
            * len(DEPTHS)
        )

        completed = 0

        for replicate in range(
            args.replicates
        ):
            for depth in DEPTHS:
                result = sim.simulate_one_depth(
                    rng=rng,
                    depth=depth,
                    n_participants=(
                        args.participants
                    ),
                    templates=templates,
                    population_means=(
                        population_means
                    ),
                    stable_var=(
                        scenario_stable
                    ),
                    transient_var=(
                        scenario_transient
                    ),
                    day_var=day_var,
                    priors=priors,
                )

                result[
                    "replicate"
                ] = replicate

                result[
                    "stable_signal_scale"
                ] = stable_scale

                result[
                    "cycle_variance_scale"
                ] = cycle_scale

                rows.append(result)

                completed += 1

            if (
                (replicate + 1) % 100 == 0
                or replicate + 1
                == args.replicates
            ):
                print(
                    "Completed",
                    completed,
                    "/",
                    total,
                )

        results = pd.DataFrame(
            rows
        )

        summary = summarize_scenario(
            results
        )

        summary[
            "stable_signal_scale"
        ] = stable_scale

        summary[
            "cycle_variance_scale"
        ] = cycle_scale

        (
            expected_depth,
            reliable_depth,
        ) = find_thresholds(
            summary
        )

        threshold_rows.append(
            {
                "stable_signal_scale": (
                    stable_scale
                ),
                "cycle_variance_scale": (
                    cycle_scale
                ),
                "expected_log_loss_crossover": (
                    expected_depth
                ),
                "reliable_log_loss_crossover": (
                    reliable_depth
                ),
            }
        )

        all_results.append(
            results
        )

        all_summaries.append(
            summary
        )

    all_results = pd.concat(
        all_results,
        ignore_index=True,
    )

    all_summaries = pd.concat(
        all_summaries,
        ignore_index=True,
    )

    thresholds = pd.DataFrame(
        threshold_rows
    )

    all_results.to_csv(
        args.output_dir
        / "replicate_metrics.csv",
        index=False,
    )

    all_summaries.to_csv(
        args.output_dir
        / "log_loss_summary.csv",
        index=False,
    )

    thresholds.to_csv(
        args.output_dir
        / "cycle_thresholds.csv",
        index=False,
    )

    make_heatmap(
        thresholds,
        args.output_dir,
    )

    print("\n" + "=" * 82)
    print("RELIABLE PERSONALIZATION THRESHOLDS")
    print("=" * 82)

    display = thresholds.copy()

    display[
        "reliable_log_loss_crossover"
    ] = display[
        "reliable_log_loss_crossover"
    ].apply(
        lambda value: (
            ">16"
            if pd.isna(value)
            else str(int(value))
        )
    )

    print(
        display.to_string(
            index=False
        )
    )

    print(
        "\nObserved mcPHASES scenario = "
        "stable_signal_scale 1.0, "
        "cycle_variance_scale 1.0."
    )

    print(
        "\nReliable threshold means the "
        "97.5th percentile of personalized-minus-"
        "population log loss is below zero."
    )

    print(
        "\nOutputs:",
        args.output_dir,
    )


if __name__ == "__main__":
    main()
