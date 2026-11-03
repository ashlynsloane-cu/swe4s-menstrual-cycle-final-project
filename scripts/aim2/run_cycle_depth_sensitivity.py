"""Sensitivity analysis for Aim 2 cycle-depth simulation."""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_cycle_depth_simulation as sim


DEFAULT_OUTPUT = Path(
    "results/aim2/cycle_depth_sensitivity"
)

SCALES = [
    0.0,
    0.5,
    1.0,
    2.0,
]


def parse_args():
    """Parse command-line options."""

    parser = argparse.ArgumentParser(
        description=(
            "Vary stable person-specific phase "
            "variance in cycle-depth simulation."
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


def threshold_table(
    summary,
):
    """Find prespecified log-loss thresholds."""

    rows = []

    for scale in SCALES:
        subset = summary[
            summary["stable_signal_scale"]
            == scale
        ]

        expected, reliable = (
            sim.find_log_loss_thresholds(
                subset
            )
        )

        rows.append(
            {
                "stable_signal_scale": scale,
                "expected_log_loss_crossover": (
                    expected
                ),
                "reliable_log_loss_crossover": (
                    reliable
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def make_figure(
    summary,
    output_dir,
):
    """Plot log-loss difference by depth and stability."""

    data = summary[
        summary["metric"]
        == "log_loss"
    ].copy()

    fig, ax = plt.subplots(
        figsize=(8.0, 5.0)
    )

    for scale in SCALES:
        subset = data[
            data["stable_signal_scale"]
            == scale
        ].sort_values(
            "depth"
        )

        label = (
            f"{scale:g}× stable signal"
        )

        ax.plot(
            subset["depth"],
            subset["mean_difference"],
            marker="o",
            label=label,
        )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_xticks(
        sim.DEPTHS
    )

    ax.set_xlabel(
        "Prior calibration cycles"
    )

    ax.set_ylabel(
        "Personalized − population log loss"
    )

    ax.set_title(
        "Sensitivity to person-specific "
        "phase stability"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "log_loss_sensitivity.png",
        dpi=300,
    )

    plt.close(
        fig
    )


def main():
    """Run sensitivity analysis."""

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

    seed_sequence = np.random.SeedSequence(
        args.seed
    )

    child_seeds = seed_sequence.spawn(
        len(SCALES)
    )

    all_results = []
    all_summaries = []

    for scale_index, scale in enumerate(
        SCALES
    ):
        print("\n" + "=" * 78)
        print(
            "STABLE SIGNAL SCALE:",
            scale,
        )
        print("=" * 78)

        scaled_stable_var = (
            stable_var
            * scale
        )

        rng = np.random.default_rng(
            child_seeds[
                scale_index
            ]
        )

        rows = []

        total = (
            args.replicates
            * len(sim.DEPTHS)
        )

        completed = 0

        for replicate in range(
            args.replicates
        ):
            for depth in sim.DEPTHS:
                result = (
                    sim.simulate_one_depth(
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
                            scaled_stable_var
                        ),
                        transient_var=(
                            transient_var
                        ),
                        day_var=day_var,
                        priors=priors,
                    )
                )

                result[
                    "replicate"
                ] = replicate

                result[
                    "stable_signal_scale"
                ] = scale

                rows.append(
                    result
                )

                completed += 1

            if (
                (replicate + 1) % 100
                == 0
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

        summary = sim.summarize(
            results
        )

        summary[
            "stable_signal_scale"
        ] = scale

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

    thresholds = threshold_table(
        all_summaries
    )

    all_results.to_csv(
        args.output_dir
        / "sensitivity_replicate_metrics.csv",
        index=False,
    )

    all_summaries.to_csv(
        args.output_dir
        / "sensitivity_summary.csv",
        index=False,
    )

    thresholds.to_csv(
        args.output_dir
        / "log_loss_thresholds.csv",
        index=False,
    )

    make_figure(
        all_summaries,
        args.output_dir,
    )

    log_loss = all_summaries[
        all_summaries["metric"]
        == "log_loss"
    ][
        [
            "stable_signal_scale",
            "depth",
            "mean_difference",
            "ci_2.5",
            "ci_97.5",
            "probability_personalized_better",
        ]
    ]

    print("\n" + "=" * 78)
    print("LOG-LOSS SENSITIVITY")
    print("=" * 78)

    print(
        log_loss.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    print("\n" + "=" * 78)
    print("CROSSOVER DEPTHS")
    print("=" * 78)

    print(
        thresholds.to_string(
            index=False,
        )
    )

    print(
        "\nOutputs:",
        args.output_dir,
    )


if __name__ == "__main__":
    main()
