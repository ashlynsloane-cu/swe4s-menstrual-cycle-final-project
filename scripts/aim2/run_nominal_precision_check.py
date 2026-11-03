"""High-precision check of the observed mcPHASES simulation scenario."""

from pathlib import Path

import numpy as np
import pandas as pd

import run_cycle_depth_simulation as sim


OUTPUT_DIR = Path(
    "results/aim2/nominal_precision_check"
)

DEPTHS = [
    1,
    2,
]

N_REPLICATES = 5000
N_PARTICIPANTS = 40
SEED = 6118


def summarize(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize personalized-minus-population log loss."""

    rows = []

    for depth in DEPTHS:
        values = results.loc[
            results["depth"] == depth,
            "delta_log_loss",
        ].to_numpy()

        rows.append(
            {
                "depth": depth,
                "n_replicates": len(values),
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


def main():
    """Run high-precision nominal sensitivity check."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        population_means,
        stable_var,
        transient_var,
        day_var,
        _,
    ) = sim.load_parameters()

    (
        templates,
        _,
        priors,
    ) = sim.load_cycle_templates()

    rng = np.random.default_rng(
        SEED
    )

    rows = []

    total = (
        N_REPLICATES
        * len(DEPTHS)
    )

    completed = 0

    for replicate in range(
        N_REPLICATES
    ):
        for depth in DEPTHS:
            result = sim.simulate_one_depth(
                rng=rng,
                depth=depth,
                n_participants=N_PARTICIPANTS,
                templates=templates,
                population_means=(
                    population_means
                ),
                stable_var=stable_var,
                transient_var=(
                    transient_var
                ),
                day_var=day_var,
                priors=priors,
            )

            result[
                "replicate"
            ] = replicate

            rows.append(
                result
            )

            completed += 1

        if (
            (replicate + 1) % 500 == 0
            or replicate + 1
            == N_REPLICATES
        ):
            print(
                "Completed",
                completed,
                "/",
                total,
                "simulation cells",
            )

    results = pd.DataFrame(
        rows
    )

    summary = summarize(
        results
    )

    results.to_csv(
        OUTPUT_DIR
        / "replicate_metrics.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "log_loss_summary.csv",
        index=False,
    )

    print("\n" + "=" * 82)
    print("OBSERVED mcPHASES PRECISION CHECK")
    print("=" * 82)

    print(
        "\nScenario:"
        "\n  stable person-specific signal = 1.0x observed"
        "\n  cycle-to-cycle variance = 1.0x observed"
        "\n  participants per replicate = 40"
        "\n  replicates = 5000"
    )

    print(
        "\nPERSONALIZED MINUS POPULATION LOG LOSS"
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.5f}"
            ),
        )
    )

    reliable = summary[
        summary["ci_97.5"] < 0
    ]

    if reliable.empty:
        print(
            "\nReliable crossover: "
            "not reached by 2 cycles."
        )
    else:
        print(
            "\nReliable crossover:",
            int(
                reliable.iloc[0][
                    "depth"
                ]
            ),
            "cycle(s)",
        )

    print(
        "\nOutputs:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
