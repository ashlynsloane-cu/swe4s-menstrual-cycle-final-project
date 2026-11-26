"""Run the Aim 1 regularized Mahalanobis sensitivity analysis."""

import argparse
from pathlib import Path

import pandas as pd

from mcphases.features import (
    build_all_cycle_signatures,
)
from mcphases.mahalanobis import (
    calculate_mahalanobis_pairwise_distances,
    mahalanobis_signal_specific_reproducibility,
)
from mcphases.run_aim1 import load_aim1_data
from mcphases.similarity import (
    permutation_test,
    select_repeated_cycle_cohort,
)


DEFAULT_OUTPUT_DIR = Path(
    "results/aim1_mahalanobis"
)


def run_aim1_mahalanobis(
    output_dir=DEFAULT_OUTPUT_DIR,
    n_permutations=10000,
    stratify_by_study_interval=False,
):
    """Run covariance-aware Aim 1 sensitivity analysis."""
    eligible_data = load_aim1_data()

    signatures = build_all_cycle_signatures(
        eligible_data
    )

    repeated_signatures = (
        select_repeated_cycle_cohort(
            signatures
        )
    )

    pairwise_distances = (
        calculate_mahalanobis_pairwise_distances(
            repeated_signatures
        )
    )

    (
        observed_advantage,
        p_value,
        null_distribution,
    ) = permutation_test(
        repeated_signatures,
        pairwise_distances,
        n_permutations=n_permutations,
        random_state=42,
        stratify_by_study_interval=(
            stratify_by_study_interval
        ),
    )

    signal_results = (
        mahalanobis_signal_specific_reproducibility(
            repeated_signatures,
            n_permutations=n_permutations,
            random_state=42,
            stratify_by_study_interval=(
                stratify_by_study_interval
            ),
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    signatures.to_csv(
        output_dir / "cycle_signatures.csv",
        index=False,
    )

    pairwise_distances.to_csv(
        output_dir / "pairwise_distances.csv",
        index=False,
    )

    signal_results.to_csv(
        output_dir
        / "signal_specific_reproducibility.csv",
        index=False,
    )

    permutation_results = pd.DataFrame(
        {
            "permuted_advantage": (
                null_distribution
            ),
            "distance_metric": (
                "regularized_mahalanobis"
            ),
        }
    )

    permutation_results.to_csv(
        output_dir
        / "permutation_null_distribution.csv",
        index=False,
    )

    print(
        "Distance metric: "
        "regularized Mahalanobis"
    )

    print(
        f"Observed similarity advantage: "
        f"{observed_advantage:.4f}"
    )

    print(
        f"Permutation p-value: "
        f"{p_value:.4f}"
    )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--permutations",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--stratify-by-study-interval",
        action="store_true",
    )

    return parser.parse_args()


def main():
    """Run sensitivity analysis from command line."""
    args = parse_args()

    run_aim1_mahalanobis(
        output_dir=args.output_dir,
        n_permutations=args.permutations,
        stratify_by_study_interval=(
            args.stratify_by_study_interval
        ),
    )


if __name__ == "__main__":
    main()
