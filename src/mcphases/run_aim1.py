"""Run Aim 1 cycle-signature analysis."""

from pathlib import Path

import pandas as pd

from mcphases.features import build_all_cycle_signatures
from mcphases.similarity import (
    calculate_pairwise_distances,
    permutation_test,
    select_repeated_cycle_cohort,
    signal_specific_reproducibility,
)


DAILY_PATH = Path(
    "results/preprocessed/mcphases_daily_master.csv"
)

CYCLE_QC_PATH = Path(
    "results/preprocessed/mcphases_cycle_qc.csv"
)

OUTPUT_PATH = Path(
    "results/aim1/cycle_signatures.csv"
)

PAIRWISE_OUTPUT_PATH = Path(
    "results/aim1/pairwise_distances.csv"
)

SIGNAL_OUTPUT_PATH = Path(
    "results/aim1/signal_specific_reproducibility.csv"
)

PERMUTATION_OUTPUT_PATH = Path(
    "results/aim1/permutation_null_distribution.csv"
)


def load_aim1_data(
    daily_path=DAILY_PATH,
    cycle_qc_path=CYCLE_QC_PATH,
):
    """Load daily data and retain Aim 1 eligible cycles."""
    daily = pd.read_csv(daily_path)
    cycle_qc = pd.read_csv(cycle_qc_path)

    eligible_cycles = cycle_qc[
        cycle_qc["aim1_primary_eligible"]
    ][
        ["id", "study_interval", "cycle_number"]
    ]

    eligible_data = daily.merge(
        eligible_cycles,
        on=["id", "study_interval", "cycle_number"],
        how="inner",
    )

    return eligible_data


def run_aim1(
    daily_path=DAILY_PATH,
    cycle_qc_path=CYCLE_QC_PATH,
    output_path=OUTPUT_PATH,
    pairwise_output_path=PAIRWISE_OUTPUT_PATH,
    signal_output_path=SIGNAL_OUTPUT_PATH,
    permutation_output_path=PERMUTATION_OUTPUT_PATH,
    n_permutations=10000,
):
    """Run the full Aim 1 cycle-signature analysis."""
    eligible_data = load_aim1_data(
        daily_path=daily_path,
        cycle_qc_path=cycle_qc_path,
    )

    signatures = build_all_cycle_signatures(eligible_data)

    repeated_signatures = select_repeated_cycle_cohort(
        signatures
    )

    pairwise_distances = calculate_pairwise_distances(
        repeated_signatures
    )

    observed_advantage, p_value, null_distribution = permutation_test(
        repeated_signatures,
        pairwise_distances,
        n_permutations=n_permutations,
    )

    signal_results = signal_specific_reproducibility(
        repeated_signatures,
        n_permutations=n_permutations,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    signatures.to_csv(
        output_path,
        index=False,
    )

    pairwise_distances.to_csv(
        pairwise_output_path,
        index=False,
    )

    signal_results.to_csv(
        signal_output_path,
        index=False,
    )

    permutation_results = pd.DataFrame(
        {
            "permuted_advantage": null_distribution,
        }
    )

    permutation_results.to_csv(
        permutation_output_path,
        index=False,
    )

    print(
        f"Saved {len(signatures)} cycle signatures to "
        f"{output_path}"
    )

    print(
        f"Repeated-cycle cohort: "
        f"{repeated_signatures['id'].nunique()} participants, "
        f"{len(repeated_signatures)} cycles"
    )

    print(
        f"Observed similarity advantage: "
        f"{observed_advantage:.4f}"
    )

    print(
        f"Permutation p-value: {p_value:.4f}"
    )


if __name__ == "__main__":
    run_aim1()
