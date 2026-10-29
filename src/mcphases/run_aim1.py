"""Run Aim 1 cycle-signature analysis."""

from pathlib import Path

import pandas as pd

from mcphases.features import build_all_cycle_signatures

DAILY_PATH = Path(
    "results/preprocessed/mcphases_daily_master.csv"
)

CYCLE_QC_PATH = Path(
    "results/preprocessed/mcphases_cycle_qc.csv"
)

OUTPUT_PATH = Path(
    "results/aim1/cycle_signatures.csv"
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
):
    """Run Aim 1 feature extraction and save cycle signatures."""
    eligible_data = load_aim1_data(
        daily_path=daily_path,
        cycle_qc_path=cycle_qc_path,
    )
    signatures = build_all_cycle_signatures(eligible_data)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    signatures.to_csv(
        output_path,
        index=False,
    )

    print(f"Saved {len(signatures)} cycle signatures to {OUTPUT_PATH}")


if __name__ == "__main__":
    run_aim1()
