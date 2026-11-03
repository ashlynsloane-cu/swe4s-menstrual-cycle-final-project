"""Build deterministic Aim 2 split manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mcphases.aim2_split import build_primary_manifest


EXPECTED_TARGETS = 23
EXPECTED_FUTURE_CYCLES = 54
EXPECTED_TEST_DAYS = 1349


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Build the primary Aim 2 split manifest."
    )

    parser.add_argument(
        "--daily",
        type=Path,
        default=Path(
            "results/preprocessed/"
            "mcphases_daily_master.csv"
        ),
    )

    parser.add_argument(
        "--cycle-qc",
        type=Path,
        default=Path(
            "results/preprocessed/"
            "mcphases_cycle_qc.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/aim2/splits"
        ),
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Require the current mcPHASES baseline counts "
            "to match the frozen Aim 2 audit."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Build and save the Aim 2 split manifest."""

    args = parse_args()

    daily = pd.read_csv(args.daily)
    cycle_qc = pd.read_csv(args.cycle_qc)

    manifest, summary = build_primary_manifest(
        daily,
        cycle_qc,
    )

    eligible = summary[
        summary["primary_matched_eligible"]
    ].copy()

    n_targets = len(eligible)

    n_future_cycles = int(
        eligible["n_future_cycles"].sum()
    )

    n_test_days = int(
        eligible["test_days"].sum()
    )

    if args.strict:
        observed = (
            n_targets,
            n_future_cycles,
            n_test_days,
        )

        expected = (
            EXPECTED_TARGETS,
            EXPECTED_FUTURE_CYCLES,
            EXPECTED_TEST_DAYS,
        )

        if observed != expected:
            raise RuntimeError(
                "Aim 2 split no longer matches the frozen "
                f"baseline. Expected {expected}, observed "
                f"{observed}."
            )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        args.output_dir
        / "aim2_primary_split_manifest.csv"
    )

    summary_path = (
        args.output_dir
        / "aim2_target_split_summary.csv"
    )

    manifest.to_csv(
        manifest_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print("=" * 72)
    print("AIM 2 SPLIT MANIFEST COMPLETE")
    print("=" * 72)

    print(
        f"Primary matched participants: {n_targets}"
    )

    print(
        f"Held-out future cycles:       {n_future_cycles}"
    )

    print(
        f"Held-out future days:         {n_test_days}"
    )

    print(
        "\nManifest:",
        manifest_path,
    )

    print(
        "Target summary:",
        summary_path,
    )

    print("\nNo target participant appears in population training.")
    print("Cycles 1 and 2 are calibration only.")
    print("Cycles 3+ are held out for testing.")


if __name__ == "__main__":
    main()
