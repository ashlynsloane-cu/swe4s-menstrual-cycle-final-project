"""Audit existing Aim 2 fold CSVs without reading restricted raw records.

The target must be absent from population_train but present in personal
calibration and future test. Same target across calibration/test is
intentional; overlap of the *same participant-day* is not.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ("source_id", "study_interval", "day_in_study")


def inspect_fold(population: pd.DataFrame, calibration: pd.DataFrame,
                 test: pd.DataFrame, analysis="primary") -> dict:
    """Enforce population exclusion, day uniqueness, and temporal holdout."""
    for label, frame in (
        ("population_train", population),
        ("personal_calibration", calibration),
        ("future_test", test),
    ):
        missing = set(KEYS) - set(frame.columns)
        if missing:
            raise ValueError(f"{label}: missing {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"{label}: empty fold")
        if frame.duplicated(list(KEYS)).any():
            raise ValueError(f"{label}: duplicate participant-day")

    targets = set(calibration["source_id"].dropna().unique())
    if len(targets) != 1:
        raise ValueError("Calibration must contain exactly one target ID")
    target = next(iter(targets))
    if target in set(population["source_id"].dropna().unique()):
        raise ValueError("Target participant appears in population training")
    if set(test["source_id"].dropna().unique()) != {target}:
        raise ValueError(
            "Future test must contain only the target participant"
        )

    cal_keys = set(map(tuple, calibration[list(KEYS)].itertuples(
        index=False, name=None
    )))
    test_keys = set(map(tuple, test[list(KEYS)].itertuples(
        index=False, name=None
    )))
    if cal_keys & test_keys:
        raise ValueError("Calibration and future test share participant-days")

    cal_interval = set(calibration["study_interval"].unique())
    test_interval = set(test["study_interval"].unique())
    if analysis == "primary":
        if cal_interval != test_interval or len(cal_interval) != 1:
            raise ValueError("Primary fold must hold out within one interval")
        if calibration["day_in_study"].max() >= test["day_in_study"].min():
            raise ValueError(
                "Primary test precedes/overlaps calibration"
            )
    elif analysis == "longitudinal":
        if not max(cal_interval) < min(test_interval):
            raise ValueError(
                "Longitudinal test interval must follow calibration"
            )
    else:
        raise ValueError("Unknown analysis mode")

    return {
        "analysis": analysis,
        "target_id": int(target),
        "population_days": len(population),
        "calibration_days": len(calibration),
        "test_days": len(test),
        "population_excludes_target": True,
        "target_in_both_calibration_and_test_by_design": True,
        "calibration_test_day_overlap": 0,
        "chronology_verified": True,
    }


def inspect_fold_directory(directory: Path, analysis: str):
    inputs = {
        role: pd.read_csv(directory / f"{name}.csv")
        for role, name in (
            ("population", "population_train"),
            ("calibration", "personal_calibration"),
            ("test", "future_test"),
        )
    }
    result = inspect_fold(**inputs, analysis=analysis)
    expected = re.fullmatch(r"id(\d+)(?:_(\d+))?", directory.name)
    if expected and int(expected.group(1)) != result["target_id"]:
        raise ValueError("Fold folder ID disagrees with target ID")
    result.pop("target_id")  # Aggregate report avoids listing IDs.
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--analysis", choices=("primary", "longitudinal"),
                        required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fold_dirs = sorted(p for p in args.fold_root.glob("id*") if p.is_dir())
    if not fold_dirs:
        parser.error(f"No id* fold directories in {args.fold_root}")
    checked, failures = [], []
    for folder in fold_dirs:
        try:
            checked.append(inspect_fold_directory(folder, args.analysis))
        except (ValueError, KeyError, FileNotFoundError) as exc:
            failures.append({"fold": folder.name, "error": str(exc)})
    report = {
        "analysis": args.analysis,
        "folders_inspected": len(fold_dirs),
        "folds_passed": len(checked),
        "folds_failed": len(failures),
        "total_population_days_checked": sum(
            row["population_days"] for row in checked
        ),
        "total_calibration_days_checked": sum(
            row["calibration_days"] for row in checked
        ),
        "total_test_days_checked": sum(row["test_days"] for row in checked),
        "failures": failures,
        "note": (
            "Population excludes target IDs; earlier target calibration "
            "and future target testing are intentionally paired."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
