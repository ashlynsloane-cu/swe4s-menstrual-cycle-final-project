#!/usr/bin/env python3
"""
Preprocess the mcPHASES dataset for the "Beyond the Average Cycle" project.

This script builds a participant-day analysis table that can be shared by:
    Aim 1: within-person vs. between-person physiological reproducibility
    Aim 2: population vs. personalized menstrual-phase prediction

Core mcPHASES files supported automatically:
    - hormones_and_selfreport.csv
    - resting_heart_rate.csv
    - computed_temperature.csv

Additional daily or high-frequency physiological signals can be added from
the command line with --signal without editing this script.

Important design choices:
    - No imputation is performed.
    - No across-participant standardization is performed here. Scaling must be
      done inside the Aim 2 cross-validation folds to avoid data leakage.
    - Menstrual cycles are defined by transitions into the dataset's
      "Menstrual" phase, separately within each study interval.
    - Boundary/incomplete cycles are retained but explicitly flagged.
    - Raw restricted mcPHASES data should NOT be committed to a public repo.

Example
-------
python preprocess_mcphases.py \
    --data-dir data/mcphases \
    --output-dir results/preprocessed

Add another signal after inspecting its columns:
python preprocess_mcphases.py \
    --data-dir data/mcphases \
    --output-dir results/preprocessed \
    --signal "hrv=hrv.csv:day_in_study:value:median"

Signal specification:
    NAME=FILE:DAY_COLUMN:VALUE_COLUMN:AGG

AGG may be one of:
    median, mean, min, max, first
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)

EXPECTED_PHASES = (
    "Menstrual",
    "Follicular",
    "Fertility",
    "Luteal",
)

CORE_HORMONE_COLUMNS = (
    "id",
    "day_in_study",
    "study_interval",
    "phase",
    "lh",
    "estrogen",
    "pdg",
)


@dataclass(frozen=True)
class SignalSpec:
    """Describe how to reduce one physiological table to daily data."""

    name: str
    filename: str
    day_column: str
    value_column: str
    aggregation: str = "median"
    positive_only: bool = False


DEFAULT_SIGNALS = (
    SignalSpec(
        name="resting_heart_rate",
        filename="resting_heart_rate.csv",
        day_column="day_in_study",
        value_column="value",
        aggregation="median",
        positive_only=True,
    ),
    SignalSpec(
        name="nightly_temperature",
        filename="computed_temperature.csv",
        day_column="sleep_end_day_in_study",
        value_column="nightly_temperature",
        aggregation="median",
    ),
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Build analysis-ready participant-day mcPHASES tables."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing the restricted mcPHASES CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/preprocessed"),
        help="Directory for processed outputs.",
    )
    parser.add_argument(
        "--signal",
        action="append",
        default=[],
        metavar="SPEC",
        help=(
            "Optional extra signal: "
            "NAME=FILE:DAY_COLUMN:VALUE_COLUMN:AGG. "
            "May be supplied multiple times."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail if a default physiological file is missing. "
            "Without --strict, missing optional signals are skipped."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser.parse_args()


def require_columns(
    dataframe: pd.DataFrame,
    required: tuple[str, ...] | list[str],
    source: str,
) -> None:
    """Raise a clear error if expected columns are missing."""

    missing = [
        column for column in required if column not in dataframe.columns
    ]
    if missing:
        raise ValueError(
            f"{source} is missing required column(s): {', '.join(missing)}"
        )


def first_nonmissing(series: pd.Series):
    """Return the first non-missing value, otherwise NaN."""

    values = series.dropna()
    if values.empty:
        return np.nan
    return values.iloc[0]


def normalize_id(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Store participant identifiers consistently as strings."""

    output = dataframe.copy()
    output["id"] = output["id"].astype("string")
    return output


def normalize_day_column(
    dataframe: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """Coerce a relative study-day column to nullable integer."""

    output = dataframe.copy()
    output[column] = pd.to_numeric(output[column], errors="coerce").round()
    output[column] = output[column].astype("Int64")
    return output


def normalize_phase(series: pd.Series) -> pd.Series:
    """Standardize known phase spelling and capitalization."""

    aliases = {
        "menstrual": "Menstrual",
        "follicular": "Follicular",
        "fertility": "Fertility",
        "fertile": "Fertility",
        "luteal": "Luteal",
    }

    text = series.astype("string").str.strip()
    normalized = text.str.lower().map(aliases)

    # Preserve an unexpected non-null label so QC can report it explicitly.
    return normalized.fillna(text)


def collapse_daily_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse duplicate hormone/self-report rows to one participant-day row.

    Numeric fields use the median. Non-numeric fields use the first non-missing
    value. This is only applied if duplicate id/day rows are actually present.
    """

    keys = ["id", "day_in_study"]
    if not dataframe.duplicated(keys).any():
        return dataframe.sort_values(keys).reset_index(drop=True)

    LOGGER.warning(
        "Duplicate id/day rows found in hormones_and_selfreport.csv; "
        "collapsing them to one participant-day row."
    )

    aggregations: dict[str, str | Callable] = {}
    for column in dataframe.columns:
        if column in keys:
            continue

        if pd.api.types.is_numeric_dtype(dataframe[column]):
            aggregations[column] = "median"
        else:
            aggregations[column] = first_nonmissing

    collapsed = (
        dataframe.groupby(keys, as_index=False, dropna=False)
        .agg(aggregations)
        .sort_values(keys)
        .reset_index(drop=True)
    )
    return collapsed


def load_hormones(data_dir: Path) -> pd.DataFrame:
    """Load and minimally clean the hormone/self-report table."""

    path = data_dir / "hormones_and_selfreport.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Download mcPHASES from PhysioNet and point --data-dir to "
            "the folder containing hormones_and_selfreport.csv."
        )

    LOGGER.info("Loading %s", path)
    data = pd.read_csv(path)
    require_columns(data, list(CORE_HORMONE_COLUMNS), path.name)

    data = normalize_id(data)
    data = normalize_day_column(data, "day_in_study")
    data["phase"] = normalize_phase(data["phase"])

    for column in ("lh", "estrogen", "pdg", "study_interval"):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=["id", "day_in_study"]).copy()
    data = collapse_daily_rows(data)

    unexpected = sorted(
        set(data["phase"].dropna().unique()) - set(EXPECTED_PHASES)
    )
    if unexpected:
        LOGGER.warning(
            "Unexpected phase label(s) found and preserved: %s",
            ", ".join(map(str, unexpected)),
        )

    return data


def phase_order_is_valid(phases: pd.Series) -> bool:
    """
    Check whether observed phase transitions follow the expected cycle order.

    Missing phases are ignored. Repeated adjacent phase labels are collapsed.
    """

    observed = [
        phase
        for phase in phases.dropna().tolist()
        if phase in EXPECTED_PHASES
    ]
    if not observed:
        return False

    transitions: list[str] = []
    for phase in observed:
        if not transitions or phase != transitions[-1]:
            transitions.append(phase)

    expected_index = {
        phase: index for index, phase in enumerate(EXPECTED_PHASES)
    }

    for previous, current in zip(transitions, transitions[1:]):
        if expected_index[current] < expected_index[previous]:
            return False

    return True


def assign_cycles(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Assign menstrual cycles within participant and study interval.

    A new cycle starts when the phase transitions into "Menstrual".
    Rows before the first observed menstrual onset in an interval are retained
    with cycle_number missing because they are left-boundary partial cycles.
    """

    data = dataframe.sort_values(
        ["id", "study_interval", "day_in_study"]
    ).copy()

    grouping = ["id", "study_interval"]
    previous_phase = data.groupby(grouping, dropna=False)["phase"].shift(1)

    new_cycle = (
        data["phase"].eq("Menstrual")
        & previous_phase.ne("Menstrual")
    )

    data["cycle_number"] = (
        new_cycle.groupby(
            [data["id"], data["study_interval"]],
            dropna=False,
        )
        .cumsum()
        .astype("Int64")
    )

    # Leading partial segments before the first menstrual onset have cumsum 0.
    data.loc[data["cycle_number"].eq(0), "cycle_number"] = pd.NA

    valid_cycle = data["cycle_number"].notna()
    cycle_keys = ["id", "study_interval", "cycle_number"]

    data["cycle_start_day"] = pd.Series(pd.NA, index=data.index, dtype="Int64")
    data["cycle_end_day"] = pd.Series(pd.NA, index=data.index, dtype="Int64")
    data["cycle_day"] = pd.Series(pd.NA, index=data.index, dtype="Int64")
    data["cycle_span_days"] = pd.Series(pd.NA, index=data.index, dtype="Int64")
    data["cycle_progress_pct"] = np.nan

    if valid_cycle.any():
        cycle_data = data.loc[valid_cycle].copy()

        start_day = cycle_data.groupby(cycle_keys)["day_in_study"].transform(
            "min"
        )
        end_day = cycle_data.groupby(cycle_keys)["day_in_study"].transform(
            "max"
        )

        data.loc[valid_cycle, "cycle_start_day"] = start_day.astype("Int64")
        data.loc[valid_cycle, "cycle_end_day"] = end_day.astype("Int64")

        cycle_day = (
            cycle_data["day_in_study"].astype("Int64")
            - start_day.astype("Int64")
            + 1
        )
        span = (
            end_day.astype("Int64")
            - start_day.astype("Int64")
            + 1
        )

        data.loc[valid_cycle, "cycle_day"] = cycle_day
        data.loc[valid_cycle, "cycle_span_days"] = span

        denominator = (span - 1).replace(0, pd.NA).astype("Float64")
        progress = 100 * (cycle_day - 1).astype("Float64") / denominator
        data.loc[valid_cycle, "cycle_progress_pct"] = progress.astype(float)

    return add_cycle_quality_flags(data)


def add_cycle_quality_flags(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add complete-cycle and phase-order QC flags."""

    data = dataframe.copy()
    cycle_keys = ["id", "study_interval", "cycle_number"]
    valid = data["cycle_number"].notna()

    data["complete_cycle"] = False
    data["valid_phase_order"] = False

    if not valid.any():
        return data

    grouped = data.loc[valid].groupby(cycle_keys, dropna=False)

    phase_sets = grouped["phase"].agg(
        lambda values: frozenset(
            value
            for value in values.dropna()
            if value in EXPECTED_PHASES
        )
    )
    complete = phase_sets.eq(frozenset(EXPECTED_PHASES))
    valid_order = grouped["phase"].agg(phase_order_is_valid)

    quality = pd.DataFrame(
        {
            "complete_cycle": complete,
            "valid_phase_order": valid_order,
        }
    ).reset_index()

    merged = data.merge(
        quality,
        on=cycle_keys,
        how="left",
        suffixes=("", "_cycle"),
    )

    merged["complete_cycle"] = (
        merged["complete_cycle_cycle"].fillna(False).astype(bool)
    )
    merged["valid_phase_order"] = (
        merged["valid_phase_order_cycle"].fillna(False).astype(bool)
    )

    merged = merged.drop(
        columns=["complete_cycle_cycle", "valid_phase_order_cycle"]
    )
    return merged


def aggregation_function(name: str) -> str | Callable:
    """Validate and return a pandas-compatible aggregation."""

    allowed = {"median", "mean", "min", "max", "first"}
    if name not in allowed:
        raise ValueError(
            f"Unsupported aggregation '{name}'. "
            f"Choose from: {', '.join(sorted(allowed))}."
        )
    return name


def load_signal(
    data_dir: Path,
    spec: SignalSpec,
    required: bool = False,
) -> pd.DataFrame | None:
    """Load one signal and reduce it to one value per participant-day."""

    path = data_dir / spec.filename

    if not path.exists():
        message = f"Signal file not found: {path}"
        if required:
            raise FileNotFoundError(message)
        LOGGER.warning("%s; skipping %s.", message, spec.name)
        return None

    LOGGER.info("Loading signal %s from %s", spec.name, path)
    data = pd.read_csv(path)
    require_columns(
        data,
        ["id", spec.day_column, spec.value_column],
        path.name,
    )

    data = normalize_id(data)
    data = normalize_day_column(data, spec.day_column)
    data[spec.value_column] = pd.to_numeric(
        data[spec.value_column],
        errors="coerce",
    )

    data = data.dropna(
        subset=["id", spec.day_column, spec.value_column]
    ).copy()

    if spec.positive_only:
        data = data[data[spec.value_column] > 0].copy()

    output = (
        data.groupby(
            ["id", spec.day_column], as_index=False
        )[spec.value_column]
        .agg(aggregation_function(spec.aggregation))
        .rename(
            columns={
                spec.day_column: "day_in_study",
                spec.value_column: spec.name,
            }
        )
    )

    output["day_in_study"] = output["day_in_study"].astype("Int64")
    return output


def parse_signal_spec(text: str) -> SignalSpec:
    """Parse NAME=FILE:DAY_COLUMN:VALUE_COLUMN:AGG."""

    if "=" not in text:
        raise ValueError(
            f"Invalid --signal '{text}'. Expected NAME=FILE:DAY:VALUE:AGG."
        )

    name, remainder = text.split("=", 1)
    parts = remainder.split(":")

    if len(parts) not in (3, 4):
        raise ValueError(
            f"Invalid --signal '{text}'. Expected NAME=FILE:DAY:VALUE:AGG."
        )

    filename, day_column, value_column = parts[:3]
    aggregation = parts[3] if len(parts) == 4 else "median"

    if not name.strip():
        raise ValueError("Signal NAME cannot be empty.")

    aggregation_function(aggregation)

    return SignalSpec(
        name=name.strip(),
        filename=filename.strip(),
        day_column=day_column.strip(),
        value_column=value_column.strip(),
        aggregation=aggregation.strip(),
    )


def merge_signals(
    daily: pd.DataFrame,
    signals: list[pd.DataFrame],
) -> pd.DataFrame:
    """Left-join daily physiological signals onto the hormone table."""

    merged = daily.copy()

    for signal in signals:
        value_columns = [
            column
            for column in signal.columns
            if column not in ("id", "day_in_study")
        ]
        LOGGER.info("Merging signal column(s): %s", ", ".join(value_columns))
        merged = merged.merge(
            signal,
            on=["id", "day_in_study"],
            how="left",
            validate="one_to_one",
        )

    return merged


def add_participant_eligibility(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Annotate whether participants have repeated complete cycles."""

    data = dataframe.copy()

    cycle_rows = data[data["complete_cycle"]].dropna(
        subset=["cycle_number"]
    )

    complete_counts = (
        cycle_rows[
            ["id", "study_interval", "cycle_number"]
        ]
        .drop_duplicates()
        .groupby("id")
        .size()
        .rename("n_complete_cycles_participant")
    )

    interval_counts = (
        cycle_rows[
            ["id", "study_interval"]
        ]
        .drop_duplicates()
        .groupby("id")
        .size()
        .rename("n_intervals_with_complete_cycle")
    )

    data = data.merge(complete_counts, on="id", how="left")
    data = data.merge(interval_counts, on="id", how="left")

    data["n_complete_cycles_participant"] = (
        data["n_complete_cycles_participant"].fillna(0).astype(int)
    )
    data["n_intervals_with_complete_cycle"] = (
        data["n_intervals_with_complete_cycle"].fillna(0).astype(int)
    )

    data["aim1_repeated_cycle_candidate"] = (
        data["n_complete_cycles_participant"] >= 2
    )
    data["long_term_candidate"] = (
        data["n_intervals_with_complete_cycle"] >= 2
    )

    return data


def build_cycle_summary(
    dataframe: pd.DataFrame,
    signal_columns: list[str],
) -> pd.DataFrame:
    """Create one QC row per observed menstrual cycle."""

    cycle_data = dataframe.dropna(subset=["cycle_number"]).copy()
    if cycle_data.empty:
        return pd.DataFrame()

    keys = ["id", "study_interval", "cycle_number"]

    summary = (
        cycle_data.groupby(keys, as_index=False)
        .agg(
            cycle_start_day=("day_in_study", "min"),
            cycle_end_day=("day_in_study", "max"),
            n_rows=("day_in_study", "size"),
            n_unique_days=("day_in_study", "nunique"),
            n_phases=("phase", "nunique"),
            complete_cycle=("complete_cycle", "first"),
            valid_phase_order=("valid_phase_order", "first"),
        )
    )

    summary["cycle_span_days"] = (
        summary["cycle_end_day"] - summary["cycle_start_day"] + 1
    )
    summary["day_coverage"] = (
        summary["n_unique_days"] / summary["cycle_span_days"]
    )

    for signal in signal_columns:
        if signal not in cycle_data.columns:
            continue

        coverage = (
            cycle_data.assign(
                _present=cycle_data[signal].notna().astype(int)
            )
            .groupby(keys, as_index=False)["_present"]
            .mean()
            .rename(columns={"_present": f"{signal}_coverage"})
        )
        summary = summary.merge(coverage, on=keys, how="left")

    return summary.sort_values(keys).reset_index(drop=True)


def build_participant_summary(
    dataframe: pd.DataFrame,
    signal_columns: list[str],
) -> pd.DataFrame:
    """Create one QC row per participant."""

    base = (
        dataframe.groupby("id", as_index=False)
        .agg(
            n_study_days=("day_in_study", "nunique"),
            first_day=("day_in_study", "min"),
            last_day=("day_in_study", "max"),
            n_study_intervals=("study_interval", "nunique"),
        )
    )

    complete = (
        dataframe[dataframe["complete_cycle"]]
        [["id", "study_interval", "cycle_number"]]
        .dropna()
        .drop_duplicates()
        .groupby("id")
        .size()
        .rename("n_complete_cycles")
        .reset_index()
    )
    base = base.merge(complete, on="id", how="left")
    base["n_complete_cycles"] = base["n_complete_cycles"].fillna(0).astype(int)

    for signal in ("lh", "estrogen", "pdg", *signal_columns):
        if signal not in dataframe.columns:
            continue

        coverage = (
            dataframe.assign(
                _present=dataframe[signal].notna().astype(int)
            )
            .groupby("id", as_index=False)["_present"]
            .mean()
            .rename(columns={"_present": f"{signal}_coverage"})
        )
        base = base.merge(coverage, on="id", how="left")

    base["aim1_repeated_cycle_candidate"] = base["n_complete_cycles"] >= 2

    intervals_with_complete = (
        dataframe[dataframe["complete_cycle"]]
        [["id", "study_interval"]]
        .drop_duplicates()
        .groupby("id")
        .size()
        .rename("n_intervals_with_complete_cycle")
        .reset_index()
    )
    base = base.merge(intervals_with_complete, on="id", how="left")
    base["n_intervals_with_complete_cycle"] = (
        base["n_intervals_with_complete_cycle"].fillna(0).astype(int)
    )
    base["long_term_candidate"] = (
        base["n_intervals_with_complete_cycle"] >= 2
    )

    return base.sort_values("id").reset_index(drop=True)


def write_outputs(
    daily: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    participant_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write analysis-ready and QC tables."""

    output_dir.mkdir(parents=True, exist_ok=True)

    daily_path = output_dir / "mcphases_daily_master.csv"
    complete_path = output_dir / "mcphases_daily_complete_cycles.csv"
    cycle_path = output_dir / "mcphases_cycle_qc.csv"
    participant_path = output_dir / "mcphases_participant_qc.csv"

    daily.to_csv(daily_path, index=False)
    daily[daily["complete_cycle"]].to_csv(complete_path, index=False)
    cycle_summary.to_csv(cycle_path, index=False)
    participant_summary.to_csv(participant_path, index=False)

    LOGGER.info("Wrote %s", daily_path)
    LOGGER.info("Wrote %s", complete_path)
    LOGGER.info("Wrote %s", cycle_path)
    LOGGER.info("Wrote %s", participant_path)


def print_summary(
    daily: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    participant_summary: pd.DataFrame,
    signal_columns: list[str],
) -> None:
    """Print a compact run summary."""

    n_participants = daily["id"].nunique()
    n_complete = 0
    if not cycle_summary.empty:
        n_complete = int(cycle_summary["complete_cycle"].sum())

    repeated = int(
        participant_summary["aim1_repeated_cycle_candidate"].sum()
    )
    long_term = int(participant_summary["long_term_candidate"].sum())

    print("\n" + "=" * 72)
    print("mcPHASES PREPROCESSING COMPLETE")
    print("=" * 72)
    print(f"Participants:                         {n_participants}")
    print(f"Participant-days:                     {len(daily)}")
    print(f"Complete cycles detected:              {n_complete}")
    print(f"Participants with >=2 complete cycles: {repeated}")
    print(f"Long-term candidates (both intervals): {long_term}")
    print("Merged physiological signals:          "
          + (", ".join(signal_columns) if signal_columns else "none"))
    print("=" * 72)
    print(
        "\nNext step: inspect mcphases_participant_qc.csv and "
        "mcphases_cycle_qc.csv BEFORE choosing coverage thresholds."
    )


def main() -> None:
    """Run preprocessing."""

    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    daily = load_hormones(data_dir)
    daily = assign_cycles(daily)

    specs = list(DEFAULT_SIGNALS)
    specs.extend(parse_signal_spec(text) for text in args.signal)

    signal_frames: list[pd.DataFrame] = []
    signal_columns: list[str] = []

    for spec in specs:
        signal = load_signal(
            data_dir=data_dir,
            spec=spec,
            required=args.strict and spec in DEFAULT_SIGNALS,
        )
        if signal is not None:
            signal_frames.append(signal)
            signal_columns.append(spec.name)

    daily = merge_signals(daily, signal_frames)
    daily = add_participant_eligibility(daily)

    # Keep a stable column order for the most important analysis variables.
    preferred = [
        "id",
        "study_interval",
        "day_in_study",
        "cycle_number",
        "cycle_day",
        "cycle_span_days",
        "cycle_progress_pct",
        "phase",
        "complete_cycle",
        "valid_phase_order",
        "cycle_start_day",
        "cycle_end_day",
        "lh",
        "estrogen",
        "pdg",
        *signal_columns,
        "n_complete_cycles_participant",
        "n_intervals_with_complete_cycle",
        "aim1_repeated_cycle_candidate",
        "long_term_candidate",
    ]
    preferred = [column for column in preferred if column in daily.columns]
    remainder = [column for column in daily.columns if column not in preferred]
    daily = daily[preferred + remainder]

    cycle_summary = build_cycle_summary(daily, signal_columns)
    participant_summary = build_participant_summary(
        daily,
        signal_columns,
    )

    write_outputs(
        daily=daily,
        cycle_summary=cycle_summary,
        participant_summary=participant_summary,
        output_dir=output_dir,
    )
    print_summary(
        daily=daily,
        cycle_summary=cycle_summary,
        participant_summary=participant_summary,
        signal_columns=signal_columns,
    )


if __name__ == "__main__":
    main()
