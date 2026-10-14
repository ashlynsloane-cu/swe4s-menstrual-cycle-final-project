"""Deterministic train/calibration/test splitting for Aim 2."""

from __future__ import annotations

import pandas as pd


PHASES = (
    "Menstrual",
    "Follicular",
    "Fertility",
    "Luteal",
)

PREDICTORS = (
    "resting_heart_rate",
    "hrv_rmssd",
    "nightly_temperature",
    "respiratory_rate",
)

DAY_KEYS = [
    "id",
    "study_interval",
    "day_in_study",
]

CYCLE_KEYS = [
    "id",
    "study_interval",
    "cycle_number",
]


def add_aim2_complete_case(
    daily: pd.DataFrame,
) -> pd.DataFrame:
    """Flag days usable as four-predictor phase observations."""

    data = daily.copy()

    required = {
        "id",
        "study_interval",
        "day_in_study",
        "phase",
        *PREDICTORS,
    }

    missing = sorted(required - set(data.columns))

    if missing:
        raise ValueError(
            "Daily data are missing required column(s): "
            + ", ".join(missing)
        )

    if data.duplicated(DAY_KEYS).any():
        raise ValueError(
            "Daily master contains duplicate participant-days."
        )

    data["aim2_complete_case"] = (
        data["phase"].isin(PHASES)
        & data[list(PREDICTORS)].notna().all(axis=1)
    )

    return data


def prepare_valid_cycles(
    cycle_qc: pd.DataFrame,
) -> pd.DataFrame:
    """Select and chronologically number valid biological cycles."""

    required = {
        "id",
        "study_interval",
        "cycle_number",
        "cycle_start_day",
        "complete_cycle",
        "valid_phase_order",
    }

    missing = sorted(required - set(cycle_qc.columns))

    if missing:
        raise ValueError(
            "Cycle QC is missing required column(s): "
            + ", ".join(missing)
        )

    valid = cycle_qc[
        cycle_qc["complete_cycle"]
        & cycle_qc["valid_phase_order"]
    ].copy()

    valid = valid.sort_values(
        [
            "id",
            "study_interval",
            "cycle_start_day",
            "cycle_number",
        ]
    ).reset_index(drop=True)

    valid["target_cycle_index"] = (
        valid.groupby("id").cumcount() + 1
    )

    return valid


def select_cycle_days(
    daily: pd.DataFrame,
    cycles: pd.DataFrame,
) -> pd.DataFrame:
    """Return complete-case days belonging to selected cycles."""

    if cycles.empty:
        return daily.iloc[0:0].copy()

    selected = daily.merge(
        cycles[
            CYCLE_KEYS
            + ["target_cycle_index"]
        ],
        on=CYCLE_KEYS,
        how="inner",
    )

    return selected[
        selected["aim2_complete_case"]
    ].copy()


def phase_counts(
    data: pd.DataFrame,
) -> dict[str, int]:
    """Count usable observations in each menstrual phase."""

    counts = (
        data["phase"]
        .value_counts()
        .reindex(PHASES, fill_value=0)
    )

    return {
        phase: int(counts[phase])
        for phase in PHASES
    }


def build_target_summary(
    daily: pd.DataFrame,
    valid_cycles: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize one- versus two-cycle calibration by participant."""

    records = []

    for participant_id, cycles in valid_cycles.groupby("id"):
        cycles = cycles.sort_values(
            "target_cycle_index"
        ).reset_index(drop=True)

        if len(cycles) < 3:
            continue

        person = daily[
            daily["id"] == participant_id
        ].copy()

        cycle_1 = cycles.iloc[[0]]
        cycle_2 = cycles.iloc[[1]]
        test_cycles = cycles.iloc[2:]

        cal_1 = select_cycle_days(
            person,
            cycle_1,
        )

        cal_2_extra = select_cycle_days(
            person,
            cycle_2,
        )

        cal_2_total = pd.concat(
            [
                cal_1,
                cal_2_extra,
            ],
            ignore_index=True,
        )

        test = select_cycle_days(
            person,
            test_cycles,
        )

        cal_1_counts = phase_counts(cal_1)
        cal_2_counts = phase_counts(cal_2_total)
        test_counts = phase_counts(test)

        record = {
            "target_id": participant_id,
            "n_valid_cycles": len(cycles),
            "n_future_cycles": len(test_cycles),
            "cal1_days": len(cal_1),
            "cycle2_added_days": len(cal_2_extra),
            "cal2_days": len(cal_2_total),
            "test_days": len(test),
            "cal1_n_phases": sum(
                value > 0
                for value in cal_1_counts.values()
            ),
            "cal2_n_phases": sum(
                value > 0
                for value in cal_2_counts.values()
            ),
            "test_n_phases": sum(
                value > 0
                for value in test_counts.values()
            ),
        }

        for phase in PHASES:
            label = phase.lower()

            record[
                f"cal1_{label}_days"
            ] = cal_1_counts[phase]

            record[
                f"cal2_{label}_days"
            ] = cal_2_counts[phase]

            record[
                f"test_{label}_days"
            ] = test_counts[phase]

        record["population_testable"] = (
            record["test_days"] > 0
        )

        record["hybrid1_testable"] = (
            record["cal1_days"] > 0
            and record["test_days"] > 0
        )

        record["hybrid2_testable"] = (
            record["cal2_days"] > 0
            and record["test_days"] > 0
        )

        record["primary_matched_eligible"] = (
            record["hybrid1_testable"]
            and record["hybrid2_testable"]
        )

        record["second_cycle_adds_data"] = (
            record["cycle2_added_days"] > 0
        )

        record["personal1_four_class"] = (
            record["cal1_n_phases"] == 4
            and record["test_days"] > 0
        )

        record["personal2_four_class"] = (
            record["cal2_n_phases"] == 4
            and record["test_days"] > 0
        )

        records.append(record)

    return pd.DataFrame(records)


def make_manifest_rows(
    data: pd.DataFrame,
    target_id,
    role: str,
) -> pd.DataFrame:
    """Convert selected participant-days to split-manifest rows."""

    if data.empty:
        return pd.DataFrame(
            columns=[
                "target_id",
                "source_id",
                "study_interval",
                "day_in_study",
                "cycle_number",
                "target_cycle_index",
                "phase",
                "split_role",
            ]
        )

    output = data[
        [
            "id",
            "study_interval",
            "day_in_study",
            "cycle_number",
            "phase",
        ]
    ].copy()

    if "target_cycle_index" in data.columns:
        output["target_cycle_index"] = (
            data["target_cycle_index"].values
        )
    else:
        output["target_cycle_index"] = pd.NA

    output = output.rename(
        columns={
            "id": "source_id",
        }
    )

    output.insert(
        0,
        "target_id",
        target_id,
    )

    output["split_role"] = role

    return output


def build_primary_manifest(
    daily: pd.DataFrame,
    cycle_qc: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the matched primary Aim 2 split manifest."""

    daily = add_aim2_complete_case(daily)
    valid_cycles = prepare_valid_cycles(cycle_qc)

    summary = build_target_summary(
        daily,
        valid_cycles,
    )

    eligible_ids = summary.loc[
        summary["primary_matched_eligible"],
        "target_id",
    ].tolist()

    population_pool = daily[
        daily["aim2_complete_case"]
    ].copy()

    manifest_parts = []

    for target_id in eligible_ids:
        cycles = valid_cycles[
            valid_cycles["id"] == target_id
        ].sort_values(
            "target_cycle_index"
        )

        person = daily[
            daily["id"] == target_id
        ]

        cal_1 = select_cycle_days(
            person,
            cycles.iloc[[0]],
        )

        cal_2_extra = select_cycle_days(
            person,
            cycles.iloc[[1]],
        )

        future_test = select_cycle_days(
            person,
            cycles.iloc[2:],
        )

        population_train = population_pool[
            population_pool["id"] != target_id
        ].copy()

        manifest_parts.extend(
            [
                make_manifest_rows(
                    population_train,
                    target_id,
                    "population_train",
                ),
                make_manifest_rows(
                    cal_1,
                    target_id,
                    "personal_calibration_cycle1",
                ),
                make_manifest_rows(
                    cal_2_extra,
                    target_id,
                    "personal_calibration_cycle2",
                ),
                make_manifest_rows(
                    future_test,
                    target_id,
                    "future_test",
                ),
            ]
        )

    manifest = pd.concat(
        manifest_parts,
        ignore_index=True,
    )

    validate_manifest(manifest)

    return manifest, summary


def validate_manifest(
    manifest: pd.DataFrame,
) -> None:
    """Check the split manifest for leakage and chronology errors."""

    required = {
        "target_id",
        "source_id",
        "study_interval",
        "day_in_study",
        "split_role",
    }

    missing = sorted(required - set(manifest.columns))

    if missing:
        raise ValueError(
            "Manifest is missing required column(s): "
            + ", ".join(missing)
        )

    day_identity = [
        "target_id",
        "source_id",
        "study_interval",
        "day_in_study",
    ]

    if manifest.duplicated(day_identity).any():
        raise ValueError(
            "A participant-day appears in multiple split roles "
            "within the same target fold."
        )

    for target_id, fold in manifest.groupby("target_id"):
        population = fold[
            fold["split_role"] == "population_train"
        ]

        if (population["source_id"] == target_id).any():
            raise ValueError(
                f"Target {target_id} leaked into population training."
            )

        personal = fold[
            fold["split_role"].isin(
                [
                    "personal_calibration_cycle1",
                    "personal_calibration_cycle2",
                    "future_test",
                ]
            )
        ]

        if not (
            personal["source_id"] == target_id
        ).all():
            raise ValueError(
                f"Non-target data appear in target {target_id}'s "
                "personal/test rows."
            )

        cycle_1 = fold[
            fold["split_role"]
            == "personal_calibration_cycle1"
        ]

        cycle_2 = fold[
            fold["split_role"]
            == "personal_calibration_cycle2"
        ]

        future = fold[
            fold["split_role"] == "future_test"
        ]

        if not (
            cycle_1["target_cycle_index"] == 1
        ).all():
            raise ValueError(
                f"Target {target_id} has invalid cycle-1 rows."
            )

        if not (
            cycle_2["target_cycle_index"] == 2
        ).all():
            raise ValueError(
                f"Target {target_id} has invalid cycle-2 rows."
            )

        if not (
            future["target_cycle_index"] >= 3
        ).all():
            raise ValueError(
                f"Target {target_id} has pre-test cycles in future_test."
            )
