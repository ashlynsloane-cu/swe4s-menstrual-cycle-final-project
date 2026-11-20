"""Generate a descriptive Aim 1 permutation-exchangeability audit.

Uses existing cycle signatures, never raw restricted participant-day data.
Does not change the frozen test, reuse results, or select an optimal null.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mcphases.similarity import (
    calculate_pairwise_distances,
    calculate_similarity_advantage,
    get_feature_columns,
    permutation_test,
    select_repeated_cycle_cohort,
)


def audit_signatures(
    signatures: pd.DataFrame,
    permutations: int = 999,
    random_state: int = 42,
) -> dict:
    """Compare frozen and stratified nulls with sampling diagnostics.

    The global null permutes participant IDs across all cycles. The interval
    null preserves the empirical multiplicity of each ID per interval.
    Neither is necessarily valid when participant signatures are dependent
    across time or observations have confounding structure.
    """
    required = {"id", "study_interval", "cycle_number"}
    missing = sorted(required.difference(signatures.columns))
    if missing:
        raise ValueError(f"Missing signature columns: {', '.join(missing)}")
    if not isinstance(permutations, int) or permutations < 1:
        raise ValueError("permutations must be a positive integer")

    repeated = select_repeated_cycle_cohort(signatures)
    features = get_feature_columns(repeated)
    if not features:
        raise ValueError("No physiological signature features were found")

    complete = repeated.dropna().reset_index(drop=True)
    if complete.empty:
        raise ValueError("No complete physiological signatures were found")

    variances = complete[features].var(ddof=1)
    if not np.isfinite(variances).all() or (variances <= 0).any():
        raise ValueError("Constant or non-finite signature feature variance")

    pairs = calculate_pairwise_distances(complete, features)
    if pairs.empty or pairs["same_person"].all():
        raise ValueError("Both within- and between-participant pairs required")

    # Join by the indices exported from calculate_pairwise_distances, which
    # reset the index after dropping complete-case missing signatures.
    intervals = complete["study_interval"].to_numpy()
    pairs = pairs.copy()
    pairs["same_interval"] = (
        intervals[pairs["index_a"].to_numpy()]
        == intervals[pairs["index_b"].to_numpy()]
    )
    cross = pairs.loc[~pairs["same_interval"]]

    # Paired data are not independent observations: the same signature occurs
    # in many pairs. These counts are descriptive, not an effective sample N.
    observed = calculate_similarity_advantage(pairs)
    _, global_p, global_null = permutation_test(
        complete,
        pairs,
        n_permutations=permutations,
        random_state=random_state,
        stratify_by_study_interval=False,
    )
    _, interval_p, interval_null = permutation_test(
        complete,
        pairs,
        n_permutations=permutations,
        random_state=random_state,
        stratify_by_study_interval=True,
    )

    if not (np.isfinite(global_null).all()
            and np.isfinite(interval_null).all()):
        raise ValueError(
            "Non-finite permutation null values: a degenerate label "
            "shuffle makes the requested p-value undefined"
        )

    interval_counts = (
        complete.groupby("study_interval", dropna=False)
        .agg(n_cycles=("id", "size"), n_participants=("id", "nunique"))
        .reset_index()
    )
    return {
        "analysis": (
            "Aim1 descriptive assumption audit; not a new primary test"
        ),
        "random_state": int(random_state),
        "permutations": permutations,
        "n_complete_signatures": len(complete),
        "n_participants": int(complete["id"].nunique()),
        "n_features": len(features),
        "n_within_pairs": int(pairs["same_person"].sum()),
        "n_between_pairs": int((~pairs["same_person"]).sum()),
        "n_cross_interval_pairs": len(cross),
        "n_cross_interval_same_person_pairs": int(cross["same_person"].sum()),
        "interval_counts": interval_counts.to_dict(orient="records"),
        "within_person_similarity_advantage": float(observed),
        "global_permutation_p": float(global_p),
        "interval_stratified_permutation_p": float(interval_p),
        "global_null_finite": bool(np.isfinite(global_null).all()),
        "interval_null_finite": bool(np.isfinite(interval_null).all()),
        "interpretation_caution": (
            "Different p-values describe different exchangeability "
            "assumptions; do not choose the smaller p-value or replace the "
            "frozen null without design-based justification."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--signatures", type=Path,
        default=Path("results/aim1/cycle_signatures.csv"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/aim1/audit/assumptions.json"),
    )
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = audit_signatures(
        pd.read_csv(args.signatures), args.permutations, args.seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"Saved assumption audit: {args.output}")
    print(
        "Global versus interval-stratified p-values: "
        f"{summary['global_permutation_p']:.4f} versus "
        f"{summary['interval_stratified_permutation_p']:.4f}"
    )


if __name__ == "__main__":
    main()
