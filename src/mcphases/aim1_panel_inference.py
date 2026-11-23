"""Exploratory interval-specific Aim 1 permutation contrasts.

Panel A tests cycle-label exchangeability *within each interval*.
Panel B tests whether identity matching persists between two intervals,
permuting whole participant-profile labels in the later interval.

Both tests use a prespecified directional similarity advantage (between
minus within mean standardized Euclidean distance). Pairwise distances are
not independent; p-values come from conditional label permutations, not a
pairwise rank test. Valid inference requires the relevant exchangeability
assumptions, which cannot be established by numerical calibration alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mcphases.similarity import select_repeated_cycle_cohort

CONTRASTS = ("within_interval", "cross_interval")


def _statistic(distances, mask, same_person):
    """Return a signed similarity advantage for a preselected pair set."""
    same = mask & same_person
    different = mask & ~same_person
    if not np.any(same) or not np.any(different):
        raise ValueError("Both same- and different-person pairs are required")
    return float(distances[different].mean() - distances[same].mean())


def _shuffle_cycle_labels(ids, intervals, rng):
    """Shuffle identities among cycles separately in each study interval."""
    shuffled = ids.copy()
    for interval in np.unique(intervals):
        positions = np.flatnonzero(intervals == interval)
        shuffled[positions] = rng.permutation(ids[positions])
    return shuffled


def _shuffle_later_participant_blocks(
    ids, intervals, rng, eligible_ids=None
):
    """Reassign entire later-interval participant histories as blocks.

    When eligible_ids is supplied, only remap profiles of people who
    contributed cycles in *both* periods (matched longitudinal cohort).
    """
    shuffled = ids.copy()
    later = np.unique(intervals)[-1]
    mask = intervals == later
    if eligible_ids is not None:
        mask &= np.isin(ids, eligible_ids)
    original = np.unique(ids[mask])
    reassigned = rng.permutation(original)
    shuffled[mask] = reassigned[np.searchsorted(original, ids[mask])]
    return shuffled


def holm_adjust(p_values):
    """Holm family-wise adjusted values, maintaining monotonicity."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("Expected a nonempty one-dimensional p-value vector")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def _loo_diagnostics(ids, index_a, index_b, distances, pair_mask):
    """Descriptive leave-one-participant-out sensitivity, not a CI."""
    values = []
    influential_ids = np.unique(
        np.concatenate((ids[index_a[pair_mask]], ids[index_b[pair_mask]]))
    )
    for identifier in influential_ids:
        keep = pair_mask & (ids[index_a] != identifier) & (
            ids[index_b] != identifier
        )
        same = ids[index_a] == ids[index_b]
        try:
            values.append(_statistic(distances, keep, same))
        except ValueError:
            continue
    if not values:
        raise ValueError("LOO cannot retain both distance groups")
    return {
        "loo_n": len(values),
        "loo_advantage_min": float(np.min(values)),
        "loo_advantage_max": float(np.max(values)),
        "loo_fraction_positive": float(np.mean(np.asarray(values) > 0)),
    }


def compare_panels(signatures, pairs, permutations=9999, seed=42):
    """Compute two directional, exploratory, participant-aware contrasts.

    The supplied pairs must be indexed to the complete-case repeated-cycle
    cohort produced by the existing plotting function. This assertion
    prevents an undetected mismatch between plotted and tested distances.

    Returns
    -------
    summary : pandas.DataFrame
        Exactly two prespecified contrasts with Holm-adjusted p-values.
    nulls : pandas.DataFrame
        Each simulated null statistic, for full reproducibility.
    """
    if permutations < 1:
        raise ValueError("permutations must be positive")
    complete = (
        select_repeated_cycle_cohort(signatures)
        .dropna()
        .reset_index(drop=True)
    )
    intervals = complete["study_interval"].to_numpy()
    if len(np.unique(intervals)) != 2:
        raise ValueError("Panel B requires exactly two study intervals")
    ids = complete["id"].to_numpy()
    a = pairs["index_a"].to_numpy(dtype=int)
    b = pairs["index_b"].to_numpy(dtype=int)
    if len(a) < 2 or not (
        (a >= 0) & (b >= 0) & (a < len(ids)) & (b < len(ids))
    ).all():
        raise ValueError("Pair indices do not match complete-case signatures")
    distances = pairs["distance"].to_numpy(dtype=float)
    if not np.isfinite(distances).all():
        raise ValueError("All pairwise distances must be finite")

    observed_same = ids[a] == ids[b]
    same_interval = intervals[a] == intervals[b]
    if not np.array_equal(observed_same, pairs["same_person"].to_numpy()):
        raise ValueError("Participant identity labels do not match pairs")
    if not np.array_equal(
        same_interval, pairs["same_interval"].to_numpy()
    ):
        raise ValueError("Study-interval labels do not match pairs")

    # Restrict cross-interval testing to contributors in both years.
    # Otherwise selection/attrition can make "same person" a special
    # subset of the panel's unmatched between-person comparison group.
    years = np.unique(intervals)
    matched_ids = np.intersect1d(
        np.unique(ids[intervals == years[0]]),
        np.unique(ids[intervals == years[1]]),
    )
    if len(matched_ids) < 3:
        raise ValueError("At least three shared participants are required")
    matched_pairs = np.isin(ids[a], matched_ids) & np.isin(
        ids[b], matched_ids
    )
    rngs = [
        np.random.default_rng(child)
        for child in np.random.SeedSequence(seed).spawn(2)
    ]
    masks = (same_interval, (~same_interval) & matched_pairs)
    shufflers = (
        _shuffle_cycle_labels,
        lambda observed, periods, rng: _shuffle_later_participant_blocks(
            observed, periods, rng, eligible_ids=matched_ids
        ),
    )
    rows = []
    null_tables = []

    for name, mask, shuffle, rng in zip(
        CONTRASTS, masks, shufflers, rngs
    ):
        observed = _statistic(distances, mask, observed_same)
        simulated = np.empty(permutations, dtype=float)
        for index in range(permutations):
            permuted_ids = shuffle(ids, intervals, rng)
            simulated[index] = _statistic(
                distances, mask, permuted_ids[a] == permuted_ids[b]
            )

        same = mask & observed_same
        different = mask & ~observed_same
        p_value = (1 + int(np.count_nonzero(simulated >= observed))) / (
            permutations + 1
        )
        rows.append(
            {
                "contrast": name,
                "n_same_person_pairs": int(same.sum()),
                "n_different_person_pairs": int(different.sum()),
                "mean_same_person_distance": float(distances[same].mean()),
                "mean_different_person_distance": float(
                    distances[different].mean()
                ),
                "similarity_advantage": observed,
                "p_one_sided": float(p_value),
                "null_mean": float(np.mean(simulated)),
                "null_q025": float(np.quantile(simulated, 0.025)),
                "null_q975": float(np.quantile(simulated, 0.975)),
                "permutations": int(permutations),
                "seed": int(seed),
                "permutation_unit": (
                    "cycle labels within study interval"
                    if name == "within_interval"
                    else "later-interval identity blocks among shared people"
                ),
                "n_participants_in_contrast": int(
                    len(np.unique(np.r_[ids[a[mask]], ids[b[mask]]]))
                ),
                **_loo_diagnostics(ids, a, b, distances, mask),
            }
        )
        null_tables.append(
            pd.DataFrame(
                {
                    "contrast": name,
                    "iteration": np.arange(1, permutations + 1),
                    "null_similarity_advantage": simulated,
                }
            )
        )

    summary = pd.DataFrame(rows)
    summary["p_holm_two_contrasts"] = holm_adjust(
        summary["p_one_sided"].to_numpy()
    )
    return summary, pd.concat(null_tables, ignore_index=True)
