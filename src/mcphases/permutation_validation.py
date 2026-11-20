"""Validation helpers for the Aim 1 participant-label null model."""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def permute_participant_ids(
    participant_ids: ArrayLike,
    rng: np.random.Generator,
    study_intervals: ArrayLike | None = None,
) -> np.ndarray:
    """Shuffle cycle-level IDs while preserving their multiplicities.

    If ``study_intervals`` is supplied, IDs are permuted separately within
    each interval, preserving that interval's original ID multiplicities.
    This is a sensitivity null, not a replacement for the frozen primary
    analysis, which permutes across all complete cycle signatures.
    """
    ids = np.asarray(participant_ids)

    if ids.ndim != 1:
        raise ValueError("participant_ids must be one-dimensional")

    if study_intervals is None:
        return rng.permutation(ids)

    intervals = np.asarray(study_intervals)
    if intervals.ndim != 1 or len(intervals) != len(ids):
        raise ValueError(
            "study_intervals must match participant_ids in length"
        )

    shuffled = ids.copy()
    for interval in pd.unique(intervals):
        positions = np.flatnonzero(intervals == interval)
        shuffled[positions] = rng.permutation(ids[positions])

    return shuffled
