"""Pure, synthetic-data diagnostics for the mcPHASES code review.

These methods do not access participant data. Power depends on the stated
simulation assumptions, not solely on sample size or number of permutations.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.spatial.distance import pdist


def bootstrap_id_diagnostics(ids, draws=5000, seed=6118):
    """Audit diversity under ordinary participant-cluster resampling.

    Sampling with replacement is intentional. Repeated IDs are valid draws;
    this function reports concentration without changing the estimator.
    """
    ids = np.asarray(ids)
    unique_ids = np.unique(ids)
    n = len(unique_ids)
    if n < 2 or draws < 1:
        raise ValueError("Need >=2 distinct participants and positive draws")
    rng = np.random.default_rng(seed)
    n_unique = np.empty(draws, dtype=int)
    maximum_count = np.empty(draws, dtype=int)
    effective_n = np.empty(draws, dtype=float)
    for iteration in range(draws):
        sampled = rng.choice(unique_ids, size=n, replace=True)
        counts = np.unique(sampled, return_counts=True)[1]
        n_unique[iteration] = len(counts)
        maximum_count[iteration] = counts.max()
        effective_n[iteration] = n ** 2 / np.square(counts).sum()
    return {
        "participants": int(n),
        "bootstrap_replicates": int(draws),
        "expected_unique_analytic": float(n * (1 - (1 - 1 / n) ** n)),
        "observed_mean_unique": float(n_unique.mean()),
        "unique_5th_percentile": float(np.quantile(n_unique, 0.05)),
        "unique_median": float(np.median(n_unique)),
        "unique_95th_percentile": float(np.quantile(n_unique, 0.95)),
        "mean_fraction_unique": float((n_unique / n).mean()),
        "maximum_count_95th_percentile": float(
            np.quantile(maximum_count, 0.95)
        ),
        "probability_participant_sampled_at_least_10_times": float(
            (maximum_count >= 10).mean()
        ),
        "effective_n_median": float(np.median(effective_n)),
        "method": "n participants drawn with replacement per replicate",
        "interpretation": (
            "Repeated IDs are mathematically expected in a cluster bootstrap. "
            "These are concentration diagnostics, not acceptance thresholds."
        ),
    }


def _center_phase_blocks(matrix):
    """Center each four-phase block as in within-cycle signatures."""
    arr = np.asarray(matrix, dtype=float)
    if arr.shape[-1] != 16:
        raise ValueError("Require 16 features (4 signals x 4 phases)")
    shape = arr.shape[:-1]
    four = arr.reshape(*shape, 4, 4)
    return (four - four.mean(axis=-1, keepdims=True)).reshape(
        *shape, 16
    )


def synthetic_cycle_signatures(
    participants=25,
    cycles_per_person=4,
    stable_sd=0.75,
    day_noise_sd=1.0,
    long_term_drift_sd=0.0,
    seed=42,
):
    """Generate phase-centered person/cycle profiles, not raw recordings.

    Each individual has a stable 4-signal x 4-phase profile. Each cycle has
    separate noise; 2024 can include an additional person-specific drift.
    No cross-signal reliability structure is assumed.
    """
    if participants < 3 or cycles_per_person < 2:
        raise ValueError("Need >=3 participants and >=2 cycles/person")
    if min(stable_sd, day_noise_sd, long_term_drift_sd) < 0:
        raise ValueError("Standard deviations must be nonnegative")
    if day_noise_sd == 0 and stable_sd == 0:
        raise ValueError("Cannot generate entirely constant signatures")
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(participants), cycles_per_person)
    years = np.tile(
        np.array([2022] * (cycles_per_person // 2)
                 + [2024] * (cycles_per_person - cycles_per_person // 2)),
        participants,
    )
    stable = _center_phase_blocks(
        rng.normal(size=(participants, 16))
    ) * stable_sd
    drift = _center_phase_blocks(
        rng.normal(size=(participants, 16))
    ) * long_term_drift_sd
    noise = _center_phase_blocks(
        rng.normal(size=(len(ids), 16))
    ) * day_noise_sd
    values = stable[ids] + noise + drift[ids] * (years == 2024)[:, None]
    return ids, years, values


def fast_permutation_test(ids, years, features, permutations=199,
                          seed=42, stratified=False):
    """Vectorized distance statistic; exact original participant-label null.

    Standardization uses pooled sample variance (ddof=1), matching Aim 1.
    """
    ids = np.asarray(ids)
    years = np.asarray(years)
    values = np.asarray(features, dtype=float)
    if len(ids) != len(values) or len(years) != len(ids):
        raise ValueError("Rows of IDs, intervals and features must match")
    if permutations < 1 or len(np.unique(ids)) < 2:
        raise ValueError("Require permutations and >=2 participants")
    variance = values.var(axis=0, ddof=1)
    if not np.isfinite(variance).all() or (variance <= 0).any():
        raise ValueError("Nonfinite or constant signature feature")
    standardized = values / np.sqrt(variance)
    distances = pdist(standardized, metric="euclidean")
    a, b = np.triu_indices(len(ids), 1)

    def advantage(labels):
        same = labels[a] == labels[b]
        if not same.any() or same.all():
            raise ValueError("Undefined within/between pair comparison")
        return distances[~same].mean() - distances[same].mean()

    observed = advantage(ids)
    rng = np.random.default_rng(seed)
    count_exceed = 0
    unique_years = np.unique(years)
    for _ in range(permutations):
        if stratified:
            shuffled = ids.copy()
            for year in unique_years:
                mask = years == year
                shuffled[mask] = rng.permutation(ids[mask])
        else:
            shuffled = rng.permutation(ids)
        count_exceed += advantage(shuffled) >= observed
    return float(observed), (1 + count_exceed) / (1 + permutations)


def wilson_interval(successes, trials, z=1.95996398454):
    """Wilson binomial interval for simulated rejection proportions."""
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("Invalid successes/trials")
    p = successes / trials
    den = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / den
    half = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials ** 2)) / den
    return float(center - half), float(center + half)


def simulate_power(participants, cycles_per_person, stable_sd,
                   simulations=50, permutations=199, seed=42,
                   stratified=False, drift_sd=0.0, alpha=0.05):
    """Estimate scenario-specific rejection rate and Monte Carlo interval."""
    if simulations < 1 or permutations < 1 or not 0 < alpha < 1:
        raise ValueError("Invalid simulation count, permutations, or alpha")
    rng = np.random.default_rng(seed)
    pvalues = []
    for _ in range(simulations):
        replicate_seed = int(rng.integers(0, 2 ** 32 - 1))
        ids, years, values = synthetic_cycle_signatures(
            participants=participants,
            cycles_per_person=cycles_per_person,
            stable_sd=stable_sd,
            long_term_drift_sd=drift_sd,
            seed=replicate_seed,
        )
        _, p = fast_permutation_test(
            ids, years, values, permutations=permutations,
            seed=replicate_seed, stratified=stratified,
        )
        pvalues.append(p)
    rejects = int((np.asarray(pvalues) < alpha).sum())
    low, high = wilson_interval(rejects, simulations)
    return {
        "participants": participants,
        "cycles_per_person": cycles_per_person,
        "stable_profile_sd": stable_sd,
        "cycle_noise_sd": 1.0,
        "interval_drift_sd": drift_sd,
        "stratified": stratified,
        "simulated_datasets": simulations,
        "permutations_per_dataset": permutations,
        "alpha": alpha,
        "rejections": rejects,
        "rejection_fraction": rejects / simulations,
        "wilson_ci_95_low": low,
        "wilson_ci_95_high": high,
        "meaning": (
            "Type-I error scenario" if stable_sd == 0 else "Power scenario"
        ),
        "caution": (
            "Simulated behavior is conditional on synthetic phase-profile "
            "and noise assumptions; not empirical power for mcPHASES."
        ),
    }
