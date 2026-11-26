"""Unit tests for interval-specific Aim 1 randomization comparisons."""

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mcphases.aim1_panel_inference import (
    _shuffle_cycle_labels,
    _shuffle_later_participant_blocks,
    compare_panels,
    holm_adjust,
)

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/aim1/plot_aim1_distance_boxes.py"
)


def load_plot():
    """Load the project plotting script by path."""
    spec = importlib.util.spec_from_file_location("aim1_plot", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_signatures(seed=31, stable_sd=0.0):
    """Cycle signatures for six people measured twice in both periods."""
    rng = np.random.default_rng(seed)
    records = []
    for identifier in range(1, 7):
        profile = rng.normal(0, stable_sd, 16)
        for interval in (2022, 2024):
            for cycle in (1, 2):
                features = profile + rng.normal(0, 0.35, 16)
                record = {
                    "id": identifier,
                    "study_interval": interval,
                    "cycle_number": cycle,
                }
                record.update(
                    {f"signal_{j}": float(x) for j, x in enumerate(features)}
                )
                records.append(record)
    return pd.DataFrame(records)


def test_holm_adjustment_known_example_and_ties():
    """Holm adjustment controls two tests without selecting lowest p."""
    assert np.allclose(holm_adjust([0.02, 0.06]), [0.04, 0.06])
    assert np.allclose(holm_adjust([0.07, 0.01]), [0.07, 0.02])
    assert np.allclose(holm_adjust([0.02, 0.02]), [0.04, 0.04])
    with pytest.raises(ValueError):
        holm_adjust([-0.01, 0.4])


def test_cycle_permutation_preserves_each_interval_multiplicities():
    """Panel A shuffles cycles but retains 2022/2024 label counts."""
    ids = np.array([1, 1, 2, 2, 1, 2, 2, 3])
    intervals = np.array([2022] * 4 + [2024] * 4)
    changed = _shuffle_cycle_labels(ids, intervals, np.random.default_rng(11))
    for period in (2022, 2024):
        original = np.unique(ids[intervals == period], return_counts=True)
        new = np.unique(changed[intervals == period], return_counts=True)
        assert np.array_equal(original[0], new[0])
        assert np.array_equal(original[1], new[1])


def test_block_permutation_preserves_later_within_person_structure():
    """Panel B must move complete 2024 profiles, not individual cycles."""
    ids = np.array([1, 1, 2, 2, 1, 1, 2, 2, 3, 3])
    intervals = np.array([2022] * 4 + [2024] * 6)
    remapped = _shuffle_later_participant_blocks(
        ids, intervals, np.random.default_rng(2)
    )
    assert np.array_equal(remapped[:4], ids[:4])
    assert remapped[4] == remapped[5]
    assert remapped[6] == remapped[7]
    assert remapped[8] == remapped[9]
    assert len(np.unique(remapped[4:])) == 3


def test_comparisons_reproducible_and_data_are_not_mutated():
    """P-values use actual pair indices and fixed random seeds."""
    plot = load_plot()
    signatures = synthetic_signatures()
    original = signatures.copy(deep=True)
    pairs, _ = plot.summarize_distance_pairs(signatures)
    one, null_one = compare_panels(signatures, pairs, 99, seed=44)
    two, null_two = compare_panels(signatures, pairs, 99, seed=44)
    pd.testing.assert_frame_equal(one, two)
    pd.testing.assert_frame_equal(null_one, null_two)
    pd.testing.assert_frame_equal(signatures, original)
    assert one["contrast"].tolist() == ["within_interval", "cross_interval"]
    assert (one["p_holm_two_contrasts"] >= one["p_one_sided"]).all()
    assert one["loo_n"].ge(5).all()
    assert len(null_one) == 198
    assert all(float(p) >= 0.01 for p in one["p_one_sided"])
    assert int(one["n_same_person_pairs"].iloc[0]) == 12
    assert int(one["n_same_person_pairs"].iloc[1]) == 24


def test_detects_stable_profiles_with_large_synthetic_effect():
    """Strong deliberately simulated identity effects should be detectable."""
    plot = load_plot()
    signatures = synthetic_signatures(stable_sd=8.0)
    pairs, _ = plot.summarize_distance_pairs(signatures)
    summary, _ = compare_panels(signatures, pairs, 299, seed=12)
    assert (summary["similarity_advantage"] > 0).all()
    assert (summary["p_one_sided"] < 0.05).all()
    assert (summary["p_holm_two_contrasts"] < 0.05).all()


def test_plot_annotates_real_comparisons_not_stars(tmp_path):
    """Brackets show exact adjusted p, preserving two-panel styling."""
    plot = load_plot()
    signatures = synthetic_signatures(stable_sd=2.0)
    pairs, _ = plot.summarize_distance_pairs(signatures)
    comparisons, _ = compare_panels(signatures, pairs, 99, seed=37)
    fig = plot.make_figure(pairs, comparisons)
    try:
        assert len(fig.axes) == 2
        for axis in fig.axes:
            assert any("Holm-adjusted p =" in t.get_text() for t in axis.texts)
        paths = plot.save_figure(fig, tmp_path)
        assert all(p.exists() for p in paths)
    finally:
        plt.close(fig)


def test_mismatched_indices_are_rejected():
    """Fail closed rather than annotate a test from different pair rows."""
    plot = load_plot()
    signatures = synthetic_signatures()
    pairs, _ = plot.summarize_distance_pairs(signatures)
    broken = pairs.copy()
    broken.loc[0, "same_person"] = not bool(broken.loc[0, "same_person"])
    with pytest.raises(ValueError, match="Participant identity"):
        compare_panels(signatures, broken, 9)


def test_cross_panel_only_uses_people_present_in_both_intervals():
    """Prevent the cross-interval null from mixing attrition groups."""
    plot = load_plot()
    base = synthetic_signatures(stable_sd=1.0)
    year_2022_only = base.loc[
        (base["id"] == 1) & (base["study_interval"] == 2022)
    ].copy()
    year_2022_only["id"] = 7
    year_2024_only = base.loc[
        (base["id"] == 2) & (base["study_interval"] == 2024)
    ].copy()
    year_2024_only["id"] = 8
    signatures = pd.concat(
        [base, year_2022_only, year_2024_only], ignore_index=True
    )
    pairs, descriptive = plot.summarize_distance_pairs(signatures)
    comparisons, _ = compare_panels(signatures, pairs, 49, seed=10)
    cross = comparisons.set_index("contrast").loc["cross_interval"]
    assert cross["n_participants_in_contrast"] == 6
    assert cross["n_same_person_pairs"] == 24
    assert cross["n_different_person_pairs"] == 120
    assert descriptive.iloc[3]["pair_count_not_independent_n"] > 120

    fig = plot.make_figure(pairs, comparisons)
    try:
        assert "120 pairs" in fig.axes[1].get_xticklabels()[1].get_text()
        assert "24 pairs" in fig.axes[1].get_xticklabels()[0].get_text()
    finally:
        plt.close(fig)


def test_requires_two_periods_and_both_pair_categories():
    """Fail clearly on designs that cannot support both contrasts."""
    plot = load_plot()
    only_2022 = synthetic_signatures().query("study_interval == 2022")
    pairs, _ = plot.summarize_distance_pairs(only_2022)
    with pytest.raises(ValueError, match="exactly two"):
        compare_panels(only_2022, pairs, permutations=10)
