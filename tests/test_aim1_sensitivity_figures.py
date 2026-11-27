"""Test Aim 1 sensitivity analysis visualizations."""

import importlib.util
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mcphases.aim1_panel_inference import compare_panels

matplotlib.use("Agg")

SCRIPT_DIR = (
    Path(__file__).resolve().parents[1]
    / "scripts/aim1"
)

SIGNALS = (
    "resting_heart_rate",
    "hrv_rmssd",
    "nightly_temperature",
    "respiratory_rate",
)

PHASES = (
    "menstrual",
    "follicular",
    "fertility",
    "luteal",
)


def load_script(name):
    """Import a standalone plotting script."""
    path = SCRIPT_DIR / f"{name}.py"

    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def example_signatures():
    """Generate synthetic repeated-cycle signatures."""
    rng = np.random.default_rng(42)

    features = [
        f"{signal}_{phase}"
        for signal in SIGNALS
        for phase in PHASES
    ]

    rows = []

    for participant in range(1, 5):
        profile = rng.normal(size=len(features))

        for interval in (2022, 2024):
            for cycle in (1, 2):
                values = profile + rng.normal(
                    scale=0.5,
                    size=len(features),
                )

                row = {
                    "id": participant,
                    "study_interval": interval,
                    "cycle_number": cycle,
                }

                row.update(zip(features, values))
                rows.append(row)

    return pd.DataFrame(rows)


def test_metrics_use_same_pairs():
    """Both metrics must compare identical cycle pairs."""
    plot = load_script("plot_aim1_distance_boxes")
    signatures = example_signatures()

    euclidean, _ = plot.summarize_distance_pairs(
        signatures,
        distance_metric="standardized_euclidean",
    )

    mahalanobis, _ = plot.summarize_distance_pairs(
        signatures,
        distance_metric="regularized_mahalanobis",
    )

    columns = [
        "index_a",
        "index_b",
        "same_person",
        "same_interval",
    ]

    pd.testing.assert_frame_equal(
        euclidean[columns],
        mahalanobis[columns],
    )

    assert np.isfinite(mahalanobis["distance"]).all()
    assert (mahalanobis["distance"] >= 0).all()

    assert not np.allclose(
        euclidean["distance"],
        mahalanobis["distance"],
    )


def test_mahalanobis_panel_inference():
    """Mahalanobis supports the interval-specific tests."""
    plot = load_script("plot_aim1_distance_boxes")
    signatures = example_signatures()

    pairs, _ = plot.summarize_distance_pairs(
        signatures,
        distance_metric="regularized_mahalanobis",
    )

    comparisons, nulls = compare_panels(
        signatures,
        pairs,
        permutations=19,
        seed=42,
    )

    assert len(comparisons) == 2
    assert len(nulls) == 38

    assert comparisons[
        "p_holm_two_contrasts"
    ].between(0, 1).all()

    fig = plot.make_figure(
        pairs,
        comparisons,
        distance_metric="regularized_mahalanobis",
    )

    try:
        assert len(fig.axes) == 2
        assert "Mahalanobis" in fig.axes[0].get_ylabel()
    finally:
        plt.close(fig)


def test_metric_comparison_plot():
    """Check signal ordering and combined-signals emphasis."""
    plot = load_script("plot_aim1_metric_comparison")

    results = pd.DataFrame(
        {
            "signal": list(SIGNALS) + ["all_signals"],
            "similarity_advantage": [
                0.17, 0.10, 0.12, -0.06, 0.11
            ],
            "p_value": [
                0.06, 0.25, 0.15, 0.65, 0.23
            ],
        }
    )

    alternative = results.copy()
    alternative["similarity_advantage"] *= 0.8

    comparison = plot.prepare_comparison(
        results,
        alternative,
    )

    assert len(comparison) == 5
    assert comparison.iloc[-1]["signal"] == "all_signals"

    fig = plot.make_figure(comparison)

    try:
        labels = fig.axes[0].get_yticklabels()

        assert labels[-1].get_text() == "All signals"
        assert labels[-1].get_fontweight() == "bold"
    finally:
        plt.close(fig)


def test_feature_correlation_matrix():
    """Check shape, symmetry, diagonal, and valid range."""
    plot = load_script(
        "plot_aim1_feature_correlations"
    )

    matrix = plot.calculate_correlations(
        example_signatures()
    )

    assert matrix.shape == (16, 16)

    np.testing.assert_allclose(
        matrix,
        matrix.T,
    )

    np.testing.assert_allclose(
        np.diag(matrix),
        np.ones(16),
    )

    assert np.isfinite(matrix.to_numpy()).all()
    assert (matrix.abs() <= 1.000001).all().all()


def test_feature_correlation_figure(tmp_path):
    """Check that the heatmap can be rendered and saved."""
    plot = load_script(
        "plot_aim1_feature_correlations"
    )

    matrix = plot.calculate_correlations(
        example_signatures()
    )

    fig = plot.make_figure(matrix)

    try:
        output = tmp_path / "correlations.png"
        fig.savefig(output)

        assert output.is_file()
        assert output.stat().st_size > 0
    finally:
        plt.close(fig)
