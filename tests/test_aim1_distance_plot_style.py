"""Check Aim 1 plot output and Aim 2-consistent visual conventions."""

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/review/plot_aim1_distance_boxes.py"
)


def load_script():
    """Load a standalone script without adding scripts/ to sys.path."""
    spec = importlib.util.spec_from_file_location("aim1_distance_plot", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def example_signatures():
    """Construct 12 synthetic cycles with finite variable features."""
    rng = np.random.default_rng(19)
    rows = []
    for participant in (1, 2, 3):
        profile = rng.normal(size=16)
        for interval in (2022, 2024):
            for cycle in (1, 2):
                row = {
                    "id": participant,
                    "study_interval": interval,
                    "cycle_number": cycle,
                }
                row.update(
                    {
                        f"feature_{index}": value
                        for index, value in enumerate(
                            profile + rng.normal(scale=0.7, size=16)
                        )
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def test_distance_plot_preserves_summary_and_saves_all_formats(tmp_path):
    """Original four-way summary and PNG/PDF/SVG should be present."""
    plot = load_script()
    pairs, summary = plot.summarize_distance_pairs(example_signatures())
    assert len(summary) == 4
    assert int(summary["pair_count_not_independent_n"].sum()) == len(
        pairs
    )

    fig = plot.make_figure(pairs)
    try:
        assert len(fig.axes) == 2
        expected = (plot.SAME_PERSON_COLOR, plot.DIFFERENT_PERSON_COLOR)
        for axis in fig.axes:
            assert len(axis.patches) == 2
            colors = [patch.get_facecolor() for patch in axis.patches]
            for actual, hex_code in zip(colors, expected):
                assert np.allclose(actual[:3], plt.matplotlib.colors.to_rgb(
                    hex_code
                ))
            assert not axis.spines["top"].get_visible()
            assert not axis.spines["right"].get_visible()

        paths = plot.save_figure(fig, tmp_path)
        assert sorted(path.suffix for path in paths) == [
            ".pdf", ".png", ".svg"
        ]
        assert all(
            path.is_file() and path.stat().st_size > 0
            for path in paths
        )
    finally:
        plt.close(fig)


def test_clean_inference_annotation_and_caption():
    """Check adjusted p-values and effects; caveats belong in the caption."""
    plot = load_script()
    pairs, _ = plot.summarize_distance_pairs(example_signatures())
    comparisons = pd.DataFrame(
        {
            "contrast": ["within_interval", "cross_interval"],
            "similarity_advantage": [0.112, 0.269],
            "p_holm_two_contrasts": [0.2282, 0.2188],
            "n_participants_in_contrast": [25, 14],
        }
    )
    fig = plot.make_figure(pairs, comparisons)
    try:
        for axis, p_value, advantage in zip(
            fig.axes,
            ("0.2282", "0.2188"),
            ("+0.112", "+0.269"),
        ):
            axis_text = [item.get_text() for item in axis.texts]
            assert any(p_value in item for item in axis_text)
            assert any(advantage in item for item in axis_text)
            assert any("Mean advantage" in item for item in axis_text)
            assert all(
                "Exploratory one-sided" not in item
                for item in axis_text
            )
        assert len(fig.texts) <= 2
        caption = plot.build_figure_caption(comparisons)
        assert "1.5 × IQR" in caption
        assert "25 participants" in caption
        assert "14 participants" in caption
        assert "reassigned as blocks" in caption
        assert "Holm-adjusted" in caption
    finally:
        plt.close(fig)


def test_descriptive_caption_omits_p_values():
    """Unannotated plots cannot claim a permutation p-value."""
    plot = load_script()
    caption = plot.build_figure_caption(None)
    assert "descriptive only" in caption
    assert "adjusted p" not in caption


def test_cli_writes_reproducible_caption_and_existing_exports(
    tmp_path, monkeypatch
):
    """The caption is generated from inference, not pasted results."""
    plot = load_script()
    signatures = tmp_path / "cycles.csv"
    out = tmp_path / "figures"
    example_signatures().to_csv(signatures, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_aim1_distance_boxes.py",
            "--signatures", str(signatures),
            "--output-dir", str(out),
            "--permutations", "19",
            "--seed", "42",
            "--annotate-p",
        ],
    )
    plot.main()
    assert (out / "aim1_distance_boxes.png").is_file()
    assert (out / "aim1_distance_boxes.pdf").is_file()
    assert (out / "aim1_distance_boxes.svg").is_file()
    assert (out / "aim1_panel_comparisons.csv").is_file()
    caption = (out / "aim1_distance_boxes_caption.md").read_text()
    assert "Holm-adjusted" in caption
    assert "Panel A" in caption and "Panel B" in caption


def test_figure_caption_follows_significance_result():
    """Caption must never hardcode a nonsignificant conclusion."""
    plot = load_script()
    comparisons = pd.DataFrame(
        {
            "contrast": ["within_interval", "cross_interval"],
            "similarity_advantage": [0.5, 0.6],
            "p_holm_two_contrasts": [0.01, 0.2],
            "n_participants_in_contrast": [15, 14],
        }
    )
    caption = plot.build_figure_caption(comparisons)
    assert "Adjusted p < 0.05 for panel(s) A" in caption
    assert "Neither adjusted p-value" not in caption
