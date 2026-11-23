"""Fast tests of shared Aim 2 output functions without fitting models."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/aim2/hierarchical_shared.py"
)
SPEC = importlib.util.spec_from_file_location(
    "hierarchical_shared", SCRIPT_PATH
)
SHARED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SHARED)


def test_combine_results_preserves_fold_layout(tmp_path):
    primary = tmp_path / "primary"
    longitudinal = tmp_path / "longitudinal"

    for root, names in (
        (primary, ("id1_2022", "id2_2024")),
        (longitudinal, ("id1", "id2")),
    ):
        for index, name in enumerate(names):
            folder = root / name
            folder.mkdir(parents=True)
            pd.DataFrame({"fold": [index]}).to_csv(
                folder / "fold_result.csv", index=False
            )
            pd.DataFrame({"row": [index]}).to_csv(
                folder / "predictions.csv", index=False
            )

    SHARED.combine_results(primary, "id*_*")
    SHARED.combine_results(longitudinal, "id*")

    for root in (primary, longitudinal):
        assert pd.read_csv(root / "all_fold_results.csv")[
            "fold"
        ].tolist() == [0, 1]
        assert pd.read_csv(root / "all_predictions.csv")[
            "row"
        ].tolist() == [0, 1]


def test_probability_extraction_retains_class_and_observation_order():
    probabilities = xr.DataArray(
        np.array(
            [
                [[[0.8, 0.2], [0.3, 0.7]]],
                [[[0.6, 0.4], [0.5, 0.5]]],
            ]
        ),
        dims=("chain", "draw", "obs_id", "phase_dim"),
        coords={"phase_dim": ["Menstrual", "Luteal"]},
    )

    class StubModel:
        def predict(self, idata, **kwargs):
            assert kwargs["kind"] == "response_params"
            assert kwargs["include_group_specific"] is False
            assert len(kwargs["data"]) == 2
            return SimpleNamespace(posterior={"p": probabilities})

    result, labels = SHARED.extract_probabilities(
        StubModel(), None, pd.DataFrame({"x": [1, 2]}), False
    )

    assert labels == ["Menstrual", "Luteal"]
    np.testing.assert_allclose(result, [[0.7, 0.3], [0.4, 0.6]])


def test_probability_extraction_rejects_ambiguous_observation_axis():
    probabilities = xr.DataArray(
        np.ones((1, 1, 2, 2)) / 2,
        dims=("chain", "draw", "row", "class"),
    )

    class StubModel:
        def predict(self, idata, **kwargs):
            return SimpleNamespace(posterior={"p": probabilities})

    with pytest.raises(RuntimeError, match="observation dimension"):
        SHARED.extract_probabilities(StubModel(), None, pd.DataFrame(), True)


def test_aligned_log_loss_respects_category_order():
    """Bambi coordinates must determine probability-column order."""
    truth = np.array(["Menstrual", "Luteal"])
    labels = ["Menstrual", "Luteal"]  # intentionally not sorted
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8]])
    observed = SHARED.aligned_log_loss(truth, probabilities, labels)
    assert observed == pytest.approx(-np.log([0.9, 0.8]).mean())

    # Reversing the matrix columns and labels must preserve the score.
    reversed_score = SHARED.aligned_log_loss(
        truth, probabilities[:, ::-1], labels[::-1]
    )
    assert reversed_score == pytest.approx(observed)


def test_aligned_log_loss_rejects_invalid_probabilities_and_labels():
    with pytest.raises(ValueError, match="valid categorical"):
        SHARED.aligned_log_loss(
            ["Menstrual"], [[0.5, 0.6]], ["Menstrual", "Luteal"]
        )
    with pytest.raises(ValueError, match="absent"):
        SHARED.aligned_log_loss(
            ["Follicular"], [[0.5, 0.5]], ["Menstrual", "Luteal"]
        )
