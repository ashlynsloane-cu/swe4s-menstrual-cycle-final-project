"""Run exact Aim 2 model utility functions without sampling posteriors.

The real source function bodies are compiled from the scripts' AST, so these
regression checks avoid importing Bambi/PyMC and avoid fitting any model.
This does not prove model posterior equivalence after refactoring.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RUNNERS = (
    "run_primary_hierarchical_models.py",
    "run_longitudinal_hierarchical_models.py",
)
NAMES = {
    "add_model_group",
    "standardize_fold",
    "multiclass_brier",
    "evaluate",
    "prediction_table",
}
CONSTANTS = {
    "PHASES",
    "RAW_COLUMNS",
    "RAW_Z_COLUMNS",
    "WITHIN_COLUMNS",
    "WITHIN_Z_COLUMNS",
    "FORMULA",
}


def load_model_utilities(filename):
    """Load real pure source functions without importing model toolchains."""
    source = (ROOT / "scripts" / "aim2" / filename).read_text()
    tree = ast.parse(source)
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id in CONSTANTS
                for target in node.targets
            ):
                nodes.append(node)
        if isinstance(node, ast.FunctionDef) and node.name in NAMES:
            nodes.append(node)
    shared = ROOT / "scripts/aim2/hierarchical_shared.py"
    shared_tree = ast.parse(shared.read_text(encoding="utf-8"))
    helpers = [
        node for node in shared_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "aligned_log_loss"
    ]
    assert len(helpers) == 1
    nodes.extend(helpers)
    found = {node.name for node in nodes if isinstance(node, ast.FunctionDef)}
    assert found == NAMES | {"aligned_log_loss"}, (
        "Expected the model utilities to remain available"
    )
    # Postponed annotations allow a small, dependency-free namespace.
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, *nodes], type_ignores=[])
    )
    namespace = {
        "np": np,
        "pd": pd,
        "StandardScaler": StandardScaler,
        "accuracy_score": accuracy_score,
        "log_loss": log_loss,
    }
    exec(compile(module, filename, "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.fixture(params=RUNNERS)
def utilities(request):
    return load_model_utilities(request.param)


def make_fold(utilities, rhr, within, *, source_id=1, year=2022):
    values = {
        "resting_heart_rate": rhr,
        "hrv_rmssd": rhr + 2,
        "nightly_temperature": rhr / 10,
        "respiratory_rate": rhr / 5,
        "rhr_within": within,
        "hrv_within": within / 2,
        "temp_within": within / 10,
        "resp_within": within * 2,
        "source_id": source_id,
        "study_interval": year,
    }
    return pd.DataFrame([values])


def test_no_calibration_or_test_leakage_in_scaler(utilities):
    population = pd.concat(
        [
            make_fold(utilities, rhr, within)
            for rhr, within in [(60, 0), (70, 2), (80, 4)]
        ],
        ignore_index=True,
    )
    calibration = make_fold(utilities, 500, 100)
    test = make_fold(utilities, 1000, 200)
    before_pop = population.copy(deep=True)
    before_cal = calibration.copy(deep=True)

    p1, c1, t1 = utilities.standardize_fold(population, calibration, test)
    _, c2, _ = utilities.standardize_fold(
        population, make_fold(utilities, -500, -100), test
    )
    assert p1[utilities.RAW_Z_COLUMNS].mean().abs().max() < 1e-12
    assert p1[utilities.WITHIN_Z_COLUMNS].mean().abs().max() < 1e-12
    assert t1["rhr_z"].iloc[0] > 10  # not scaled using test statistics
    assert c1["rhr_z"].iloc[0] != c2["rhr_z"].iloc[0]
    pd.testing.assert_frame_equal(population, before_pop)
    pd.testing.assert_frame_equal(calibration, before_cal)


def test_group_id_uses_participant_and_interval_without_mutation(utilities):
    data = pd.DataFrame({
        "source_id": [5, 5, 8], "study_interval": [2022, 2024, 2022]
    })
    grouped = utilities.add_model_group(data)
    assert grouped["model_id"].astype(str).tolist() == [
        "5_2022", "5_2024", "8_2022"
    ]
    assert str(grouped["model_id"].dtype) == "category"
    assert "model_id" not in data


def test_brier_and_log_loss_match_known_probabilities(utilities):
    labels = ["Menstrual", "Luteal"]
    truth = np.array(["Menstrual", "Luteal"])
    probs = np.array([[0.9, 0.1], [0.2, 0.8]])
    expected_brier = (2 * 0.1 ** 2 + 2 * 0.2 ** 2) / 2
    assert utilities.multiclass_brier(truth, probs, labels) == pytest.approx(
        expected_brier
    )
    score = utilities.evaluate(pd.DataFrame({"phase": truth}), probs, labels)
    assert score["accuracy"] == 1.0
    assert score["log_loss"] == pytest.approx(-np.mean(np.log([0.9, 0.8])))
    assert score["brier_score"] == pytest.approx(expected_brier)


def test_probability_and_prediction_column_order_is_stable(utilities):
    test = pd.DataFrame({
        "source_id": [7, 7], "study_interval": [2024, 2024],
        "day_in_study": [905, 906], "phase": ["Luteal", "Menstrual"],
    })
    probs = np.array([[0.1, 0.9], [0.8, 0.2]])
    result = utilities.prediction_table(
        test, probs, ["Menstrual", "Luteal"], "population_only", 7, 2024
    )
    assert result["target_id"].tolist() == [7, 7]
    assert result["target_interval"].tolist() == [2024, 2024]
    assert result["predicted_phase"].tolist() == ["Luteal", "Menstrual"]
    assert result["prob_menstrual"].tolist() == [0.1, 0.8]
    assert result["prob_luteal"].tolist() == [0.9, 0.2]
    assert test.columns.tolist() == [
        "source_id", "study_interval", "day_in_study", "phase"
    ]
