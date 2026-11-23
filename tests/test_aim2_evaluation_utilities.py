"""Regression checks on actual Aim 2 evaluation and resampling functions."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/aim2/evaluate_primary_predictions.py"
FUNCTIONS = {
    "probability_columns", "brier_score", "true_phase_probability",
    "calculate_metrics", "prepare_paired", "paired_metric_values",
    "cluster_bootstrap", "bootstrap_summary",
}
CONSTANTS = {"PHASES", "MODELS", "KEY_COLUMNS", "N_BOOTSTRAPS", "SEED"}


def load_evaluation_module():
    """Execute the exact pure helper bodies in an isolated namespace."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in CONSTANTS
            for target in node.targets
        ):
            selected.append(node)
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS:
            selected.append(node)
    found = {
        node.name for node in selected if isinstance(node, ast.FunctionDef)
    }
    assert found == FUNCTIONS
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(ast.Module(
        body=[future, *selected], type_ignores=[]
    ))
    ns = {
        "pd": pd, "np": np, "accuracy_score": accuracy_score,
        "balanced_accuracy_score": balanced_accuracy_score,
        "f1_score": f1_score,
    }
    exec(compile(module, str(SOURCE), "exec"), ns)
    return SimpleNamespace(**ns)


@pytest.fixture
def evaluation():
    return load_evaluation_module()


def make_predictions():
    """Build two models' fully paired held-out predictions."""
    rows = []
    phases = ["Menstrual", "Follicular", "Fertility", "Luteal"]
    for target in (1, 2, 3):
        for day in (10, 11):
            phase = phases[(target + day) % 4]
            for model in ("population_only", "hybrid_personalized"):
                probs = {f"prob_{p.lower()}": 0.1 for p in phases}
                probs[f"prob_{phase.lower()}"] = 0.7
                rows.append({
                    "target_id": target,
                    "target_interval": 2022,
                    "source_id": target,
                    "study_interval": 2022,
                    "day_in_study": day,
                    "cycle_number": 1,
                    "interval_cycle_index": 3,
                    "phase": phase,
                    "model": model,
                    "predicted_phase": phase,
                    **probs,
                })
    return pd.DataFrame(rows)


def test_brier_and_log_loss_for_perfect_predictions(evaluation):
    phases = evaluation.PHASES
    rows = []
    for phase in phases:
        values = {f"prob_{p.lower()}": float(p == phase) for p in phases}
        rows.append({"phase": phase, "predicted_phase": phase, **values})
    result = evaluation.calculate_metrics(pd.DataFrame(rows))
    assert result["accuracy"] == 1
    assert result["balanced_accuracy"] == 1
    assert result["macro_f1"] == 1
    assert result["brier_score"] == 0
    assert result["log_loss"] == pytest.approx(0)
    assert result["mean_true_phase_probability"] == 1


def test_preparation_rejects_both_directions_of_mismatched_pairs(evaluation):
    predictions = make_predictions()
    assert len(evaluation.prepare_paired(predictions)) == 6
    missing_hybrid = predictions.loc[~(
        (predictions["model"] == "hybrid_personalized")
        & (predictions["target_id"] == 3)
        & (predictions["day_in_study"] == 11)
    )]
    with pytest.raises(RuntimeError, match="not perfectly matched"):
        evaluation.prepare_paired(missing_hybrid)
    missing_population = predictions.loc[~(
        (predictions["model"] == "population_only")
        & (predictions["target_id"] == 2)
        & (predictions["day_in_study"] == 10)
    )]
    with pytest.raises(RuntimeError, match="not perfectly matched"):
        evaluation.prepare_paired(missing_population)


def test_bootstrap_draws_entire_participant_clusters(evaluation):
    paired = evaluation.prepare_paired(make_predictions())
    # Compare the implementation against an independently seeded reference
    # that samples participant IDs as units (not individual held-out days).
    evaluation.cluster_bootstrap.__globals__["N_BOOTSTRAPS"] = 15
    observed = evaluation.cluster_bootstrap(paired)
    assert len(observed) == 15
    rng = np.random.default_rng(evaluation.SEED)
    for iteration in range(15):
        sampled_ids = rng.choice([1, 2, 3], size=3, replace=True)
        reference = pd.concat(
            [paired.loc[paired["target_id"] == x] for x in sampled_ids],
            ignore_index=True,
        )
        assert len(reference) == 6  # 2 complete days for each selected cluster
        expected = evaluation.paired_metric_values(reference)
        for metric, value in expected.items():
            assert observed.loc[iteration, metric] == pytest.approx(value)


def test_bootstrap_summary_matches_percentiles(evaluation):
    synthetic = pd.DataFrame({"accuracy": [-0.3, 0.0, 0.1, 0.4]})
    table = evaluation.bootstrap_summary({"accuracy": 0.05}, synthetic)
    assert table.loc[0, "hybrid_minus_population"] == 0.05
    assert table.loc[0, "ci_2.5"] == pytest.approx(
        np.quantile(synthetic["accuracy"], 0.025)
    )
    assert table.loc[0, "ci_97.5"] == pytest.approx(
        np.quantile(synthetic["accuracy"], 0.975)
    )
